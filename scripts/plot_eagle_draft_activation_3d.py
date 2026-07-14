#!/usr/bin/env python
"""ACTIVATION-ONLY 3D distribution visualizations for the EAGLE draft.

Every tensor plotted here is a PRE-LAYER ACTIVATION (the input fed INTO a draft
module, captured immediately before that module executes) during REAL EAGLE tree
generation. NO weights are plotted in any figure produced by this script.

Axes for every 3D figure:
    x = channel index
    y = token index      (= row index of the actual activation matrix the layer
                            saw in that forward call; tree-batched nodes)
    z = abs(activation)

Configs (draft precision; target is fake-W4A4 for C2/C3/C4):
    C2 = pure-R1 draft fp16              (reference)
    C3 = pure-R1 draft fake W4A4         (collapses acceptance)
    C4 = pure-R1 draft fake W8A8         (recovers acceptance)

12 activation types captured per (config, cycle, depth):
    1  embedding_activation          embed_tokens output  (e_R = e@R1)
    2  embedding_branch_activation   fc input [:D]         (embed half of concat)
    3  hidden_branch_activation      fc input [D:]         (hidden half of concat)
    4  projection_input_activation   fc input  (concat[e_R, h_R]/[e_R, f_R])
    5  q_proj_input_activation       q_proj input  (residual stream; no in-norm on layer 0)
    6  k_proj_input_activation       k_proj input
    7  v_proj_input_activation       v_proj input
    8  o_proj_input_activation       o_proj input  (attention output)
    9  up_proj_input_activation      up_proj input  (post_attention_layernorm output)
    10 gate_proj_input_activation    gate_proj input (post_attention_layernorm output)
    11 down_proj_input_activation    down_proj input (MLP intermediate, pre online-Hadamard)
    12 lm_head_input_activation      head input  (final draft feature f_R)

Usage:
  CUDA_VISIBLE_DEVICES=4 python scripts/plot_eagle_draft_activation_3d.py \
      --run-dir runs/eagle_draft_activation_3d_<ts> --num-prompts 6 \
      --max-new-tokens 48 --capture-cycle 1
"""

import argparse, json, os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "EAGLE"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import numpy as np  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: E402,F401  (registers 3d proj)
from eagle_spinquant import (eagle_bridge, experiment, logging_utils, study,  # noqa: E402
                             w4a4_impl_fix as wf, spinquant_draft as spd,
                             fake_w4a4_draft as f4, fake_w8a8_draft as f8)

DEV = "cuda:0"
D = 4096

# The 12 activation types (all PRE-LAYER activations) in canonical order.
ACT_TYPES = [
    "embedding_activation", "embedding_branch_activation", "hidden_branch_activation",
    "projection_input_activation", "q_proj_input_activation", "k_proj_input_activation",
    "v_proj_input_activation", "o_proj_input_activation", "up_proj_input_activation",
    "gate_proj_input_activation", "down_proj_input_activation", "lm_head_input_activation",
]
# activation types that are R1-basis-identical between C2 (R1-only) and C3/C4
# (R1+R2+R4). v/o/down live in an R2/R4-rotated basis in C3/C4, so a direct
# C2-vs-C3 pointwise error there mixes rotation with quantization -> flagged.
BASIS_MATCHED = {"embedding_activation", "embedding_branch_activation",
                 "hidden_branch_activation", "projection_input_activation",
                 "q_proj_input_activation", "k_proj_input_activation",
                 "up_proj_input_activation", "gate_proj_input_activation",
                 "lm_head_input_activation"}
# montage minimum set (section 15)
MONTAGE_MIN = ["projection_input_activation", "q_proj_input_activation",
               "v_proj_input_activation", "o_proj_input_activation",
               "down_proj_input_activation", "lm_head_input_activation"]

# plot downsampling caps (PLOT ONLY; full-res stats saved separately)
MAX_TOK_PLOT = 96
MAX_CH_PLOT = 384


# ----------------------------- statistics --------------------------------- #
def tstats(x):
    x = x.detach().float().reshape(-1, x.shape[-1])
    flat = x.flatten(); a = flat.abs()
    m, s = flat.mean().item(), flat.std().item()
    xc = flat - flat.mean(); var = xc.pow(2).mean()
    kurt = (xc.pow(4).mean() / (var**2 + 1e-12) - 3).item()
    skew = (xc.pow(3).mean() / (var**1.5 + 1e-12)).item()
    return dict(mean=m, std=s, min=flat.min().item(), max=flat.max().item(),
                abs_mean=a.mean().item(), abs_max=a.max().item(),
                median_abs=a.median().item(),
                p95_abs=torch.quantile(a, 0.95).item(),
                p99_abs=torch.quantile(a, 0.99).item(),
                p99_9_abs=torch.quantile(a.float(), 0.999).item(),
                kurtosis=kurt, skewness=skew)


def channel_stats(x):
    x = x.detach().float().reshape(-1, x.shape[-1])
    cabs = x.abs().amax(0); cstd = x.std(0)
    return dict(per_channel_absmax_max=cabs.max().item(),
                per_channel_absmax_median=cabs.median().item(),
                per_channel_std_max=cstd.max().item(),
                per_channel_std_median=cstd.median().item(),
                max_to_median_channel_absmax_ratio=(cabs.max()/(cabs.median()+1e-12)).item(),
                _cabs=cabs.cpu().numpy())


def token_stats(x):
    x = x.detach().float().reshape(-1, x.shape[-1])
    tabs = x.abs().amax(1); tstd = x.std(1)
    return dict(per_token_absmax_max=tabs.max().item(),
                per_token_absmax_median=tabs.median().item(),
                per_token_std_max=tstd.max().item(),
                per_token_std_median=tstd.median().item(), n_tokens=x.shape[0])


def align_err(xq, x):
    xq = xq.detach().float().flatten(); x = x.detach().float().flatten()
    n = min(len(xq), len(x)); xq, x = xq[:n], x[:n]
    e = xq - x
    return dict(rel_l2=(e.norm()/(x.norm()+1e-12)).item(),
                cosine=F.cosine_similarity(xq, x, 0).item(),
                mean_abs_error=e.abs().mean().item(), max_abs_error=e.abs().max().item())


# ----------------------------- plotting ----------------------------------- #
def _prep(Z):
    """abs, downsample tokens/channels for plotting; return (Zds, ch_stride, tok_take)."""
    Z = np.abs(np.asarray(Z, dtype=np.float32))
    if Z.ndim == 1:
        Z = Z[None, :]
    T, C = Z.shape
    tok_take = min(T, MAX_TOK_PLOT)
    Zt = Z[:tok_take]
    ch_stride = max(1, C // MAX_CH_PLOT)
    Zds = Zt[:, ::ch_stride]
    return Zds, ch_stride, tok_take


def plot_3d(Z, title, path):
    Zds, ch_stride, tok_take = _prep(Z)
    T, C = Zds.shape
    X, Y = np.meshgrid(np.arange(C), np.arange(T))
    fig = plt.figure(figsize=(9, 6)); ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Zds, cmap="viridis", linewidth=0, antialiased=False,
                    rcount=min(T, 96), ccount=min(C, 256))
    ax.set_xlabel("channel index" + (f" (/{ch_stride})" if ch_stride > 1 else ""))
    ax.set_ylabel("token index"); ax.set_zlabel("abs(activation)")
    ax.set_title(title, fontsize=8); ax.view_init(elev=32, azim=-58)
    fig.tight_layout(); fig.savefig(path, dpi=115); plt.close(fig)


def plot_heatmap(Z, title, path, vmax=None):
    Zds, ch_stride, _ = _prep(Z)
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(Zds, aspect="auto", cmap="magma", origin="lower", vmax=vmax)
    ax.set_xlabel("channel index" + (f" (/{ch_stride})" if ch_stride > 1 else ""))
    ax.set_ylabel("token index"); ax.set_title(title, fontsize=8)
    fig.colorbar(im, label="abs(activation)")
    fig.tight_layout(); fig.savefig(path, dpi=115); plt.close(fig)


def plot_channel_curve(cabs, title, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.sort(np.asarray(cabs))[::-1], lw=1.3)
    ax.set_yscale("log"); ax.set_xlabel("channel (sorted descending)")
    ax.set_ylabel("per-channel absmax  |activation|")
    ax.set_title(title, fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=115); plt.close(fig)


def _robust_vmax(arrs, pct=99.5):
    """cap at a high percentile so a single spike doesn't wash the panels out."""
    v = max(float(np.percentile(np.abs(z), pct)) for z in arrs)
    return v if v > 0 else 1.0


def montage(zs_by_depth, title, path, kind="heatmap"):
    depths = sorted(zs_by_depth)
    if not depths:
        return
    vmax = _robust_vmax(zs_by_depth.values())
    dtag = f"depths {depths[0]}-{depths[-1]}" + (" (depth 5 not reached at cycle 1)"
                                                 if 5 not in depths else "")
    n = len(depths)
    if kind == "heatmap":
        fig, axes = plt.subplots(1, n, figsize=(3.9*n, 4.2), squeeze=False,
                                 constrained_layout=True)
        for ax, d in zip(axes[0], depths):
            Zds, chs, _ = _prep(zs_by_depth[d])
            im = ax.imshow(Zds, aspect="auto", cmap="magma", origin="lower", vmin=0.0, vmax=vmax)
            ax.set_title(f"depth {d}", fontsize=9)
            ax.set_xlabel("channel"); ax.set_ylabel("token")
        fig.colorbar(im, ax=axes[0], label="abs(activation)  [z clipped at p99.5]",
                     shrink=0.85, pad=0.02)
    else:
        fig = plt.figure(figsize=(4.4*n, 4.2), constrained_layout=True)
        for i, d in enumerate(depths):
            Zds, chs, _ = _prep(zs_by_depth[d])
            T, C = Zds.shape
            Xg, Yg = np.meshgrid(np.arange(C), np.arange(T))
            ax = fig.add_subplot(1, n, i+1, projection="3d")
            ax.plot_surface(Xg, Yg, Zds, cmap="viridis", linewidth=0, antialiased=False,
                            rcount=min(T, 64), ccount=min(C, 160), vmin=0.0, vmax=vmax)
            ax.set_zlim(0.0, vmax)   # shared z-scale across depths
            ax.set_title(f"depth {d}", fontsize=9)
            ax.set_xlabel("ch"); ax.set_ylabel("tok"); ax.set_zlabel("|act|")
            ax.view_init(elev=32, azim=-58)
    fig.suptitle(title + f" | {dtag}", fontsize=9)
    fig.savefig(path, dpi=105); plt.close(fig)


# --------------------------- capture manager ------------------------------ #
class Capture:
    """Registers pre-layer activation hooks on the draft. Stores tensors ONLY for
    the target verification cycle (bounded memory); records stats for every
    captured (act_type, cycle, depth)."""

    def __init__(self, adapter, config_name, target_cycle):
        self.ad = adapter
        self.cfg = config_name
        self.target_cycle = target_cycle
        self.handles = []
        self.state = {"depth": 0, "cycle": 0}
        self.store = {}          # (act_type, depth) -> cpu tensor  [tokens, ch]
        self.stat_rows = []      # distribution stats
        self.chan_rows = []      # per-channel stats
        self.tok_rows = []       # per-token stats
        self.cabs = {}           # (act_type, depth) -> full-res per-channel absmax
        self.post_actq = {}      # (act_type, depth) -> post-actquant tensor (C3/C4)

    # -- record one captured pre-layer activation --
    def _record(self, act_type, x):
        d, c = self.state["depth"], self.state["cycle"]
        st = tstats(x)
        self.stat_rows.append(dict(config=self.cfg, activation_type=act_type,
                                   tensor_kind="activation", capture_point="pre_layer_input",
                                   verification_cycle=c, tree_depth=d, **st))
        cs = channel_stats(x)
        self.chan_rows.append(dict(config=self.cfg, activation_type=act_type,
                                   tensor_kind="activation", verification_cycle=c, tree_depth=d,
                                   **{k: v for k, v in cs.items() if not k.startswith("_")}))
        ts = token_stats(x)
        self.tok_rows.append(dict(config=self.cfg, activation_type=act_type,
                                  tensor_kind="activation", verification_cycle=c, tree_depth=d, **ts))
        if c == self.target_cycle and (act_type, d) not in self.store:
            xm = x.detach().float().reshape(-1, x.shape[-1]).cpu()
            self.store[(act_type, d)] = xm
            self.cabs[(act_type, d)] = cs["_cabs"]

    def install(self):
        ad = self.ad
        ea = ad.ea_layer
        # module handles (FakeW4A4Linear for C3/C4 lives in fq_modules; plain
        # nn.Linear for C2). Either way register_forward_hook sees inp[0] = the
        # pre-layer activation.
        fqm = getattr(ad, "fq_modules", None)

        def get(name, parent, attr):
            if fqm and name in fqm:
                return fqm[name]
            return getattr(parent, attr)

        attn = ea.layers[0].self_attn
        mlp = ea.layers[0].mlp
        fc_mod = get("fc", ea, "fc")
        mods = {
            "q_proj_input_activation": get("layers.0.self_attn.q_proj", attn, "q_proj"),
            "k_proj_input_activation": get("layers.0.self_attn.k_proj", attn, "k_proj"),
            "v_proj_input_activation": get("layers.0.self_attn.v_proj", attn, "v_proj"),
            "o_proj_input_activation": get("layers.0.self_attn.o_proj", attn, "o_proj"),
            "gate_proj_input_activation": get("layers.0.mlp.gate_proj", mlp, "gate_proj"),
            "up_proj_input_activation": get("layers.0.mlp.up_proj", mlp, "up_proj"),
            "down_proj_input_activation": get("layers.0.mlp.down_proj", mlp, "down_proj"),
        }

        # embed_tokens: fires BEFORE fc -> read depth/cycle directly from adapter
        def embed_hook(m, inp, out):
            self.state["depth"] = ad._fc_idx
            self.state["cycle"] = ad._cycle
            self._record("embedding_activation", out)
        self.handles.append(ea.embed_tokens.register_forward_hook(embed_hook))

        # fc pre-hook: lock in depth/cycle for the rest of this draft forward
        def fc_pre(m, inp):
            self.state["depth"] = ad._fc_idx
            self.state["cycle"] = ad._cycle
        self.handles.append(fc_mod.register_forward_pre_hook(fc_pre))

        # fc forward hook: projection input + the two concat branches
        def fc_hook(m, inp, out):
            z = inp[0]
            self._record("projection_input_activation", z)
            self._record("embedding_branch_activation", z[..., :D])
            self._record("hidden_branch_activation", z[..., D:])
        self.handles.append(fc_mod.register_forward_hook(fc_hook))

        # the 7 attn/mlp linears
        def mk(act_type, module):
            def hook(m, inp, out):
                self._record(act_type, inp[0])
                # optional pre/post-actquant capture (C3/C4 only)
                if self.state["cycle"] == self.target_cycle and getattr(m, "aq", None) is not None:
                    try:
                        x2 = inp[0].detach().reshape(-1, m.in_features)
                        if getattr(m, "online_had", False):
                            from eagle_spinquant import spinquant_bridge as sb
                            sb.add_spinquant_to_syspath()
                            from utils import hadamard_utils
                            x2 = hadamard_utils.matmul_hadU_cuda(x2, m.had_K, m.K)
                        key = (act_type, self.state["depth"])
                        if key not in self.post_actq:
                            self.post_actq[key] = m.aq(x2).detach().float().cpu()
                    except Exception:
                        pass
            return module.register_forward_hook(hook)
        for at, module in mods.items():
            self.handles.append(mk(at, module))

        # lm_head input = final draft feature (pre-hook on adapter.head)
        def head_pre(m, inp):
            self._record("lm_head_input_activation", inp[0])
        self.handles.append(ad.head.register_forward_pre_hook(head_pre))
        return self

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


# --------------------------- generation helper ---------------------------- #
@torch.no_grad()
def eagle_accept(model, ids, ilen, tree, max_new):
    deltas, prev, out_last = [], ilen, None
    for out in model.ea_generate(ids, temperature=0.0, max_steps=max_new + 8, tree_choices=tree):
        cur = out.shape[1]
        if cur > prev:
            deltas.append(cur - prev); prev = cur
        out_last = out
        if cur - ilen >= max_new:
            break
    return (sum(deltas)/len(deltas)) if deltas else 0.0


def make_draft(kind, model, stash):
    if kind == "C2":
        return spd.SpinquantDraftPureR1Adapter(model, stash, DEV, torch.float16, trace=True)
    if kind == "C3":
        return f4.FakeW4A4DraftAdapter(model, stash, DEV, torch.float16, trace=True).configure_fq(w_bits=4, a_bits=4)
    if kind == "C4":
        return f8.FakeW8A8DraftAdapter(model, stash, DEV, torch.float16, trace=True).configure_fq(w_bits=8, a_bits=8)
    raise ValueError(kind)


# --------------------------------- main ----------------------------------- #
@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--num-prompts", type=int, default=6, help="candidate pool for representative selection")
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--capture-cycle", type=int, default=1)
    ap.add_argument("--representative-prompt-index", type=int, default=-1,
                    help="override auto-selection with a fixed candidate index")
    ap.add_argument("--configs", default="C2,C3,C4")
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    study.assert_gpu_policy()
    assert torch.cuda.device_count() == 1, "one physical GPU per job"

    rd = args.run_dir if os.path.isabs(args.run_dir) else os.path.join(PROJECT_ROOT, args.run_dir)
    subdirs = ["figures_3d", "figures_heatmap", "figures_channel_curves", "stats", "samples"]
    for s in subdirs:
        os.makedirs(os.path.join(rd, s), exist_ok=True)
    with open(os.path.join(rd, "commands.sh"), "a") as f:
        f.write("CUDA_VISIBLE_DEVICES=" + os.environ.get("CUDA_VISIBLE_DEVICES", "?")
                + " " + " ".join(sys.argv) + "\n")
    with open(os.path.join(rd, "environment.txt"), "w") as f:
        f.write(json.dumps(logging_utils.env_summary(), indent=2) + "\n")
    for p in ("git_status.txt", "git_diff.patch"):
        if not os.path.isfile(os.path.join(rd, p)):
            open(os.path.join(rd, p), "w").write("not a git repository\n")

    cfg = experiment.load_config(args.config)
    paths = experiment.resolve_paths(cfg)
    rr = cfg.get("paths", {}).get("rotations_root")
    tmpl = cfg.get("model", {}).get("chat_template", "llama2")
    build_prompt = eagle_bridge.PROMPT_BUILDERS[tmpl]
    prompts = experiment.load_mt_bench_prompts(args.num_prompts)
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]

    print("[act3d] building target full/w4a4 ...", flush=True)
    model, stash, _ = study.build_study_target(
        paths["target_path"], paths["draft_path"], cfg["model"]["target"],
        "full", "random_hadamard", "w4a4", 0, device=DEV, rotations_root=rr)
    tok = eagle_bridge.get_tokenizer(model)
    from eagle.model.choices import mc_sim_7b_63 as tree_full
    tree = [list(p) for p in tree_full]
    study.set_draft_tree(model, tree, DEV)
    # target's unfused tail exposes original post-norm hidden h (needed for pure-R1)
    tail = wf.UnfusedTailAdapter(model, stash).install()

    ids_list = [build_prompt(tok, p["text"]).to(DEV) for p in prompts]

    # ---- Stage A selection: pick prompt where C2 healthy & C3 collapses ----
    if args.representative_prompt_index >= 0:
        rep = args.representative_prompt_index
        sel_rows = [dict(candidate_index=rep, note="fixed via --representative-prompt-index")]
        c2_acc = c3_acc = float("nan")
    else:
        print("[act3d] representative-prompt scan (C2 vs C3) ...", flush=True)
        sel_rows = []
        acc = {"C2": [], "C3": []}
        for k in ("C2", "C3"):
            dr = make_draft(k, model, stash); dr.install()
            for pi, ids in enumerate(ids_list):
                if hasattr(dr, "set_context"):
                    dr.set_context(prompts[pi]["question_id"])
                acc[k].append(eagle_accept(model, ids, ids.shape[1], tree, args.max_new_tokens))
            dr.uninstall()
        for pi in range(len(ids_list)):
            gap = acc["C2"][pi] - acc["C3"][pi]
            sel_rows.append(dict(candidate_index=pi, prompt_id=prompts[pi]["question_id"],
                                 c2_acceptance=round(acc["C2"][pi], 4),
                                 c3_acceptance=round(acc["C3"][pi], 4), c2_minus_c3=round(gap, 4)))
        # prefer C2 healthy (>=2) and C3 collapsed (<=1.6), else max gap
        good = [r for r in sel_rows if r["c2_acceptance"] >= 2.0 and r["c3_acceptance"] <= 1.6]
        pool = good if good else sel_rows
        best = max(pool, key=lambda r: r["c2_minus_c3"])
        rep = best["candidate_index"]
        c2_acc, c3_acc = best["c2_acceptance"], best["c3_acceptance"]
    rep_prompt = prompts[rep]
    rep_ids = ids_list[rep]
    print(f"[act3d] representative candidate index={rep} prompt_id={rep_prompt['question_id']} "
          f"C2={c2_acc} C3={c3_acc} cycle={args.capture_cycle}", flush=True)

    # ---- Stage B capture: real generation, per config ----
    caps = {}
    for k in configs:
        dr = make_draft(k, model, stash); dr.install()
        if hasattr(dr, "set_context"):
            dr.set_context(rep_prompt["question_id"])
        cap = Capture(dr, k, args.capture_cycle).install()
        _ = eagle_accept(model, rep_ids, rep_ids.shape[1], tree, args.max_new_tokens)
        cap.remove(); caps[k] = cap
        dr.uninstall()
        n_types = len({at for (at, d) in cap.store})
        print(f"[act3d] {k}: captured {len(cap.store)} (act_type,depth) tensors, "
              f"{n_types}/12 activation types at cycle {args.capture_cycle}", flush=True)

    tail.uninstall()

    # ---------------------------- figures ---------------------------------- #
    _make_figures(rd, caps, configs, rep_prompt, args.capture_cycle)

    # ------------------------------ CSVs ----------------------------------- #
    all_stat = [r for c in caps.values() for r in c.stat_rows]
    all_chan = [r for c in caps.values() for r in c.chan_rows]
    all_tok = [r for c in caps.values() for r in c.tok_rows]
    logging_utils.write_csv(os.path.join(rd, "stats", "activation_distribution_stats.csv"), all_stat)
    logging_utils.write_csv(os.path.join(rd, "stats", "per_channel_stats.csv"), all_chan)
    logging_utils.write_csv(os.path.join(rd, "stats", "per_token_stats.csv"), all_tok)

    def err_csv(ref, other, fname):
        rows = []
        if ref in caps and other in caps:
            for (at, d), xr in caps[ref].store.items():
                xo = caps[other].store.get((at, d))
                if xo is None or xr.shape != xo.shape:
                    continue
                rows.append(dict(activation_type=at, tree_depth=d,
                                 ref_config=ref, other_config=other,
                                 basis_matched=bool(at in BASIS_MATCHED),
                                 tensor_kind="activation", **align_err(xo, xr)))
        logging_utils.write_csv(os.path.join(rd, "stats", fname), rows)
    err_csv("C2", "C3", "c2_vs_c3_activation_error.csv")
    err_csv("C2", "C4", "c2_vs_c4_activation_error.csv")

    # representative case manifest
    rep_rows = [dict(role="representative_prompt", candidate_index=rep,
                     prompt_id=rep_prompt["question_id"],
                     prompt_text=rep_prompt["text"][:200],
                     c2_acceptance=c2_acc, c3_acceptance=c3_acc,
                     capture_cycle=args.capture_cycle,
                     token_axis_definition="row index of the actual activation matrix the layer saw in that forward (tree-batched nodes)")]
    for r in sel_rows:
        r["role"] = "candidate"
    logging_utils.write_csv(os.path.join(rd, "stats", "representative_case_manifest.csv"), rep_rows + sel_rows)

    # capture manifest: every stored tensor
    man = []
    for k, cap in caps.items():
        for (at, d), x in cap.store.items():
            man.append(dict(config=k, activation_type=at, tensor_kind="activation",
                            capture_point="pre_layer_input", verification_cycle=args.capture_cycle,
                            tree_depth=d, n_tokens=x.shape[0], n_channels=x.shape[1],
                            plot_token_cap=MAX_TOK_PLOT, plot_channel_cap=MAX_CH_PLOT))
    logging_utils.write_csv(os.path.join(rd, "stats", "capture_manifest.csv"), man)

    # save full-res per-channel absmax NPY for plotted tensors (downsample audit)
    for k, cap in caps.items():
        for (at, d), cabs in cap.cabs.items():
            np.save(os.path.join(rd, "samples", f"{k}__{at}__depth{d:02d}__per_channel_absmax.npy"), cabs)

    # ------------------------------ summary -------------------------------- #
    _write_summary_md(rd, caps, configs, rep_prompt, rep, c2_acc, c3_acc, args.capture_cycle)
    print(f"[act3d] DONE -> {rd}", flush=True)
    return 0


def _fig_paths(rd, kind_dir, config, depth, act_type, suffix, prompt_id, cycle):
    base = f"{act_type}__{suffix}__prompt{prompt_id:02d}_cycle{cycle:02d}_depth{depth:02d}.png"
    if kind_dir == "figures_channel_curves":
        d = os.path.join(rd, kind_dir, config)
        name = f"{act_type}__channel_absmax_curve__depth{depth:02d}.png"
    else:
        d = os.path.join(rd, kind_dir, config, f"depth{depth}")
        name = base
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, name)


def _title(config, act_type, depth, prompt_id, cycle, extra=""):
    # the EAGLE draft is a single decoder layer (index 0) + fc + head, so the
    # numeric layer index is 0 for every attn/mlp tap; stated explicitly here.
    return (f"{config} | {act_type} | pre-layer ACTIVATION | draft L0 | abs(value) | "
            f"prompt {prompt_id} | cycle {cycle} | depth {depth}{extra}")


def _make_figures(rd, caps, configs, rep_prompt, cycle):
    pid = int(rep_prompt["question_id"]) % 100
    depths_present = sorted({d for c in caps.values() for (at, d) in c.store})
    # per-type shared vmax at depth 1 for fair overlay
    for k in configs:
        cap = caps[k]
        # 3D + heatmap + channel curve for every (act_type, depth)
        for (at, d), x in cap.store.items():
            Z = x.numpy()
            try:
                plot_3d(Z, _title(k, at, d, pid, cycle),
                        _fig_paths(rd, "figures_3d", k, d, at, "3d", pid, cycle))
            except Exception as e:
                print(f"[act3d] 3d fail {k}/{at}/d{d}: {e}", flush=True)
            try:
                plot_heatmap(Z, _title(k, at, d, pid, cycle),
                             _fig_paths(rd, "figures_heatmap", k, d, at, "heatmap", pid, cycle))
            except Exception as e:
                print(f"[act3d] heat fail {k}/{at}/d{d}: {e}", flush=True)
        # channel curves (one per act_type; use depth 1 if present else min depth)
        by_type = {}
        for (at, d), cabs in cap.cabs.items():
            by_type.setdefault(at, {})[d] = cabs
        for at, dd in by_type.items():
            d0 = 1 if 1 in dd else min(dd)
            try:
                plot_channel_curve(dd[d0], _title(k, at, d0, pid, cycle, " | per-channel absmax"),
                                   _fig_paths(rd, "figures_channel_curves", k, d0, at, "curve", pid, cycle))
            except Exception as e:
                print(f"[act3d] curve fail {k}/{at}: {e}", flush=True)
        # montages depth1..5
        for at in ACT_TYPES:
            zs = {d: cap.store[(at, d)].numpy() for (a2, d) in cap.store if a2 == at and 1 <= d <= 5}
            if len(zs) >= 2:
                try:
                    mdir = os.path.join(rd, "figures_heatmap", "montages", k)
                    os.makedirs(mdir, exist_ok=True)
                    montage(zs, f"{k} | {at} | pre-layer ACTIVATION abs | depth1..5 montage | prompt {pid}",
                            os.path.join(mdir, f"{at}__depth1to5__montage.png"), kind="heatmap")
                except Exception as e:
                    print(f"[act3d] heat-montage fail {k}/{at}: {e}", flush=True)
                if at in MONTAGE_MIN:
                    try:
                        m3 = os.path.join(rd, "figures_3d", "montages", k)
                        os.makedirs(m3, exist_ok=True)
                        montage(zs, f"{k} | {at} | pre-layer ACTIVATION abs | depth1..5 3D montage | prompt {pid}",
                                os.path.join(m3, f"{at}__depth1to5__3d_montage.png"), kind="3d")
                    except Exception as e:
                        print(f"[act3d] 3d-montage fail {k}/{at}: {e}", flush=True)

    # C2 vs C3 vs C4 overlay at depth 1 (identical axis + z-limit) for key types
    for at in set(MONTAGE_MIN + ["projection_input_activation"]):
        panels = {k: caps[k].store.get((at, 1)) for k in configs if (at, 1) in caps[k].store}
        if len(panels) < 2:
            continue
        try:
            vmax = _robust_vmax([v.numpy() for v in panels.values()])   # p99.5, outlier-robust
            odir = os.path.join(rd, "figures_heatmap", "overlays_depth1")
            os.makedirs(odir, exist_ok=True)
            n = len(panels)
            fig, axes = plt.subplots(1, n, figsize=(4.4*n, 4.2), squeeze=False,
                                     constrained_layout=True)
            for ax, (k, v) in zip(axes[0], panels.items()):
                Zds, chs, _ = _prep(v.numpy())
                im = ax.imshow(Zds, aspect="auto", cmap="magma", origin="lower", vmin=0.0, vmax=vmax)
                ax.set_title(k, fontsize=10); ax.set_xlabel("channel"); ax.set_ylabel("token")
            fig.colorbar(im, ax=axes[0], label="abs(activation)  [z clipped at p99.5]",
                         shrink=0.85, pad=0.02)
            fig.suptitle(f"{at} | pre-layer ACTIVATION abs | draft L0 | depth 1 | "
                         f"C2/C3/C4 shared z-limit | prompt {pid}", fontsize=9)
            fig.savefig(os.path.join(odir, f"{at}__C2C3C4_depth1_overlay.png"), dpi=110); plt.close(fig)
        except Exception as e:
            print(f"[act3d] overlay fail {at}: {e}", flush=True)


def _write_summary_md(rd, caps, configs, rep_prompt, rep, c2_acc, c3_acc, cycle):
    import pandas as pd
    stat = pd.DataFrame([r for c in caps.values() for r in c.stat_rows])
    lines = ["# EAGLE draft ACTIVATION-only 3D distribution — run summary", "",
             f"Representative prompt: candidate index {rep}, prompt_id "
             f"{rep_prompt['question_id']} (C2 acc={c2_acc}, C3 acc={c3_acc}); capture cycle={cycle}.",
             "",
             "ALL plotted tensors are PRE-LAYER ACTIVATIONS (input fed into the draft "
             "module, captured before it executes). No weights are plotted.", "",
             "## Activation types captured (per config)"]
    for k in configs:
        n = len({at for (at, d) in caps[k].store})
        depths = sorted({d for (at, d) in caps[k].store})
        lines.append(f"- {k}: {n}/12 activation types; depths captured = {depths}")
    lines += ["", "## Key figure locations",
              "- 3D: `figures_3d/<C>/depth<d>/<act>__3d__promptNN_cycleNN_depthNN.png`",
              "- heatmap: `figures_heatmap/<C>/depth<d>/<act>__heatmap__...png`",
              "- channel curves: `figures_channel_curves/<C>/<act>__channel_absmax_curve__depthNN.png`",
              "- depth montages: `figures_heatmap/montages/<C>/<act>__depth1to5__montage.png`",
              "- C2/C3/C4 overlay: `figures_heatmap/overlays_depth1/<act>__C2C3C4_depth1_overlay.png`", ""]
    open(os.path.join(rd, "summary.md"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    sys.exit(main())

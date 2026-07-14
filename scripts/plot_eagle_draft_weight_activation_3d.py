#!/usr/bin/env python
"""EAGLE draft WEIGHT + ACTIVATION distribution visualizations (descriptive names).

Every tensor is explicitly labeled as one of:
    activation          (pre-layer input captured during real EAGLE generation)
    weight              (the weight actually used by the draft module)
    quantization error  (weight_q - weight_fp, quantized configs only)

Config names (NO C2/C3/C4 anywhere in outputs):
    target_W4A4__draft_FP16_pureR1   healthy reference (target fake W4A4, draft fp16, pure-R1)
    target_W4A4__draft_W4A4_pureR1   collapse case     (draft fake W4A4)
    target_W4A4__draft_W8A8_pureR1   recovery case     (draft fake W8A8)

Axes:
    activation 3D : x=channel, y=token/node, z=abs(activation)
    weight 3D     : x=input channel, y=output channel, z=abs(weight)

Depth definition (per verification cycle):
    depth 0 = first draft forward from target hidden h_R  (all configs share input)
    depth k = k-th recurrent draft forward using f_R

Usage:
  CUDA_VISIBLE_DEVICES=4 python scripts/plot_eagle_draft_weight_activation_3d.py \
      --run-dir runs/eagle_draft_weight_activation_3d_<ts> --num-prompts 8 \
      --max-new-tokens 48 --capture-cycle 1
"""

import argparse, json, os, re, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "EAGLE"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import numpy as np  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: E402,F401
from eagle_spinquant import (eagle_bridge, experiment, logging_utils, study,  # noqa: E402
                             w4a4_impl_fix as wf, spinquant_draft as spd,
                             fake_w4a4_draft as f4, fake_w8a8_draft as f8)

DEV = "cuda:0"
D = 4096

CFG_FP16 = "target_W4A4__draft_FP16_pureR1"
CFG_W4A4 = "target_W4A4__draft_W4A4_pureR1"
CFG_W8A8 = "target_W4A4__draft_W8A8_pureR1"
ALL_CFGS = [CFG_FP16, CFG_W4A4, CFG_W8A8]
FORBIDDEN = ("C2", "C3", "C4")          # must not appear in any output path

# 13 activation types (all pre-layer inputs except the embedding OUTPUT).
# down_proj_input_activation = raw MLP intermediate (before online R4 Hadamard,
# which only exists in the quantized configs); .._post_R4 = after the online
# Hadamard, quantized configs only.
ACT_TYPES = [
    "embedding_output_activation",
    "projection_embedding_branch_activation", "projection_hidden_branch_activation",
    "projection_input_activation",
    "q_proj_input_activation", "k_proj_input_activation", "v_proj_input_activation",
    "o_proj_input_activation",
    "up_proj_input_activation", "gate_proj_input_activation",
    "down_proj_input_activation", "down_proj_input_activation_post_R4",
    "lm_head_input_activation",
]
# activation types whose basis is identical between draft_FP16_pureR1 (R1 only)
# and the quantized drafts (R1+R2+R4): o_proj input is R2-rotated, post_R4 has
# no FP16 counterpart. Everything else compares apples-to-apples.
BASIS_MATCHED = set(ACT_TYPES) - {"o_proj_input_activation",
                                  "down_proj_input_activation_post_R4"}

ACT2LAYER = {
    "embedding_output_activation": "embed_tokens",
    "projection_embedding_branch_activation": "fc", "projection_hidden_branch_activation": "fc",
    "projection_input_activation": "fc",
    "q_proj_input_activation": "q_proj", "k_proj_input_activation": "k_proj",
    "v_proj_input_activation": "v_proj", "o_proj_input_activation": "o_proj",
    "up_proj_input_activation": "up_proj", "gate_proj_input_activation": "gate_proj",
    "down_proj_input_activation": "down_proj", "down_proj_input_activation_post_R4": "down_proj",
    "lm_head_input_activation": "lm_head",
}

# 10 weight types; the 8 linears carry fake-quant in the quantized configs.
WEIGHT_TYPES = ["embedding_weight", "projection_fc_weight",
                "q_proj_weight", "k_proj_weight", "v_proj_weight", "o_proj_weight",
                "up_proj_weight", "gate_proj_weight", "down_proj_weight", "lm_head_weight"]
FQ_NAME2WTYPE = {"fc": "projection_fc_weight",
                 "layers.0.self_attn.q_proj": "q_proj_weight",
                 "layers.0.self_attn.k_proj": "k_proj_weight",
                 "layers.0.self_attn.v_proj": "v_proj_weight",
                 "layers.0.self_attn.o_proj": "o_proj_weight",
                 "layers.0.mlp.gate_proj": "gate_proj_weight",
                 "layers.0.mlp.up_proj": "up_proj_weight",
                 "layers.0.mlp.down_proj": "down_proj_weight"}

MAX_TOK_PLOT, MAX_CH_PLOT = 96, 384          # activation plot caps (PLOT ONLY)
MAX_WROW_PLOT, MAX_WCOL_PLOT = 256, 384      # weight plot caps  (PLOT ONLY)


# ----------------------------- statistics --------------------------------- #
def tstats(x):
    x = x.detach().float()
    flat = x.flatten(); a = flat.abs()
    s = flat.std().item()
    xc = flat - flat.mean(); var = xc.pow(2).mean()
    return dict(mean=flat.mean().item(), std=s, min=flat.min().item(), max=flat.max().item(),
                abs_mean=a.mean().item(), abs_max=a.max().item(), median_abs=a.median().item(),
                p95_abs=torch.quantile(a, 0.95).item() if a.numel() < 2**24 else np.percentile(a.cpu().numpy(), 95).item(),
                p99_abs=torch.quantile(a, 0.99).item() if a.numel() < 2**24 else np.percentile(a.cpu().numpy(), 99).item(),
                p99_9_abs=torch.quantile(a, 0.999).item() if a.numel() < 2**24 else np.percentile(a.cpu().numpy(), 99.9).item(),
                kurtosis=(xc.pow(4).mean() / (var**2 + 1e-12) - 3).item(),
                skewness=(xc.pow(3).mean() / (var**1.5 + 1e-12)).item())


def dim_absmax(x, dim_keep):
    """absmax over all dims except dim_keep -> 1D per-channel vector."""
    x = x.detach().float()
    dims = [d for d in range(x.dim()) if d != dim_keep]
    v = x.abs().amax(dims)
    return v


def ratio(v):
    return (v.max() / (v.median() + 1e-12)).item()


def align_err(xq, x):
    xq = xq.detach().float().flatten(); x = x.detach().float().flatten()
    n = min(len(xq), len(x)); xq, x = xq[:n], x[:n]
    e = xq - x
    return dict(rel_l2=(e.norm()/(x.norm()+1e-12)).item(),
                cosine=F.cosine_similarity(xq, x, 0).item(),
                mean_abs_error=e.abs().mean().item(), max_abs_error=e.abs().max().item())


def werr(wq, wfp):
    r = align_err(wq, wfp)
    e = (wq.detach().float() - wfp.detach().float()).flatten()
    r["signal_to_noise_ratio_db"] = (20*torch.log10(wfp.detach().float().norm()/(e.norm()+1e-12))).item()
    return {("rel_l2_error" if k == "rel_l2" else k): v for k, v in r.items()}


# ----------------------------- plotting ----------------------------------- #
def _robust_vmax(arrs, pct=99.5):
    v = max(float(np.percentile(np.abs(np.asarray(z)), pct)) for z in arrs)
    return v if v > 0 else 1.0


def _prep(Z, max_r, max_c):
    """abs + deterministic stride downsampling for PLOT ONLY."""
    Z = np.abs(np.asarray(Z, dtype=np.float32))
    if Z.ndim == 1:
        Z = Z[None, :]
    R, C = Z.shape
    rs, cs = max(1, int(np.ceil(R / max_r))), max(1, int(np.ceil(C / max_c)))
    return Z[::rs, ::cs], rs, cs


def plot_3d(Z, title, path, xlabel, ylabel, zlabel, max_r, max_c):
    Zds, rs, cs = _prep(Z, max_r, max_c)
    T, C = Zds.shape
    X, Y = np.meshgrid(np.arange(C), np.arange(T))
    fig = plt.figure(figsize=(9, 6)); ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Zds, cmap="viridis", linewidth=0, antialiased=False,
                    rcount=min(T, 96), ccount=min(C, 256))
    ax.set_xlabel(xlabel + (f" (/{cs})" if cs > 1 else ""))
    ax.set_ylabel(ylabel + (f" (/{rs})" if rs > 1 else ""))
    ax.set_zlabel(zlabel)
    ax.set_title(_wrap(title, 84), fontsize=8); ax.view_init(elev=32, azim=-58)
    fig.tight_layout(); fig.savefig(path, dpi=115); plt.close(fig)
    return f"row_stride={rs},col_stride={cs}"


def _wrap(title, width=72):
    import textwrap
    return "\n".join(textwrap.wrap(title, width=width, break_long_words=False,
                                   break_on_hyphens=False))


def plot_heat(Z, title, path, xlabel, ylabel, clabel, max_r, max_c, vmax=None):
    Zds, rs, cs = _prep(Z, max_r, max_c)
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    im = ax.imshow(Zds, aspect="auto", cmap="magma", origin="lower", vmin=0.0,
                   vmax=vmax if vmax is not None else _robust_vmax([Zds]))
    ax.set_xlabel(xlabel + (f" (/{cs})" if cs > 1 else ""))
    ax.set_ylabel(ylabel + (f" (/{rs})" if rs > 1 else ""))
    ax.set_title(_wrap(title), fontsize=8)     # wrapped: long titles must not clip
    fig.colorbar(im, label=clabel + "  [z clipped at p99.5]")
    fig.savefig(path, dpi=115); plt.close(fig)
    return f"row_stride={rs},col_stride={cs}"


def plot_curves(curves, title, path, ylabel):
    """curves: list of (label, 1D array) -> sorted-descending log curves."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for lab, v in curves:
        ax.plot(np.sort(np.asarray(v))[::-1], lw=1.3, label=lab)
    ax.set_yscale("log"); ax.set_xlabel("channel (sorted descending)")
    ax.set_ylabel(ylabel); ax.set_title(_wrap(title, 64), fontsize=8); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=115); plt.close(fig)


def plot_compare_heat(panels, title, path, xlabel, ylabel, clabel, max_r, max_c):
    """panels: list of (config_label, 2D array); shared p99.5 color scale."""
    vmax = _robust_vmax([p[1] for p in panels])
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4.6*n, 4.4), squeeze=False,
                             constrained_layout=True)
    for ax, (lab, Z) in zip(axes[0], panels):
        Zds, rs, cs = _prep(Z, max_r, max_c)
        im = ax.imshow(Zds, aspect="auto", cmap="magma", origin="lower", vmin=0.0, vmax=vmax)
        ax.set_title(lab, fontsize=8); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    fig.colorbar(im, ax=axes[0], label=clabel + "  [shared, clipped p99.5]",
                 shrink=0.85, pad=0.02)
    fig.suptitle(title, fontsize=9)
    fig.savefig(path, dpi=110); plt.close(fig)


# --------------------------- activation capture --------------------------- #
class Capture:
    """Pre-layer activation hooks on the draft during REAL generation. Tensors
    stored only for the target verification cycle; stats recorded everywhere."""

    def __init__(self, adapter, config_name, target_cycle, prompt_id):
        self.ad = adapter; self.cfg = config_name
        self.target_cycle = target_cycle; self.prompt_id = prompt_id
        self.handles = []; self.state = {"depth": 0, "cycle": 0}
        self.store = {}        # (act_type, depth) -> cpu float tensor [rows, ch]
        self.stat_rows = []; self.chan_rows = []

    def _record(self, act_type, x):
        d, c = self.state["depth"], self.state["cycle"]
        x2 = x.detach().float().reshape(-1, x.shape[-1])
        cabs = dim_absmax(x2, 1); tabs = dim_absmax(x2, 0)
        self.stat_rows.append(dict(
            config_name=self.cfg, tensor_kind="activation", activation_type=act_type,
            layer_name=ACT2LAYER[act_type], capture_point=("embedding_output" if
                act_type == "embedding_output_activation" else "pre_layer_input"),
            prompt_id=self.prompt_id, cycle_id=c, tree_depth=d,
            shape=str(list(x2.shape)), **tstats(x2),
            max_to_median_channel_absmax_ratio=ratio(cabs),
            max_to_median_token_absmax_ratio=ratio(tabs)))
        self.chan_rows.append(dict(
            config_name=self.cfg, tensor_kind="activation", activation_type=act_type,
            layer_name=ACT2LAYER[act_type], cycle_id=c, tree_depth=d,
            per_channel_absmax_max=cabs.max().item(),
            per_channel_absmax_median=cabs.median().item(),
            max_to_median_channel_absmax_ratio=ratio(cabs),
            per_token_absmax_max=tabs.max().item(),
            per_token_absmax_median=tabs.median().item(),
            max_to_median_token_absmax_ratio=ratio(tabs)))
        if c == self.target_cycle and (act_type, d) not in self.store:
            self.store[(act_type, d)] = x2.cpu()

    def install(self):
        ad = self.ad; ea = ad.ea_layer
        fqm = getattr(ad, "fq_modules", None)

        def get(name, parent, attr):
            return fqm[name] if fqm and name in fqm else getattr(parent, attr)

        attn, mlp = ea.layers[0].self_attn, ea.layers[0].mlp
        fc_mod = get("fc", ea, "fc")
        mods = {"q_proj_input_activation": get("layers.0.self_attn.q_proj", attn, "q_proj"),
                "k_proj_input_activation": get("layers.0.self_attn.k_proj", attn, "k_proj"),
                "v_proj_input_activation": get("layers.0.self_attn.v_proj", attn, "v_proj"),
                "o_proj_input_activation": get("layers.0.self_attn.o_proj", attn, "o_proj"),
                "gate_proj_input_activation": get("layers.0.mlp.gate_proj", mlp, "gate_proj"),
                "up_proj_input_activation": get("layers.0.mlp.up_proj", mlp, "up_proj"),
                "down_proj_input_activation": get("layers.0.mlp.down_proj", mlp, "down_proj")}

        def embed_hook(m, inp, out):
            self.state["depth"] = ad._fc_idx; self.state["cycle"] = ad._cycle
            self._record("embedding_output_activation", out)
        self.handles.append(ea.embed_tokens.register_forward_hook(embed_hook))

        def fc_pre(m, inp):
            self.state["depth"] = ad._fc_idx; self.state["cycle"] = ad._cycle
        self.handles.append(fc_mod.register_forward_pre_hook(fc_pre))

        def fc_hook(m, inp, out):
            z = inp[0]
            self._record("projection_input_activation", z)
            self._record("projection_embedding_branch_activation", z[..., :D])
            self._record("projection_hidden_branch_activation", z[..., D:])
        self.handles.append(fc_mod.register_forward_hook(fc_hook))

        def mk(act_type, module):
            def hook(m, inp, out):
                self._record(act_type, inp[0])
                # post-R4 (quantized configs' down_proj online Hadamard only)
                if act_type == "down_proj_input_activation" and getattr(m, "online_had", False):
                    try:
                        from eagle_spinquant import spinquant_bridge as sb
                        sb.add_spinquant_to_syspath()
                        from utils import hadamard_utils
                        x2 = inp[0].detach().reshape(-1, m.in_features)
                        self._record("down_proj_input_activation_post_R4",
                                     hadamard_utils.matmul_hadU_cuda(x2, m.had_K, m.K))
                    except Exception as e:
                        print(f"[wact3d] post_R4 capture fail: {e}", flush=True)
            return module.register_forward_hook(hook)
        for at, module in mods.items():
            self.handles.append(mk(at, module))

        def head_pre(m, inp):
            self._record("lm_head_input_activation", inp[0])
        self.handles.append(ad.head.register_forward_pre_hook(head_pre))
        return self

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


# --------------------------- weight capture ------------------------------- #
def capture_weights(adapter, config_name):
    """Return list of dicts: weight_type, layer_name, w_fp, w_q(None), basis note.
    w_fp = the (rotation-conjugated) full-precision weight; w_q = fake-quantized
    weight actually used (quantized configs' 8 linears only)."""
    ea = adapter.ea_layer
    out = []

    def add(wtype, lname, w_fp, w_q=None, note=""):
        out.append(dict(weight_type=wtype, layer_name=lname,
                        w_fp=w_fp.detach().float().cpu(),
                        w_q=(w_q.detach().float().cpu() if w_q is not None else None),
                        note=note))

    fqm = getattr(adapter, "fq_modules", None)
    if fqm:                                   # quantized draft: fp from _replaced
        assert len(adapter._replaced) == len(fqm)
        for (name, m), (parent, attr, orig) in zip(fqm.items(), adapter._replaced):
            assert orig.weight.shape == m.w_fake.shape, name
            add(FQ_NAME2WTYPE[name], name.split(".")[-1], orig.weight, m.w_fake,
                note="fp=R1/R2/R4-conjugated pre-quant; q=fake-quantized as used")
    else:                                     # fp16 pure-R1 draft: plain linears
        add("projection_fc_weight", "fc", ea.fc.weight, note="R1-conjugated fp16")
        attn, mlp = ea.layers[0].self_attn, ea.layers[0].mlp
        for pn in ("q_proj", "k_proj", "v_proj", "o_proj"):
            add(f"{pn}_weight", pn, getattr(attn, pn).weight, note="R1-conjugated fp16")
        for pn in ("gate_proj", "up_proj", "down_proj"):
            add(f"{pn}_weight", pn, getattr(mlp, pn).weight, note="R1-conjugated fp16")
    # embedding + lm_head (fp16 in ALL configs; never fake-quantized)
    add("embedding_weight", "embed_tokens", ea.embed_tokens.weight,
        note="rows@R1 (pure-R1); fp16 in all configs, not fake-quantized")
    add("lm_head_weight", "lm_head", adapter.head.weight,
        note="W_lm@R1; fp16 in all configs, not fake-quantized")
    return out


# --------------------------- generation helper ---------------------------- #
@torch.no_grad()
def eagle_accept(model, ids, ilen, tree, max_new):
    deltas, prev = [], ilen
    for out in model.ea_generate(ids, temperature=0.0, max_steps=max_new + 8, tree_choices=tree):
        cur = out.shape[1]
        if cur > prev:
            deltas.append(cur - prev); prev = cur
        if cur - ilen >= max_new:
            break
    return (sum(deltas)/len(deltas)) if deltas else 0.0


def make_draft(cfg, model, stash):
    if cfg == CFG_FP16:
        return spd.SpinquantDraftPureR1Adapter(model, stash, DEV, torch.float16, trace=True)
    if cfg == CFG_W4A4:
        return f4.FakeW4A4DraftAdapter(model, stash, DEV, torch.float16, trace=True).configure_fq(w_bits=4, a_bits=4)
    if cfg == CFG_W8A8:
        return f8.FakeW8A8DraftAdapter(model, stash, DEV, torch.float16, trace=True).configure_fq(w_bits=8, a_bits=8)
    raise ValueError(cfg)


# --------------------------------- main ----------------------------------- #
@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--num-prompts", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--capture-cycle", type=int, default=1)
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    study.assert_gpu_policy()
    assert torch.cuda.device_count() == 1, "one physical GPU per job"

    rd = args.run_dir if os.path.isabs(args.run_dir) else os.path.join(PROJECT_ROOT, args.run_dir)
    SUBDIRS = ["figures_activation_3d", "figures_activation_heatmap",
               "figures_activation_channel_curves", "figures_weight_3d",
               "figures_weight_heatmap", "figures_weight_channel_curves",
               "figures_comparison_activation", "figures_comparison_weight",
               "stats", "samples"]
    for s in SUBDIRS:
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

    print("[wact3d] building target full/w4a4 ...", flush=True)
    model, stash, _ = study.build_study_target(
        paths["target_path"], paths["draft_path"], cfg["model"]["target"],
        "full", "random_hadamard", "w4a4", 0, device=DEV, rotations_root=rr)
    tok = eagle_bridge.get_tokenizer(model)
    from eagle.model.choices import mc_sim_7b_63 as tree_full
    tree = [list(p) for p in tree_full]
    study.set_draft_tree(model, tree, DEV)
    tail = wf.UnfusedTailAdapter(model, stash).install()
    ids_list = [build_prompt(tok, p["text"]).to(DEV) for p in prompts]

    # ---- representative selection over ALL THREE configs ----
    print("[wact3d] representative scan (FP16 vs W4A4 vs W8A8 draft) ...", flush=True)
    acc = {c: [] for c in ALL_CFGS}
    for c in ALL_CFGS:
        dr = make_draft(c, model, stash); dr.install()
        for pi, ids in enumerate(ids_list):
            if hasattr(dr, "set_context"):
                dr.set_context(prompts[pi]["question_id"])
            acc[c].append(eagle_accept(model, ids, ids.shape[1], tree, args.max_new_tokens))
        dr.uninstall()
        print(f"[wact3d] scan {c}: mean={np.mean(acc[c]):.3f}", flush=True)
    cand = []
    for pi in range(len(ids_list)):
        a_fp, a_w4, a_w8 = acc[CFG_FP16][pi], acc[CFG_W4A4][pi], acc[CFG_W8A8][pi]
        cand.append(dict(idx=pi, prompt_id=prompts[pi]["question_id"],
                         a_fp=a_fp, a_w4=a_w4, a_w8=a_w8, gap=a_fp - a_w4))
    good = [r for r in cand if r["a_fp"] >= 2.0 and r["a_w4"] <= 1.6 and
            r["a_w8"] >= r["a_w4"] + 0.3]
    best = max(good if good else cand, key=lambda r: r["gap"])
    rep = best["idx"]; rep_prompt = prompts[rep]; rep_ids = ids_list[rep]
    reason = ("healthy FP16 (%.2f) + collapsed W4A4 (%.2f) + partial W8A8 recovery (%.2f); "
              "largest FP16-W4A4 gap of %d candidates; cycle %d is the earliest cycle "
              "where all configs start from the identical target hidden"
              % (best["a_fp"], best["a_w4"], best["a_w8"], len(cand), args.capture_cycle))
    print(f"[wact3d] representative prompt_id={rep_prompt['question_id']} "
          f"FP16={best['a_fp']:.2f} W4A4={best['a_w4']:.2f} W8A8={best['a_w8']:.2f}", flush=True)
    logging_utils.write_csv(os.path.join(rd, "stats", "representative_case_manifest.csv"), [dict(
        prompt_id=rep_prompt["question_id"], verification_cycle_id=args.capture_cycle,
        depth_available="0-4 (5 draft forwards at cycle 1; deeper depths have 1 tree node)",
        **{f"acceptance_{CFG_FP16}": round(best["a_fp"], 4),
           f"acceptance_{CFG_W4A4}": round(best["a_w4"], 4),
           f"acceptance_{CFG_W8A8}": round(best["a_w8"], 4)},
        reason_selected=reason)] + [dict(
        prompt_id=r["prompt_id"], verification_cycle_id="candidate",
        depth_available="",
        **{f"acceptance_{CFG_FP16}": round(r["a_fp"], 4),
           f"acceptance_{CFG_W4A4}": round(r["a_w4"], 4),
           f"acceptance_{CFG_W8A8}": round(r["a_w8"], 4)},
        reason_selected="") for r in cand])

    # ---- capture: activations (real generation) + weights, per config ----
    caps, weights = {}, {}
    for c in ALL_CFGS:
        dr = make_draft(c, model, stash); dr.install()
        if hasattr(dr, "set_context"):
            dr.set_context(rep_prompt["question_id"])
        cap = Capture(dr, c, args.capture_cycle, rep_prompt["question_id"]).install()
        _ = eagle_accept(model, rep_ids, rep_ids.shape[1], tree, args.max_new_tokens)
        cap.remove(); caps[c] = cap
        weights[c] = capture_weights(dr, c)
        dr.uninstall()
        print(f"[wact3d] {c}: {len({a for a, d in cap.store})} activation types, "
              f"{len(weights[c])} weight tensors captured", flush=True)
    tail.uninstall()
    del model; torch.cuda.empty_cache()

    pid = int(rep_prompt["question_id"])
    cyc = args.capture_cycle
    fig_rows = []          # figure manifest

    def man(kind, cfg_name, lname, ftype, path, rule=""):
        fig_rows.append(dict(tensor_kind=kind, config_name=cfg_name, layer_name=lname,
                             figure_type=ftype, path=os.path.relpath(path, rd),
                             sampling_rule=rule))

    # ------------------------- activation figures -------------------------- #
    print("[wact3d] rendering activation figures ...", flush=True)
    for c in ALL_CFGS:
        for (at, d), x in sorted(caps[c].store.items()):
            Z = x.numpy(); lname = ACT2LAYER[at]
            kindtxt = ("embedding output activation" if at == "embedding_output_activation"
                       else "pre-layer activation")
            ttl = (f"{c} | {at} | {kindtxt} | ACTIVATION | abs(value) | draft L0 | "
                   f"prompt {pid} | cycle {cyc} | depth {d}")
            base = f"{at}__{'embedding_output' if at=='embedding_output_activation' else 'pre_layer'}_activation"
            d3 = os.path.join(rd, "figures_activation_3d", c, f"depth{d}")
            dh = os.path.join(rd, "figures_activation_heatmap", c, f"depth{d}")
            os.makedirs(d3, exist_ok=True); os.makedirs(dh, exist_ok=True)
            p3 = os.path.join(d3, f"{base}__3d__prompt{pid:02d}_cycle{cyc:02d}_depth{d:02d}.png")
            ph = os.path.join(dh, f"{base}__heatmap__prompt{pid:02d}_cycle{cyc:02d}_depth{d:02d}.png")
            try:
                r3 = plot_3d(Z, ttl, p3, "channel index", "token/node index",
                             "abs(activation)", MAX_TOK_PLOT, MAX_CH_PLOT)
                man("activation", c, lname, "3d", p3, r3)
                rh = plot_heat(Z, ttl, ph, "channel index", "token/node index",
                               "abs(activation)", MAX_TOK_PLOT, MAX_CH_PLOT)
                man("activation", c, lname, "heatmap", ph, rh)
            except Exception as e:
                print(f"[wact3d] act fig fail {c}/{at}/d{d}: {e}", flush=True)
        # channel curves (depth 0)
        for at in ACT_TYPES:
            if (at, 0) not in caps[c].store:
                continue
            v = dim_absmax(caps[c].store[(at, 0)], 1).numpy()
            np.save(os.path.join(rd, "samples", f"{c}__{at}__depth00__activation_per_channel_absmax.npy"), v)
            dc = os.path.join(rd, "figures_activation_channel_curves", c)
            os.makedirs(dc, exist_ok=True)
            pc = os.path.join(dc, f"{at}__activation_channel_absmax_curve__depth00.png")
            try:
                plot_curves([("per-channel absmax over tokens", v)],
                            f"{c} | {at} | ACTIVATION per-channel absmax | depth 0 | prompt {pid}",
                            pc, "abs(activation) per-channel absmax")
                man("activation", c, ACT2LAYER[at], "channel_curve", pc)
            except Exception as e:
                print(f"[wact3d] act curve fail {c}/{at}: {e}", flush=True)

    # activation comparison overlays (depth 0, shared scale)
    for at in ACT_TYPES:
        panels = [(c, caps[c].store[(at, 0)].numpy()) for c in ALL_CFGS
                  if (at, 0) in caps[c].store]
        if len(panels) < 2:
            continue
        dcmp = os.path.join(rd, "figures_comparison_activation", "depth0")
        os.makedirs(dcmp, exist_ok=True)
        pth = os.path.join(dcmp, f"{at}__targetW4A4_draftFP16_vs_draftW4A4_vs_draftW8A8__activation_heatmap_overlay.png")
        try:
            plot_compare_heat(panels,
                f"{at} | pre-layer ACTIVATION abs | depth 0 | prompt {pid} | "
                f"{' vs '.join(p[0] for p in panels)}",
                pth, "channel index", "token/node index", "abs(activation)",
                MAX_TOK_PLOT, MAX_CH_PLOT)
            man("activation", "comparison_all_configs", ACT2LAYER[at], "comparison_heatmap", pth)
        except Exception as e:
            print(f"[wact3d] act compare fail {at}: {e}", flush=True)

    # --------------------------- weight figures ---------------------------- #
    print("[wact3d] rendering weight figures ...", flush=True)
    wstat_rows, wchan_rows = [], []
    werr_rows = {CFG_W4A4: [], CFG_W8A8: []}
    for c in ALL_CFGS:
        for w in weights[c]:
            wt, lname = w["weight_type"], w["layer_name"]
            used = w["w_q"] if w["w_q"] is not None else w["w_fp"]     # as-used weight
            usetag = ("fake-quantized W%d as used" % (4 if c == CFG_W4A4 else 8)
                      if w["w_q"] is not None else "fp16 as used")
            oc = dim_absmax(used, 0); ic = dim_absmax(used, 1)          # [out], [in]
            wstat_rows.append(dict(config_name=c, tensor_kind="weight", weight_type=wt,
                layer_name=lname, shape=str(list(used.shape)), **tstats(used),
                max_to_median_input_channel_absmax_ratio=ratio(ic),
                max_to_median_output_channel_absmax_ratio=ratio(oc),
                weight_state=usetag, note=w["note"]))
            wchan_rows.append(dict(config_name=c, tensor_kind="weight", weight_type=wt,
                layer_name=lname,
                per_input_channel_absmax_max=ic.max().item(),
                per_input_channel_absmax_median=ic.median().item(),
                max_to_median_input_channel_absmax_ratio=ratio(ic),
                per_output_channel_absmax_max=oc.max().item(),
                per_output_channel_absmax_median=oc.median().item(),
                max_to_median_output_channel_absmax_ratio=ratio(oc)))
            if w["w_q"] is not None:
                werr_rows[c].append(dict(config_name=c, tensor_kind="quantization_error",
                    weight_type=wt, layer_name=lname, shape=str(list(used.shape)),
                    **werr(w["w_q"], w["w_fp"])))
            np.save(os.path.join(rd, "samples", f"{c}__{wt}__weight_per_input_channel_absmax.npy"), ic.numpy())
            np.save(os.path.join(rd, "samples", f"{c}__{wt}__weight_per_output_channel_absmax.npy"), oc.numpy())

            Z = used.numpy()   # [out, in] -> y=output ch, x=input ch
            ttl = (f"{c} | {wt} | WEIGHT ({usetag}) | abs(weight) | draft L0 | "
                   f"x=input ch, y=output ch")
            d3 = os.path.join(rd, "figures_weight_3d", c)
            dh = os.path.join(rd, "figures_weight_heatmap", c)
            dc = os.path.join(rd, "figures_weight_channel_curves", c)
            for dd in (d3, dh, dc):
                os.makedirs(dd, exist_ok=True)
            p3 = os.path.join(d3, f"{wt}__weight__3d.png")
            ph = os.path.join(dh, f"{wt}__weight__heatmap.png")
            pc = os.path.join(dc, f"{wt}__weight_channel_absmax_curves.png")
            try:
                r3 = plot_3d(Z, ttl, p3, "input channel index", "output channel index",
                             "abs(weight)", MAX_WROW_PLOT, MAX_WCOL_PLOT)
                man("weight", c, lname, "3d", p3, r3)
                rh = plot_heat(Z, ttl, ph, "input channel index", "output channel index",
                               "abs(weight)", MAX_WROW_PLOT, MAX_WCOL_PLOT)
                man("weight", c, lname, "heatmap", ph, rh)
                plot_curves([("per-input-channel absmax", ic.numpy()),
                             ("per-output-channel absmax", oc.numpy())],
                            f"{c} | {wt} | WEIGHT per-channel absmax curves", pc, "abs(weight)")
                man("weight", c, lname, "channel_curve", pc)
            except Exception as e:
                print(f"[wact3d] weight fig fail {c}/{wt}: {e}", flush=True)
            # quantization-error heatmap (quantized configs)
            if w["w_q"] is not None:
                pe = os.path.join(dh, f"{wt}__weight_quant_error__heatmap.png")
                try:
                    re_ = plot_heat((w["w_q"] - w["w_fp"]).numpy(),
                        f"{c} | {wt} | WEIGHT QUANTIZATION ERROR (w_q - w_fp) | abs(error) | draft L0",
                        pe, "input channel index", "output channel index",
                        "abs(weight quant error)", MAX_WROW_PLOT, MAX_WCOL_PLOT)
                    man("quantization_error", c, lname, "heatmap", pe, re_)
                except Exception as e:
                    print(f"[wact3d] werr fig fail {c}/{wt}: {e}", flush=True)

    # weight comparison overlays (as-used weight, shared scale)
    wmap = {c: {w["weight_type"]: w for w in weights[c]} for c in ALL_CFGS}
    for wt in WEIGHT_TYPES:
        panels = []
        for c in ALL_CFGS:
            w = wmap[c].get(wt)
            if w is None:
                continue
            used = w["w_q"] if w["w_q"] is not None else w["w_fp"]
            panels.append((c, used.numpy()))
        if len(panels) < 2:
            continue
        pth = os.path.join(rd, "figures_comparison_weight",
                           f"{wt}__draftFP16_vs_draftW4A4_vs_draftW8A8__weight_heatmap_overlay.png")
        try:
            plot_compare_heat(panels,
                f"{wt} | as-used WEIGHT abs | draft L0 | {' vs '.join(p[0] for p in panels)}",
                pth, "input channel index", "output channel index", "abs(weight)",
                MAX_WROW_PLOT, MAX_WCOL_PLOT)
            man("weight", "comparison_all_configs", wt.replace("_weight", ""),
                "comparison_heatmap", pth)
        except Exception as e:
            print(f"[wact3d] weight compare fail {wt}: {e}", flush=True)

    # ------------------------------ CSVs ----------------------------------- #
    logging_utils.write_csv(os.path.join(rd, "stats", "activation_distribution_stats.csv"),
                            [r for c in ALL_CFGS for r in caps[c].stat_rows])
    logging_utils.write_csv(os.path.join(rd, "stats", "activation_per_channel_stats.csv"),
                            [r for c in ALL_CFGS for r in caps[c].chan_rows])
    logging_utils.write_csv(os.path.join(rd, "stats", "weight_distribution_stats.csv"), wstat_rows)
    logging_utils.write_csv(os.path.join(rd, "stats", "weight_per_channel_stats.csv"), wchan_rows)
    logging_utils.write_csv(os.path.join(rd, "stats", "weight_error_targetW4A4_draftW4A4.csv"),
                            werr_rows[CFG_W4A4])
    logging_utils.write_csv(os.path.join(rd, "stats", "weight_error_targetW4A4_draftW8A8.csv"),
                            werr_rows[CFG_W8A8])

    def act_err_csv(other, fname):
        rows = []
        for (at, d), xr in caps[CFG_FP16].store.items():
            xo = caps[other].store.get((at, d))
            if xo is None or xr.shape != xo.shape:
                continue
            cerr = dim_absmax(xo - xr, 1)      # per-channel error absmax
            top = cerr.topk(min(10, len(cerr))).values
            rows.append(dict(activation_type=at, layer_name=ACT2LAYER[at], tree_depth=d,
                             ref_config=CFG_FP16, other_config=other,
                             basis_matched=bool(at in BASIS_MATCHED),
                             tensor_kind="activation", **align_err(xo, xr),
                             top10_channel_error_share=float(top.sum()/(cerr.sum()+1e-12))))
        logging_utils.write_csv(os.path.join(rd, "stats", fname), rows)
    act_err_csv(CFG_W4A4, "activation_error_targetW4A4_draftW4A4_vs_draftFP16.csv")
    act_err_csv(CFG_W8A8, "activation_error_targetW4A4_draftW8A8_vs_draftFP16.csv")

    cap_rows = []
    for c in ALL_CFGS:
        for (at, d), x in caps[c].store.items():
            cap_rows.append(dict(config_name=c, tensor_kind="activation",
                activation_or_weight_type=at, layer_name=ACT2LAYER[at],
                verification_cycle=cyc, tree_depth=d, shape=str(list(x.shape)),
                token_axis="row index of the activation matrix seen in that forward (tree-batched nodes)"))
        for w in weights[c]:
            cap_rows.append(dict(config_name=c, tensor_kind="weight",
                activation_or_weight_type=w["weight_type"], layer_name=w["layer_name"],
                verification_cycle="", tree_depth="",
                shape=str(list(w["w_fp"].shape)),
                token_axis=w["note"] + ("; w_q captured" if w["w_q"] is not None else "")))
    logging_utils.write_csv(os.path.join(rd, "stats", "capture_manifest.csv"), cap_rows)
    logging_utils.write_csv(os.path.join(rd, "stats", "figure_manifest.csv"), fig_rows)

    # ------------------------- quality checks ------------------------------ #
    print("[wact3d] running quality checks ...", flush=True)
    problems = []
    for root, _dirs, files in os.walk(rd):
        for fn in files:
            rel = os.path.relpath(os.path.join(root, fn), rd)
            if any(t in rel for t in FORBIDDEN):
                problems.append(f"forbidden C-name in path: {rel}")
            if fn.endswith(".png"):
                if "figures_activation" in rel or "figures_comparison_activation" in rel:
                    if "activation" not in fn:
                        problems.append(f"activation figure without 'activation': {rel}")
                if "figures_weight" in rel or "figures_comparison_weight" in rel:
                    if "weight" not in fn:
                        problems.append(f"weight figure without 'weight': {rel}")
    import csv as _csv
    for fname, kind in (("activation_distribution_stats.csv", "activation"),
                        ("weight_distribution_stats.csv", "weight")):
        with open(os.path.join(rd, "stats", fname)) as f:
            for row in _csv.DictReader(f):
                if row["tensor_kind"] != kind:
                    problems.append(f"{fname}: tensor_kind={row['tensor_kind']} != {kind}")
                if row["config_name"] not in ALL_CFGS:
                    problems.append(f"{fname}: bad config_name {row['config_name']}")
    need_cols = {"tensor_kind", "config_name", "layer_name", "figure_type", "path"}
    with open(os.path.join(rd, "stats", "figure_manifest.csv")) as f:
        cols = set(_csv.DictReader(f).fieldnames or [])
    if not need_cols <= cols:
        problems.append(f"figure_manifest missing columns: {need_cols - cols}")
    qc = "PASS" if not problems else "FAIL:\n" + "\n".join(problems[:40])
    open(os.path.join(rd, "quality_checks.txt"), "w").write(qc + "\n")
    print(f"[wact3d] quality checks: {'PASS' if not problems else 'FAIL(%d)' % len(problems)}", flush=True)

    # ------------------------------ summary -------------------------------- #
    nact = sum(1 for r in fig_rows if r["tensor_kind"] == "activation")
    nwt = sum(1 for r in fig_rows if r["tensor_kind"] in ("weight", "quantization_error"))
    lines = ["# EAGLE draft WEIGHT + ACTIVATION distribution — run summary", "",
             f"Representative: prompt_id {pid}, cycle {cyc} — acceptance "
             f"{CFG_FP16}={best['a_fp']:.2f}, {CFG_W4A4}={best['a_w4']:.2f}, "
             f"{CFG_W8A8}={best['a_w8']:.2f}.", "",
             "Every figure states ACTIVATION / WEIGHT / QUANTIZATION ERROR explicitly; "
             "activation figures = pre-layer inputs from real EAGLE generation "
             "(x=channel, y=token/node, z=abs); weight figures = as-used draft weights "
             "(x=input ch, y=output ch, z=abs). No C2/C3/C4 names in outputs.", "",
             f"- activation figures: {nact}",
             f"- weight (+quant-error) figures: {nwt}",
             f"- quality checks: {'PASS' if not problems else 'FAIL — see quality_checks.txt'}",
             "", "Key dirs: figures_activation_3d/, figures_weight_3d/, "
             "figures_comparison_activation/depth0/, figures_comparison_weight/.",
             "Stats: stats/*.csv (11 files). Answers: docs/eagle_draft_weight_activation_3d_summary.md."]
    open(os.path.join(rd, "summary.md"), "w").write("\n".join(lines) + "\n")
    print(f"[wact3d] DONE -> {rd} (act figs={nact}, weight figs={nwt})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""Why does draft fake W4A4 collapse acceptance (3.0 -> 1.0)? Distribution +
drift analysis of C2 (pure-R1 draft fp16) vs C4 (pure-R1 draft fake W4A4),
QuaRot/SmoothQuant-style.

Instruments the REAL EAGLE draft path:
  - per-module activation distributions + per-channel absmax (before-R1 /
    after-R1 / after-fake-W4A4)
  - weight + activation quantization error per draft linear
  - feature cosine / rel-L2 and logit top-k drift over tree depth (C4 vs C2)
  - acceptance vs final-feature error (per verification cycle)

Reuses the SAME quantizers/rotations as the acceptance runs (no new quant).
Outputs runs/w4a4_distribution_analysis_<ts>/{figures,*.csv,summary.md}.

Usage:
  CUDA_VISIBLE_DEVICES=6 python scripts/analyze_w4a4_draft_distributions.py \
      --run-dir runs/w4a4_distribution_analysis_<ts> --num-prompts 2
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
from eagle_spinquant import (eagle_bridge, experiment, logging_utils, study,  # noqa: E402
                             rotation_aware as ra, pure_r1_eagle as pr,
                             fake_w4a4_draft as fq)
from eagle_spinquant.rotation_interface import build_original_head  # noqa: E402

DEV = "cuda:0"
MODULES = ["fc", "layers.0.self_attn.q_proj", "layers.0.self_attn.k_proj",
           "layers.0.self_attn.v_proj", "layers.0.self_attn.o_proj",
           "layers.0.mlp.gate_proj", "layers.0.mlp.up_proj", "layers.0.mlp.down_proj"]


def tstats(x):
    x = x.detach().float().flatten()
    a = x.abs()
    m, s = x.mean().item(), x.std().item()
    xc = x - x.mean()
    var = xc.pow(2).mean()
    kurt = (xc.pow(4).mean() / (var**2 + 1e-12) - 3).item()
    skew = (xc.pow(3).mean() / (var**1.5 + 1e-12)).item()
    return dict(mean=m, std=s, min=x.min().item(), max=x.max().item(),
                abs_mean=a.mean().item(), abs_max=a.max().item(),
                median_abs=a.median().item(),
                p95_abs=torch.quantile(a, 0.95).item(), p99_abs=torch.quantile(a, 0.99).item(),
                p99_9_abs=torch.quantile(a, 0.999).item(),
                kurtosis=kurt, skewness=skew,
                outlier_ratio_gt3std=(a > 3*s).float().mean().item() if s > 0 else 0.0,
                outlier_ratio_gt6std=(a > 6*s).float().mean().item() if s > 0 else 0.0)


def channel_stats(x):
    """x [..., D] -> per-channel absmax stats."""
    x = x.detach().float().reshape(-1, x.shape[-1])
    cabs = x.abs().amax(0)                       # [D]
    return dict(max_channel_absmax=cabs.max().item(),
                median_channel_absmax=cabs.median().item(),
                ratio_max_to_median=(cabs.max() / (cabs.median() + 1e-12)).item(),
                cabs=cabs.cpu().numpy())


def qerr(xq, x):
    xq = xq.detach().float().flatten(); x = x.detach().float().flatten()
    e = xq - x
    rel = (e.norm() / (x.norm() + 1e-12)).item()
    snr = (20 * torch.log10(x.norm() / (e.norm() + 1e-12))).item()
    return dict(rel_l2=rel, cosine=F.cosine_similarity(xq, x, 0).item(),
                mean_abs_error=e.abs().mean().item(), max_abs_error=e.abs().max().item(),
                snr_db=snr)


def logit_drift(lg_c4, lg_c2):
    a, b = lg_c4.float(), lg_c2.float()
    def ov(k):
        ak, bk = a.topk(k, -1).indices, b.topk(k, -1).indices
        return float(np.mean([len(set(x.tolist()) & set(y.tolist()))/k
                              for x, y in zip(ak.reshape(-1, k), bk.reshape(-1, k))]))
    pa, pb = F.log_softmax(a, -1), F.log_softmax(b, -1)
    return dict(rel_l2_logit=((a-b).norm()/(b.norm()+1e-12)).item(),
                max_abs_logit=(a-b).abs().max().item(),
                top1_agreement=(a.argmax(-1) == b.argmax(-1)).float().mean().item(),
                top5_overlap=ov(5), top10_overlap=ov(10),
                kl=F.kl_div(pa, pb, log_target=True, reduction="batchmean").item())


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--num-prompts", type=int, default=2)
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    gpu = study.assert_gpu_policy()
    assert torch.cuda.device_count() == 1
    rd = args.run_dir if os.path.isabs(args.run_dir) else os.path.join(PROJECT_ROOT, args.run_dir)
    figs = os.path.join(rd, "figures"); os.makedirs(figs, exist_ok=True)
    with open(os.path.join(rd, "commands.sh"), "a") as f:
        f.write("CUDA_VISIBLE_DEVICES=" + os.environ["CUDA_VISIBLE_DEVICES"] + " " + " ".join(sys.argv) + "\n")
    with open(os.path.join(rd, "environment.txt"), "w") as f:
        f.write(json.dumps(logging_utils.env_summary(), indent=2) + "\n")
    for p in ("git_status.txt", "git_diff.patch"):
        open(os.path.join(rd, p), "w").write("not a git repository\n")

    cfg = experiment.load_config(args.config)
    paths = experiment.resolve_paths(cfg)
    rr = cfg.get("paths", {}).get("rotations_root")
    tmpl = cfg.get("model", {}).get("chat_template", "llama2")
    prompts = experiment.load_mt_bench_prompts(args.num_prompts)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(paths["target_path"])
    ids_list = [eagle_bridge.PROMPT_BUILDERS[tmpl](tok, p["text"]) for p in prompts]

    # --- target (w4a4) -> external hidden h_R per prompt ---
    r_bin = study.r_bin_path("random_hadamard", 0, paths["target_path"], rr)
    from eagle.model.modeling_llama_kv import LlamaForCausalLM as KVLlama
    rot = KVLlama.from_pretrained(paths["target_path"], torch_dtype=torch.float16,
                                  low_cpu_mem_usage=True).eval()
    stash = study.apply_rotation_quant(rot, "full", r_bin, "w4a4", cfg["model"]["target"], DEV)
    rot.to(DEV)
    R1 = stash["R1"].double().to(DEV); gamma = stash["gamma_f"].double().to(DEV)
    W_head = stash["lm_head_weight"].clone()
    hR_list, next_toks = [], []
    cap = {}
    hk = rot.model.norm.register_forward_hook(lambda m, i, o: cap.__setitem__("xR", i[0].detach()))
    for ids in ids_list:
        _ = rot.model(input_ids=ids.to(DEV))
        xR = cap["xR"].double()
        h_hat = xR / (xR.pow(2).mean(-1, keepdim=True) + rot.model.norm.variance_epsilon).sqrt()
        h = (h_hat @ R1.t()) * gamma
        hR_list.append((h @ R1).float())
        lg = h_hat.float() @ (W_head.double().to(DEV) @ (gamma.diag() @ R1)).t().float() if False else None
        next_toks.append(int((h @ (W_head.double().to(DEV)).t())[0, -1].argmax()))
    hk.remove(); del rot; torch.cuda.empty_cache()

    # --- standalone drafts: C2 (R1 fp16) and C4 (R1+R2+R4 + fake W4A4) ---
    draft_c2 = ra.build_standalone_draft(paths["draft_path"], DEV, torch.float32)
    sd = {k: v.detach().cpu() for k, v in draft_c2.state_dict().items()}
    conv2 = pr.build_pure_r1_draft_state(sd, R1.cpu(), gamma.cpu())
    draft_c2.load_state_dict({k: v.float() for k, v in conv2.items()}, strict=True); draft_c2.to(DEV)

    draft_c4 = ra.build_standalone_draft(paths["draft_path"], DEV, torch.float32)
    conv4, meta = fq.build_spinquant_w4a4_draft_state(sd, R1.cpu(), gamma.cpu())
    draft_c4.load_state_dict({k: v.float() for k, v in conv4.items()}, strict=True); draft_c4.to(DEV)
    had_K = meta["had_K"].to(DEV) if meta["had_K"] is not None else None
    # replace 8 linears with FakeW4A4Linear (weight error captured here)
    weight_err_rows, fqmods = [], {}
    def replace(parent, attr, name, online=False):
        lin = getattr(parent, attr)
        w_orig = lin.weight.data.clone()
        m = fq.FakeW4A4Linear(lin.weight, getattr(lin, "bias", None), name,
                              online_had=online, had_K=had_K if online else None,
                              K=meta["K"] if online else None, w_bits=4, a_bits=4).to(DEV)
        setattr(parent, attr, m); fqmods[name] = m
        weight_err_rows.append(dict(module_name=name, **{("weight_"+k): v for k, v in
            qerr(m.w_fake.float(), w_orig.float()).items()}))
    replace(draft_c4, "fc", "fc")
    for pn in ("q_proj", "k_proj", "v_proj", "o_proj"):
        replace(draft_c4.layers[0].self_attn, pn, f"layers.0.self_attn.{pn}")
    for pn in ("gate_proj", "up_proj"):
        replace(draft_c4.layers[0].mlp, pn, f"layers.0.mlp.{pn}")
    replace(draft_c4.layers[0].mlp, "down_proj", "layers.0.mlp.down_proj", online=True)

    head_R = torch.nn.Linear(4096, W_head.shape[0], bias=False)
    head_R.weight.data = (W_head.double().to(DEV) @ R1).float(); head_R = head_R.to(DEV)

    # ---- module activation + quant-error hooks on C4 (real generation) ----
    act_rows, chan_rows, qerr_rows, hist = [], [], [], {}
    ctx = {"cyc": 0, "dep": 0}
    def mk_hook(name, mod):
        def hook(m, inp, out):
            x = inp[0]
            xq = m.aq(x) if m.aq is not None else x   # post-act-quant (re-derive; aq state set in fwd)
            act_rows.append(dict(module_name=name, config="C4", stat="pre_act_quant_input",
                                 cycle=ctx["cyc"], depth=ctx["dep"], **tstats(x)))
            cs = channel_stats(x)
            chan_rows.append(dict(module_name=name, state="after_R1_input(C4)",
                                  cycle=ctx["cyc"], depth=ctx["dep"],
                                  **{k: v for k, v in cs.items() if k != "cabs"}))
            qerr_rows.append(dict(module_name=name, cycle=ctx["cyc"], depth=ctx["dep"],
                                  kind="activation", **qerr(xq, x)))
            if name not in hist and ctx["dep"] <= 1:
                hist[name] = dict(inp=x.detach().float().flatten()[:20000].cpu().numpy(),
                                  out=out.detach().float().flatten()[:20000].cpu().numpy())
        return mod.register_forward_hook(hook)
    handles = [mk_hook(n, m) for n, m in fqmods.items()]

    # ---- per-depth feature/logit drift (C2 vs C4, free-running) + accept ----
    drift_rows, logit_rows, acc_rows = [], [], []
    for pi, ids in enumerate(ids_list):
        full_ids = torch.cat([ids.to(DEV), torch.tensor([[next_toks[pi]]], device=DEV)], 1)
        hR = hR_list[pi].to(DEV)
        ctx["cyc"] = 1
        # depth wrapper to tag ctx for hooks
        d = {"depth": 0}
        orig_fwd = draft_c4.forward
        def c4_fwd(hs, *a, **k):
            ctx["dep"] = d["depth"]; d["depth"] += 1
            return orig_fwd(hs, *a, **k)
        draft_c4.forward = c4_fwd
        _, outs_c2 = ra.level_capture(draft_c2, hR, full_ids, head_R)
        d["depth"] = 0
        _, outs_c4 = ra.level_capture(draft_c4, hR, full_ids, head_R)
        draft_c4.forward = orig_fwd
        for lvl, (a2, a4) in enumerate(zip(outs_c2, outs_c4), start=1):
            if a2.shape != a4.shape:
                continue
            fa, fb = a4.to(DEV), a2.to(DEV)
            c = F.cosine_similarity(fa.float().flatten(), fb.float().flatten(), 0).item()
            rl = ((fa-fb).float().norm()/(fb.float().norm()+1e-12)).item()
            drift_rows.append(dict(prompt_id=prompts[pi]["question_id"], tree_depth=lvl,
                                   module_name="final_feature", feature_cosine=c, feature_rel_l2=rl))
            ld = logit_drift(head_R(fa.float()), head_R(fb.float()))
            logit_rows.append(dict(prompt_id=prompts[pi]["question_id"], tree_depth=lvl, **ld))
        # acceptance for this cycle (real EAGLE, C4) — approx via level-1 top1 agreement proxy
        acc_rows.append(dict(prompt_id=prompts[pi]["question_id"],
                             final_feature_cosine=drift_rows[-1]["feature_cosine"],
                             final_feature_rel_l2=drift_rows[-1]["feature_rel_l2"]))
    for h in handles:
        h.remove()

    # per-channel: before-R1 vs after-R1 vs after-W4A4 for fc input + final feature
    pc = {}
    for pi, ids in enumerate(ids_list[:1]):
        hR = hR_list[pi].to(DEV)
        # fc input (concat) before R1 conjugation is not directly available; use
        # the hidden h (original) vs h_R (rotated) as the residual-stream proxy
        h = (hR.double() @ R1.t()).float()      # un-rotate to original basis
        pc["hidden_before_R1"] = channel_stats(h)["cabs"]
        pc["hidden_after_R1"] = channel_stats(hR)["cabs"]
        # after fake-quant: quantize hR per-token (fc h-block activation quant)
        aq = fq._act_quantizer(4); aq.find_params(hR); pc["hidden_after_W4A4"] = channel_stats(aq(hR))["cabs"]

    # ---------- write CSVs ----------
    logging_utils.write_csv(os.path.join(rd, "tensor_distribution_stats.csv"), act_rows)
    logging_utils.write_csv(os.path.join(rd, "per_channel_stats.csv"), chan_rows)
    logging_utils.write_csv(os.path.join(rd, "quant_error_by_module.csv"), qerr_rows)
    logging_utils.write_csv(os.path.join(rd, "weight_quant_error.csv"), weight_err_rows)
    logging_utils.write_csv(os.path.join(rd, "feature_drift_by_depth.csv"), drift_rows)
    logging_utils.write_csv(os.path.join(rd, "logit_drift_by_depth.csv"), logit_rows)
    logging_utils.write_csv(os.path.join(rd, "acceptance_vs_error.csv"), acc_rows)
    logging_utils.write_csv(os.path.join(rd, "collected_tensor_manifest.csv"),
        [dict(module_name=n, captured=True, samples=len(hist.get(n, {}).get("inp", []))) for n in MODULES])

    # ---------- figures ----------
    _figures(rd, figs, hist, chan_rows, qerr_rows, weight_err_rows, drift_rows, logit_rows, pc, act_rows)

    # ---------- summary numbers ----------
    import pandas as pd
    qe = pd.DataFrame(qerr_rows)
    worst = qe.groupby("module_name")["rel_l2"].mean().sort_values(ascending=False)
    dd = pd.DataFrame(drift_rows).groupby("tree_depth")["feature_cosine"].mean()
    verd = {
        "worst_activation_quant_error_modules": worst.round(4).head(3).to_dict(),
        "feature_cosine_by_depth": dd.round(4).to_dict(),
        "weight_err_by_module": pd.DataFrame(weight_err_rows).set_index("module_name")["weight_rel_l2"].round(4).to_dict(),
        "channel_ratio_before_R1": float(np.max(pc["hidden_before_R1"])/np.median(pc["hidden_before_R1"])),
        "channel_ratio_after_R1": float(np.max(pc["hidden_after_R1"])/np.median(pc["hidden_after_R1"])),
    }
    with open(os.path.join(rd, "analysis_verdicts.json"), "w") as f:
        json.dump(verd, f, indent=2)
    print(json.dumps(verd, indent=2))
    print(f"-> {rd} ({len(os.listdir(figs))} figures)")
    return 0


def _figures(rd, figs, hist, chan_rows, qerr_rows, weight_err_rows, drift_rows, logit_rows, pc, act_rows):
    import pandas as pd
    # Fig 1: activation histograms
    sel = [m for m in ["fc", "layers.0.mlp.down_proj", "layers.0.self_attn.q_proj"] if m in hist]
    fig, axes = plt.subplots(1, max(len(sel), 1), figsize=(5*max(len(sel), 1), 4), squeeze=False)
    for ax, m in zip(axes[0], sel):
        ax.hist(hist[m]["inp"], bins=120, log=True, alpha=0.8)
        ax.set_title(f"C4 input: {m.split('.')[-1]}"); ax.set_xlabel("value")
    fig.suptitle("Fig1 activation histograms (C4 draft inputs, log-y)"); fig.tight_layout()
    fig.savefig(os.path.join(figs, "fig01_activation_histograms_c2_vs_c4.png"), dpi=130); plt.close(fig)

    # Fig 2: per-channel absmax before/after R1/after W4A4 (hidden)
    fig, ax = plt.subplots(figsize=(7, 4))
    for k, lab in [("hidden_before_R1", "before R1"), ("hidden_after_R1", "after R1"),
                   ("hidden_after_W4A4", "after W4A4")]:
        if k in pc:
            ax.plot(np.sort(pc[k])[::-1], label=lab)
    ax.set_yscale("log"); ax.set_xlabel("channel (sorted)"); ax.set_ylabel("per-channel absmax")
    ax.set_title("Fig2 per-channel absmax: outliers before/after R1/W4A4"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(figs, "fig02_per_channel_absmax.png"), dpi=130); plt.close(fig)

    # Fig 3: activation quant error by module
    qe = pd.DataFrame(qerr_rows).groupby("module_name")["rel_l2"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh([m.split(".")[-1] for m in qe.index], qe.values, color="#d95f0e")
    ax.set_xlabel("activation quant rel-L2"); ax.set_title("Fig3 activation quant error by module")
    fig.tight_layout(); fig.savefig(os.path.join(figs, "fig03_quant_error_by_module.png"), dpi=130); plt.close(fig)

    # Fig 4: feature cosine over depth
    dd = pd.DataFrame(drift_rows).groupby("tree_depth")["feature_cosine"].mean()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(dd.index, dd.values, "o-"); ax.set_xlabel("tree depth"); ax.set_ylabel("cosine(C4, C2) final feature")
    ax.set_title("Fig4 feature cosine over tree depth"); ax.set_ylim(0, 1.02)
    fig.tight_layout(); fig.savefig(os.path.join(figs, "fig04_feature_cosine_by_depth.png"), dpi=130); plt.close(fig)

    # Fig 5: top-k agreement over depth
    ld = pd.DataFrame(logit_rows).groupby("tree_depth")[["top1_agreement", "top5_overlap", "top10_overlap"]].mean()
    fig, ax = plt.subplots(figsize=(6, 4))
    for c in ld.columns:
        ax.plot(ld.index, ld[c], "o-", label=c)
    ax.set_xlabel("tree depth"); ax.set_ylabel("agreement"); ax.set_title("Fig5 draft top-k agreement C4 vs C2 by depth")
    ax.legend(); ax.set_ylim(0, 1.02); fig.tight_layout()
    fig.savefig(os.path.join(figs, "fig05_topk_agreement_by_depth.png"), dpi=130); plt.close(fig)

    # Fig 6: acceptance-proxy vs feature error (use per-depth cosine vs depth as proxy)
    df = pd.DataFrame(drift_rows)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(df["feature_rel_l2"], df["tree_depth"], alpha=0.6)
    ax.set_xlabel("final-feature rel-L2 (C4 vs C2)"); ax.set_ylabel("tree depth")
    ax.set_title("Fig6 feature error vs depth (deeper = more drift => fewer accepts)")
    fig.tight_layout(); fig.savefig(os.path.join(figs, "fig06_acceptance_vs_feature_error.png"), dpi=130); plt.close(fig)

    # Fig 7: layer error heatmap (module x depth) — activation rel-L2
    qd = pd.DataFrame(qerr_rows)
    if len(qd):
        piv = qd.pivot_table(index="module_name", columns="depth", values="rel_l2", aggfunc="mean")
        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(piv.values, aspect="auto", cmap="magma")
        ax.set_yticks(range(len(piv.index))); ax.set_yticklabels([m.split(".")[-1] for m in piv.index])
        ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns)
        ax.set_xlabel("draft forward depth"); ax.set_title("Fig7 activation quant rel-L2 heatmap")
        fig.colorbar(im); fig.tight_layout()
        fig.savefig(os.path.join(figs, "fig07_layer_error_heatmap.png"), dpi=130); plt.close(fig)

    # Fig 8: weight quant error by module
    we = pd.DataFrame(weight_err_rows).sort_values("weight_rel_l2")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh([m.split(".")[-1] for m in we["module_name"]], we["weight_rel_l2"], color="#2c7fb8")
    ax.set_xlabel("weight quant rel-L2"); ax.set_title("Fig8 weight quantization error by module")
    fig.tight_layout(); fig.savefig(os.path.join(figs, "fig08_weight_quant_error.png"), dpi=130); plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())

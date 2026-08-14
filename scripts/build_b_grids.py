#!/usr/bin/env python
"""Phase-B constructed-code grids (qanchor study sections B1/B2).

From the frozen anchor (c0, s) and the aggressive checkpoint's folded
codes c_bad, generate override-code files:

B1 revert:
  global_f{F}_s{S} : revert a random fraction F of ALL changed codes
                     (seeds 0/1/2), F in {0.25,0.5,0.75,0.9}
  down_f{F}       : revert fraction F of down_proj changed codes only
                     (other sites stay c_bad), F in {0.25,0.5,0.75,1.0}
  nondown_f{F}    : revert fraction F of changed codes on all sites
                     EXCEPT down (down stays c_bad)
  endpoints       : f0 (= pure c_bad) and f100 (= c0 == PTQ, no file)

B2 interpolation (masters-space, folds are linear so equal to folded-
space interpolation): alpha in {0.1,0.25,0.5,0.75,0.9}; codes =
frozen-quantize(alpha*fold(W_agg) + (1-alpha)*fold(W0)).

Each grid entry records H_Q, D_Q, per-site flips in
tables/b_grid_meta.json. Code files: ckpts/bgrid__<name>.pt
"""
import argparse, glob, json, os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "EAGLE"))
import torch
from safetensors import safe_open
from eagle_spinquant import experiment, study, anchor_quant as aq
from eagle_spinquant.exact_qat_rotated_draft import ExactQATRotatedDraft
from eagle_spinquant.residual_rotation import SharedRotation

KIND = "learned_chat_w4a4kv16"


def folded(sd, R_T, gamma, W_lm, rotM, dev):
    rot = SharedRotation(rotM).to(dev)
    m = ExactQATRotatedDraft(sd, R_T, gamma, W_lm, rot,
                             alpha_init=32.89964245299412,
                             train_alpha=False, w_bits=4, a_bits=4,
                             draft_kv_bits=16, train_draft_core=False,
                             device=dev, first_fold_R=R_T)
    with torch.no_grad():
        tw = {k: v.detach().float().cpu()
              for k, v in m.transformed_weights(exact=False).items()
              if k in aq.QSITES}
    del m
    torch.cuda.empty_cache()
    return tw


def grid_meta(codes_t, anchor):
    flips = tot = 0
    dq_n = dq_d = 0.0
    per = {}
    for k in aq.QSITES:
        c0 = anchor[k]["c0"]
        s = anchor[k]["scale"]
        f = (codes_t[k] != c0)
        per[k] = round(float(f.float().mean()), 5)
        flips += int(f.sum())
        tot += f.numel()
        dq_n += float((s * (codes_t[k].float() - c0.float())).pow(2)
                      .sum())
        dq_d += float((s * c0.float()).pow(2).sum())
    return dict(H_Q=round(flips / tot, 6), D_Q=round(dq_n / dq_d, 7),
                per_site=per)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--anchor", required=True)
    ap.add_argument("--aggressive-sd", required=True)
    ap.add_argument("--rd-ckpt", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = args.device
    cfg = experiment.load_config(None)
    paths = experiment.resolve_paths(cfg)
    rr = cfg.get("paths", {}).get("rotations_root")
    R = torch.load(study.r_bin_path(KIND, 0, paths["target_path"], rr),
                   map_location="cpu", weights_only=False)
    R_T = R["R1"].float()
    rotM = torch.load(args.rd_ckpt, map_location="cpu",
                      weights_only=False)["R_D"].float()
    gamma = W_lm = None
    for p in sorted(glob.glob(os.path.join(paths["target_path"],
                                           "*.safetensors"))):
        with safe_open(p, framework="pt") as f:
            if "model.norm.weight" in f.keys() and gamma is None:
                gamma = f.get_tensor("model.norm.weight").float()
            if "lm_head.weight" in f.keys() and W_lm is None:
                W_lm = f.get_tensor("lm_head.weight").float()
    anchor = aq.load_anchor(args.anchor)
    sd0 = torch.load(os.path.join(paths["draft_path"],
                                  "pytorch_model.bin"),
                     map_location="cpu", weights_only=True)
    agg = torch.load(args.aggressive_sd, map_location="cpu",
                     weights_only=False)
    agg = agg.get("draft_state_dict", agg)
    tw0 = folded(sd0, R_T, gamma, W_lm, rotM, dev)
    twA = folded(agg, R_T, gamma, W_lm, rotM, dev)
    cbad = {k: aq.codes(twA[k], anchor[k]["scale"])
            for k in aq.QSITES}
    meta = {}
    ck = os.path.join(args.run_dir, "ckpts")
    os.makedirs(ck, exist_ok=True)

    def save(name, codes_t):
        torch.save(codes_t, os.path.join(ck, f"bgrid__{name}.pt"))
        meta[name] = grid_meta(codes_t, anchor)
        print(f"[bgrid] {name}: H_Q {meta[name]['H_Q']:.5f} "
              f"D_Q {meta[name]['D_Q']:.6f} "
              f"down {meta[name]['per_site']['down']:.4f}", flush=True)

    save("b1_global_f0", {k: v.clone() for k, v in cbad.items()})
    # B1 global reverts
    for f in (0.25, 0.5, 0.75, 0.9):
        for s in (0, 1, 2):
            g = torch.Generator().manual_seed(1000 + s)
            out = {}
            for k in aq.QSITES:
                c0 = anchor[k]["c0"]
                cb = cbad[k].clone()
                ch = (cb != c0).nonzero(as_tuple=False)
                n = ch.shape[0]
                nrev = int(round(f * n))
                idx = ch[torch.randperm(n, generator=g)[:nrev]]
                cb[idx[:, 0], idx[:, 1]] = c0[idx[:, 0], idx[:, 1]]
                out[k] = cb
            save(f"b1_global_f{int(f*100)}_s{s}", out)
    # B1 down-only / non-down families (seed 0)
    for fam, sites in (("down", ("down",)),
                       ("nondown", tuple(k for k in aq.QSITES
                                         if k != "down"))):
        for f in (0.25, 0.5, 0.75, 1.0):
            g = torch.Generator().manual_seed(2000)
            out = {k: cbad[k].clone() for k in aq.QSITES}
            for k in sites:
                c0 = anchor[k]["c0"]
                cb = out[k]
                ch = (cb != c0).nonzero(as_tuple=False)
                n = ch.shape[0]
                nrev = int(round(f * n))
                idx = ch[torch.randperm(n, generator=g)[:nrev]]
                cb[idx[:, 0], idx[:, 1]] = c0[idx[:, 0], idx[:, 1]]
            save(f"b1_{fam}_f{int(f*100)}", out)
    # B2 interpolation under frozen scales
    for a in (0.1, 0.25, 0.5, 0.75, 0.9):
        out = {}
        for k in aq.QSITES:
            w = a * twA[k] + (1 - a) * tw0[k]
            out[k] = aq.codes(w, anchor[k]["scale"])
        save(f"b2_alpha{int(a*100)}", out)
    json.dump(meta, open(os.path.join(args.run_dir, "tables",
                                      "b_grid_meta.json"), "w"),
              indent=1)
    print(f"[bgrid] {len(meta)} grids -> tables/b_grid_meta.json")


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""Closure sections 5-6: uniform drift metrics for EVERY model under
the exact causal-study anchor + frozen quantizer, with the corrected
accounting:

  H_Q_all         all 9 deployment-visible fold sites
  H_Q_controlled  the 8 hard-budget-constrained folds
                  (q,k,v,o,gate,up,down,W_rec)
  W_first total / e-half / h-half decomposition
  per-site rates, D_Q (all sites, frozen scales), D_FP

Same adapter-fold capture as the causal study (cell space = deployed
adapter folds). Writes one JSON keyed by checkpoint basename.
"""
import argparse
import glob
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "EAGLE"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))
import torch

from eagle_spinquant import experiment, study, anchor_quant as aq
from eagle_spinquant import fake_w4a4_draft as fq
from eagle_spinquant.concat_selective_projection import (
    ConcatSelectiveDraftAdapter)
from capture_adapter_anchor import NAME2SITE

CONTROLLED = ("q", "k", "v", "o", "gate", "up", "down", "W_rec")


def metrics(folds, anchor, tw0, D):
    per = {}
    agg = dict(all=[0, 0], controlled=[0, 0])
    dq_num = dq_den = dfp_num = dfp_den = 0.0
    for k in aq.QSITES:
        s = anchor[k]["scale"]
        c = aq.codes(folds[k], s)
        c0 = anchor[k]["c0"]
        f = (c != c0)
        per[k] = dict(flip=float(f.float().mean()), n=int(f.numel()),
                      n_flips=int(f.sum()))
        agg["all"][0] += int(f.sum())
        agg["all"][1] += f.numel()
        if k in CONTROLLED:
            agg["controlled"][0] += int(f.sum())
            agg["controlled"][1] += f.numel()
        if k == "W_first":
            fe, fh = f[:, :D], f[:, D:]
            per["W_first_e"] = dict(flip=float(fe.float().mean()),
                                    n=int(fe.numel()),
                                    n_flips=int(fe.sum()))
            per["W_first_h"] = dict(flip=float(fh.float().mean()),
                                    n=int(fh.numel()),
                                    n_flips=int(fh.sum()))
        qw, qw0 = s * c.float(), s * c0.float()
        dq_num += float((qw - qw0).pow(2).sum())
        dq_den += float(qw0.pow(2).sum())
        dfp_num += float((folds[k].float() - tw0[k].float()).pow(2).sum())
        dfp_den += float(tw0[k].float().pow(2).sum())
    return dict(H_Q_all=agg["all"][0] / agg["all"][1],
                H_Q_controlled=agg["controlled"][0] / agg["controlled"][1],
                n_flips_all=agg["all"][0],
                n_flips_controlled=agg["controlled"][0],
                D_Q=dq_num / max(dq_den, 1e-12),
                D_FP=dfp_num / max(dfp_den, 1e-12),
                per_site=per)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--anchor", required=True)
    ap.add_argument("--rd-ckpt", required=True)
    ap.add_argument("--alpha", type=float, default=32.89964245299412)
    ap.add_argument("--ckpt-glob", action="append", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    anchor = aq.load_anchor(args.anchor)
    cfg = experiment.load_config(None)
    paths = experiment.resolve_paths(cfg)
    rr = cfg.get("paths", {}).get("rotations_root")
    model, stash, _ = study.build_study_target(
        paths["target_path"], paths["draft_path"],
        cfg["model"]["target"], "full", "learned_chat_w4a4kv16",
        "w4a4", 0, device=args.device, rotations_root=rr)
    ea = model.ea_layer
    sd0 = {k: v.detach().clone() for k, v in ea.state_dict().items()}
    D = ea.fc.weight.shape[0]
    ck = torch.load(args.rd_ckpt, map_location="cpu", weights_only=False)
    st = dict(stash)
    st["R1"] = ck["R_D"].double()
    kw = dict(quant_first="fake_w4a4", quant_recurrent="fake_w4a4",
              quant_ar="fake_w4a4", ar_r2r4=True,
              embed_scale_alpha=args.alpha,
              first_fold_R=stash["R1"].clone())

    def capture(draft_sd):
        ea.load_state_dict(sd0, strict=True)
        if draft_sd:
            sd_new = torch.load(draft_sd, map_location="cpu",
                                weights_only=False)
            sd_new = sd_new.get("draft_state_dict",
                                sd_new.get("model", sd_new))
            ea.load_state_dict({k: v.to(ea.fc.weight.dtype)
                                for k, v in sd_new.items()}, strict=True)
        ea.to(args.device)
        captured = {}
        orig = fq._weight_fake_quant

        def spy(w, bits=4, name=None):
            if name in NAME2SITE:
                captured[NAME2SITE[name]] = w.detach().float().cpu()
            return orig(w, bits, name=name)

        fq._weight_fake_quant = spy
        try:
            ad = ConcatSelectiveDraftAdapter(
                model, st, args.device, torch.float16, variant="folded",
                first_hidden_mode="gamma_R1", trace=False, **kw)
            ad.install()
            ad.uninstall()
        finally:
            fq._weight_fake_quant = orig
        assert all(s in captured for s in aq.QSITES)
        return captured

    tw0 = capture(None)
    out = {}
    files = sorted({f for g in args.ckpt_glob for f in glob.glob(g)})
    print(f"[cdrift] {len(files)} candidates", flush=True)
    for f in files:
        folds = capture(f)
        m = metrics(folds, anchor, tw0, D)
        out[os.path.basename(f)] = m
        ps = m["per_site"]
        print(f"[cdrift] {os.path.basename(f)}: all={m['H_Q_all']:.5f}"
              f" ctrl={m['H_Q_controlled']:.5f}"
              f" Wf_e={ps['W_first_e']['flip']:.5f}"
              f" Wf_h={ps['W_first_h']['flip']:.5f}"
              f" down={ps['down']['flip']:.5f} D_Q={m['D_Q']:.6f}",
              flush=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    print(f"[cdrift] wrote {args.out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())

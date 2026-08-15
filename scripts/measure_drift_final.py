#!/usr/bin/env python
"""Drift metrics (H_Q / D_Q / D_FP / per-site flips) for the qanchor
study's FINAL candidates, measured on the DEPLOYED ADAPTER's own folds
(the anchor cell space) with the target model built once.

Writes <run-dir>/tables/final_drift_metrics.json.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--rd-ckpt", required=True)
    ap.add_argument("--alpha", type=float, default=32.89964245299412)
    ap.add_argument("--ckpt-glob", action="append", default=[],
                    help="glob(s) of draft-sd ckpts to measure")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    anchor = aq.load_anchor(os.path.join(args.run_dir, "ckpts",
                                         "anchor_gsr5_ptq.pt"))
    cfg = experiment.load_config(None)
    paths = experiment.resolve_paths(cfg)
    rr = cfg.get("paths", {}).get("rotations_root")
    model, stash, _ = study.build_study_target(
        paths["target_path"], paths["draft_path"],
        cfg["model"]["target"], "full", "learned_chat_w4a4kv16",
        "w4a4", 0, device=args.device, rotations_root=rr)
    ea = model.ea_layer
    sd0 = {k: v.detach().clone() for k, v in ea.state_dict().items()}
    ck = torch.load(args.rd_ckpt, map_location="cpu",
                    weights_only=False)
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
                                for k, v in sd_new.items()},
                               strict=True)
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
                model, st, args.device, torch.float16,
                variant="folded", first_hidden_mode="gamma_R1",
                trace=False, **kw)
            ad.install()
            ad.uninstall()
        finally:
            fq._weight_fake_quant = orig
        missing = [s for s in aq.QSITES if s not in captured]
        assert not missing, f"sites not captured: {missing}"
        return captured

    tw0 = capture(None)          # anchor folds (public draft masters)
    out = {}
    files = sorted({f for g in args.ckpt_glob for f in glob.glob(g)})
    print(f"[drift] {len(files)} candidates", flush=True)
    for f in files:
        name = os.path.basename(f)
        folds = capture(f)
        m = aq.drift_metrics(folds, anchor, tw0=tw0)
        m["per_site"] = {k: v for k, v in m["per_site"].items()}
        out[name] = m
        print(f"[drift] {name}: H_Q={m['H_Q']:.5f} D_Q={m['D_Q']:.6f}"
              f" D_FP={m['D_FP']:.6f}"
              f" down={m['per_site']['down']['flip']:.5f}", flush=True)
    p = os.path.join(args.run_dir, "tables",
                     "final_drift_metrics.json")
    with open(p, "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    print(f"[drift] wrote {p}", flush=True)


if __name__ == "__main__":
    sys.exit(main())

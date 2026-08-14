#!/usr/bin/env python
"""Capture the qanchor study anchor from the DEPLOYED ADAPTER's own
folded weights (definitive fix for the tie-boundary mismatch between
trainer-replica folds and adapter folds: the anchor cell space is
defined on the deployment artifact itself).

Spies fq._weight_fake_quant inputs during a real adapter install
(optionally with a --draft-sd loaded), captures all 9 pre-quant folded
site weights, runs the official MSE-clip quantizer once per site, and
saves the anchor state (+ expected codes). Also usable to extract the
adapter-fold codes of ANY checkpoint (--codes-only) for B-grid
construction.
"""
import argparse, glob, hashlib, json, os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "EAGLE"))
import torch
from eagle_spinquant import experiment, study, anchor_quant as aq
from eagle_spinquant import fake_w4a4_draft as fq
from eagle_spinquant.concat_selective_projection import (
    ConcatSelectiveDraftAdapter)

NAME2SITE = {
    "projection_first_preR": "W_first",
    "projection_recurrent_preR": "W_rec",
    "ar.q_proj": "q",
    "ar.k_proj": "k",
    "ar.v_proj": "v",
    "ar.o_proj": "o",
    "ar.gate_proj": "gate",
    "ar.up_proj": "up",
    "ar.down_proj": "down",
}


def capture_adapter_folds(draft_sd=None, rd_ckpt=None,
                          alpha=32.89964245299412, alpha_rec=None,
                          device="cuda:0"):
    cfg = experiment.load_config(None)
    paths = experiment.resolve_paths(cfg)
    rr = cfg.get("paths", {}).get("rotations_root")
    model, stash, _ = study.build_study_target(
        paths["target_path"], paths["draft_path"], cfg["model"]["target"],
        "full", "learned_chat_w4a4kv16", "w4a4", 0, device=device,
        rotations_root=rr)
    if draft_sd:
        sd_new = torch.load(draft_sd, map_location="cpu",
                            weights_only=False)
        sd_new = sd_new.get("draft_state_dict",
                            sd_new.get("model", sd_new))
        ea = model.ea_layer
        ea.load_state_dict({k: v.to(ea.fc.weight.dtype)
                            for k, v in sd_new.items()}, strict=True)
        ea.to(device)
    st = stash
    kw = dict(quant_first="fake_w4a4", quant_recurrent="fake_w4a4",
              quant_ar="fake_w4a4", ar_r2r4=True,
              embed_scale_alpha=alpha)
    if alpha_rec is not None:
        kw["embed_scale_alpha_rec"] = alpha_rec
    if rd_ckpt:
        ck = torch.load(rd_ckpt, map_location="cpu", weights_only=False)
        st = dict(stash)
        kw["first_fold_R"] = stash["R1"].clone()
        st["R1"] = ck["R_D"].double()
    captured = {}
    orig = fq._weight_fake_quant

    def spy(w, bits=4, name=None):
        if name in NAME2SITE:
            captured[NAME2SITE[name]] = w.detach().float().cpu()
        return orig(w, bits, name=name)

    fq._weight_fake_quant = spy
    try:
        ad = ConcatSelectiveDraftAdapter(
            model, st, device, torch.float16, variant="folded",
            first_hidden_mode="gamma_R1", trace=False, **kw)
        ad.install()
        ad.uninstall()
    finally:
        fq._weight_fake_quant = orig
    missing = [s for s in aq.QSITES if s not in captured]
    assert not missing, f"sites not captured: {missing}"
    del model
    torch.cuda.empty_cache()
    return captured


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--rd-ckpt", required=True)
    ap.add_argument("--alpha", type=float, default=32.89964245299412)
    ap.add_argument("--draft-sd", default=None)
    ap.add_argument("--out-prefix", default="anchor_gsr5_ptq")
    ap.add_argument("--codes-only", action="store_true",
                    help="extract codes of --draft-sd under an EXISTING "
                         "anchor (ckpts/<out-prefix>.pt) instead of "
                         "capturing a new anchor")
    args = ap.parse_args()
    ck = os.path.join(args.run_dir, "ckpts")
    folds = capture_adapter_folds(args.draft_sd, args.rd_ckpt,
                                  args.alpha)
    if args.codes_only:
        anchor = aq.load_anchor(os.path.join(ck,
                                             "anchor_gsr5_ptq.pt"))
        codes = {k: aq.codes(folds[k], anchor[k]["scale"])
                 for k in aq.QSITES}
        out = os.path.join(ck, f"codes__{args.out_prefix}.pt")
        torch.save(codes, out)
        m = {k: float((codes[k] != anchor[k]["c0"]).float().mean())
             for k in aq.QSITES}
        print(f"[cap] codes -> {out}; flips vs anchor: "
              + " ".join(f"{k}:{v:.4f}" for k, v in m.items()),
              flush=True)
        return
    anchor = aq.capture_anchor(folds)
    p = os.path.join(ck, f"{args.out_prefix}.pt")
    aq.save_anchor(anchor, p)
    torch.save({k: anchor[k]["c0"] for k in aq.QSITES},
               os.path.join(ck, "expected_c0.pt"))
    sha = hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]
    print(f"[cap] ADAPTER-fold anchor -> {p} sha {sha}", flush=True)


if __name__ == "__main__":
    sys.exit(main())

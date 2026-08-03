#!/usr/bin/env python
"""Stage calibration tensors for the R-EP3-P study (spec §10).

Reuses the validated held-out EP3-P calibration captures — no new GPU
collection needed, identical manifests:
  int4 (primary): EP3-P visualization run tensors
      X_first_raw [4096, 8192], X_rec{1..4}_raw, W_before [4096,8192],
      bias  (raw e-slice, gamma_R1 first interface, c4:16 calib)
  fp16 (secondary control): GQ-study p3exp capture
      first [., 8192], rec [., 8192], W_first (identity interface)

Writes tensors/calib_int4.pt, tensors/calib_fp16.pt + SHA manifest.
"""
import hashlib, json, os, sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main():
    rd = os.path.join(ROOT, open(os.path.join(
        ROOT, "runs", "REP3P_RUN_DIR")).read().strip())
    viz = os.path.join(ROOT, open(os.path.join(
        ROOT, "runs", "EP3PVIZ_RUN_DIR")).read().strip())
    src_int4 = os.path.join(viz, "plot_data", "ep3p_tensors.pt")
    b = torch.load(src_int4, map_location="cpu", weights_only=False)
    int4 = dict(X_first=b["X_first_raw"].half(),
                W=b["W_before"].float(), bias=b.get("bias"))
    for k in (1, 2, 3, 4):
        int4[f"X_rec{k}"] = b[f"X_rec{k}_raw"].half()
    int4["X_rec_all"] = torch.cat([int4[f"X_rec{k}"]
                                   for k in (1, 2, 3, 4)])
    src_fp16 = os.path.join(
        ROOT, "runs",
        "eagle1_generic_qat_pathwise_p3exp_rcal_20260731_154652",
        "tables", "p3exp_capture_fp16.pt")
    g = torch.load(src_fp16, map_location="cpu", weights_only=False)
    fp16 = dict(X_first=g["first"].half(), X_rec_all=g["rec"].half(),
                W=g["W_first"].float(), bias=None)
    os.makedirs(os.path.join(rd, "tensors"), exist_ok=True)
    p1 = os.path.join(rd, "tensors", "calib_int4.pt")
    p2 = os.path.join(rd, "tensors", "calib_fp16.pt")
    torch.save(int4, p1)
    torch.save(fp16, p2)
    json.dump(dict(int4_source=src_int4, int4_source_sha=sha(src_int4),
                   fp16_source=src_fp16, fp16_source_sha=sha(src_fp16),
                   calib_int4_sha=sha(p1), calib_fp16_sha=sha(p2),
                   shapes={k: list(v.shape) for k, v in int4.items()
                           if hasattr(v, "shape")}),
              open(os.path.join(rd, "manifests",
                                "tensor_manifest.json"), "w"),
              indent=1)
    print(f"[collect] staged int4 first={tuple(int4['X_first'].shape)}"
          f" rec_all={tuple(int4['X_rec_all'].shape)} + fp16 control")
    return 0


if __name__ == "__main__":
    sys.exit(main())

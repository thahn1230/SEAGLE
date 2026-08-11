#!/usr/bin/env python
"""R6 smoke gate: after a 60-step training run verify (1) R6 moved away
from R2_base (training is live), (2) R6 stayed orthogonal, (3) R_D in
the checkpoint is BITWISE the frozen R5 (SharedRotation never trained),
(4) no core_weights key (draft weights untouched), (5) loss/alpha log
present. Writes gradchecks/smokecheck.json only on PASS."""
import json, os, sys
import torch

rd = sys.argv[1]
R5 = ("runs/eagle1_draft_residual_rotation_ep3p_20260804_165317/"
      "rotations/RD_HYB_s2.pt")
ck = torch.load(os.path.join(rd, "rotations", "R6_SMOKE.pt"),
                map_location="cpu", weights_only=False)
r5 = torch.load(R5, map_location="cpu", weights_only=False)["R_D"]
fails = []
HD = 128
R6 = ck.get("R6")
if R6 is None:
    fails.append("no R6 key in smoke ckpt")
else:
    g = torch.Generator().manual_seed(0)
    R2B = torch.linalg.qr(torch.randn(HD, HD, generator=g,
                                      dtype=torch.float64))[0]
    d = float((R6.double() - R2B).abs().max())
    if d == 0.0:
        fails.append("R6 did not move from R2_base after 60 steps")
    I = torch.eye(HD, dtype=torch.float64)
    oe = float((R6.double().t() @ R6.double() - I).abs().max())
    if oe > 1e-4:
        fails.append(f"R6 orthogonality {oe}")
d5 = float((ck["R_D"] - r5).abs().max())
if d5 != 0.0:
    fails.append(f"R_D changed vs frozen R5 (max {d5})")
if "core_weights" in ck:
    fails.append("core_weights present — draft weights were trained")
if not any("val_expected_tau" in r for r in ck.get("log", [])):
    fails.append("no validation record in log")
out = dict(fails=fails, verdict="PASS" if not fails else "FAIL",
           r6_move_from_base=d if R6 is not None else None,
           r6_orth=oe if R6 is not None else None,
           r6_generator_fro=ck.get("meta", {}).get("r6_generator_fro"),
           best_val=ck.get("meta", {}).get("best_val"))
print(json.dumps(out, indent=1))
if fails:
    sys.exit(1)
json.dump(out, open(os.path.join(rd, "gradchecks", "smokecheck.json"),
                    "w"), indent=1)

#!/usr/bin/env python
"""§19 basis-geometry: R5_preQAT vs R5_new_afterQAT(armD) vs
R5_fresh_on_B(P3C) vs R5_reopt(P3D). Writes
<run>/geometry/r5_order_geometry.json."""
import json, os, sys
import torch
sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))), "src"))
from eagle_spinquant.residual_rotation import rotation_geometry  # noqa

rd = sys.argv[1]
R5_OLD = ("runs/eagle1_draft_residual_rotation_ep3p_20260804_165317/"
          "rotations/RD_HYB_s2.pt")


def load(p):
    return torch.load(p, map_location="cpu",
                      weights_only=False)["R_D"].double()


def eig_angles(Q):
    ev = torch.linalg.eigvals(Q)
    return torch.angle(ev).abs()


old = load(R5_OLD)
out = {}
KIND = "learned_chat_w4a4kv16"
RT = torch.load("outputs/rotations/learned_chat_w4a4kv16/R.bin",
                map_location="cpu", weights_only=False)["R1"].double()
for name, path in (("armD_qat_first", f"{rd}/rotations/R5_QATFIRST.pt.best.pt"),
                   ("p3C_fresh_on_B", f"{rd}/rotations/R5NEW_C.pt.best.pt"),
                   ("p3D_reopt_on_B", f"{rd}/rotations/R5REOPT_D.pt.best.pt")):
    if not os.path.exists(path):
        out[name] = "missing"
        continue
    new = load(path)
    Q = old.t() @ new
    ang = eig_angles(Q)
    QT = RT.t() @ new
    angT = eig_angles(QT)
    out[name] = dict(
        vs_old_r5=dict(fro=round(float((new - old).norm()), 4),
                       max_elem=round(float((new - old).abs().max()), 5),
                       eigphase_median=round(float(ang.median()), 4),
                       eigphase_max=round(float(ang.max()), 4),
                       planes_gt_0p1rad=int((ang > 0.1).sum()) // 2),
        vs_R_T=dict(fro=round(float((new - RT).norm()), 4),
                    eigphase_median=round(float(angT.median()), 4)),
        orth_err=round(float((new.t() @ new
                              - torch.eye(4096, dtype=torch.float64))
                             .abs().max()), 8))
old_vs_rt = rotation_geometry(old.float(), RT.float())
out["old_r5_vs_R_T"] = {k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in old_vs_rt.items()}
dst = os.path.join(rd, "geometry", "r5_order_geometry.json")
os.makedirs(os.path.dirname(dst), exist_ok=True)
json.dump(out, open(dst, "w"), indent=1)
print(json.dumps(out, indent=1))

#!/usr/bin/env python
"""RQ1: is R_D actually different from R_T — and where does it matter?

For every rotation checkpoint under <run>/rotations (plus the LK
transfer rotation and R_T itself), reports:

A. Geometry: geodesic distance ||log(R_T^T R_D)||_F, Frobenius
   ||R_D - R_T||_F, max |element| of (R_D - R_T), orthogonality error.
B. Eigenangle spectrum of the relative rotation Q = R_T^T R_D:
   eigenvalues come in pairs e^{+-i theta}; reports the distribution of
   theta (max, mean, count above thresholds) — "how many 2D planes were
   rotated, and by how much".
C. Quantization-relevant functional stats per QSITE weight, in the
   R_T gauge vs the R_D gauge (official SpinQuant W4 quantizer):
   NMSE(Q(W), W), excess kurtosis, absmax — the quantities a rotation
   can actually change at the quantization boundary.
D. Activation stats through the exact-path forward on real corpus
   windows: per-linear input absmax / excess kurtosis (via an _ActQ
   recording shim), R_T vs R_D gauge.

fp16-gauge framing: with quantization OFF, R_D is a gauge and none of
C/D can differ beyond roundoff — every reported difference lives at the
quantization boundary. Writes <run>/geometry/rq1_geometry.json.
"""
import argparse, glob, json, os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "EAGLE"))

import torch

from eagle_spinquant import experiment, study
from eagle_spinquant import fake_w4a4_draft as fq
from eagle_spinquant.exact_qat_rotated_draft import ExactQATRotatedDraft
from eagle_spinquant.residual_rotation import (SharedRotation,
                                               FullRotation,
                                               rotation_geometry)

KIND = "learned_chat_w4a4kv16"
D = 4096
ALPHA_F, ALPHA_R = D ** 0.40, D ** 0.45


def eigenangles(R_T, R_D):
    """theta spectrum of Q = R_T^T R_D (radians)."""
    Q = (R_T.double().t() @ R_D.double())
    ev = torch.linalg.eigvals(Q)
    th = torch.atan2(ev.imag, ev.real).abs()
    th = th.sort(descending=True).values
    return dict(
        max=float(th[0]), mean=float(th.mean()),
        median=float(th.median()),
        n_gt_01=int((th > 0.1).sum()), n_gt_05=int((th > 0.5).sum()),
        n_gt_10=int((th > 1.0).sum()),
        top16=[round(float(x), 4) for x in th[:16]])


def weight_stats(w):
    wf = w.detach().float()
    qw = fq._weight_fake_quant(wf.half().cuda(), 4).float().to(wf.device)
    nmse = float(((qw - wf) ** 2).sum() / (wf ** 2).sum())
    x = wf.flatten()
    k = float(((x - x.mean()) ** 4).mean() / (x.var() ** 2)) - 3.0
    return dict(w4_nmse=round(nmse, 6), kurtosis_excess=round(k, 3),
                absmax=round(float(x.abs().max()), 4))


class _ActRecorder:
    """Wraps the module's _ActQ to record per-call input stats."""

    def __init__(self, aq):
        self.aq = aq
        self.absmax, self.kurt, self.n = 0.0, 0.0, 0

    def __call__(self, x):
        xf = x.detach().float()
        self.absmax = max(self.absmax, float(xf.abs().max()))
        v = xf.flatten()
        self.kurt += float(((v - v.mean()) ** 4).mean()
                           / (v.var() ** 2)) - 3.0
        self.n += 1
        return self.aq(x)


def act_stats(eq, windows, K=4):
    rec = _ActRecorder(eq.aq)
    eq.aq = rec
    with torch.no_grad():
        for w in windows:
            tok = w["tok_ids"].long()[None].to(eq.dev)
            a = w["a_seq"].float()[None].to(eq.dev)
            teach = w["teacher_tokens"][:K].long()[None].to(eq.dev)
            eq.forward_chain(tok, a, K, recur_tokens=teach)
    eq.aq = rec.aq
    return dict(absmax=round(rec.absmax, 3),
                kurtosis_excess_mean=round(rec.kurt / max(rec.n, 1), 3),
                n_calls=rec.n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--extra-ckpts", default="",
                    help="csv of extra rotation ckpts (e.g. LK transfer)")
    ap.add_argument("--n-act-windows", type=int, default=16)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") in \
        tuple(str(i) for i in range(8))
    dev = args.device
    torch.set_grad_enabled(False)

    cfg = experiment.load_config(None)
    paths = experiment.resolve_paths(cfg)
    rr = cfg.get("paths", {}).get("rotations_root")
    R = torch.load(study.r_bin_path(KIND, 0, paths["target_path"], rr),
                   map_location="cpu", weights_only=False)
    R_T = R["R1"].float()

    from safetensors.torch import load_file, safe_open
    sd = None
    for fn in ("model.safetensors", "pytorch_model.bin"):
        p = os.path.join(paths["draft_path"], fn)
        if os.path.exists(p):
            sd = (load_file(p) if fn.endswith("safetensors") else
                  torch.load(p, map_location="cpu", weights_only=True))
            break
    assert sd is not None, paths["draft_path"]
    gamma = W_lm = None
    for p in sorted(glob.glob(os.path.join(paths["target_path"],
                                           "*.safetensors"))):
        with safe_open(p, framework="pt") as f:
            if "model.norm.weight" in f.keys() and gamma is None:
                gamma = f.get_tensor("model.norm.weight").float()
            if "lm_head.weight" in f.keys() and W_lm is None:
                W_lm = f.get_tensor("lm_head.weight").float()

    # activation windows from the merged corpus (first shard)
    man = json.load(open(os.path.join(
        args.run_dir, "manifests", "lkcorpus__rd_t4_all.json")))
    sh = torch.load(os.path.join(args.run_dir, "rotations",
                                 man["shards"][0]["shard"]),
                    map_location="cpu", weights_only=False)
    windows = sh["windows"][:args.n_act_windows]

    cks = sorted(glob.glob(os.path.join(args.run_dir, "rotations",
                                        "RD_*.pt")))
    cks += [c for c in args.extra_ckpts.split(",") if c]

    def build_eq(rot):
        return ExactQATRotatedDraft(
            sd, R_T, gamma, W_lm, rot, alpha_init=ALPHA_F,
            alpha_rec_init=ALPHA_R, w_bits=4, a_bits=4,
            draft_kv_bits=16, device=dev, first_fold_R=R_T)

    out = {}
    QS = ("W_first", "W_rec", "q", "k", "v", "o", "gate", "up", "down")

    def entry(name, rot_matrix):
        rot = (SharedRotation(R_T) if rot_matrix is None
               else FullRotation(rot_matrix.float())).to(dev)
        eq = build_eq(rot)
        tw = eq.transformed_weights(exact=False)
        e = dict(
            geometry=(rotation_geometry(rot_matrix, R_T)
                      if rot_matrix is not None
                      else rotation_geometry(R_T, R_T)),
            weights={k: weight_stats(tw[k]) for k in QS},
            activations=act_stats(eq, windows))
        if rot_matrix is not None:
            e["eigenangles"] = eigenangles(R_T, rot_matrix)
        out[name] = e
        print(f"[rq1] {name}: geo={e['geometry']}", flush=True)
        del eq
        torch.cuda.empty_cache()

    entry("SHARED_RT", None)
    for ck in cks:
        name = os.path.basename(ck)[:-3]
        d = torch.load(ck, map_location="cpu", weights_only=False)
        entry(name, d["R_D"].float())

    op = os.path.join(args.run_dir, "geometry", "rq1_geometry.json")
    os.makedirs(os.path.dirname(op), exist_ok=True)
    json.dump(out, open(op, "w"), indent=1)
    print(f"[rq1] -> {op}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

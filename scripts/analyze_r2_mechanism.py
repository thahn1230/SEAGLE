#!/usr/bin/env python
"""R2 mechanism analysis for the GS R1/R2 factorial study (spec section 17).

For each named checkpoint (arm) plus the implicit BASE (R1_T + R2_B):

A. R2 geometry vs R2_B: relative eigenangle spectrum of Q = R2_B^T R2_D
   (128-dim), Frobenius distance, max element change, orthogonality
   error, generator ||B||_F (from ck['R2_W']).
B. R1 geometry vs R1_T (for A1/A3 context): geodesic/Frobenius.
C. Quantization proxies in the arm's deployed gauge (GS fold, W4A4):
   per-QSITE W4 NMSE / excess kurtosis / absmax — v/o are the R2-relevant
   sites; the rest isolate R1 effects.
D. o_proj-input activation stats through the exact-path forward on held-
   out corpus windows: absmax, excess kurtosis, official per-token-asym
   A4 code statistics (act-quant NMSE, mean code entropy over 16 levels,
   fraction of values in the two extreme codes) — captured at the o_in
   call site specifically (call-index shim; forward call order per depth
   is [proj_z, q_in, k_in, v_in, o_in, gate_in, up_in, down_in]).
E. Teacher agreement per depth k=1..4 on HELD-OUT (val-tail) windows:
   full-vocab overlap alpha_k = sum min(p,q) and KL(p||q), teacher-forced
   — the per-depth panel for A0/A1/A2/A3.

Writes <run>/geometry/r2_mechanism.json.
"""
import argparse, glob, json, os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "EAGLE"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from eagle_spinquant import experiment, study, lk_losses as L
from eagle_spinquant import fake_w4a4_draft as fq
from eagle_spinquant.exact_qat_rotated_draft import ExactQATRotatedDraft
from eagle_spinquant.pmg_configs import EP3G
from eagle_spinquant.residual_rotation import (SharedRotation,
                                               FullRotation,
                                               rotation_geometry)
from analyze_rd_rotation_geometry import eigenangles, weight_stats
from check_r1r2_gs_parity import FixedR2

KIND = "learned_chat_w4a4kv16"
ALPHA_GS = float(EP3G["int4"])
SITES = ("proj_z", "q_in", "k_in", "v_in", "o_in", "gate_in", "up_in",
         "down_in")


class _SiteActRecorder:
    """Per-call-site activation recorder wrapping the shared _ActQ."""

    def __init__(self, aq):
        self.aq = aq
        self.i = 0
        self.stats = {s: dict(absmax=0.0, kurt=0.0, qnmse=0.0,
                              entropy=0.0, extreme_frac=0.0, n=0)
                      for s in SITES}

    def __call__(self, x):
        site = SITES[self.i % len(SITES)]
        self.i += 1
        st = self.stats[site]
        xf = x.detach().float()
        st["absmax"] = max(st["absmax"], float(xf.abs().max()))
        v = xf.flatten()
        st["kurt"] += float(((v - v.mean()) ** 4).mean()
                            / (v.var() ** 2 + 1e-12)) - 3.0
        # official per-token asym A4 code stats
        q = fq._act_quantizer(4)
        x2 = xf.reshape(-1, xf.shape[-1]).half()
        q.find_params(x2)
        xq = q(x2).float()
        q.free()
        st["qnmse"] += float(((xq - x2.float()) ** 2).sum()
                             / (x2.float() ** 2).sum().clamp(min=1e-9))
        # code index per element (16 levels per token via min/max scaling)
        mn = x2.float().min(-1, keepdim=True).values
        mx = x2.float().max(-1, keepdim=True).values
        code = ((x2.float() - mn) / (mx - mn + 1e-9) * 15).round()
        hist = torch.stack([(code == c).float().mean(-1)
                            for c in range(16)], -1)
        ent = -(hist.clamp(min=1e-9) * hist.clamp(min=1e-9).log()).sum(-1)
        st["entropy"] += float(ent.mean())
        st["extreme_frac"] += float(((code == 0) | (code == 15))
                                    .float().mean())
        st["n"] += 1
        return self.aq(x)

    def summary(self):
        out = {}
        for s, st in self.stats.items():
            n = max(st["n"], 1)
            out[s] = dict(absmax=round(st["absmax"], 3),
                          kurtosis_excess=round(st["kurt"] / n, 3),
                          a4_qnmse=round(st["qnmse"] / n, 6),
                          a4_code_entropy=round(st["entropy"] / n, 4),
                          a4_extreme_frac=round(st["extreme_frac"] / n,
                                                4),
                          n_calls=st["n"])
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--ckpts", required=True,
                    help="csv name=path (e.g. A1=.../RD_GS_A1_s1001.pt)")
    ap.add_argument("--n-act-windows", type=int, default=16)
    ap.add_argument("--n-depth-windows", type=int, default=128)
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
    gamma = W_lm = None
    for p in sorted(glob.glob(os.path.join(paths["target_path"],
                                           "*.safetensors"))):
        with safe_open(p, framework="pt") as f:
            if "model.norm.weight" in f.keys() and gamma is None:
                gamma = f.get_tensor("model.norm.weight").float()
            if "lm_head.weight" in f.keys() and W_lm is None:
                W_lm = f.get_tensor("lm_head.weight").float()

    man = json.load(open(os.path.join(
        args.run_dir, "manifests", "lkcorpus__rd_gs_all.json")))
    # activation windows: first shard head; depth windows: VAL TAIL
    first = torch.load(os.path.join(args.run_dir, "rotations",
                                    man["shards"][0]["shard"]),
                       map_location="cpu", weights_only=False)["windows"]
    last = torch.load(os.path.join(args.run_dir, "rotations",
                                   man["shards"][-1]["shard"]),
                      map_location="cpu", weights_only=False)["windows"]
    act_windows = first[:args.n_act_windows]
    depth_windows = last[-args.n_depth_windows:]

    R2_B = fq.baseline_r2(0)
    arms = dict(BASE=(None, None))
    for spec in args.ckpts.split(","):
        name, _, path = spec.partition("=")
        ck = torch.load(path, map_location="cpu", weights_only=False)
        arms[name] = (ck.get("R_D"), ck)

    out = {}
    QS = ("W_first", "W_rec", "q", "k", "v", "o", "gate", "up", "down")
    for name, (R_D, ck) in arms.items():
        rot = (SharedRotation(R_T) if R_D is None
               else FullRotation(R_D.float())).to(dev)
        R2_D = None if ck is None else ck.get("R2_D")
        r2_rot = (FixedR2(R2_D).to(dev) if R2_D is not None else None)
        eq = ExactQATRotatedDraft(
            sd, R_T, gamma, W_lm, rot, alpha_init=ALPHA_GS, w_bits=4,
            a_bits=4, draft_kv_bits=16, device=dev, first_fold_R=R_T,
            r2_rot=r2_rot)
        e = dict(r1_geometry=rotation_geometry(
            R_T if R_D is None else R_D.float(), R_T))
        if R2_D is not None:
            e["r2_geometry"] = rotation_geometry(R2_D.float(),
                                                 R2_B.float())
            e["r2_eigenangles"] = eigenangles(R2_B.float(),
                                              R2_D.float())
            Q = R2_B.double().t() @ R2_D.double()
            ev = torch.linalg.eigvals(Q)
            e["r2_eigenangles_full"] = [
                round(float(t), 6) for t in
                torch.atan2(ev.imag, ev.real).abs().sort(
                    descending=True).values]
            if ck.get("R2_W") is not None:
                B = ck["R2_W"] - ck["R2_W"].t()
                e["r2_generator_frob"] = float(B.norm())
        else:
            e["r2_geometry"] = rotation_geometry(R2_B.float(),
                                                 R2_B.float())
        tw = eq.transformed_weights(exact=False)
        e["weights"] = {k: weight_stats(tw[k]) for k in QS}
        rec = _SiteActRecorder(eq.aq)
        eq.aq = rec
        for w in act_windows:
            tok = w["tok_ids"].long()[None].to(dev)
            a = w["a_seq"].float()[None].to(dev)
            teach = w["teacher_tokens"][:4].long()[None].to(dev)
            eq.forward_chain(tok, a, 4, recur_tokens=teach)
        eq.aq = rec.aq
        e["activations"] = rec.summary()
        # per-depth teacher agreement on held-out windows
        kl_k = [0.0] * 4
        al_k = [0.0] * 4
        nb = 0
        for i0 in range(0, len(depth_windows), 16):
            ws = depth_windows[i0:i0 + 16]
            tok = torch.stack([w["tok_ids"].long() for w in ws]).to(dev)
            a = torch.stack([w["a_seq"].float() for w in ws]).to(dev)
            teach = torch.stack([w["teacher_tokens"][:4].long()
                                 for w in ws]).to(dev)
            zT = torch.stack([w["teacher_logits"][:4].float()
                              for w in ws]).to(dev)
            outs = eq.forward_chain(tok, a, 4, recur_tokens=teach)
            for k, (lg, _h, _c) in enumerate(outs):
                kl_k[k] += float(L.kl_full(zT[:, k], lg).mean())
                al_k[k] += float(L.overlap_alpha(zT[:, k], lg).mean())
            nb += 1
        e["depth_kl"] = [round(x / nb, 5) for x in kl_k]
        e["depth_overlap"] = [round(x / nb, 5) for x in al_k]
        out[name] = e
        print(f"[r2mech] {name}: v_nmse={e['weights']['v']['w4_nmse']} "
              f"o_nmse={e['weights']['o']['w4_nmse']} "
              f"o_in={e['activations']['o_in']} "
              f"overlap={e['depth_overlap']}", flush=True)
        del eq
        torch.cuda.empty_cache()

    op = os.path.join(args.run_dir, "geometry", "r2_mechanism.json")
    os.makedirs(os.path.dirname(op), exist_ok=True)
    json.dump(out, open(op, "w"), indent=1)
    print(f"[r2mech] -> {op}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

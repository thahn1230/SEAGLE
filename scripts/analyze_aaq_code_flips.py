#!/usr/bin/env python
"""AA-QAT study sections 25-26: per-module weight movement and W4
code-flip fractions for trained drafts vs their init, in the DEPLOYED
folded basis.

For each named checkpoint (draft_state_dict export): build the exact
folded W4 sites for init and trained masters under the SAME structural
config (R_D / alpha / alpha_rec), then per site report
  - rel_dF = ||W_tr - W_init||_F / ||W_init||_F   (folded fp16 weights)
  - flip   = fraction of elements whose INT code under the INIT scales
             changes (round(w/s_init) comparison, per-out-channel s)
Writes tables/code_flips.json in --run-dir.
"""
import argparse, glob, json, os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "EAGLE"))
import torch
from eagle_spinquant import experiment, study
from eagle_spinquant.exact_qat_rotated_draft import ExactQATRotatedDraft
from eagle_spinquant.residual_rotation import SharedRotation

KIND = "learned_chat_w4a4kv16"


def folded(sd, R_T, gamma, W_lm, rotM, alpha, alpha_rec, dev):
    rot = SharedRotation(rotM).to(dev)
    m = ExactQATRotatedDraft(sd, R_T, gamma, W_lm, rot, alpha_init=alpha,
                             train_alpha=False, w_bits=4, a_bits=4,
                             draft_kv_bits=16, train_draft_core=False,
                             device=dev, first_fold_R=R_T,
                             alpha_rec_init=alpha_rec)
    tw = m.transformed_weights(exact=False)
    out = {k: tw[k].detach().float() for k in m.QSITES}
    del m
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--jobs", required=True,
                    help="json list of {name, sd, rot(optional ckpt), "
                         "alpha, alpha_rec(optional)}")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = args.device
    torch.set_grad_enabled(False)
    cfg = experiment.load_config(None)
    paths = experiment.resolve_paths(cfg)
    rr = cfg.get("paths", {}).get("rotations_root")
    R = torch.load(study.r_bin_path(KIND, 0, paths["target_path"], rr),
                   map_location="cpu", weights_only=False)
    R_T = R["R1"].float()
    base_sd = torch.load(os.path.join(paths["draft_path"],
                                      "pytorch_model.bin"),
                         map_location="cpu", weights_only=True)
    from safetensors import safe_open
    gamma = W_lm = None
    for p in sorted(glob.glob(os.path.join(paths["target_path"],
                                           "*.safetensors"))):
        with safe_open(p, framework="pt") as f:
            if "model.norm.weight" in f.keys() and gamma is None:
                gamma = f.get_tensor("model.norm.weight").float()
            if "lm_head.weight" in f.keys() and W_lm is None:
                W_lm = f.get_tensor("lm_head.weight").float()

    res = {}
    for job in json.load(open(args.jobs)):
        rotM = R_T
        if job.get("rot"):
            rotM = torch.load(job["rot"], map_location="cpu",
                              weights_only=False)["R_D"].float()
        tr = torch.load(job["sd"], map_location="cpu",
                        weights_only=False)
        tr = tr.get("draft_state_dict", tr)
        init_sd = job.get("init_sd")
        sd0 = base_sd
        if init_sd:
            sd0 = torch.load(init_sd, map_location="cpu",
                             weights_only=False)
            sd0 = sd0.get("draft_state_dict", sd0.get("model", sd0))
        W0 = folded(sd0, R_T, gamma, W_lm, rotM, job["alpha"],
                    job.get("alpha_rec"), dev)
        W1 = folded(tr, R_T, gamma, W_lm, rotM, job["alpha"],
                    job.get("alpha_rec"), dev)
        row = {}
        for k in W0:
            a, b = W0[k], W1[k]
            rel = float((b - a).norm() / a.norm().clamp_min(1e-12))
            s = a.abs().amax(dim=1, keepdim=True) / 7.0  # per-out RTN
            flip = float((torch.round(a / s) != torch.round(b / s))
                         .float().mean())
            row[k] = dict(rel_dF=round(rel, 5), flip=round(flip, 5))
        res[job["name"]] = row
        print(f"[flip] {job['name']}: " + " ".join(
            f"{k}:{v['flip']:.3f}" for k, v in row.items()), flush=True)
        del W0, W1
        torch.cuda.empty_cache()
    out = os.path.join(args.run_dir, "tables", "code_flips.json")
    json.dump(res, open(out, "w"), indent=1)
    print(f"[flip] -> {out}")


if __name__ == "__main__":
    sys.exit(main())

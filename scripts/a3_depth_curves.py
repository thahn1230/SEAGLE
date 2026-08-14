#!/usr/bin/env python
"""A3 depth-collapse control (qanchor study): depth-wise teacher-forced
vs free-running acceptance for a given draft configuration.

Configs:
  --mode w4a4   : deployed GS+R5 W4A4 chain (optionally --draft-sd)
  --mode fp16   : FP16 draft + FP16-target corpus (t16 teacher,
                  identity first-mode, alpha 1.0, w/a bits 16) — the
                  generic-EAGLE control that separates quantization-
                  induced depth degradation from generic depth decay.

Writes tables/a3_depth__<tag>.json with per-depth alpha (tf and free)
and expected_tau, over the last --n-windows of the given corpus.
"""
import argparse, glob, json, os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "EAGLE"))
import torch
from safetensors import safe_open
from eagle_spinquant import experiment, study, lk_losses as L
from eagle_spinquant.exact_qat_rotated_draft import ExactQATRotatedDraft
from eagle_spinquant.residual_rotation import SharedRotation

KIND = "learned_chat_w4a4kv16"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["w4a4", "fp16"])
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--corpus-run-dir", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--draft-sd", default=None)
    ap.add_argument("--rd-ckpt", default=None)
    ap.add_argument("--alpha", type=float, default=32.89964245299412)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--n-windows", type=int, default=128)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    dev = "cuda:0"
    torch.set_grad_enabled(False)
    cfg = experiment.load_config(None)
    paths = experiment.resolve_paths(cfg)
    rr = cfg.get("paths", {}).get("rotations_root")
    R = torch.load(study.r_bin_path(KIND, 0, paths["target_path"], rr),
                   map_location="cpu", weights_only=False)
    R_T = R["R1"].float()
    sd = torch.load(os.path.join(paths["draft_path"],
                                 "pytorch_model.bin"),
                    map_location="cpu", weights_only=True)
    if args.draft_sd:
        t = torch.load(args.draft_sd, map_location="cpu",
                       weights_only=False)
        sd = t.get("draft_state_dict", t.get("model", t))
    gamma = W_lm = None
    for p in sorted(glob.glob(os.path.join(paths["target_path"],
                                           "*.safetensors"))):
        with safe_open(p, framework="pt") as f:
            if "model.norm.weight" in f.keys() and gamma is None:
                gamma = f.get_tensor("model.norm.weight").float()
            if "lm_head.weight" in f.keys() and W_lm is None:
                W_lm = f.get_tensor("lm_head.weight").float()
    if args.mode == "w4a4":
        rotM = R_T
        if args.rd_ckpt:
            rotM = torch.load(args.rd_ckpt, map_location="cpu",
                              weights_only=False)["R_D"].float()
        rot = SharedRotation(rotM).to(dev)
        m = ExactQATRotatedDraft(sd, R_T, gamma, W_lm, rot,
                                 alpha_init=args.alpha,
                                 train_alpha=False, w_bits=4, a_bits=4,
                                 draft_kv_bits=16,
                                 train_draft_core=False, device=dev,
                                 first_fold_R=R_T)
    else:
        # FP16 control: identity interface (fp16 target corpus),
        # no quantization anywhere in the draft chain
        rot = SharedRotation(torch.eye(R_T.shape[0])).to(dev)
        m = ExactQATRotatedDraft(sd, torch.eye(R_T.shape[0]),
                                 torch.ones_like(gamma), W_lm, rot,
                                 alpha_init=1.0, train_alpha=False,
                                 w_bits=16, a_bits=16, draft_kv_bits=16,
                                 train_draft_core=False, device=dev,
                                 first_fold_R=torch.eye(R_T.shape[0]))
    man = json.load(open(args.corpus))
    ws = []
    for shd in man["shards"]:
        ws += torch.load(os.path.join(args.corpus_run_dir, "rotations",
                                      shd["shard"]),
                         map_location="cpu",
                         weights_only=False)["windows"]
    ws = ws[-args.n_windows:]
    res = {}
    for mode in ("tf", "free"):
        als = []
        for i0 in range(0, len(ws), 16):
            b = ws[i0:i0 + 16]
            if len(b) < 2:
                continue
            tok = torch.stack([w["tok_ids"].long() for w in b]).to(dev)
            a = torch.stack([w["a_seq"].float() for w in b]).to(dev)
            teach = torch.stack([w["teacher_tokens"][:args.K].long()
                                 for w in b]).to(dev)
            zT = torch.stack([w["teacher_logits"][:args.K].float()
                              for w in b]).to(dev)
            if mode == "tf":
                outs = m.forward_chain(tok, a, args.K,
                                       recur_tokens=teach)
            else:
                outs = m.forward_chain(tok, a, args.K,
                                       rollout=lambda lg: lg.argmax(-1))
            als.append(torch.stack(
                [L.overlap_alpha(zT[:, k], lg)
                 for k, (lg, _h, _c) in enumerate(outs)], dim=-1))
        A = torch.cat(als)
        res[mode] = dict(
            alpha=[round(float(x), 4) for x in A.mean(0)],
            expected_tau=round(float(L.expected_tau(A).mean()), 4),
            n=int(A.shape[0]))
    res["gap_by_depth"] = [round(a - b, 4) for a, b in
                           zip(res["tf"]["alpha"], res["free"]["alpha"])]
    out = os.path.join(args.run_dir, "tables",
                       f"a3_depth__{args.tag}.json")
    json.dump(res, open(out, "w"), indent=1)
    print(f"[a3] {args.tag}: tf={res['tf']['alpha']} "
          f"free={res['free']['alpha']} gap={res['gap_by_depth']} "
          f"-> {out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())

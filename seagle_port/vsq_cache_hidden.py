"""Cache deployed-target hidden features for draft rotation/QAT training
(§18/§19 H1 policy: the trainer must see the DEPLOYED target interface).

For each corpus row: forward the specified target arm, store the concat
selected-hidden [S, 20480] (basis = that target's residual basis) as bf16.
Separate cache per target basis; provenance JSON with rbin sha.
"""
import argparse
import hashlib
import json
import os

import numpy as np
import torch

from . import spinquant_target as sq

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
SRC = [1, 8, 15, 22, 29]


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--target-mode", default="w4a4")
    ap.add_argument("--rbin", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    target = sq.build_target(MODEL, args.target_mode, rbin_path=args.rbin,
                             device=args.device)
    rows = [json.loads(l) for l in open(args.corpus)]
    for i, r in enumerate(rows):
        ids = torch.tensor([r["input_ids"][:args.max_len]],
                           device=args.device)
        o = target(ids, output_hidden_states=True, use_cache=False)
        H = torch.cat([o.hidden_states[l + 1][0] for l in SRC], dim=-1)
        np.savez_compressed(
            f"{args.out_dir}/row{i:04d}.npz",
            hidden=H.to(torch.bfloat16).float().numpy().astype(np.float16)
            if False else H.float().cpu().numpy().astype(np.float16),
            input_ids=np.array(r["input_ids"][:args.max_len]),
            gen_start=r["gen_start"])
        if (i + 1) % 20 == 0:
            print(f"[cache] {i+1}/{len(rows)}", flush=True)
        del o, H
    sha = hashlib.sha256(open(args.rbin, "rb").read()).hexdigest()[:16] \
        if args.rbin else "none"
    json.dump({"corpus": args.corpus, "target_mode": args.target_mode,
               "rbin": args.rbin, "rbin_sha16": sha, "n_rows": len(rows),
               "max_len": args.max_len, "basis": "target residual basis "
               "of the specified arm (rotated if rbin given)"},
              open(f"{args.out_dir}/provenance.json", "w"), indent=1)
    print(f"[cache] DONE {len(rows)} rows -> {args.out_dir}")


if __name__ == "__main__":
    main()

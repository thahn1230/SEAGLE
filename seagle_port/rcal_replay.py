"""RCAL for quantized-target DFlash (§21): exact linear-block replay.

Replays every recorded proposal block against the FP16 reference target T_0,
lockstep on the DEPLOYED trajectory (T_0 KV is cropped to the deployed
accepted prefix after each cycle, mirroring SEAGLE's KV splice).

Conventions (SEAGLE compute_eagle_rcal_metrics):
  R_q = deployed accepted length (tau - 1, proposal-only, no bonus)
  R_0 = T_0 accepted length on the same block
  R_RC = LCP of accepted prefixes = min(R_q, R_0)  (linear block: identical
         proposal tokens, so the accepted prefixes are nested by construction)
  same_branch := R_q == R_0; div_depth := min(R_q,R_0)+1 otherwise.

Gate J: --gate-j replays a FP16-target cycle file; requires R_0 == R_q for
every cycle (bitwise self-consistency of the replay path).

Usage:
  python -m seagle_port.rcal_replay --cycles cyc__G_T4D4__mtbench.jsonl \
      --run-dir <RD> [--gate-j]
"""
import argparse
import importlib.util
import json
import os

import torch

from . import spinquant_target as sq

MODEL = "meta-llama/Llama-3.1-8B-Instruct"

_spec = importlib.util.spec_from_file_location(
    "seagle_rcal_metrics",
    "/home/thahn1230/eagle_spinquant_w4a4/scripts/compute_eagle_rcal_metrics.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
seagle_metrics = _mod.metrics


@torch.inference_mode()
def replay(cycles_path, device, block_size=10):
    from transformers import DynamicCache
    target = sq.build_target(MODEL, "fp16", device=device)
    turns = {}
    order = []
    for ln in open(cycles_path):
        r = json.loads(ln)
        key = (r["prompt_id"], r["turn"])
        if r.get("type") == "turn_header":
            turns[key] = {"input_ids": r["input_ids"], "cycles": []}
            order.append(key)
        else:
            turns[key]["cycles"].append(r)

    recs = []
    for key in order:
        t = turns[key]
        ids = torch.tensor([t["input_ids"]], device=device)
        n_in = ids.shape[1]
        kv = DynamicCache()
        maxlen = n_in + sum(c["tau"] for c in t["cycles"]) + block_size + 8
        pos = torch.arange(maxlen, device=device).unsqueeze(0)
        target(ids, position_ids=pos[:, :n_in], past_key_values=kv,
               use_cache=True, logits_to_keep=1)
        cur = n_in
        for c in t["cycles"]:
            assert c["prefix_len"] == cur, \
                f"trajectory drift {key}: {c['prefix_len']} != {cur}"
            block = torch.tensor([c["block"]], device=device)
            out = target(block, position_ids=pos[:, cur:cur + block.shape[1]],
                         past_key_values=kv, use_cache=True)
            post0 = out.logits.argmax(-1)
            match0 = (block[:, 1:] == post0[:, :-1])
            R0 = int(match0.cumprod(dim=1).sum(dim=1)[0].item())
            Rq = c["tau"] - 1
            recs.append({
                "prompt_id": key[0], "turn": key[1],
                "prefix_len": cur, "R_q": Rq, "R_0": R0,
                "R_RC": min(Rq, R0), "same_branch": Rq == R0,
                "div_depth": (min(Rq, R0) + 1) if Rq != R0 else -1,
                "S_q": Rq, "S_0": R0,
                "post0_bonus": int(post0[0, min(Rq, R0)].item()),
            })
            cur += Rq + 1
            kv.crop(cur)
    del target
    torch.cuda.empty_cache()
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--gate-j", action="store_true")
    args = ap.parse_args()
    recs = replay(args.cycles, args.device)
    base = os.path.basename(args.cycles).replace("cyc__", "").replace(
        ".jsonl", "")
    os.makedirs(os.path.join(args.run_dir, "cycles"), exist_ok=True)
    rp = os.path.join(args.run_dir, "cycles", f"rcal__{base}.jsonl")
    with open(rp, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    m = seagle_metrics(recs, kmax=9)
    out = os.path.join(args.run_dir, "tables", f"rcal__{base}.json")
    json.dump(m, open(out, "w"), indent=1)
    print(json.dumps({k: v for k, v in m.items()
                      if k != "depth_survival"}, indent=1))
    if args.gate_j:
        bad = sum(1 for r in recs if r["R_q"] != r["R_0"])
        print(f"GATE J: {'PASS' if bad == 0 else f'FAIL ({bad} mismatches)'}")
    print("[rcal] DONE", rp)


if __name__ == "__main__":
    main()

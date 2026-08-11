"""FIDI §16: component-wise W4A4 sensitivity via the block-CE proxy.

Grid over RotQuantDraft cfg["fp_components"] in the deployed M3 basis
(R1_T-folded fc + learned R1_D/R2_D, no R_C), scored by SpecForge block CE on
the SAME held-out hcache val rows the trainers use (tail --val-rows, never
test data). Directions:

  restore_X : full W4A4 except X kept FP        (class and per-layer)
  only_X    : everything FP except X quantized  (class and per-layer)

Class cells additionally get validation-AL runs via eval_al
--vsq-fp-components (queued separately); this proxy covers the full
per-layer grid cheaply. Writes tables/qkvo_component_sensitivity.csv.
"""
import argparse
import csv
import os

import numpy as np
import torch

from . import spinquant_target as sq          # noqa: F401 (sys.path setup)
from . import vsq_specforge_port as sf
from .vsq_train_qat import build
from .vsq_train_draft_rot import load_shared_modules

ALL = ("fc", "q", "k", "v", "o", "gate", "up", "down")
CLASSES = {"q": ("q",), "k": ("k",), "v": ("v",), "o": ("o",),
           "mlp": ("gate", "up", "down"), "fc": ("fc",)}
NL = 5


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--rbin", required=True)
    ap.add_argument("--rot-ckpt", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--val-rows", type=int, default=15)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = args.device
    torch.manual_seed(0)
    rq = build("Q3", args.rbin, args.rot_ckpt, dev)   # M3 basis, weights fp32
    for p in rq.parameters():
        p.requires_grad_(False)
    embed, head = load_shared_modules(dev)

    files = sorted(f for f in os.listdir(args.cache_dir)
                   if f.endswith(".npz"))[-args.val_rows:]
    rows_data = []
    for f in files:
        z = np.load(os.path.join(args.cache_dir, f))
        ids = torch.tensor(z["input_ids"], device=dev).unsqueeze(0)
        H = torch.tensor(z["hidden"], device=dev,
                         dtype=torch.bfloat16).unsqueeze(0)
        lm = torch.zeros(1, ids.shape[1], device=dev)
        lm[:, int(z["gen_start"]):] = 1.0
        rows_data.append((ids, H, lm))

    def val_ce(fp_set):
        rq.cfg["fp_components"] = frozenset(fp_set)
        tot = 0.0
        for ids, H, lm in rows_data:
            l, _ = sf.specforge_block_forward(
                rq, embed, head, ids, H, lm, num_anchors=16,
                block_size=rq.block_size, mask_token_id=rq.mask_token_id,
                gamma=5.0)[:2]
            tot += l.item()
        return tot / len(rows_data)

    out_rows = []

    def cell(name, comp, layer, direction, fp_set):
        ce = val_ce(fp_set)
        out_rows.append({"cell": name, "comp": comp, "layer": layer,
                         "direction": direction, "val_ce": round(ce, 4)})
        print(out_rows[-1], flush=True)
        return ce

    ce_fp = cell("ref_fp16", "-", -1, "ref", ALL)
    ce_q = cell("base_w4a4", "-", -1, "ref", ())
    for cname, comps in CLASSES.items():
        cell(f"restore_{cname}", cname, -1, "restore", comps)
        cell(f"only_{cname}", cname, -1, "only",
             tuple(c for c in ALL if c not in comps))
    for c in ("q", "k", "v", "o"):
        for L in range(NL):
            cell(f"restore_{c}@{L}", c, L, "restore", (f"{c}@{L}",))
            fp = tuple(x for x in ALL if x != c) + tuple(
                f"{c}@{m}" for m in range(NL) if m != L)
            cell(f"only_{c}@{L}", c, L, "only", fp)

    for r in out_rows:
        r["delta_vs_fp"] = round(r["val_ce"] - ce_fp, 4)
        r["delta_vs_w4a4"] = round(r["val_ce"] - ce_q, 4)
    os.makedirs(os.path.join(args.run_dir, "tables"), exist_ok=True)
    with open(os.path.join(args.run_dir, "tables",
                           "qkvo_component_sensitivity.csv"), "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"[qkvo_ce] DONE {len(out_rows)} cells "
          f"(fp16 {ce_fp:.3f} / w4a4 {ce_q:.3f})")


if __name__ == "__main__":
    main()

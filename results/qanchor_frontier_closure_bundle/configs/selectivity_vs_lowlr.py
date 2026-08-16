#!/usr/bin/env python3
"""Closure sections 25-26: down_proj claim audit with explicit
denominators + flipped-index overlap between HB and matched low-LR
plain QAT (Jaccard, intersection, unique counts, per module).
Writes results/qanchor_closure/selectivity/selectivity_closure.json."""
import json
import os

import torch

REPO = "/home/thahn1230/SEAGLE"
QRUN = os.path.join(REPO, "runs/eagle1_qat_qanchor_causal_20260814")
CKD = os.path.join(QRUN, "ckpts")
SITES = ("W_first", "W_rec", "q", "k", "v", "o", "gate", "up", "down")
CONTROLLED = ("q", "k", "v", "o", "gate", "up", "down", "W_rec")
D = 4096

anchor = torch.load(os.path.join(CKD, "anchor_gsr5_ptq.pt"),
                    map_location="cpu", weights_only=True)
c0 = {s: anchor[s]["c0"] for s in SITES}

MODELS = {
    "HB_hybrid_b0.005_s0": "codes__HB_B_hybrid_b0.005_s0_st3000.pt",
    "HB_conv_b0.02_s0": "codes__HB_B_conv_b0.02_s0_st3000.pt",
    "T6conv_1e-6_s1": "codes__CAN_T6_gsr5_conv_s1_st1500.pt",
    "T6hybrid_1e-6_s1": "codes__CAN_T6_gsr5_hybrid_s1_st1000.pt",
    "aggressive_P2conv_s0": "bgrid__b1_global_f0.pt",
}


def load_flips(fname):
    d = torch.load(os.path.join(CKD, fname), map_location="cpu",
                   weights_only=True)
    if isinstance(d, dict) and "codes" in d:
        d = d["codes"]
    return {s: (d[s] != c0[s]) for s in SITES}


masks = {}
for n, f in MODELS.items():
    p = os.path.join(CKD, f)
    if not os.path.exists(p):
        print(f"  [skip] {n}: {f} missing")
        continue
    masks[n] = load_flips(f)

# ---- section 25: down audit with explicit denominators ----
audit = {}
for n, m in masks.items():
    tot = sum(int(m[s].sum()) for s in SITES)
    down = int(m["down"].sum())
    ctrl = sum(int(m[s].sum()) for s in CONTROLLED)
    n_all = sum(m[s].numel() for s in SITES)
    n_ctrl = sum(m[s].numel() for s in CONTROLLED)
    audit[n] = dict(
        total_changed_codes=tot,
        down_changed_codes=down,
        down_fraction_of_all_flips=round(down / max(tot, 1), 6),
        down_flip_rate_within_down=round(
            down / m["down"].numel(), 6),
        H_Q_all=round(tot / n_all, 6),
        H_Q_controlled=round(ctrl / n_ctrl, 6),
        wfirst_h_flips=int(m["W_first"][:, D:].sum()),
        wfirst_h_fraction_of_all=round(
            int(m["W_first"][:, D:].sum()) / max(tot, 1), 6))

# ---- section 26: flipped-index overlap per module ----
overlaps = {}
PAIRS = [("HB_hybrid_b0.005_s0", "T6hybrid_1e-6_s1"),
         ("HB_hybrid_b0.005_s0", "T6conv_1e-6_s1"),
         ("HB_conv_b0.02_s0", "T6conv_1e-6_s1"),
         ("HB_hybrid_b0.005_s0", "HB_conv_b0.02_s0")]
for a, b in PAIRS:
    if a not in masks or b not in masks:
        continue
    per = {}
    ti = tu = ta = tb = 0
    for s in SITES:
        ma, mb = masks[a][s], masks[b][s]
        inter = int((ma & mb).sum())
        union = int((ma | mb).sum())
        per[s] = dict(jaccard=round(inter / max(union, 1), 4),
                      intersection=inter,
                      unique_a=int(ma.sum()) - inter,
                      unique_b=int(mb.sum()) - inter)
        ti += inter
        tu += union
        ta += int(ma.sum())
        tb += int(mb.sum())
    overlaps[f"{a}__vs__{b}"] = dict(
        jaccard_all=round(ti / max(tu, 1), 4),
        intersection=ti, unique_a=ta - ti, unique_b=tb - ti,
        per_module=per)

out = dict(down_audit=audit, overlaps=overlaps)
op = os.path.join(REPO,
                  "results/qanchor_closure/selectivity/"
                  "selectivity_closure.json")
os.makedirs(os.path.dirname(op), exist_ok=True)
json.dump(out, open(op, "w"), indent=1, sort_keys=True)
print(f"wrote {op}")
for n, a in audit.items():
    print(f"{n}: flips={a['total_changed_codes']} "
          f"down_frac_of_flips={a['down_fraction_of_all_flips']} "
          f"down_rate={a['down_flip_rate_within_down']} "
          f"Wf_h_frac={a['wfirst_h_fraction_of_all']}")
for k, v in overlaps.items():
    print(f"{k}: J={v['jaccard_all']} inter={v['intersection']} "
          f"uA={v['unique_a']} uB={v['unique_b']}")

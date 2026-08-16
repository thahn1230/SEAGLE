#!/usr/bin/env python3
"""Closure F4 (sections 15-21, 38): exact flip partition of the best
HB model + deployment-code counterfactual construction with hard unit
verification.

C0  = anchor codes (expected_c0.pt)
CHB = HB hybrid b0.5% s0 st3000 deployed codes

Counterfactuals (deployment-code interventions, NOT trainable ckpts):
  HB_no_WfirstH    CHB, W_first h-half -> C0
  HB_WfirstH_only  C0,  W_first h-half -> CHB
  HB_no_Wfirst     CHB, all W_first -> C0
  HB_Wfirst_only   C0,  all W_first -> CHB
  HB_controlled_only  C0, 8 controlled folds -> CHB
NOTE: with the 9-site partition, HB_controlled_only == HB_no_Wfirst
(S_other is empty). Asserted below; evaluated once.
"""
import hashlib
import json
import os

import torch

QRUN = "/home/thahn1230/SEAGLE/runs/eagle1_qat_qanchor_causal_20260814"
NR = "/home/thahn1230/SEAGLE/runs/eagle1_qanchor_closure_20260816"
OUTD = os.path.join(NR, "ckpts")
SITES = ("W_first", "W_rec", "q", "k", "v", "o", "gate", "up", "down")
CONTROLLED = ("q", "k", "v", "o", "gate", "up", "down", "W_rec")
D = 4096

c0 = torch.load(os.path.join(QRUN, "ckpts", "expected_c0.pt"),
                map_location="cpu", weights_only=True)
chb = torch.load(os.path.join(
    QRUN, "ckpts", "codes__HB_B_hybrid_b0.005_s0_st3000.pt"),
    map_location="cpu", weights_only=True)
for s in SITES:
    assert c0[s].shape == chb[s].shape and c0[s].dtype == torch.int8

# ---- exact flip partition (section 15) ----
def cnt(mask):
    return int(mask.sum())

flips = {s: (chb[s] != c0[s]) for s in SITES}
n_all = sum(f.numel() for f in flips.values())
k_all = sum(cnt(f) for f in flips.values())
part = dict(
    S_controlled=dict(
        elements=sum(flips[s].numel() for s in CONTROLLED),
        flips=sum(cnt(flips[s]) for s in CONTROLLED)),
    S_Wfirst_e=dict(elements=flips["W_first"][:, :D].numel(),
                    flips=cnt(flips["W_first"][:, :D])),
    S_Wfirst_h=dict(elements=flips["W_first"][:, D:].numel(),
                    flips=cnt(flips["W_first"][:, D:])),
    S_down=dict(elements=flips["down"].numel(),
                flips=cnt(flips["down"]),
                note="subset of S_controlled"),
    S_other=dict(elements=0, flips=0),
)
for k, v in part.items():
    v["flip_rate"] = round(v["flips"] / max(v["elements"], 1), 6)
    v["fraction_of_all_flips"] = round(v["flips"] / max(k_all, 1), 6)
part["totals"] = dict(elements=n_all, flips=k_all,
                      H_Q_all=round(k_all / n_all, 6))
os.makedirs("/home/thahn1230/SEAGLE/results/qanchor_closure/wfirst",
            exist_ok=True)
pp = ("/home/thahn1230/SEAGLE/results/qanchor_closure/wfirst/"
      "flip_partition.json")
json.dump(part, open(pp, "w"), indent=1)
print(f"wrote {pp}")
for k, v in part.items():
    if k != "totals":
        print(f"  {k}: flips={v['flips']} rate={v['flip_rate']} "
              f"frac_of_all={v['fraction_of_all_flips']}")
print(f"  totals: {part['totals']}")

# ---- counterfactual construction ----
def clone(codes):
    return {s: codes[s].clone() for s in SITES}

CF = {}
x = clone(chb)
x["W_first"][:, D:] = c0["W_first"][:, D:]
CF["HB_no_WfirstH"] = x
x = clone(c0)
x["W_first"][:, D:] = chb["W_first"][:, D:]
CF["HB_WfirstH_only"] = x
x = clone(chb)
x["W_first"] = c0["W_first"].clone()
CF["HB_no_Wfirst"] = x
x = clone(c0)
x["W_first"] = chb["W_first"].clone()
CF["HB_Wfirst_only"] = x
x = clone(c0)
for s in CONTROLLED:
    x[s] = chb[s].clone()
CF["HB_controlled_only"] = x

# ---- unit verification (section 38) ----
def eq(a, b):
    return torch.equal(a, b)

v = CF["HB_no_WfirstH"]
assert eq(v["W_first"][:, D:], c0["W_first"][:, D:])
assert eq(v["W_first"][:, :D], chb["W_first"][:, :D])
for s in CONTROLLED:
    assert eq(v[s], chb[s]), s
v = CF["HB_WfirstH_only"]
assert eq(v["W_first"][:, D:], chb["W_first"][:, D:])
assert eq(v["W_first"][:, :D], c0["W_first"][:, :D])
for s in CONTROLLED:
    assert eq(v[s], c0[s]), s
v = CF["HB_no_Wfirst"]
assert eq(v["W_first"], c0["W_first"])
for s in CONTROLLED:
    assert eq(v[s], chb[s]), s
v = CF["HB_Wfirst_only"]
assert eq(v["W_first"], chb["W_first"])
for s in CONTROLLED:
    assert eq(v[s], c0[s]), s
# equivalence C3 == C5
for s in SITES:
    assert eq(CF["HB_no_Wfirst"][s], CF["HB_controlled_only"][s]), s
print("unit gates PASS (incl. HB_no_Wfirst == HB_controlled_only)")

shas = {}
for name, codes in CF.items():
    p = os.path.join(OUTD, f"cf__{name}.pt")
    torch.save(codes, p)
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]
    r = torch.load(p, map_location="cpu", weights_only=True)
    for s in SITES:
        assert torch.equal(r[s], codes[s]), f"reload mismatch {name}/{s}"
    shas[name] = h
    print(f"cf__{name}.pt sha16={h}")
json.dump(shas, open(os.path.join(
    "/home/thahn1230/SEAGLE/results/qanchor_closure/wfirst",
    "counterfactual_shas.json"), "w"), indent=1)

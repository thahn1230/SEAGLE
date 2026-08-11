# Does optimization order explain the QAT-DraRot interaction? (GS, W4A4)

Run `runs/eagle1_r5qat_causal_gs_20260811_141143` · git `dc4fc334bba880d200c2d525d7f8ac30a908a0d2` · hypothesis tested: "large-space weight adaptation should precede small-space acceptance-aware basis adaptation" — stated as CONDITIONAL optimization (R5* depends on W), never justified by parameter count alone.

## Main ordering table

| Optimization order | MT | GSM8K | ShareGPT | HumanEval | Avg |
|---|---:|---:|---:|---:|---:|
| R5 | 3.1645 | 3.6875 | 3.2351 | 3.8282 | 3.4788 |
| R5 -> QAT | 3.1989 | 3.6265 | 3.2845 | 3.7065 | 3.4541 |
| QAT (no R5) | 3.1923 | 3.6126 | 3.2679 | 3.6994 | 3.4431 |
| QAT -> R5 (ARM D) | 3.1882 | 3.6180 | 3.2586 | 3.6302 | 3.4237 |
| R5 -> QAT -> R5(reopt) (ARM F) | 3.2271 | 3.6734 | 3.2875 | 3.6589 | 3.4617 |
| R5 -> QAT -> R5(fresh ctl) | 3.2104 | 3.6549 | 3.2712 | 3.7160 | 3.4631 |

**Delta_order = Avg(QAT->R5) - Avg(R5->QAT) = -0.0304**
(aggregate bootstrap: CI [-0.0574, -0.0027], p = 0.032)

## Statistics

**QAT->R5 vs R5->QAT (primary)**

| dataset | delta [95% CI] | raw p | Holm |
|---|---|---:|---|
| mtbench | -0.0107 [-0.0738, +0.0524] | 0.7267 | n.s. |
| gsm8k | -0.0085 [-0.0483, +0.0316] | 0.6947 | n.s. |
| sharegpt | -0.0259 [-0.0938, +0.0369] | 0.4547 | n.s. |
| humaneval | -0.0764 [-0.1240, -0.0295] | 0.0013 | SIG |

**QAT->R5 vs R5-only**

| dataset | delta [95% CI] | raw p | Holm |
|---|---|---:|---|
| mtbench | +0.0237 [-0.0346, +0.0821] | 0.4220 | n.s. |
| gsm8k | -0.0695 [-0.1112, -0.0271] | 0.0007 | SIG |
| sharegpt | +0.0235 [-0.0426, +0.0884] | 0.4647 | n.s. |
| humaneval | -0.1980 [-0.2481, -0.1441] | 0.0000 | SIG |

**QAT->R5 vs QAT-only**

| dataset | delta [95% CI] | raw p | Holm |
|---|---|---:|---|
| mtbench | -0.0041 [-0.0638, +0.0579] | 0.8967 | n.s. |
| gsm8k | +0.0054 [-0.0376, +0.0476] | 0.8167 | n.s. |
| sharegpt | -0.0094 [-0.0749, +0.0588] | 0.7973 | n.s. |
| humaneval | -0.0693 [-0.1183, -0.0204] | 0.0067 | SIG |

**R5->QAT->R5reopt vs R5->QAT**

| dataset | delta [95% CI] | raw p | Holm |
|---|---|---:|---|
| mtbench | +0.0283 [-0.0397, +0.0955] | 0.4080 | n.s. |
| gsm8k | +0.0469 [+0.0078, +0.0853] | 0.0213 | n.s. |
| sharegpt | +0.0030 [-0.0592, +0.0621] | 0.9527 | n.s. |
| humaneval | -0.0477 [-0.0960, +0.0003] | 0.0507 | n.s. |

**warm-reopt(F) vs QAT->R5(D)**

| dataset | delta [95% CI] | raw p | Holm |
|---|---|---:|---|
| mtbench | +0.0389 [-0.0205, +0.0961] | 0.1880 | n.s. |
| gsm8k | +0.0554 [+0.0159, +0.0941] | 0.0053 | SIG |
| sharegpt | +0.0289 [-0.0339, +0.0899] | 0.3673 | n.s. |
| humaneval | +0.0287 [-0.0191, +0.0731] | 0.2313 | n.s. |

**fresh vs warm R5 on same weights**

| dataset | delta [95% CI] | raw p | Holm |
|---|---|---:|---|
| mtbench | +0.0167 [-0.0511, +0.0799] | 0.6247 | n.s. |
| gsm8k | +0.0185 [-0.0208, +0.0569] | 0.3500 | n.s. |
| sharegpt | +0.0162 [-0.0498, +0.0845] | 0.6627 | n.s. |
| humaneval | -0.0572 [-0.1061, -0.0063] | 0.0233 | n.s. |

## Basis geometry (§19)

```json
{
 "armD_qat_first": {
  "vs_old_r5": {
   "fro": 81.2673,
   "max_elem": 0.1113,
   "eigphase_median": 1.2255,
   "eigphase_max": 3.1411,
   "planes_gt_0p1rad": 1959
  },
  "vs_R_T": {
   "fro": 69.5256,
   "eigphase_median": 0.8935
  },
  "orth_err": 1.68e-06
 },
 "p3C_fresh_on_B": {
  "vs_old_r5": {
   "fro": 81.749,
   "max_elem": 0.10503,
   "eigphase_median": 1.2373,
   "eigphase_max": 3.141,
   "planes_gt_0p1rad": 1960
  },
  "vs_R_T": {
   "fro": 73.3285,
   "eigphase_median": 1.0299
  },
  "orth_err": 1.57e-06
 },
 "p3D_reopt_on_B": {
  "vs_old_r5": {
   "fro": 73.9999,
   "max_elem": 0.11368,
   "eigphase_median": 1.0539,
   "eigphase_max": 3.0554,
   "planes_gt_0p1rad": 1929
  },
  "vs_R_T": {
   "fro": 85.2817,
   "eigphase_median": 1.3512
  },
  "orth_err": 1.82e-06
 },
 "old_r5_vs_R_T": {
  "geodesic_dist": 68.0216,
  "frob_dist": 81.0936,
  "max_elem_change": 0.0982,
  "orth_error": 0.0001,
  "max_abs_elem": 0.0823
 }
}
```

## ARM D training trajectory (mtbench official tau at rotation snapshots)

| step | 500 | 1000 | 1500 | 2000 | 2500 | 3000 |
| tau | 3.1776 | 3.1764 | 3.1778 | 3.1681 | 3.2427 | 3.1866 |


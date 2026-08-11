# Why does QAT slightly reduce average AL after DraRot R5? — causal study (GS, W4A4)

Run `runs/eagle1_r5qat_causal_gs_20260811_141143` · git `dc4fc334bba880d200c2d525d7f8ac30a908a0d2` · all taus recomputed from raw cycle records. R5/R6 semantics and arm order verified in `audit/latest_drarot_lineage.json`.

## Headline answers

**1. Is the average degradation significant?** No. mean4 3.4788 -> 3.4541, delta -0.0247, 95% CI [-0.0524, 0.004], p = 0.0993 (paired within-dataset prompt-cluster resampling of the arithmetic 4-dataset mean).

## Table 4 — functional vs quantization degradation (2x2)

| State | FP draft | W4A4 draft |
|---|---|---|
| pre-QAT R5 | 3.5742 | 3.4788 |
| post-QAT R5 | 3.5542 | 3.4541 |

| Dataset | FP pre | FP post | dFP | W4 pre | W4 post | dW4 |
|---|---:|---:|---:|---:|---:|---:|
| mtbench | 3.2710 | 3.3064 | +0.0354 | 3.1645 | 3.1989 | +0.0344 |
| gsm8k | 3.7077 | 3.6860 | -0.0217 | 3.6875 | 3.6265 | -0.0610 |
| sharegpt | 3.3550 | 3.4095 | +0.0546 | 3.2351 | 3.2845 | +0.0494 |
| humaneval | 3.9629 | 3.8147 | -0.1482 | 3.8282 | 3.7065 | -0.1217 |

Every dataset moves the SAME direction in FP as in W4A4: the tradeoff exists at the functional level (OUTCOME A / CASE 3) — QAT changed the draft function itself, not primarily its quantization geometry.

## Table 3 — R5 order test (key experiment)

| Order | MT | GSM | Share | HE | Avg |
|---|---:|---:|---:|---:|---:|
| R5 (PTQ) | 3.1645 | 3.6875 | 3.2351 | 3.8282 | 3.4788 |
| R5 -> QAT | 3.1989 | 3.6265 | 3.2845 | 3.7065 | 3.4541 |
| QAT -> new R5 | 3.2104 | 3.6549 | 3.2712 | 3.7160 | 3.4631 |
| R5 -> QAT -> reoptimized R5 | 3.2271 | 3.6734 | 3.2875 | 3.6589 | 3.4617 |

Neither fresh nor re-optimized R5 recovers the lost acceptance (best 3.4631 vs R5->QAT 3.4541): the stale-R5 hypothesis is weakened; the change is in the weights' function, not the rotation optimum.

## Quantization-grid forensics

| site | rel W move | W4 code flips | NMSE pre->post | sat pre->post | act absmax pre->post |
|---|---:|---:|---|---|---|
| W_first | 0.0762 | 0.0955 | 0.02003->0.02107 | 0.0030->0.0029 | 5.0664->5.0742 |
| W_rec | 0.0716 | 0.0907 | 0.01816->0.01816 | 0.0031->0.0031 | 7.9297->8.5859 |
| q | 0.0229 | 0.0645 | 0.01210->0.01215 | 0.0036->0.0036 | 2.9141->2.8184 |
| k | 0.0231 | 0.0594 | 0.01211->0.01215 | 0.0036->0.0036 | 2.9141->2.8184 |
| v | 0.0293 | 0.0736 | 0.01208->0.01226 | 0.0036->0.0036 | 2.9141->2.8184 |
| o | 0.0513 | 0.1038 | 0.01368->0.01393 | 0.0031->0.0031 | 3.8652->6.3672 |
| gate | 0.0509 | 0.1277 | 0.01209->0.01223 | 0.0036->0.0036 | 5.1992->5.9219 |
| up | 0.0528 | 0.1315 | 0.01210->0.01225 | 0.0036->0.0036 | 5.1992->5.9219 |
| down | 0.2728 | 0.4801 | 0.01215->0.01191 | 0.0035->0.0036 | 69.4375->78.75 |

## Quantizer-staleness audit (§16)

Structural answer: NO staleness is possible in this pipeline. Weight-quantizer scales/clipping are recomputed at every adapter install from the CURRENT folded weights (`FakeW4A4Linear.__init__` -> `_weight_fake_quant`), and A4 activation quantization is dynamic per token per forward. No calibration artifact is inherited from the pre-QAT state; a post-QAT recalibration control is therefore an identity operation.

## Implementation parity

R5 frozen during QAT (trainer freezes rot; gate-verified), single GS application, first bridge pinned at R_T, fake-quant placement gate-verified bitwise core==deploy (Gate G / Gate R6), no R6 in the R5+QAT baseline (predates R6 code), GS-only artifacts (LS alphas never passed), quantizer granularity fixed. Deterministic eval verified bitwise across reruns earlier this session.


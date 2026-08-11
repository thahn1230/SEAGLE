# VSQ study - figure index

All figures saved as PDF+PNG (150 dpi) in `/home/thahn1230/dflash_workspace/dflash/runs/dflash_vanilla_spinquant_novelty_20260810_104957/figs`.

## FIG_A_tppl

Target-model WikiText-2 perplexity for the W4A4 quantization ladder (fp16 reference, RTN w4a4 without rotation, Hadamard, R1-only, R1+R2), three rotation seeds per arm; y is log-scaled because RTN (~190 PPL) dwarfs the rotated arms (~8.8-10.9). Rows deduplicated by (seed, arm) keeping the last append. Source: tables/target_ppl.csv.

## FIG_B_draft_ladder

Draft-model block cross-entropy per quantization arm: fp16 reference, W4A4 RTN without rotation, random rotation, R1D, and R1D+R2D. Rotations recover part of the RTN gap (7.23 to 4.47) but remain far from the fp16 draft (2.41). Source: tables/draft_quality.csv.

## FIG_C_qat

Draft-rotation QAT. Left: 100-step pilot LR sweep on arm Q2 - best validation CE vs learning rate (log x); open markers sit at the init CE level for runs that never improved (3e-3, 1e-2, 1e0); the orange square is the repeat run 1e-1b at lr=1e-1. Right: main runs as init->best dumbbells - Q2 and Q3 improve at lr 3e-1, Q5 does not move at 3e-1 or 1e-1 and only improves at 3e-2 (2.644->2.389). Source: rotations/draft/QAT*.pt.summary.json.

## FIG_D_al_ladder

Accepted-length (AL) ladder on mtbench/gsm8k/humaneval/sharegpt with the 4-dataset mean overlaid (black diamonds): FP16, naive W4A4 RTN, vanilla SpinQuant (M3), +R_C (M5_rconly), +P2+R_C (M5_p2rc), and the final system with QAT draft weights (M6_q5bp2). The interface-broken M2_vsqraw row (mtbench-only, tau=1.00) is excluded and noted as an annotation. Source: FIDI tables/final_4dataset_al.csv.

## FIG_E_validation

Cycle-pooled accepted length on the gsm8kvalid split for every validation arm with shard data, sorted ascending and colored by family (M-family blue, Q-family green, component arms grey, G1 gate orange, I7 interface red). Dashed/dotted reference lines mark V_M3 and V_M5_rc. Values are a snapshot; validation jobs were still appending shards at render time. Source: shards/al__V_*__w4a4__gsm8kvalid.csv.

## FIG_F_mechanism

Why R_C works. Left: H_t kurtosis (log y) in the h-cache counterfactual (M1 69.5 / M3 69.4 / M5+R_C 3.0; tables/mechanism_stats.csv) and in the deployed forward on gsm8k (R1 Ht_dep 180.5, R3 Ht_dep 195.0, R3 Ht_rc_dep 2.9; FIDI tables/ht_stats.csv). Right: A4 NMSE for the same arms (mechanism_stats a4_nmse; FIDI ht_qparam_stats nmse_mean). R_C collapses the heavy tails by ~20-70x in kurtosis and cuts A4 NMSE by an order of magnitude in both the counterfactual and the deployed forward.

## FIG_G_ctxbits

Context-precision ablation on mtbench (cycle-pooled tau), sorted ascending: M3_vsq (ctx A4, no R_C), HP2_ctxA8_noRC (ctx A8, no R_C), HP1_ctxFP16 (ctx FP16), M5_rconly (ctx A4 + R_C), and HP0_ctxA8. Raising context activations from 4 to 8 bits without R_C already recovers most of R_C's benefit, locating the failure in ctx quantization of heavy-tailed H_t. Source: shards/al__{M3_vsq,HP*,M5_rconly}__w4a4__mtbench.csv.

## FIG_H_forest

Forest plot of the key preregistered comparisons (P1 M0vM3, P4b M3vM5p2rc, P3 M5rcVsM5p2rc, P5 M5vM6, GAP M0vM6), one row per (comparison, dataset): delta pooled tau with 95% bootstrap CI whiskers around the zero line. Solid markers are Holm-adjusted significant; hollow markers are not (only humaneval GAP_M0vM6 is non-significant - the final system closes the FP16 gap there). Sources: stats/bootstrap_{mtbench,gsm8k,humaneval,sharegpt}.json, stats/holm_adjusted.json.

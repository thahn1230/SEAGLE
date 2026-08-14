# Cross-server compliance audit (directive of 2026-08-13, rules R1-R5)

Env (R5): gpusystem, 8x RTX 4090, driver 550.163.01, CUDA 12.4,
torch 2.6.0+cu124, transformers 4.51.3, env `seagle`
(recorded in manifests/provenance.json).

## R4 re-run audit — verdict per completed experiment

| Experiment | Baseline used | Same-server pairing? | Verdict |
|---|---|---|---|
| Phase-1 pilots + calib (Q-selection) | CAL_A_ptq / CAL_B_ptq calib evals, run THIS server | yes | VALID |
| Phase-2 matched chain (24 runs) + selections | c4-calib refs, this server | yes | VALID |
| Q0h conv-QAT (15 runs) + selections | same | yes | VALID |
| PTQ 4-ds baselines Naive/LS/LS+R5 | generated this campaign | yes | VALID |
| GS PTQ (A0), GS+R5 PTQ (A1) 4-ds shards | R1R2GS run 20260810, SAME host gpusystem, same seagle env | yes (same server, earlier run dir) | VALID subject to GATE-REPRO below |
| LR pilot + T6 tuned runs | c4-calib refs, this server | yes | VALID |
| Reference table (prev-server B1..B9, QF_rdg) | — | quoted as CONTEXT ONLY, never paired | COMPLIANT (R1/R3) |

**Re-run list: EMPTY**, conditional on GATE-REPRO passing. If GATE-REPRO
fails (A0 mtbench shard not reproduced to 4 decimals), the A0/A1
4-dataset panels are re-run inside this campaign before any final
bootstrap (8 eval jobs, queued automatically).

## Gates being executed now (section 3 of the directive)

1. GATE-REPRO (R2/baseline reproduction): re-run A0 (GS PTQ) mtbench:80
   with the campaign evaluator; require pooled micro-tau equal to the
   R1R2GS A0 mtbench shard to 4 decimals (same-server determinism was
   proven on the original server; this proves it across run-dirs/days
   here and certifies A0/A1 shard reuse).
2. Gate D (deploy parity): check_qat_deploy_parity.py on a trained
   campaign checkpoint (trainer forward vs runtime adapter).
3. Gate GS-R1/R2-C: check_r1r2_gs_parity.py (rotation folds, bytewise).

## Protocol-contract check (section 2)

- greedy, batch 1, max_new 128, mc_sim_7b_63, cycle-pooled micro-tau,
  acceptance_list stores accepted+1 (audited): MATCH.
- Pools: mtbench 80 / gsm8k 200 / sharegpt 80 / humaneval 164, offsets
  eval 0 / calib 500 / valid 1000, ShareGPT Aeala first-human-turn
  60-1200 chars: MATCH (manifests pinned + Gate F passed 2026-08-12).
- NOTE mtbench turns: the EAGLE-side frozen pool (this codebase's
  load_eval_prompts, used unchanged by the original-server grid study)
  is TURN-1 prompts; "MT-Bench 80(2턴)" in the directive matches the
  DFlash-side contract. EAGLE-side stays turn-1 = the contract its own
  original-server baselines used. Recorded here per R5.
- 4-ds mean = mean of 4 dataset-level taus (never cycle-pooled across
  datasets): MATCH (prereg section 2).
- Selection on calib/validation only; median-seed reporting: MATCH.
- Bootstrap >=3000 paired same-prompt clusters + Holm, p floored at
  1/reps: MATCH (bootstrap_eagle_tau.py + holm_adjust_bootstrap.py).
- GS alpha 32.89964245299412: MATCH. All quant is fake-quant; no real
  INT4 speed claims: MATCH.

## Post-restoration finding (2026-08-13 12:45)

Baseline gates under canonical R.bin: B1/B3/B7/B9 = 13/13 EXACT
(4-decimal) vs original-server values — bit-level cross-server
reproduction confirmed. B5 initially FAILED systematically (-0.03..-0.11):
root cause = evaluator `--draft-cfg d4p3` branch IGNORES `--alpha-rec`
(pathwise EP3-P scale silently dropped). Correct invocation for LS PTQ is
`--draft-cfg d4p3_deploy` (handles alpha-rec; no --draft-sd = public
draft). B5 re-queued with the fix. NOTE: the archived pilot-track
(20260812 run) PTQ_ls 4-dataset shards used the same buggy invocation
and are mis-specified (LS-first scale without recurrent rescale) — do
not reuse them; the pilot track's LS-PTQ row is invalid. All other
pilot-track arms (naive/gs/gsr5/lsr5, all QAT arms, FIN_AA_ls which used
d4p3_deploy) are unaffected.

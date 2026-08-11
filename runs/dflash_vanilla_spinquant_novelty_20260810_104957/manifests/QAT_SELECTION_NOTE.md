# Q-family (Operation B) selection provenance — frozen 2026-08-11 09:14

All selection on gsm8kvalid (validation) or the trainers' held-out corpus-CE
split. No test-set (mtbench/gsm8k/humaneval/sharegpt) number was seen before
the M6 4-ds jobs were queued.

## LR pilots (Q2 arm, 100 steps, SGD-mom 0.9, init CE 7.0572)
3e-3: 7.057 / 1e-2: 7.057 / 3e-2: 6.361 / 1e-1: 5.445 (prev server) and
5.400 (this server, regenerated hcache — confirms cache reproducibility) /
**3e-1: 5.218 (best)** / 1e0: 7.057 (diverged, best_step 0). → mains at 3e-1.

## Mains (400 steps, lr 3e-1)
Q2 7.057→3.602 / Q3 4.513→3.479 / Q5 2.644→no improvement (init already low
thanks to R_C; 3e-1 overshoots from this init).

AMENDMENT (documented deviation): the pilot LR was calibrated at Q2's init
scale; for Q5 only, lower LRs were additionally tried using the same
train-val CE criterion: 1e-1 no improvement, **3e-2: 2.644→2.389** → Q5
checkpoint = QAT_Q5_lr3e-2.pt. Selection signal is training-side CE only.

## Deployed validation AL (gsm8kvalid n60 T512, cycle-pooled tau)
V_M3 2.017 (reference) / V_Q2 2.494 / V_Q3 2.742 / V_Q5(lr3e-1: weights
== init) 3.148 ≈ V_M5_rconly 3.213 (consistency check) / **V_Q5b 3.501** /
V_M5_rc(P2+RC, no QAT) 3.407 / **V_Q5bp2 3.669 ← selected M6-analog**.

Caveat carried to the report: Q5 was TRAINED without fc_p2; V_Q5bp2 applies
P2 at eval only (granularity mismatch train→eval). It nevertheless wins
validation; the mismatch is disclosed, and the honest label for M6 is
"QAT(Q5,3e-2) + P2 + R_C (eval-composed)".

## Frozen M6 4-ds configuration (queued 09:14, tags M6_q5bp2)
--target-mode w4a4 --rbin s1 --vsq-draft R1D_s1r1.pt --vsq-rc rt --vsq-p2
--qat-ckpt QAT_Q5_lr3e-2.pt ; mtbench80(+record-cycles)/gsm8k200/
humaneval164/sharegpt80, max_new 1024, greedy, B=10.

## sharegpt.jsonl reconstruction caveat (2026-08-11 09:5x)
cache/sharegpt.jsonl was never committed and did not survive the server move;
the frozen checksum (DFST provenance_final.txt f512d036...) could NOT be
byte-reproduced from today's Aeala/ShareGPT_Vicuna_unfiltered snapshot under
33 recipe variants (selection rule x strip timing x json formatting x row
count). The file was reconstructed by the documented SEAGLE recipe (first
human turn, 60<=len(raw)<=1200, eval offset 0, first 80, stripped, upstream
jsonl format). Fidelity gate queued: M0chk_sharegpt (fp16 n80) — its
per-prompt n_cycles/tau vector is compared against the canonical
al__M0_fp16__fp16__sharegpt.csv; M6 sharegpt pairing is only used if that
gate shows prompt-level agreement. New file sha256 recorded in this note's
companion command output and in provenance at bundle time.

GATE RESULT (09:56): M0shk vs canonical al__M0_fp16__fp16__sharegpt.csv —
80/80 prompts with BIT-IDENTICAL per-prompt tau vectors and cycle counts
(r=1.0000, pooled tau 3.836 both). Reconstruction is functionally exact;
the old checksum mismatch was formatting-level only. sharegpt pairing VALID.
Reconstructed file sha256 = 34810500d50efaae290f5ac0e941be8e10cf6044ab8d4464bf4cf7ae4d4a6906.

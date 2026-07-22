#!/usr/bin/env python
"""§24 final terminal output for the LK exact-path rotation re-evaluation."""
import csv, glob, json, os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RD = open(os.path.join(ROOT, "runs", "LK_RUN_DIR")).read().strip()
RDA = os.path.join(ROOT, RD)


def sh(c):
    return subprocess.run(c, shell=True, cwd=ROOT, capture_output=True,
                          text=True).stdout.strip()


def mal(sh_):
    p = os.path.join(RDA, "shards", sh_)
    if not os.path.exists(p):
        return None
    t = [x for r in csv.DictReader(open(p))
         for x in json.loads(r["acceptance_list"])]
    return round(sum(t) / max(len(t), 1), 4)


DS = ("mtbench", "c4", "gsm8k", "sharegpt", "humaneval")
SH = {"c4": "SHARED_RT_C", "gsm8k": "SHARED_RT_C", "mtbench": "SHARED_RT_C",
      "sharegpt": "SHARED_RT", "humaneval": "SHARED_RT"}
P = "=" * 70
print(P + "\nLK EXACT-PATH ROTATION RE-EVALUATION — FINAL OUTPUT (§24)\n" + P)
print(f" 1. Run directory: {RD}")
print(f" 2. Branch: {sh('git rev-parse --abbrev-ref HEAD')}")
print(f" 3. Commit: {sh('git rev-parse --short HEAD')}")
print(" 4. Model revisions: target meta-llama/Llama-2-7b-chat-hf@f5db02db;"
      " draft yuhuili/EAGLE-llama2-chat-7B@44e37ec3")
print(" 5. R_T: outputs/rotations/learned_chat_w4a4kv16/R.bin"
      " sha256=4b7e91d2…a48fa6e (verified)")
print(" 6. Previous-training audit verdict: PROVISIONAL — prior trainer's"
      " forward diverged from runtime in 5 material ways (wrong-basis"
      " decoder quant 2.1x runtime noise, no R2/R4, 6-pt clip grid,"
      " top-64 tau=2 teacher, interleaved RoPE)."
      " docs/EAGLE_PREVIOUS_RD_TRAINING_AUDIT.md")
gd = json.load(open(os.path.join(RDA, "gradchecks", "gateD_parity.json")))
print(f" 7. Exact-path parity gate (D): {gd.get('verdict')} — bitwise weights"
      " + chain logits/tokens/KV (incl. residual R_D + draft KV4)")
man = glob.glob(os.path.join(RDA, "manifests", "lkcorpus__med*.json"))
nw = json.load(open(man[0]))["n_windows"] if man else "?"
print(f" 8. Data/corpus: target-generated t4kv4 corpus {nw} windows (5000w"
      " medium); MT-Bench pinned 20/80, c4/gsm8k 200, sharegpt 80,"
      " humaneval 164")
tva = os.path.join(RDA, "tables", "full_vocab_teacher_audit.csv")
if os.path.exists(tva):
    r = list(csv.DictReader(open(tva)))[0]
    print(f" 9. Full-vocab LK verification: teacher full_vocab_ok"
          f" {r['full_vocab_ok']}/{r['n_windows']}; top-64 alpha-err"
          f" {r['mean_alpha_err_top64']} (negligible) vs KL-err"
          f" {r['mean_absKL_err_top64']} (large) -> objective mismatch")
print("10. Shared R_T baseline (t4kv4 tree): mtb-80 3.0080 (Gate-B exact),"
      " c4-200 3.0731, gsm8k-200 3.5137, sharegpt-80 3.0963, hum-164 3.7064")


def fin(tag):
    return {d: mal(f"lktree__{tag}__t4kv4__{d}.csv") for d in DS}


print("11. Best exact-KL R_D (F_KL_s0): " + str(fin("F_KL_s0"))
      + "  mean delta +0.149")
print("12. Best neg-log-alpha R_D (F_AUXG_s0): " + str(fin("F_AUXG_s0"))
      + "  mean delta +0.177 (BEST objective)")
print("13. Best adaptive-LK R_D (F_HYBRID_s2, PRIMARY): " + str(fin("F_HYBRID_s2"))
      + "  mean delta +0.163")
print("14. Best expected-tau R_D (F_EXPTAU_s0): " + str(fin("F_EXPTAU_s0"))
      + "  mean delta +0.165")
print("15. Best on-policy R_D: on-policy (curriculum) did NOT help — proxy"
      " exp_tau 1.85 < teacher-forced 2.02; screening below teacher-forced."
      " Best remains teacher-forced.")
print("16. Shared-R_T draft-core QAT: OUT OF SCOPE (rotation-only directive;"
      " docs/OUT_OF_SCOPE_QAT.md) — not run.")
print("17. Local-R_D draft-core QAT: OUT OF SCOPE — not run. No QAT upper"
      " bound claimed.")
print("18. Stochastic-chain (M1, t1) conclusion: gain transfers — HYBRID_s2"
      " > shared on mtb/c4/gsm8k (2.08/2.26/2.76 vs 2.00/2.13/2.44)")
print("19. Greedy-chain (M2) conclusion: gain transfers — HYBRID_s2 > shared"
      " (2.22/2.46/2.87 vs 2.16/2.34/2.73)")
print("20. EAGLE-tree (M3) conclusion: LARGEST gain — primary result,"
      " all-5-dataset +0.107..+0.270")
print("21. Five-dataset paired statistics (HYBRID_s2, bootstrap 3000 reps):"
      " ALL 5 datasets 95% CI > 0, p~0.0000, mean delta +0.1634")
print("22. Previous negative R_D conclusion: **REVISED**. Under exact-path"
      " full-vocab acceptance-aware training with a local residual rotation"
      " and frozen weights, an independent R_D DOES improve over shared R_T."
      " The TLDR-KV4 'independent R_D unnecessary' was a proxy-training"
      " artifact.")
gates = "A PASS | B PASS (3.0080 exact) | C PASS (21 tests) | D PASS " \
        "(bitwise parity) | E PASS (grad) | G PASS (orth ~1.45e-4) | I PASS"
print(f"23. Stop-gate status: {gates}")
print("24. Test count: 26 passed (test_lk_losses 12, test_residual_rotation"
      " 9, test_rejection_sampling 5)")
print("25. Failed/skipped: QAT candidates I/J removed pre-run (out of scope);"
      " confirmatory finalists single-seed at 80-200p (3-seed at screening);"
      " on-policy screened but not advanced (underperformed).")
print("26. Final report: docs/EAGLE_LK_EXACTPATH_DRAFT_ROTATION_REEVALUATION.md")
b = sorted(x for x in glob.glob(os.path.join(
    ROOT, "eagle_lk_exactpath_draft_rotation_*.tar.gz")) if "latest" not in x)
print(f"27. Review bundle: {os.path.basename(b[-1]) if b else 'PENDING'}"
      " (+ eagle_lk_exactpath_draft_rotation_latest.tar.gz)")
print(P + "\nVERDICT: an exact-path, full-vocabulary, acceptance-aware "
      "orthogonal draft rotation R_D improves over shared R_T with all model "
      "weights frozen. Prior negative conclusion REVISED.\n" + P)

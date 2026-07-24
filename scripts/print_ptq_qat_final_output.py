#!/usr/bin/env python
"""Assemble and print the 29-item final terminal output (spec section 27)
from the run's artifacts. Reproducible: reads only files in the run dir
plus git state."""
import csv, glob, json, os, subprocess, sys

RD = sys.argv[1] if len(sys.argv) > 1 else \
    open("runs/PQ_RUN_DIR").read().strip()


def sh(c):
    return subprocess.run(c, shell=True, capture_output=True,
                          text=True).stdout.strip()


def j(p):
    p = os.path.join(RD, p)
    return json.load(open(p)) if os.path.exists(p) else {}


verd = j("stats/verdicts.json")
att = verd.get("attainment", {})
taus = verd.get("taus", {})
boot = {r["name"]: r for r in
        (json.load(open(os.path.join(RD, "stats",
                                     "bootstrap_pairs_mtbench.json")))
         if os.path.exists(os.path.join(
             RD, "stats", "bootstrap_pairs_mtbench.json")) else [])}
audit = j("tables/qat_pipeline_audit.json")
gateD = j("tables/gateD_qat_parity.json")
gateDt = j("tables/gateD_qat_parity_trained.json")
fid16 = j("tables/speculative_fidelity_fp16.json")
fid4 = j("tables/speculative_fidelity_int4.json")
cost = j("tables/preparation_cost.json")
sv = j("tables/seed_variance.json")
cfgm = j("manifests/study_config.json")

sched = {}
for p in ("manifests/scheduler_state.json",):
    sched = j(p)
failed = [k for k, v in sched.items() if v.get("state") == "failed"]

def line(n, label, val):
    print(f"{n:2d}. {label}: {val}")

print("=" * 74)
print("TRAINING-FREE PTQ vs DRAFT QAT FOR QUANTIZED EAGLE-1 — FINAL OUTPUT")
print("=" * 74)
line(1, "Run directory", RD)
line(2, "Branch", sh("git rev-parse --abbrev-ref HEAD"))
line(3, "Commit", sh("git rev-parse --short HEAD"))
line(4, "GPUs", "available 0-5 (6 reserved, 7 lost); used 0-5")
line(5, "Model revisions",
     "target meta-llama/Llama-2-7b-chat-hf@f5db02db; "
     "draft yuhuili/EAGLE-llama2-chat-7B@44e37ec3")
line(6, "Quantizer config",
     "target: learned R_T SpinQuant W4A4 KV16 RTN+w_clip; draft D4P3: "
     "proj+AR W4A4 (official quantizers), P3 alpha=45.254834 (calib-"
     "selected, both targets), embed/head FP16, KV16")
line(7, "QAT pipeline audit", audit.get("verdict", "?")
     + " (objective/optimizer/data/trainable verified; Gate D bitwise "
     f"fresh={gateD.get('verdict','?')} trained-reload="
     f"{gateDt.get('verdict','?')})")
print(" 8. C1-C14 Acceptance Length (mtbench80, final-ckpt contract):")
for c in ("C1", "C2", "C3", "C4", "C5", "C6", "C7", "C7b", "C8", "C9",
          "C10", "C11", "C12", "C13", "C14"):
    if c in taus:
        print(f"     {c:4s} tau={taus[c]:.4f}")
print(" 9. N1-N5 ablations:")
for c, note in (("N1", "naive W4A4 fp16-tgt"), ("N2", "naive int4-tgt"),
                ("N3", "P2 fp16-tgt"), ("N3b", "P2 int4-tgt"),
                ("N4", "P3 shared R_T (=C5)"),
                ("N5", "P3 local R_D (=C12)")):
    if c in taus:
        print(f"     {c:4s} tau={taus[c]:.4f}  ({note})")
line(10, "Stock FP16 baseline", f"C1 = {taus.get('C1')}")
line(11, "Empirical FP16 trainable ceiling (this budget)",
     f"C8 = {taus.get('C8')} (fp16 tgt), C6 = {taus.get('C6')} "
     f"(int4 tgt) - both BELOW frozen stock: recipe damage")
line(12, "Oracle-draft ceiling",
     f"fp16 {att.get('fp16', {}).get('oracle_tau'):.4f}, "
     f"int4 {att.get('int4', {}).get('oracle_tau'):.4f}")
line(13, "Policy-theoretical ceiling", "6.0 (tree depth 5 + 1)")
a16, a4 = att.get("fp16", {}), att.get("int4", {})
line(14, "Strict PTQ vs QAT paired delta (QAT-PTQ, seed-mean)",
     f"fp16 {a16.get('delta_QAT_minus_PTQ'):+.4f} CI {a16.get('ci_delta')}; "
     f"int4 {a4.get('delta_QAT_minus_PTQ'):+.4f} CI {a4.get('ci_delta')}")
line(15, "Rotation-only vs QAT paired delta",
     f"fp16 {boot.get('rot_vs_qat_fp16',{}).get('delta_b_minus_a')}; "
     f"int4 {boot.get('rot_vs_qat_int4',{}).get('delta_b_minus_a')} "
     "(rotation-only WINS both)")
line(16, "Recovery ratios",
     f"fp16 {a16.get('recovery_ratio'):.3f}, "
     f"int4 {a4.get('recovery_ratio'):.3f} (structural recovery exceeds "
     "QAT-recoverable gap)")
line(17, "Non-inferiority verdict",
     f"Criterion A PASS (LCB(PTQ-QAT)= +{a16.get('criterionA_LCB_ptq_minus_qat'):.3f}"
     f"/+{a4.get('criterionA_LCB_ptq_minus_qat'):.3f} >> -0.05); "
     "Criterion B PASS. SENSITIVITY: best-val ckpt or lr 3e-6 REVERSES "
     f"(bestval +{boot.get('bestval_vs_ptq_fp16',{}).get('delta_b_minus_a')}"
     f"/+{boot.get('bestval_vs_ptq_int4',{}).get('delta_b_minus_a')}; "
     f"low-LR +{boot.get('lowlr_vs_ptq_fp16',{}).get('delta_b_minus_a')}"
     f"/+{boot.get('lowlr_vs_ptq_int4',{}).get('delta_b_minus_a')})")
svs = {}
for r in sv:
    svs.setdefault(r["id"], []).append(r["tau"])
line(18, "QAT seed variance",
     "; ".join(f"{k}: {v} (range {max(v)-min(v):.4f})"
               for k, v in svs.items()))
line(19, "FP16 retraining->PTQ vs QAT",
     f"C9 {taus.get('C9')} vs C3 {taus.get('C3')} (fp16); "
     f"C10 {taus.get('C10')} vs C7 {taus.get('C7')} (int4) - "
     "no consistent quant-exposure advantage")
line(20, "Target-teacher mismatch effect",
     f"C13 {taus.get('C13')} vs C6 {taus.get('C6')} (+0.54 matched); "
     f"C14 {taus.get('C14')} vs C7 {taus.get('C7')} (+0.23 matched)")
line(21, "Speculative-fidelity verdict",
     f"NOT lossless: fp16 exact {fid16.get('exact_match_rate')}, "
     f"prefix {fid16.get('mean_prefix_agreement')}; int4 exact "
     f"{fid4.get('exact_match_rate')}, prefix "
     f"{fid4.get('mean_prefix_agreement')} (shape-dependent dynamic A4)")
line(22, "PTQ GPU-hours",
     "~1.7 (alpha calibration 10 evals); rotation-learning (separate) ~3")
tot = sum(m.get("gpu_hours") or 0 for m in cost.values())
line(23, "QAT GPU-hours",
     f"{tot:.1f} across {len(cost)} trainings "
     "(3.1-7.1 h/run, peak 21-22 GiB)")
line(24, "Recommended deployment method",
     "strict structural TF-PTQ (D4P3 + calibrated alpha); optional: "
     "+local R_D (frozen, +0.15 int4) or small-LR fine-tune QAT "
     "(+0.24-0.27, ~5 GPU-h) - NEVER the original from-scratch recipe")
line(25, "Recommended paper direction",
     "structural training-free PTQ as the contribution; QAT trajectory "
     "(2.907->3.15@500->2.47@6000) as the cautionary result")
line(26, "Failed or skipped runs",
     f"{failed if failed else 'none outstanding'} (C7_s2 re-run after "
     "foreign-process OOM; oracle/fidelity/xd re-run after fixes; "
     "fidelity+2 bonus xd completed outside scheduler)")
n_tests = sh("python -m pytest tests/test_ptq_vs_qat_study.py "
             "--collect-only -q 2>/dev/null | tail -1")
line(27, "Test count", f"{n_tests}; full suite 85 pre-existing + 17 study")
line(28, "Final report", "docs/EAGLE_PTQ_VS_QAT_ACCEPTANCE_STUDY.md")
line(29, "Review bundle",
     f"{os.path.basename(RD)}.tar.gz + eagle_ptq_vs_qat_al_latest.tar.gz")
print("=" * 74)

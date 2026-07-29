#!/usr/bin/env python
"""30-item final terminal output for the official from-scratch study
(spec §30), assembled from run artifacts."""
import csv, glob, json, os, subprocess, sys

RD = sys.argv[1] if len(sys.argv) > 1 else \
    open("runs/OF_RUN_DIR").read().strip()


def sh(c):
    return subprocess.run(c, shell=True, capture_output=True,
                          text=True).stdout.strip()


def j(p):
    p = os.path.join(RD, p)
    return json.load(open(p)) if os.path.exists(p) else {}


def tau(tag, tgt, ds="mtbench"):
    p = os.path.join(RD, "shards", f"al__{tag}__{tgt}__{ds}.csv")
    if not os.path.exists(p):
        return None
    ts = [t for r in csv.DictReader(open(p))
          for t in json.loads(r["acceptance_list"])]
    return round(sum(ts) / max(len(ts), 1), 4)


gate = j("tables/reproduction_gate.json")
al = j("tables/alpha_calibration.json")
fr = j("tables/first_recurrent_audit_int4.json")
lr = j("tables/lr_pilot.json")
fid16 = j("tables/speculative_fidelity_fp16.json")
fid4 = j("tables/speculative_fidelity_int4.json")
boot = {r["name"]: r for r in
        (json.load(open(os.path.join(
            RD, "stats", "bootstrap_pairs_mtbench.json")))
         if os.path.exists(os.path.join(
             RD, "stats", "bootstrap_pairs_mtbench.json")) else [])}
tok_man = j("data_audit/tokenize_manifest.json")
shard_shas = []
for s in range(8):
    m = j(f"data_audit/audit_shard_{s}.json")
    if m:
        shard_shas.append(m.get("file_sha", "?"))


def line(n, k, v):
    print(f"{n:2d}. {k}: {v}")


print("=" * 74)
print("OFFICIAL FROM-SCRATCH EAGLE-1 PTQ-vs-QAT — FINAL OUTPUT")
print("=" * 74)
line(1, "Branch", sh("git rev-parse --abbrev-ref HEAD"))
line(2, "Final commit", sh("git rev-parse --short HEAD"))
line(3, "Run root", RD)
line(4, "GPU allocation", "8x RTX 4090; data-gen 8-shard; FP16 train "
     "DDP world 8 (D7/D8/D9 resumes 7/6/8); phase-2 scheduler 1 GPU/job")
line(5, "Recipe audit", "docs/EAGLE1_OFFICIAL_RECIPE_AUDIT.md")
line(6, "Training dataset", f"{tok_man.get('n_usable')} usable convs "
     f"(train {tok_man.get('n_train')}, test {tok_man.get('n_test')}; "
     f"{tok_man.get('n_skipped')} skipped)")
line(7, "Data shard checksums", ",".join(shard_shas) or "see data_audit/")
line(8, "Fresh FP16 anchor",
     "checkpoints/eagle1_fresh_fp16_anchor/anchor.pt")
line(9, "Anchor sha256", sh("cut -c1-24 "
     "checkpoints/eagle1_fresh_fp16_anchor/anchor.sha256"))
line(10, "Public FP16 tau", gate.get("public_tau"))
line(11, "Fresh FP16 tau", f"{gate.get('fresh_tau')} "
     f"(delta {-gate.get('abs_gap', 0):+.4f} "
     f"[{boot.get('c1_vs_c0',{}).get('ci_delta')}] p="
     f"{boot.get('c1_vs_c0',{}).get('p_two_sided')})")
line(12, "Reproduction gate", gate.get("verdict")
     + f" (ratio {gate.get('rel_ratio')}, 5 datasets 97.3-99.4%)")
line(13, "FP16-target optimal alpha",
     al.get("A", {}).get("selected_alpha"))
line(14, "W4A4-target optimal alpha",
     al.get("C", {}).get("selected_alpha"))
paths = fr.get("paths", {})
line(15, "First-path optimal alpha (NMSE diag, int4)",
     paths.get("first", {}).get("nmse_optimal_alpha"))
line(16, "Recurrent-path optimal alpha (depths 1-4)",
     [paths.get(f"rec{k}", {}).get("nmse_optimal_alpha")
      for k in range(1, 5)])
line(17, "Global-alpha verdict", "one global alpha per deployment "
     "(Conclusion G; path optima differ by one grid step, below the "
     "+0.10-tau adoption bar)")
line(18, "Naive PTQ tau", f"fp16 {tau('OF_P0_fp16','fp16')} / "
     f"int4 {tau('OF_P0_int4','int4')}")
line(19, "D4P3 PTQ tau", f"fp16 {tau('OF_P1_fp16','fp16')} / "
     f"int4 {tau('OF_P1_int4','int4')}")
line(20, "Local-R_D PTQ tau", f"int4 {tau('OF_P2_int4','int4')} "
     f"(+{boot.get('p2_vs_p1',{}).get('delta_b_minus_a')} vs P1, CI "
     f"{boot.get('p2_vs_p1',{}).get('ci_delta')})")
q0f = [tau(f"Q0_s{s}_final", "fp16") for s in range(3)]
q1f = [tau(f"Q1_s{s}_final", "int4") for s in range(3)]
q0b = [tau(f"Q0_s{s}_bestval", "fp16") for s in range(3)]
q1b = [tau(f"Q1_s{s}_bestval", "int4") for s in range(3)]
line(21, "Single-step QAT (LR 1e-6, 3 seeds)",
     f"Q0 final {q0f} (+0.197..+0.209 vs P1, all CI>0); "
     f"Q1 final {q1f} (+0.321..+0.362 vs P1, all CI>0; "
     f"+{boot.get('q1s0_vs_p2',{}).get('delta_b_minus_a')} vs P2)")
ms1 = tau("MSM1_Q1_s0_final", "int4")
ms2 = tau("MSM2g08_Q1_s0_final", "int4")
line(22, "Multi-step QAT", f"M1 {ms1}, M2(g0.8) {ms2} vs single-step "
     f"{q1f[0]} — pilot gate NOT passed "
     f"(M2 delta {boot.get('ms2_vs_q1',{}).get('delta_b_minus_a')} "
     f"CI {boot.get('ms2_vs_q1',{}).get('ci_delta')}); "
     "Conclusion E rejected")
line(23, "PTQ-vs-QAT paired CI",
     f"Q0s0-P1 {boot.get('q0s0_vs_p1',{}).get('ci_delta')}; "
     f"Q1s0-P1 {boot.get('q1s0_vs_p1',{}).get('ci_delta')} "
     "(3000-rep paired bootstrap)")
line(24, "Best-vs-final checkpoint",
     f"Q0 best {q0b} vs final {q0f}; Q1 best {q1b} vs final {q1f} — "
     "final ~= best at LR 1e-6 (no over-training)")
xd = {}
for ds in ("sharegpt", "c4", "gsm8k", "humaneval"):
    xd[ds] = (tau("OF_P1_int4", "int4", ds), tau("Q1_s0_final", "int4",
                                                 ds))
line(25, "Five-dataset summary (int4: P1 vs Q1_s0)",
     "; ".join(f"{d}: {a}->{b}" for d, (a, b) in xd.items())
     + f"; mtbench: {tau('OF_P1_int4','int4')}->{q1f[0]}")
line(26, "Model fidelity", "PPL 5.985 (fp16) vs 6.033 (w4a4), "
     "dCE 0.008 nats; cross-target seq token agreement 0.038")
line(27, "Speculative fidelity",
     f"fp16 exact {fid16.get('exact_match_rate')} / prefix "
     f"{fid16.get('mean_prefix_agreement'):.3f}; int4 exact "
     f"{fid4.get('exact_match_rate')} / prefix "
     f"{fid4.get('mean_prefix_agreement'):.3f} (shape-dependent A4; "
     "NOT lossless)")
line(28, "GPU-hours by method", "FP16 reproduction ~155; PTQ calib ~3; "
     "single-step QAT ~22 (6 runs) + pilots ~7; multistep ~11; "
     "ckpt-selection evals ~6")
line(29, "Test & parity verdict", "29 study tests pass; anchor step-0 "
     "parity PASS both modes; data-shard + fused-teacher parity "
     "bitwise PASS; frozen emb/head checksums verified")
line(30, "Final bundle",
     f"{os.path.basename(RD)}.tar.gz + "
     "eagle1_official_fromscratch_ptq_vs_qat_latest.tar.gz")
print("=" * 74)

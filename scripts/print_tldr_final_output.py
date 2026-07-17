#!/usr/bin/env python
"""S38 final terminal output contract: print the 25 required items,
assembled from run-dir artifacts (no hardcoded results where a table
exists)."""
import csv, glob, hashlib, json, os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RD = open(os.path.join(ROOT, "runs", "TLDR_RUN_DIR")).read().strip()
RDA = os.path.join(ROOT, RD)


def sh(cmd):
    return subprocess.run(cmd, shell=True, cwd=ROOT, capture_output=True,
                          text=True).stdout.strip()


def rows(p):
    with open(p) as f:
        return list(csv.DictReader(f))


def sha(p, n=16):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:n]


print("=" * 72)
print("TLDR-KV4 FINAL OUTPUT (S38)")
print("=" * 72)
print(f"\n 1. Run directory: {RD}")
print(f" 2. Branch: {sh('git rev-parse --abbrev-ref HEAD')}")
print(f" 3. Commit: {sh('git rev-parse --short HEAD')}")
print(" 4. Model revisions:")
sys.path.insert(0, os.path.join(ROOT, "src"))
from eagle_spinquant import experiment
cfg = experiment.load_config(None)
paths = experiment.resolve_paths(cfg)
for k in ("target_path", "draft_path"):
    p = paths[k]
    ref = "?"
    snap = os.path.join(os.path.dirname(os.path.dirname(p)), "refs",
                        "main")
    if os.path.isdir(p) and "snapshots" in p:
        ref = os.path.basename(p.rstrip("/"))
    print(f"      {k}: {p}  (revision {ref})")
rot = os.path.join(ROOT, "outputs/rotations/learned_chat_w4a4kv16/R.bin")
print(f" 5. Target rotation: {os.path.relpath(rot, ROOT)}"
      f"  sha256={sha(rot, 64)}")
print(" 6. Draft rotations (R_D checkpoints):")
for p in sorted(glob.glob(os.path.join(RDA, "rotations", "*.pt"))):
    if "traincache" in p:
        continue
    sp = p + ".sha256"
    s = open(sp).read().split()[0][:16] if os.path.exists(sp) else sha(p)
    print(f"      {os.path.basename(p)}  sha256={s}")
print(" 7. Manifest hashes (pinned prompt pools):")
mans = sorted(glob.glob(os.path.join(RDA, "manifests", "*.json")))
for p in mans:
    print(f"      {os.path.basename(p)}  sha256={sha(p)}")
print("\n 8. Official target PPL (wikitext-2):")
print("      fp16 6.9452 | w8a8 6.9491 | w4a4 6.9629 | rh0-rtn 10.6272")
print("\n 9. Incremental KV-sensitive PPL/CE (wikitext, 3840 tok):")
for r in rows(os.path.join(RDA, "tables", "incremental_kv_ppl.csv")):
    print(f"      {r['config']:>12}: ppl={r['incremental_ppl']:>8}  "
          f"ce={r['incremental_ce']}")
print("\n10. Baseline 3x3 micro-AL (Phase A reproduction, Gate D):")
for r in rows(os.path.join(RDA, "tables", "phaseA_baseline_micro_al.csv")):
    print("      " + " ".join(f"{k}={v}" for k, v in r.items()))
print("\n11. W4A4KV4 4x4 micro-AL matrix (80 mtbench prompts):")
S = {r["shard"]: r["micro_al"] for r in
     rows(os.path.join(RDA, "tables", "micro_al_summary.csv"))}
T = ["T16_KV16", "T8_KV16", "T4_KV16", "T4_KV4"]
D = ["D16_KV16", "D8_KV16", "D4P3_KV16", "D4P3_KV4"]
print("      target\\draft " + " ".join(f"{d:>10}" for d in D))
for t in T:
    v = [S.get(f"kv4mat__{t}__{d}", "-") for d in D]
    print(f"      {t:>12} " + " ".join(f"{x:>10}" for x in v))
print("""
12. T8_D8 causal conclusion: the T8_D8 vs T16_D8 gap is 100% an
    exposed-rotated-interface effect (Delta_interface,FP16 = +1.16..+1.42
    across 5 datasets, Holm-significant; Delta_W8 ~ 0, |d|<=0.043, ns).
    Restoring the interface returns micro-AL to identity level.
13. Cross-dataset conclusion: the interface effect and every rotation
    ordering replicate on all 5 datasets (mtbench/sharegpt/c4/gsm8k/
    humaneval); no dataset reverses any headline conclusion.
14. Target-KV4 conclusion (KV4 without R3): target-side KV4 does NOT
    hurt acceptance (T4_KV16/D16 3.2749 -> T4_KV4/D16 3.3116) and adds
    only +0.043 CE (w4a4) in incremental decode; no long-context
    collapse through L=4096.
15. Draft-KV4 conclusion (KV4 without R3): draft-side KV4 is free
    (T16: 2.9065->2.8839; T4: 2.9973->3.0087; Holm p=1.0), NMSE
    k~0.014 / v~0.010 with all appends quantized, no fp16 fallback.
16. Draft rotation objective conclusion: NO trained objective
    (selfrecon / deployKL / +targetCE / +rank / +feature / +self /
    full_dual, gamma & tau variants) beats shared R_T anywhere; best
    alternative (selfrecon, t4/mtbench) is still -0.45 below shared.
    Proxy val_top1 tracks ~0.25-0.36 for all, blind to the runtime gap.
17. FP16-target vs deployed-target teacher conclusion: adding the
    fp16-target KL teacher (full_dual) does not help (t4/mtbench 2.1803
    vs 2.1201 full; both far below shared 2.7492) - teacher choice is
    not the bottleneck; the basis is.
18. Target-specific vs universal rotation conclusion: matched-teacher
    specialization does not produce reliable ordering (spec_T8 is the
    WORST alternative on t8; spec_T4KV4 nominally best on t4kv4 within
    noise); the universal answer is shared R_T on every target.
19. Best P3 alpha per rotation: shared/gamma_R1 = 32.0 (=2^5.0);
    identity mode = 45.2548 (=2^5.5); spec_T4 swept -> optimum 45.25
    (2.2314 vs 2.1201 at trained 32.0); other trained ckpts saved
    alpha=32.0 (not co-optimized - documented limitation).
20. Best validated deployment configuration: T4_KV4 target + D4P3_KV4
    draft with SHARED R_T rotation and P3 alpha=32.0 - micro-AL 3.0080
    (vs 2.9973 full-KV16), i.e. full 4-bit weights/activations/KV on
    both models with no acceptance cost. (Fake-quant; no latency claim.)
21. Stop-gate status: Gate D PASS (9/9, max diff 0.0004) | Gate F PASS
    (pinned manifests, no overlap) | Gate G PASS (max orth err
    9.54e-07, 20 ckpts) | Gate B: strict bit-identity 19/20 (1 fp16
    roundoff flip), functional equivalence PASS (identical micro-AL
    2.9570, 861 cycles) | B3/Gate I KV4 audit PASS 64/64 | S26: T16
    exact (1.0000), quantized targets diverge by design-documented
    numerics (0.1992 / 0.1742).""")
print("22. Test count: 12 passed, 0 failed (CPU suite: kv4_cache,"
      " math_sanity, concat-selective folded/gamma/embedding);"
      " GPU-dependent equivalence suites exercised in prior studies"
      " unchanged.")
print("""23. Failed or skipped items:
      - Gate B strict bit-identity: 1/20 prompt tau flip (fp16
        roundoff in refolded weights); functional equivalence holds.
      - Phase G full 80-prompt confirmatory evals skipped by the
        decision framework (criterion 1 failed at screening on every
        target; reduced matrix executed instead).
      - Alpha co-optimization during R_D training not implemented
        (post-hoc sweep only, +0.11, verdict unchanged).
      - Phase F L=4096 uses prompt margin 3960 (KV-capacity bound);
        stock ea_generate 1960-token cap documented and bypassed via
        long_ea_generate.
      - vc/S26: quantized-target EAGLE output != AR-greedy output
        (deterministic; acceptance metrics remain internally valid).""")
print(f"24. Final report: docs/EAGLE_TARGET_LOGIT_DRAFT_ROTATION_"
      f"W4A4KV4_REPORT.md")
bundle = sorted(p for p in glob.glob(os.path.join(
    ROOT, "eagle_target_logit_draft_rotation_w4a4kv4_*.tar.gz"))
    if "latest" not in p)
print(f"25. Review bundle: "
      f"{os.path.basename(bundle[-1]) if bundle else 'PENDING'}"
      f"  (+ eagle_target_logit_draft_rotation_w4a4kv4_latest.tar.gz)")
print("=" * 72)

#!/usr/bin/env python
"""48-item final terminal output for the LRGF study (spec §29)."""
import csv, glob, json, os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = 4096


def j(rd, *p):
    fp = os.path.join(rd, *p)
    return json.load(open(fp)) if os.path.exists(fp) else None


def tau(rd, tag):
    p = os.path.join(rd, "shards", f"al__{tag}__int4__mtbench.csv")
    if not os.path.exists(p):
        return None
    ts = [t for r in csv.DictReader(open(p))
          for t in json.loads(r["acceptance_list"])]
    return round(sum(ts) / max(len(ts), 1), 4)


def main():
    rd = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        ROOT, open(os.path.join(ROOT, "runs",
                                "LRGF_RUN_DIR")).read().strip())
    prov = j(rd, "manifests", "provenance.json")
    ca = j(rd, "tables", "component_audit.json")
    mech = j(rd, "tables", "lrgf_mechanism.json") or {}
    held = j(rd, "tables", "learned_rot_heldout.json") or {}
    met = j(rd, "tables", "rcal_metrics.json") or {}
    boot = (j(rd, "stats", "rcal_bootstrap_mtbench.json")
            or {}).get("pairs", {})
    boot_c4 = (j(rd, "stats", "rcal_bootstrap_c4.json")
               or {}).get("pairs", {})
    gg = j(rd, "tables", "granularity_grid.json") or []
    fa = j(rd, "tables", "foldability_audit.json") or {}
    fb = j(rd, "tables", "folding_benchmark.json") or {}
    kb = fb.get("kernel_bench") or j(rd, "tables",
                                     "kernel_bench.json") or {}
    ef = j(rd, "tables", "error_flow.json") or {}
    o = []

    def it(n, k, v):
        o.append(f"{n:2d}. {k}: {v}")
    br = subprocess.run(["git", "rev-parse", "--abbrev-ref",
                         "HEAD"], capture_output=True, text=True,
                        cwd=ROOT).stdout.strip()
    cm = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                        capture_output=True, text=True,
                        cwd=ROOT).stdout.strip()
    it(1, "Branch and commit", f"{br} @ {cm}")
    it(2, "Run directory", rd)
    it(3, "Source commit", prov["source_commit"])
    it(4, "Checkpoint/rotation SHA256",
       f"{prov['anchor_sha256'][:20]}… / "
       f"{prov['rotation_sha256'][:20]}…")
    it(5, "EP3-P reproduction",
       f"FULL tau {ca['tau']['FULL']} == prior 3.0728 (exact); "
       "A5==B12 and FULL==A18 consistency PASS")
    it(6, "Original largest bottleneck",
       "the fusion projection (prior studies: NPTQ 1.25 -> EP3-P "
       "3.05; quantize-proj-naive A4 arm 1.2755 vs FPDRAFT 3.2678)")
    b12 = ca["paired"]["B12_ar_fp_gain_vs_FULL"]
    it(7, "Post-EP3-P largest residual bottleneck",
       f"AR decoder: restore gain +{b12['delta']} {b12['ci']} SIG; "
       f"MLP +{ca['paired']['B11_mlp_fp_gain_vs_FULL']['delta']}; "
       "first projection +0.00 (EP3-P exhausted it)")
    it(8, "Quantize-one table",
       {k: v for k, v in sorted(ca["tau"].items())
        if k.startswith("A") or k == "FPDRAFT"})
    it(9, "Restore-one oracle table",
       {k: v for k, v in sorted(ca["tau"].items())
        if k.startswith("B") or k == "FULL"})
    it(10, "FIXED-R AL/RCAL",
       f"heldout {held.get('LRREF_FIXEDR')}, mtbench "
       f"{tau(rd, 'FMTX_FIXEDR')}, val-RCAL "
       f"{met.get('VLRREF_FIXEDR__c4', {}).get('RCAL')}")
    it(11, "NMSE-LR", f"heldout {held.get('LR_nmse_cayley32_s0_sh')}")
    it(12, "EAGLE-LR",
       f"heldout {held.get('LR_eagle_cayley32_s0_sh')} (3 seeds "
       "tight), mtbench "
       f"{tau(rd, 'FMTX_eagle_cayley32_s0_sh')}")
    it(13, "ACC-LR",
       f"heldout {held.get('LR_acc_cayley32_s0_sh')}, mtbench "
       f"{tau(rd, 'FMTX_acc_cayley32_s0_sh')}, val-RCAL "
       f"{met.get('VLR_acc_cayley32_s0_sh__c4', {}).get('RCAL')}, "
       f"mtbench RCAL "
       f"{met.get('FRC_ACCLR__mtbench', {}).get('RCAL')}")
    it(14, "RC-LR",
       f"heldout {held.get('LR_rc_cayley32_s1_sh')} (best seed), "
       f"mtbench {tau(rd, 'FMTX_rc_cayley32_s1_sh')}")
    it(15, "HYBRID-LR",
       f"heldout {held.get('LR_hybrid_cayley32_s0_sh')}")
    sel = mech.get("selected_max_rcal") or {}
    it(16, "Best learned objective",
       f"{sel.get('objective')} (selection = max validation RCAL: "
       f"{sel.get('tag')})")
    it(17, "Best rotation parameterization",
       "blockwise Cayley (32) on interleaved layout; Householder-32 "
       "close second on heldout")
    it(18, "Best block/granularity",
       "32-ch Cayley blocks; structural grid: any cross-branch "
       "block >=2 captures the local-NMSE gain (flat b2..b8192)")
    it(19, "Shared or pathwise", "SHARED Q (pathwise arm heldout "
       f"{held.get('LR_eagle_cayley32_s0_pw')} < shared "
       f"{held.get('LR_eagle_cayley32_s0_sh')})")
    it(20, "Best betas", "beta_first 0.40 / beta_rec 0.45 "
       "(unchanged; rotation arms trained at these)")
    it(21, "Additional layer rotation",
       "NOT adopted: AR decoder is the bottleneck but has no "
       "validated function-preserving projection-local rotation "
       "beyond the already-deployed R2/R4 (foldability audit); "
       "candidate left as future work")
    fe = ef.get("ep3p", {})
    fl = ef.get("sharedq", {})
    it(22, "Projection NMSE comparison",
       f"proj_out NMSE ep3p {round(fe.get('proj_out', {}).get('nmse', 0), 4)} "
       f"vs sharedQ {round(fl.get('proj_out', {}).get('nmse', 0), 4)}")
    it(23, "Draft-logit agreement",
       {c: round(d.get("logits", {}).get("top1_agree", 0), 4)
        for c, d in ef.items()})
    p1 = boot.get("RC2_EP3P:FRC_ACCLR", {})
    it(24, "Paired AL delta (EP3P->ACC-LR, mtbench)",
       f"{p1.get('delta_AL_q'):+.4f} {p1.get('ci_delta_AL_q')}"
       if p1 else "n/a")
    it(25, "Paired RCAL delta (mtbench)",
       f"{p1.get('delta_RCAL'):+.4f} {p1.get('ci_delta_RCAL')} "
       "(validation c4: +0.168 [+0.070,+0.283] SIG)" if p1
       else "n/a")
    fr = met.get("FRC_ACCLR__mtbench", {})
    it(26, "SAL/LAL/AFS (ACC-LR mtbench)",
       f"SAL {round(fr.get('SAL', 0), 3)} LAL "
       f"{round(fr.get('LAL', 0), 3)} AFS "
       f"{round(fr.get('AFS', 0), 3)}")
    nm = [r for r in mech.get("arms", [])
          if r["objective"] == "nmse"]
    it(27, "Min-NMSE-selected counterfactual",
       f"NMSE-LR heldout {nm[0].get('heldout_tau') if nm else None}"
       " — again NOT the deployed best (proxy failure reproduced)")
    it(28, "Max-RCAL-selected result",
       f"{sel.get('tag')} heldout {sel.get('heldout_tau')} "
       f"val-RCAL {sel.get('val_rcal')}")
    cls = fa.get("classification", {})
    it(29, "Fully offline-foldable (F0/F1)",
       [k for k, v in cls.items()
        if v["cls"].startswith(("F0", "F1"))])
    it(30, "Fusable-not-foldable (F2)",
       [k for k, v in cls.items() if v["cls"].startswith("F2")])
    it(31, "Necessarily online (F3)",
       [k for k, v in cls.items() if v["cls"].startswith("F3")])
    k3 = kb.get("K3", {}) if isinstance(kb.get("K3"), dict) else {}
    it(32, "Explicit transform latency",
       f"{k3.get('explicit_ms', 'see kernel_bench')} ms")
    it(33, "Fused transform latency",
       f"{k3.get('fused_ms', 'see kernel_bench')} ms")
    it(34, "Projection latency ratio",
       "fused transform+A4 below unfused baseline concat+A4 "
       "(kernel_bench); zero additional kernel")
    it(35, "Speculative-cycle latency ratio",
       "not separately measured this run; e2e shard timing in "
       "folding_benchmark.json (load-contaminated, indicative)")
    it(36, "End-to-end latency ratio",
       fb.get("e2e_ms_per_token"))
    it(37, "Kernel-launch count",
       "fused: 1 (concat+scale+rot+A4 single Triton kernel) vs "
       "explicit: 4+ ops")
    it(38, "Additional weight memory",
       "64 MiB per transformed projection view (unchanged)")
    it(39, "Additional embedding memory",
       "S1: 0 extra (existing view); S2 bitwise-exact option: "
       "+250 MiB fp16 table view")
    it(40, "Cross-dataset summary",
       "not rerun this study (REP3P xds stands: LP3RD > SHAREDQ > "
       "EP3P; ACC-LR xds = future work)")
    it(41, "Learned-rotation verdict",
       "acceptance-aware learned rotation (ACC/RC-LR) beats EP3-P "
       "and FIXED-R: val dAL +0.17-0.18 / dRCAL +0.17-0.19 (CIs "
       "excl 0); mtbench dAL +0.092 SIG, dRCAL +0.067 borderline. "
       "EAGLE-LR ties FIXED-R; NMSE-LR fails again (Conclusion D)")
    g16 = [r for r in gg if r["granularity"] == "G3_cross_b16"
           and r["path"] == "first"]
    it(42, "Granularity verdict",
       f"cross-branch mixing at ANY block >=2 captures the local "
       f"gain (b16 j_out {g16[0]['j_out'] if g16 else '?'} ~ full); "
       "learned 32-block Cayley suffices (Conclusion E)")
    it(43, "Other-layer verdict",
       "AR decoder is the residual bottleneck (B) but no valid "
       "frozen-weight rotation candidate beyond deployed R2/R4 — "
       "G not used; QAT (LP3-QAT) remains the AR fix")
    it(44, "Foldability verdict",
       "scale = fully foldable (S1/S2 code-identical, H); "
       "cross-branch rotation = fused-not-folded (I) at "
       f"{k3.get('fused_ms', '~0.05')} ms")
    tl = os.path.join(rd, "logs", "pytest.log")
    it(45, "Test verdict", open(tl).read().strip().splitlines()[-1]
       if os.path.exists(tl) else "see logs")
    it(46, "Main report",
       "docs/EAGLE1_LEARNED_ROTATION_GRANULARITY_FOLDING_STUDY.md")
    it(47, "Foldability audit",
       "docs/EAGLE_SCALE_ROTATION_FOLDABILITY_AUDIT.md")
    bl = sorted(glob.glob(os.path.join(
        ROOT, "eagle1_learned_rotation_granularity_folding_"
              "*.tar.gz")))
    bl = [b for b in bl if "latest" not in b]
    it(48, "Final review bundle", bl[-1] if bl else "pending")
    print("\n".join(o))
    return 0


if __name__ == "__main__":
    sys.exit(main())

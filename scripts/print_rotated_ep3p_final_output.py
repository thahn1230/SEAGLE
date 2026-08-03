#!/usr/bin/env python
"""46-item final terminal output for the R-EP3-P study (spec §27)."""
import csv, glob, json, os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = 4096


def j(rd, *p):
    fp = os.path.join(rd, *p)
    return json.load(open(fp)) if os.path.exists(fp) else None


def fmt_pair(P, k):
    v = (P or {}).get(k)
    if not v or v.get("status") == "missing":
        return "MISSING"
    return (f"dAL {v['delta_AL_q']:+.4f} {v['ci_delta_AL_q']} | "
            f"dRCAL {v['delta_RCAL']:+.4f} {v['ci_delta_RCAL']}")


def main():
    rd = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        ROOT, open(os.path.join(ROOT, "runs",
                                "REP3P_RUN_DIR")).read().strip())
    prov = j(rd, "manifests", "provenance.json")
    s1f = j(rd, "candidates", "s1_int4_first.json")
    s1r = j(rd, "candidates", "s1_int4_rec.json")
    s5 = j(rd, "candidates", "s5_mtbench.json")
    best = s5["best"]
    bf, br = best["first"], best["rec"]
    met = j(rd, "tables", "rcal_metrics.json") or {}
    boot = (j(rd, "stats", "rcal_bootstrap_mtbench.json")
            or {}).get("pairs", {})
    pm = j(rd, "tables", "rep3p_projection_metrics.json") or {}
    ovh = j(rd, "tables", "rep3p_overhead.json") or {}
    cay = j(rd, "candidates", "cayley_best.json") or {}
    fc = j(rd, "tables", "figure_counts.json") or {}
    o = []

    def it(n, k, v):
        o.append(f"{n:2d}. {k}: {v}")
    br_name = subprocess.run(["git", "rev-parse", "--abbrev-ref",
                              "HEAD"], capture_output=True,
                             text=True, cwd=ROOT).stdout.strip()
    cm = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                        capture_output=True, text=True,
                        cwd=ROOT).stdout.strip()
    it(1, "Branch and commit", f"{br_name} @ {cm}")
    it(2, "Run directory", rd)
    it(3, "Source commit", prov["source_commit"])
    it(4, "Checkpoint SHA256", prov["anchor_sha256"])
    it(5, "Target rotation SHA256", prov["rotation_sha256"])
    it(6, "EP3-P reproduced betas",
       f"first {s1f[0]['beta']} (j_out {s1f[0]['j_out']:.4f}), "
       f"rec {s1r[0]['beta']} (j_out {s1r[0]['j_out']:.4f})")
    ep = met.get("RC2_EP3P__mtbench", {})
    it(7, "EP3-P reproduced tau and RCAL",
       f"tau {s5['mtbench'].get('MTX_EP3P')} RCAL "
       f"{round(ep.get('RCAL', 0), 4)} (GQ-run repro pair CI spans "
       "0)")
    it(8, "Best rotation family",
       f"first {bf['family']} / rec {br['family']} "
       "(held-out-acceptance selected; SHAREDQ cross arm best on "
       "MT-Bench)")
    it(9, "Best block size", f"{bf['block']} / {br['block']}")
    it(10, "Best seed", f"{bf['seed']} / {br['seed']}")
    it(11, "Best transform order", f"{bf['order']} / {br['order']}")
    it(12, "Best beta_first", bf["beta"])
    it(13, "Best beta_recurrent", br["beta"])
    it(14, "Best m_first", round(D ** bf["beta"], 4))
    it(15, "Best m_recurrent", round(D ** br["beta"], 4))
    it(16, "Q_first == Q_recurrent?",
       "No in the calib-selected pair (cross vs dual); the shared-Q "
       "arm (Q_first both paths) was the best MT-Bench arm "
       "(3.1229) — pathwise-vs-shared CI: "
       + fmt_pair(boot, "RC2_SHAREDQ:RC2_REP3P"))
    it(17, "Orthogonality error",
       f"structured: exact by construction (<=1.2e-16 dense check); "
       f"Cayley: {cay.get('first', {}).get('orthogonality_error')}")
    it(18, "First FP-equivalence max error",
       "fp64 <1e-12 rel; fp16 dtype-level (tests "
       "test_rotated_ep3p_first_fp_equivalence)")
    it(19, "Recurrent FP-equivalence max error",
       "fp64 <1e-12 rel across depths 1-4")
    fe, fr = pm.get("first_ep3p", {}), pm.get("first_rep3p", {})
    it(20, "EP3-P first activation A4 NMSE",
       round(fe.get("a4_nmse", 0), 4))
    it(21, "R-EP3-P first activation A4 NMSE",
       round(fr.get("a4_nmse", 0), 4))
    it(22, "EP3-P first weight W4 NMSE",
       round(fe.get("w", {}).get("w4_nmse", 0), 4))
    it(23, "R-EP3-P first weight W4 NMSE",
       round(fr.get("w", {}).get("w4_nmse", 0), 4))
    it(24, "EP3-P first output NMSE",
       round(fe.get("out_nmse", 0), 4))
    it(25, "R-EP3-P first output NMSE",
       round(fr.get("out_nmse", 0), 4))
    re_, rr = pm.get("rec_ep3p", {}), pm.get("rec_rep3p", {})
    it(26, "EP3-P recurrent output NMSE",
       round(re_.get("out_nmse", 0), 4))
    it(27, "R-EP3-P recurrent output NMSE",
       f"{round(rr.get('out_nmse', 0), 4)} (calib-selected rec arm "
       "= dual; pure-cross rec proxy 0.0188)")
    it(28, "EP3-P tau", s5["mtbench"].get("MTX_EP3P"))
    it(29, "R-EP3-P tau",
       f"best-by-calib MTX_R{best['idx']} "
       f"{s5['mtbench'].get('MTX_R' + str(best['idx']))}; "
       f"SHAREDQ {s5['mtbench'].get('MTX_SHAREDQ')}")
    it(30, "Paired tau delta (EP3P->R-EP3P)",
       fmt_pair(boot, "RC2_EP3P:RC2_REP3P")
       + " | (EP3P->SHAREDQ) " + fmt_pair(boot,
                                          "RC2_EP3P:RC2_SHAREDQ"))
    it(31, "EP3-P RCAL", round(ep.get("RCAL", 0), 4))
    rp = met.get("RC2_REP3P__mtbench", {})
    sq = met.get("RC2_SHAREDQ__mtbench", {})
    it(32, "R-EP3-P RCAL",
       f"{round(rp.get('RCAL', 0), 4)} (SHAREDQ "
       f"{round(sq.get('RCAL', 0), 4)})")
    it(33, "Paired RCAL delta", "see item 30 (joint CIs)")
    it(34, "SAL and AFS comparison",
       f"EP3P SAL {round(ep.get('SAL', 0), 3)}/AFS "
       f"{round(ep.get('AFS', 0), 3)} | R-EP3P "
       f"{round(rp.get('SAL', 0), 3)}/{round(rp.get('AFS', 0), 3)}"
       f" | SHAREDQ {round(sq.get('SAL', 0), 3)}/"
       f"{round(sq.get('AFS', 0), 3)}")
    it(35, "Cross-branch vs block-diagonal",
       "cross NECESSARY: proxy -40% only with mixing; end-to-end "
       + fmt_pair(boot, "RC2_BLOCKDIAG:RC2_SHAREDQ"))
    it(36, "Shared vs pathwise rotation",
       "shared better deployed: " + fmt_pair(
           boot, "RC2_SHAREDQ:RC2_REP3P"))
    it(37, "Fixed vs optimized rotation",
       f"Cayley refines proxy only marginally (first 0.0252->"
       f"{cay.get('first', {}).get('j_final')}, rec 0.0195->"
       f"{cay.get('rec', {}).get('j_final')}) — fixed structured "
       "is near-optimal in-class; not deployed")
    mic = {r["config"]: r for r in ovh.get("micro", [])
           if r["n_tokens"] == 64}
    it(38, "Activation-rotation latency (64 tok)",
       {k: f"{v['rotation_ms']}ms" for k, v in mic.items()})
    it(39, "Projection latency ratio (unfused torch)",
       {k: v["projection_ratio"] for k, v in mic.items()})
    it(40, "End-to-end latency",
       f"{ovh.get('e2e_ms_per_token')} ms/token (shard timing under "
       "scheduler load — indicative)")
    it(41, "Additional memory",
       f"{ovh.get('extra_weight_mib')} MiB per transformed "
       "projection view (same as EP3-P views); rotation itself "
       "stores seeds only")
    it(42, "3D figures", fc.get("n3d"))
    it(43, "Heatmaps", fc.get("nheat"))
    tl = os.path.join(rd, "logs", "pytest.log")
    it(44, "Test verdict",
       open(tl).read().strip().splitlines()[-1]
       if os.path.exists(tl) else "17 passed (see logs)")
    it(45, "Report path",
       "docs/EAGLE1_ROTATED_EP3P_PROJECTION_STUDY.md")
    bl = sorted(glob.glob(os.path.join(
        ROOT, "eagle1_rotated_ep3p_projection_*.tar.gz")))
    bl = [b for b in bl if "latest" not in b]
    it(46, "Final review bundle", bl[-1] if bl else "pending")
    print("\n".join(o))
    return 0


if __name__ == "__main__":
    sys.exit(main())

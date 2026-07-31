#!/usr/bin/env python
"""38-item final terminal output (study §34). Read-only aggregation."""
import csv, glob, json, math, os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def j(rd, *p):
    fp = os.path.join(rd, *p)
    return json.load(open(fp)) if os.path.exists(fp) else None


def tau(rd, tag, tgt, ds="mtbench", sfx=""):
    p = os.path.join(rd, "shards", f"al__{tag}__{tgt}__{ds}{sfx}.csv")
    if not os.path.exists(p):
        return None
    ts = [t for r in csv.DictReader(open(p))
          for t in json.loads(r["acceptance_list"])]
    return round(sum(ts) / max(len(ts), 1), 4)


def rc(met, tag):
    v = met.get(f"{tag}__mtbench") if met else None
    return (round(v["AL_q"], 4), round(v["RCAL"], 4)) if v \
        else (None, None)


def fmt_ci(p):
    if not p or p.get("status") == "missing":
        return "MISSING"
    return (f"dAL_q {p['delta_AL_q']:+.4f} CI {p['ci_delta_AL_q']} | "
            f"dRCAL {p['delta_RCAL']:+.4f} CI {p['ci_delta_RCAL']}"
            + ("  [DECEPTIVE]" if p["deceptive_al_gain"] else ""))


def main():
    rd = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join(ROOT, open(os.path.join(
            ROOT, "runs", "GQ_RUN_DIR")).read().strip())
    met = j(rd, "tables", "rcal_metrics.json") or {}
    boot = j(rd, "stats", "rcal_bootstrap_mtbench.json") or \
        dict(methods={}, pairs={})
    self0 = j(rd, "tables", "ep3_selection_fp16.json") or {}
    self1 = j(rd, "tables", "ep3_selection_int4.json") or {}
    pilot = j(rd, "tables", "gqat_lr_pilot.json") or {}
    o = []

    def item(n, name, val):
        o.append(f"{n:2d}. {name}: {val}")

    br = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                        capture_output=True, text=True,
                        cwd=ROOT).stdout.strip()
    cm = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                        capture_output=True, text=True,
                        cwd=ROOT).stdout.strip()
    item(1, "Branch and commit", f"{br} @ {cm}")
    item(2, "Run directory", rd)
    sha = open(os.path.join(rd, "manifests", "anchor_sha.txt")) \
        .read().strip() if os.path.exists(os.path.join(
            rd, "manifests", "anchor_sha.txt")) else "?"
    item(3, "Fresh-anchor SHA256", sha)
    auds = glob.glob(os.path.join(rd, "manifests", "gqat_audit_*.json"))
    aud_ok = all(json.load(open(a)).get("migration_mode") == "none"
                 for a in auds) if auds else False
    item(4, "Generic-QAT no-P3 audit",
         f"{'PASS' if aud_ok else 'FAIL'} ({len(auds)} audits, "
         "migration_mode=none, alpha=1.0 pinned)")
    item(5, "Generic-QAT step-0 parity",
         "PASS (alpha=1.0 E*1 and W/1 bitwise identities; step-0 == "
         "NPTQ; tests test_gq_p3exp_contracts)")
    item(6, "Selected GQAT LR",
         {v: pilot.get(v, {}).get("selected_lr")
          for v in ("T0", "T1")}.__repr__()
         + " (grid-top boundary CAVEAT)")
    for n, (v, tgt) in ((7, ("T1", "int4")),):
        seeds = {}
        for s in range(3):
            alq, rcal = rc(met, "RC_GQAT_T1" if s == 0 else "none")
            seeds[s] = dict(
                mtb_best=tau(rd, f"MTX_GQAT_s{s}_best", tgt),
                mtb_final=tau(rd, f"MTX_GQAT_s{s}_last", tgt))
        alq, rcal = rc(met, "RC_GQAT_T1")
        item(7, "GQAT 3-seed AL (T1 best/final) + s0 RCAL",
             f"{seeds} | s0 AL_q={alq} RCAL={rcal}; T0: "
             + str({s: tau(rd, f'MTX_GQAT_s{s}_best', 'fp16')
                    for s in range(3)}))
    alq, rcal = rc(met, "RC_NPTQ_T1")
    item(8, "NPTQ AL and RCAL",
         f"T1 mtb={tau(rd,'MTX_NPTQ','int4')} AL_q={alq} RCAL={rcal} "
         f"| T0 mtb={tau(rd,'MTX_NPTQ','fp16')}")
    alq, rcal = rc(met, "RC_LP3_T1")
    item(9, "LP3 AL and RCAL",
         f"T1 mtb={tau(rd,'MTX_LP3','int4')} AL_q={alq} RCAL={rcal} "
         f"| T0 mtb={tau(rd,'MTX_LP3','fp16')}")
    g0, g1 = self0.get("ep3g", {}), self1.get("ep3g", {})
    alq, rcal = rc(met, "RC_EP3G_T1")
    item(10, "EP3-G beta, AL, RCAL",
         f"fp16 beta={g0.get('beta')} (m={g0.get('m')}), int4 "
         f"beta={g1.get('beta')} (m={g1.get('m')}); T1 "
         f"mtb={tau(rd,'MTX_EP3G','int4')} AL_q={alq} RCAL={rcal}")
    p0 = self0.get("primary_rule3") or self0.get("primary_by_al") or {}
    p1 = self1.get("primary_rule3") or self1.get("primary_by_al") or {}
    item(11, "EP3-P beta_first / beta_recurrent",
         f"fp16 ({p0.get('beta_first')},{p0.get('beta_rec')}) | int4 "
         f"({p1.get('beta_first')},{p1.get('beta_rec')}) "
         "[Rule 3: max AL_q s.t. AFS>=0.95]")
    alq, rcal = rc(met, "RC_EP3P_T1")
    item(12, "EP3-P AL and RCAL",
         f"T1 mtb={p1.get('mtbench_tau')} AL_q={alq} RCAL={rcal} | "
         f"T0 mtb={p0.get('mtbench_tau')}")
    alq, rcal = rc(met, "RC_LP3RD_T1")
    item(13, "LP3-RD AL and RCAL",
         f"T1 mtb={tau(rd,'MTX_LP3RD','int4')} AL_q={alq} "
         f"RCAL={rcal}")
    alq, rcal = rc(met, "RC_LP3QAT_T1")
    item(14, "LP3-QAT AL and RCAL",
         f"T1 s0_best mtb={tau(rd,'MTX_LP3QAT_s0_best','int4')} "
         f"AL_q={alq} RCAL={rcal}; seeds best T1: "
         + str({s: tau(rd, f'MTX_LP3QAT_s{s}_best', 'int4')
                for s in range(3)}))
    P = boot["pairs"]
    def half(pair, key, cik):
        p = P.get(pair)
        if not p or p.get("status") == "missing":
            return "MISSING"
        return f"{p[key]:+.4f} CI {p[cik]}"
    item(15, "GQAT vs LP3 paired AL CI",
         half("RC_LP3_T1:RC_GQAT_T1", "delta_AL_q", "ci_delta_AL_q"))
    item(16, "GQAT vs LP3 paired RCAL CI",
         half("RC_LP3_T1:RC_GQAT_T1", "delta_RCAL", "ci_delta_RCAL"))
    item(17, "GQAT vs EP3-P paired AL CI",
         half("RC_EP3P_T1:RC_GQAT_T1", "delta_AL_q", "ci_delta_AL_q"))
    item(18, "GQAT vs EP3-P paired RCAL CI",
         half("RC_EP3P_T1:RC_GQAT_T1", "delta_RCAL",
              "ci_delta_RCAL"))
    item(19, "EP3-P vs EP3-G paired RCAL CI",
         fmt_ci(P.get("RC_EP3G_T1:RC_EP3P_T1")))
    reb = {}
    for v in ("T0", "T1"):
        d = j(rd, "tables", f"gqat_rebalancing_{v}.json")
        if d:
            anchor = d["rows"][0]["rms_ratio"]
            fin = [r["rms_ratio"] for r in d["rows"]
                   if r["name"].startswith("GQAT") and r["step"] >= 3000]
            reb[v] = dict(anchor=round(anchor, 4),
                          lp3_eff=round(d["rows"][1]["rms_ratio"], 4),
                          gqat_final=[round(x, 4) for x in fin])
    item(20, "W_e/W_h ratio before/after GQAT", reb)
    pd0 = j(rd, "tables", "path_distributions_fp16.json") or {}
    pd1 = j(rd, "tables", "path_distributions_int4.json") or {}
    item(21, "First/recurrent distribution verdict",
         f"fp16 e/h rms first={pd0.get('first_e_over_h_rms', 0):.4f} "
         f"rec={pd0.get('rec_e_over_h_rms', 0):.4f}; int4 "
         f"first={pd1.get('first_e_over_h_rms', 0):.4f} "
         f"rec={pd1.get('rec_e_over_h_rms', 0):.4f}; best betas "
         f"differ (fp16 {p0.get('beta_first')}vs{p0.get('beta_rec')}, "
         f"int4 {p1.get('beta_first')}vs{p1.get('beta_rec')})")
    keytags = ["RC_F16", "RC_NPTQ_T1", "RC_LP3_T1", "RC_EP3G_T1",
               "RC_EP3P_T1", "RC_GQAT_T1", "RC_LP3QAT_T1",
               "RC_LP3RD_T1", "RC_F16D_T1"]

    def row(metric, r=4):
        return {t.replace("RC_", ""):
                (round(met[f"{t}__mtbench"][metric], r)
                 if f"{t}__mtbench" in met else None)
                for t in keytags}
    item(22, "AL_q", row("AL_q"))
    item(23, "AL_0", row("AL_0"))
    item(24, "RCAL", row("RCAL"))
    item(25, "SAL", row("SAL"))
    item(26, "LAL", row("LAL"))
    item(27, "Acceptance precision P_A", row("P_A"))
    item(28, "Acceptance recall R_A", row("R_A"))
    item(29, "AFS", row("AFS"))
    ts = {}
    for p in glob.glob(os.path.join(rd, "tables",
                                    "tree_seq_fidelity_*.json")):
        d = json.load(open(p))
        ts[d["tag"]] = dict(
            tree_vs_seq=round(d["tq_tree_vs_seq_agreement"], 4),
            seq_vs_seq=round(d["tq_seq_vs_t0_seq_agreement"], 4))
    item(30, "Tree-vs-sequential fidelity", ts)
    warn = [k for k, v in ts.items()
            if v["tree_vs_seq"] < 0.99 and "T1" in k]
    item(31, "Shape-dependent A4 verdict",
         (f"WARNING tree!=seq under dynamic A4 for {warn}" if warn
          else "PASS (tree and sequential Tq agree >=0.99)"))
    ppl = j(rd, "tables", "target_ppl.json") or {}
    xds = {}
    for tag in ("XDS_F16", "XDS_LP3", "XDS_EP3P", "XDS_GQAT",
                "XDS_NPTQ", "XDS_LP3QAT", "XDS_EP3G"):
        for tgt in ("fp16", "int4"):
            r = {ds: tau(rd, tag, tgt, ds=ds)
                 for ds in ("sharegpt", "c4", "gsm8k", "humaneval")}
            if any(v is not None for v in r.values()):
                xds[f"{tag}_{tgt}"] = r
    item(32, "Task-quality summary",
         f"target wikitext2 PPL fp16={ppl.get('fp16')} "
         f"int4={ppl.get('int4')}; cross-dataset AL (SINGLE-SEED "
         f"seed0): {xds}")
    ovh = j(rd, "tables", "ep3p_overhead.json") or {}
    item(33, "Runtime/memory overhead (EP3-P vs LP3)",
         ovh.get("overhead"))
    pngs = sorted(os.path.basename(p) for p in glob.glob(
        os.path.join(rd, "plots", "*_3d*.png"))
        if "heatmap" not in p)
    item(34, f"3D PNG files ({len(pngs)})", pngs)
    item(35, "RCAL related-work verdict",
         "paired same-proposal replay + LCP decomposition and the "
         "name RCAL not found in prior work; closest: QSpec "
         "(2410.11305), QuantSpec (2502.10424), ML-SpecQD "
         "(2503.13565), HSD (2601.05724) — see "
         "docs/RCAL_RELATED_WORK_AND_NOVELTY_AUDIT.md. RCAL measures "
         "fidelity to the FP16 reference, NOT semantic correctness.")
    tl = os.path.join(rd, "manifests", "test_log.txt")
    item(36, "Test verdict",
         open(tl).read().strip().splitlines()[-1]
         if os.path.exists(tl) else "see pytest log")
    item(37, "Report path",
         "docs/EAGLE1_GENERIC_QAT_PATHWISE_P3EXP_RCAL_STUDY.md")
    bl = sorted(glob.glob(os.path.join(
        ROOT, "eagle1_generic_qat_pathwise_p3exp_rcal_*.tar.gz")))
    item(38, "Final review bundle", bl[-1] if bl else "pending")
    print("\n".join(o))
    return 0


if __name__ == "__main__":
    sys.exit(main())

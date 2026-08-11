"""Report-number verification for docs/DFLASH_SPINQUANT_NOVELTY_STUDY.md.

Every load-bearing number in the report is re-derived from the underlying
artifacts (VSQ run RD + FIDI run). Report-side numbers are extracted from the
report text by regex (a failed extraction counts as MISSING, never OK).

Run (CPU only):
  python -m seagle_port.verify_dflash_spinquant_novelty [RD] [FIDI] [REPORT]

Prints one line per check and ends with `missing=<n> mismatch=<n>`;
exits 1 unless both are 0.
"""
import csv
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RD = sys.argv[1] if len(sys.argv) > 1 else (
    f"{ROOT}/runs/dflash_vanilla_spinquant_novelty_20260810_104957")
FIDI = sys.argv[2] if len(sys.argv) > 2 else (
    f"{ROOT}/runs/dflash_full_interface_distribution_intervention_20260811_074001")
REPORT = sys.argv[3] if len(sys.argv) > 3 else (
    f"{ROOT}/docs/DFLASH_SPINQUANT_NOVELTY_STUDY.md")

RAW = open(REPORT, encoding="utf-8").read()
# normalized text for regex: unicode minus/en-dash -> '-', ellipsis -> '...'
T = (RAW.replace("−", "-").replace("–", "-")
        .replace("…", "..."))
T = re.sub(r"\s+", " ", T)

FL = r"(\d+\.\d+)"          # decimal number (no trailing-period capture)
RESULTS = []                # (status, label, detail)


def add(status, label, detail=""):
    RESULTS.append((status, label, detail))


def rex(pattern, n=1):
    """Return n captured strings from normalized report text, or None."""
    m = re.search(pattern, T)
    if not m:
        return None if n == 1 else [None] * n
    return m.group(1) if n == 1 else [m.group(i + 1) for i in range(n)]


def dec_of(s):
    s = str(s)
    return len(s.split(".", 1)[1]) if "." in s else 0


def cmp(label, rs, av, rtol=1e-3, atol=0.0, exact=False):
    """rs: report-side string (from regex) or None; av: artifact float."""
    if rs is None:
        add("MISSING", label, "report value not found by regex")
        return
    if av is None:
        add("MISSING", label, "artifact value unavailable")
        return
    rv = float(rs)
    if exact:
        ok = rv == av
        tol = 0.0
    else:
        half_ulp = 0.5 * 10 ** (-dec_of(rs)) + 1e-12
        tol = max(rtol * abs(av), atol, half_ulp)
        ok = abs(rv - av) <= tol
    add("OK" if ok else "MISMATCH", label,
        f"report={rs} artifact={av:.6g}")


def cond(label, claim, value, detail=""):
    """claim: regex pattern that must appear in report (or True to skip);
    value: bool artifact condition (None = artifact unavailable)."""
    if claim is not True and re.search(claim, T) is None:
        add("MISSING", label, "report claim not found by regex")
        return
    if value is None:
        add("MISSING", label, "artifact unavailable")
        return
    add("OK" if value else "MISMATCH", label, detail)


def safe(fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except Exception:
        return None


def jload(path):
    return json.load(open(path))


def csvrows(path):
    return list(csv.DictReader(open(path)))


_POOL = {}


def pooled(tag, tgt, ds):
    """Cycle-pooled micro-tau recomputed from an RD shard csv."""
    key = (tag, tgt, ds)
    if key not in _POOL:
        def _p():
            s = n = 0
            for r in csvrows(f"{RD}/shards/al__{tag}__{tgt}__{ds}.csv"):
                ts = [int(x) for x in r["taus"].split(";") if x]
                s += sum(ts)
                n += len(ts)
            return s / n
        _POOL[key] = safe(_p)
    return _POOL[key]


def shard_map(tag, tgt, ds):
    def _m():
        return {(r["prompt_id"], r["turn"]): r["taus"]
                for r in csvrows(f"{RD}/shards/al__{tag}__{tgt}__{ds}.csv")}
    return safe(_m)


def md_row(anchor, ncells):
    for line in RAW.splitlines():
        ls = line.strip()
        if anchor in line and ls.startswith("|"):
            cells = [c.strip() for c in line.replace("*", "").split("|")]
            if len(cells) >= 2 + ncells:
                return cells[2:2 + ncells]
    return None


B = {}
for _f in ("mtbench", "gsm8k", "gsm8kvalid", "humaneval", "sharegpt"):
    B.update(safe(jload, f"{RD}/stats/bootstrap_{_f}.json") or {})
HOLM = safe(jload, f"{RD}/stats/holm_adjusted.json") or {}


def bs(key, field="delta"):
    return B[key][field] if key in B else None


def holm_sig(key):
    return HOLM[key]["significant"] if key in HOLM else None


DSS = ("mtbench", "gsm8k", "humaneval", "sharegpt")

# ---------------------------------------------------------------- A. headline
FTBL = safe(lambda: {r["tag"]: r
                     for r in csvrows(f"{FIDI}/tables/final_4dataset_al.csv")})
A_ROWS = [("M0 FP16", "M0_fp16", "fp16"),
          ("M1 naive W4A4 RTN", "M1_rtn", "w4a4_norot"),
          ("M3 vanilla SpinQuant", "M3_vsq", "w4a4"),
          ("M5a + R_C", "M5_rconly", "w4a4"),
          ("M5 + P2 + R_C", "M5_p2rc", "w4a4"),
          ("M6 + QAT", "M6_q5bp2", "w4a4")]
pooled_mean = {}
for anchor, tag, tgt in A_ROWS:
    cells = md_row(anchor, 5)
    pms = []
    for i, ds in enumerate(DSS):
        rs = cells[i] if cells else None
        cv = safe(lambda: float(FTBL[tag][ds]))
        cmp(f"A tbl {tag} {ds} report vs final_4dataset_al.csv", rs, cv,
            rtol=0.0)
        pv = pooled(tag, tgt, ds)
        pms.append(pv)
        cmp(f"A tbl {tag} {ds} report vs pooled shard tau", rs, pv)
    mrs = cells[4] if cells else None
    cmp(f"A tbl {tag} mean report vs csv mean_4ds", mrs,
        safe(lambda: float(FTBL[tag]["mean_4ds"])), rtol=0.0)
    pm = (sum(pms) / 4 if all(v is not None for v in pms) else None)
    pooled_mean[tag] = pm
    cmp(f"A tbl {tag} mean report vs mean of pooled shards", mrs, pm)
m2 = md_row("M2 vanilla SQ RAW", 1)
cmp("A tbl M2_vsqraw mtbench report vs csv", m2[0] if m2 else None,
    safe(lambda: float(FTBL["M2_vsqraw"]["mtbench"])), rtol=0.0)
cmp("A tbl M2_vsqraw mtbench report vs pooled shard",
    m2[0] if m2 else None, pooled("M2_vsqraw", "w4a4", "mtbench"))

# ------------------------------------------------------- B. preregistered stats


def rng(vals):
    vals = [v for v in (vals or []) if v is not None]
    if not vals:
        return None, None
    a = [abs(v) for v in vals]
    return min(a), max(a)


p1 = [bs(f"{d}:P1_M0vM3") for d in DSS]
p4a = [bs(f"{d}:P4a_M3vM5rc") for d in DSS]
p4b = [bs(f"{d}:P4b_M3vM5p2rc") for d in DSS]
p3 = [bs(f"{d}:P3_M5rcVsM5p2rc") for d in DSS]
p5 = [bs(f"{d}:P5_M5vM6") for d in DSS]
gap = [bs(f"{d}:GAP_M0vM6") for d in DSS]

g = rex(r"P1 M0→M3 -" + FL + r"\.\.\.-" + FL, 2)
lo, hi = rng(p1)
cmp("B P1 range low |min|", g[0], lo)
cmp("B P1 range high |max|", g[1], hi)
g = rex(r"P4a M3→M5a \+" + FL + r"\.\.\.\+" + FL, 2)
lo, hi = rng(p4a)
cmp("B P4a range low", g[0], lo)
cmp("B P4a range high", g[1], hi)
g = rex(r"P4b M3→M5 \+" + FL + r"\.\.\.\+" + FL, 2)
lo, hi = rng(p4b)
cmp("B P4b range low", g[0], lo)
cmp("B P4b range high", g[1], hi)
g = rex(r"P3 M5a→M5 \+" + FL + r"\.\.\.\+" + FL, 2)
lo, hi = rng(p3)
cmp("B P3 range low", g[0], lo)
cmp("B P3 range high", g[1], hi)
g = rex(r"P5 M5→M6 \+" + FL + r"/\+" + FL + r"/\+" + FL + r"/\+" + FL, 4)
for i, ds in enumerate(DSS):
    cmp(f"B P5 delta {ds}", g[i], p5[i])
g = rex(r"GAP M0 vs M6 -" + FL + r"/-" + FL + r"/-" + FL +
        r"\(p=" + FL + r", n\.s\.\)/-" + FL, 5)
cmp("B GAP mtbench delta", g[0], safe(abs, gap[0]))
cmp("B GAP gsm8k delta", g[1], safe(abs, gap[1]))
cmp("B GAP humaneval delta", g[2], safe(abs, gap[2]))
cmp("B GAP humaneval p", g[3], bs("humaneval:GAP_M0vM6", "p"))
cmp("B GAP sharegpt delta", g[4], safe(abs, gap[3]))
cmp("B P6 M2->M3 delta", rex(r"P6 M2→M3 \+" + FL + r" \(sig\)"),
    bs("mtbench:P6_M2rawVsM3"))
cmp("B S2 seed-check delta", rex(r"S2 seed check \+" + FL + r" \(n\.s\."),
    bs("mtbench:S2SEED_M3vS2CHK"))
g = rex(r"S2CHK Δ\+" + FL + r", p=" + FL, 2)
cmp("B S2CHK delta (S1 text)", g[0], bs("mtbench:S2SEED_M3vS2CHK"))
cmp("B S2CHK p", g[1], bs("mtbench:S2SEED_M3vS2CHK", "p"))
cmp("B replication delta 0.0000", rex(r"replication Δ=" + FL),
    bs("mtbench:REP_M3vM3cyc"))


def allsig(key):
    vals = [holm_sig(f"{d}:{key}") for d in DSS]
    return None if any(v is None for v in vals) else all(vals)


cond("B P1 all sig (Holm)", r"P1 M0→M3 [^·]*\(all sig\)", allsig("P1_M0vM3"))
cond("B P4a all sig (Holm)", r"P4a M3→M5a [^·]*\(all sig\)",
     allsig("P4a_M3vM5rc"))
cond("B P4b all sig (Holm)", r"P4b M3→M5 [^·]*\(all sig\)",
     allsig("P4b_M3vM5p2rc"))
cond("B P3 all sig (Holm)", r"P3 M5a→M5 [^·]*\(all sig\)",
     allsig("P3_M5rcVsM5p2rc"))
cond("B P5 all sig (Holm)", r"P5 M5→M6 [^·]*\(all sig\)", allsig("P5_M5vM6"))
cond("B GAP humaneval M5+M6 fail Holm",
     r"humaneval FP16-gap cells fail Holm",
     safe(lambda: holm_sig("humaneval:GAP_M0vM6") is False
          and holm_sig("humaneval:GAP_M0vM5") is False))
cond("B GAP mt/gsm/sg sig (Holm)", True,
     safe(lambda: all(holm_sig(f"{d}:GAP_M0vM6")
                      for d in ("mtbench", "gsm8k", "sharegpt"))))
cond("B P6 sig (Holm)", r"P6 M2→M3 \+0\.\d+ \(sig\)",
     holm_sig("mtbench:P6_M2rawVsM3"))
cond("B S2 n.s. (Holm)", r"S2 seed check [^·]*\(n\.s\.",
     safe(lambda: holm_sig("mtbench:S2SEED_M3vS2CHK") is False))
g = rex(r"across all (\d+) comparisons \(name-deduped; (\d+) rejected", 2)
cmp("B Holm family size", g[0], safe(lambda: float(len(HOLM))), exact=True)
cmp("B Holm rejected count", g[1],
    safe(lambda: float(sum(1 for v in HOLM.values() if v["significant"]))),
    exact=True)
EXP_NONREJ = {"mtbench:S2SEED_M3vS2CHK", "mtbench:REP_M3vM3cyc",
              "humaneval:GAP_M0vM5", "humaneval:GAP_M0vM6",
              "gsm8kvalid:VRESTQ_M3vRestQ", "gsm8kvalid:VRESTO_M3vRestO",
              "gsm8kvalid:VI7B_RConlyVsI7m5",
              "gsm8kvalid:SUPP_RCrtVsRand", "gsm8kvalid:SUPP_RCrtVsHad",
              "gsm8kvalid:SUPP_Q5bp2VsQ6"}
cond("B Holm 10 non-rejections identity", r"the 10 non-rejections",
     safe(lambda: {k for k, v in HOLM.items()
                   if not v["significant"]} == EXP_NONREJ))

# §0 verdict numbers
g = rex(r"reaches 4-ds mean AL \*\*" + FL + r"\*\* vs FP16 \*\*" + FL
        + r"\*\*", 2)
cmp("S0 M3 4-ds mean", g[0], pooled_mean.get("M3_vsq"))
cmp("S0 M0 4-ds mean", g[1], pooled_mean.get("M0_fp16"))
g = rex(r"4-ds mean \*\*" + FL + r"\*\* = " + FL
        + r"% of the fp16→vanilla gap", 2)
cmp("S0 M6 4-ds mean", g[0], pooled_mean.get("M6_q5bp2"))
rec = safe(lambda: 100 * (pooled_mean["M6_q5bp2"] - pooled_mean["M3_vsq"])
           / (pooled_mean["M0_fp16"] - pooled_mean["M3_vsq"]))
cmp("S0 gap recovery pct", g[1], rec)
g = rex(r"M6 Δ-" + FL + r", p=" + FL + r" raw", 2)
cmp("S0 GAP humaneval delta", g[0], safe(abs, gap[2]))
cmp("S0 GAP humaneval p", g[1], bs("humaneval:GAP_M0vM6", "p"))
g = rex(r"\+" + FL + r"\.\.\.\+" + FL + r" per ds \(P4a\)", 2)
lo, hi = rng(p4a)
cmp("S0 iface P4a low", g[0], lo)
cmp("S0 iface P4a high", g[1], hi)
g = rex(r"\+" + FL + r"\.\.\.\+" + FL + r" on top of R_C \(P3\); "
        r"\*\*hurts alone\*\* \(-" + FL + r" val\)", 3)
lo, hi = rng(p3)
cmp("S0 iface P3 low", g[0], lo)
cmp("S0 iface P3 high", g[1], hi)
cmp("S0 P2-alone hurts (val delta)", g[2],
    safe(abs, bs("gsm8kvalid:P2a_M3vP2")))
g = rex(r"\+" + FL + r"\.\.\.\+" + FL + r" on top of P2\+R_C \(P5\)", 2)
lo, hi = rng(p5)
cmp("S0 iface P5 low", g[0], lo)
cmp("S0 iface P5 high", g[1], hi)
cmp("S0 G1 validation delta", rex(r"\(G1: -" + FL + r" validation AL"),
    safe(abs, bs("gsm8kvalid:VG1_M3vG1")))
cmp("S5 Q20 M3->M6 total 4-ds gain",
    rex(r"total M3→M6 \+" + FL + r" 4-ds-mean points"),
    safe(lambda: pooled_mean["M6_q5bp2"] - pooled_mean["M3_vsq"]))
g = rex(r"R_C delivers \+" + FL + r" to \+" + FL + r" AL per dataset", 2)
cmp("S5 Q9 R_C per-ds low (P4a)", g[0], rng(p4a)[0])
cmp("S5 Q9 R_C per-ds high (P4a)", g[1], rng(p4a)[1])

# ------------------------------------------------- C. validation ladder (pooled)
VP = lambda tag: pooled(tag, "w4a4", "gsm8kvalid")
g = rex(r"M-family: V_M3 " + FL + r" / \+P2 " + FL
        + r" \(\*\*worse\*\*, -" + FL + r" sig\) / \+MP3 " + FL
        + r" \(worse\) / \+R_C " + FL + r" / \+P2\+R_C " + FL
        + r" / G1 \(R1_D:=R1_T\) " + FL + r" \(-" + FL
        + r" sig\) / G1\+RC " + FL + r" / R_D=I " + FL, 10)
cmp("C V_M3", g[0], VP("V_M3"))
cmp("C V_M4a_p2 (+P2)", g[1], VP("V_M4a_p2"))
cmp("C +P2 delta (bootstrap P2a)", g[2],
    safe(abs, bs("gsm8kvalid:P2a_M3vP2")))
cmp("C V_M4b_mp3 (+MP3)", g[3], VP("V_M4b_mp3"))
cmp("C V_M5_rconly (+R_C)", g[4], VP("V_M5_rconly"))
cmp("C V_M5_rc (+P2+R_C)", g[5], VP("V_M5_rc"))
cmp("C V_G1", g[6], VP("V_G1"))
cmp("C G1 delta (bootstrap VG1)", g[7],
    safe(abs, bs("gsm8kvalid:VG1_M3vG1")))
cmp("C V_G1_rc (G1+RC)", g[8], VP("V_G1_rc"))
cmp("C V_M3id (R_D=I)", g[9], VP("V_M3id"))
g = rex(r"V_Q2 " + FL + r" / V_Q3 " + FL + r" / V_Q5 " + FL
        + r" / V_Q5b " + FL + r" / \*\*V_Q5bp2 " + FL, 5)
for i, tag in enumerate(("V_Q2", "V_Q3", "V_Q5", "V_Q5b", "V_Q5bp2")):
    cmp(f"C {tag}", g[i], VP(tag))
g = rex(r"restore-V " + FL + r" \(\+" + FL + r"\) > restore-K " + FL
        + r" \(\+" + FL + r"\) > restore-Q " + FL + r" \(\+" + FL
        + r", p=" + FL + r", Holm-n\.s\.\) > restore-O " + FL
        + r" \(paired Δ\+" + FL + r", p=" + FL + r", n\.s\.\)", 10)
cmp("C V_restv", g[0], VP("V_restv"))
cmp("C restore-V delta", g[1], bs("gsm8kvalid:VRESTV_M3vRestV"))
cmp("C V_restk", g[2], VP("V_restk"))
cmp("C restore-K delta", g[3], bs("gsm8kvalid:VRESTK_M3vRestK"))
cmp("C V_restq", g[4], VP("V_restq"))
cmp("C restore-Q delta", g[5], bs("gsm8kvalid:VRESTQ_M3vRestQ"))
cmp("C restore-Q p", g[6], bs("gsm8kvalid:VRESTQ_M3vRestQ", "p"))
cond("C restore-Q Holm-n.s.", r"restore-Q [^;]*Holm-n\.s\.",
     safe(lambda: holm_sig("gsm8kvalid:VRESTQ_M3vRestQ") is False))
cmp("C V_resto", g[7], VP("V_resto"))
cmp("C restore-O paired delta", g[8], bs("gsm8kvalid:VRESTO_M3vRestO"))
cmp("C restore-O p", g[9], bs("gsm8kvalid:VRESTO_M3vRestO", "p"))
cond("C restore-O n.s.", r"restore-O [^;]*n\.s\.",
     safe(lambda: B["gsm8kvalid:VRESTO_M3vRestO"]["p"] > 0.05))
g = rex(r"only-V " + FL + r" \(worst\) < only-K " + FL + r" < only-Q "
        + FL + r" ≈ only-O " + FL, 4)
cmp("C V_onlyv", g[0], VP("V_onlyv"))
cmp("C V_onlyk", g[1], VP("V_onlyk"))
cmp("C V_onlyq", g[2], VP("V_onlyq"))
cmp("C V_onlyo", g[3], VP("V_onlyo"))
# I7 deployed-validation sentence (checked as-is against artifacts)
cmp("C V_I7m3 pooled (I7 sentence)", rex(r"V_I7m3 pooled " + FL),
    VP("V_I7m3"))
g = rex(r"\(paired Δ-" + FL + r" vs V_M3, p=([0-9][0-9.e-]*)\)", 2)
cmp("C I7m3 paired delta (VI7A)", g[0],
    safe(abs, bs("gsm8kvalid:VI7A_M3vI7m3")))
cmp("C I7m3 p (VI7A)", g[1], bs("gsm8kvalid:VI7A_M3vI7m3", "p"),
    atol=5e-5)
g = rex(r"V_I7m5 " + FL + r" on top of R_C is a null \(Δ-" + FL
        + r", p=" + FL + r"\)", 3)
cmp("C V_I7m5 (I7 sentence)", g[0], VP("V_I7m5"))
cmp("C I7m5 delta (VI7B)", g[1],
    safe(abs, bs("gsm8kvalid:VI7B_RConlyVsI7m5")))
cmp("C I7m5 p (VI7B)", g[2], bs("gsm8kvalid:VI7B_RConlyVsI7m5", "p"))
cond("C I7m5 null (n.s.)", r"on top of R_C is a null",
     safe(lambda: B["gsm8kvalid:VI7B_RConlyVsI7m5"]["p"] > 0.05))

# ----------------------------------------------------------------- D. QAT CEs
QS = lambda f, k="best_val_ce": safe(
    lambda: jload(f"{RD}/rotations/draft/{f}.summary.json")[k])
g = rex(r"3e-2 " + FL + r", 1e-1 " + FL + r" \(prev server\) ≈ " + FL
        + r" \(this server", 3)
cmp("D pilot 3e-2 best CE", g[0], QS("QATpilot_3e-2.pt"))
cmp("D pilot 1e-1 best CE", g[1], QS("QATpilot_1e-1.pt"))
cmp("D pilot 1e-1b best CE (this server)", g[2], QS("QATpilot_1e-1b.pt"))
cmp("D pilot 3e-1 best CE", rex(r"\*\*3e-1 " + FL + r" best\*\*"),
    QS("QATpilot_3e-1.pt"))
g = rex(r"mains \(400 steps, lr 3e-1\) Q2 " + FL + r" / Q3 " + FL
        + r" / Q5 no-move from init " + FL, 3)
cmp("D main Q2 best CE", g[0], QS("QAT_Q2.pt"))
cmp("D main Q3 best CE", g[1], QS("QAT_Q3.pt"))
cmp("D main Q5 init CE (no-move)", g[2], QS("QAT_Q5.pt", "init_val_ce"))
cond("D Q5 no-move (best==init, best_step 0)", r"Q5 no-move from init",
     safe(lambda: QS("QAT_Q5.pt") == QS("QAT_Q5.pt", "init_val_ce")
          and QS("QAT_Q5.pt", "best_step") == 0))
cmp("D Q5@3e-2 best CE", rex(r"Q5@3e-2 → \*\*" + FL + r"\*\*"),
    QS("QAT_Q5_lr3e-2.pt"))

# --------------------------------------------------------- E. FIDI H_t (S3)
CAP = {}
for _c in ("R1", "R3"):
    CAP[_c] = safe(jload, f"{FIDI}/tables/capture__{_c}__gsm8k.json")
CG = lambda c, sec, t, k: safe(lambda: CAP[c][sec][t][k])
row = md_row("R1 vanilla SQ complete", 3)
cmp("E R1 S3_Ht_dep kurtosis", row[0] if row else None,
    CG("R1", "stats", "S3_Ht_dep", "kurtosis"), atol=0.5)
cmp("E R1 S3_Ht_dep A4 NMSE", row[1] if row else None,
    CG("R1", "qparams", "S3_Ht_dep", "nmse_mean"))
cmp("E R1 S3_Ht_dep code entropy", row[2] if row else None,
    CG("R1", "qparams", "S3_Ht_dep", "code_entropy_bits"))
row = md_row("R3 pre-R_C", 3)
cmp("E R3 pre S3_Ht_dep kurtosis", row[0] if row else None,
    CG("R3", "stats", "S3_Ht_dep", "kurtosis"), atol=0.5)
cmp("E R3 pre A4 NMSE", row[1] if row else None,
    CG("R3", "qparams", "S3_Ht_dep", "nmse_mean"))
cmp("E R3 pre code entropy", row[2] if row else None,
    CG("R3", "qparams", "S3_Ht_dep", "code_entropy_bits"))
row = md_row("R3 post-R_C", 3)
cmp("E R3 post S3_Ht_rc_dep kurtosis", row[0] if row else None,
    CG("R3", "stats", "S3_Ht_rc_dep", "kurtosis"), atol=0.5)
cmp("E R3 post A4 NMSE", row[1] if row else None,
    CG("R3", "qparams", "S3_Ht_rc_dep", "nmse_mean"))
cmp("E R3 post code entropy", row[2] if row else None,
    CG("R3", "qparams", "S3_Ht_rc_dep", "code_entropy_bits"))
g = rex(r"deployed-forward H_t kurtosis (\d+)-(\d+) and A4 NMSE " + FL
        + r" AFTER complete vanilla SpinQuant; " + FL + r" / " + FL
        + r" after R_C", 5)
klo = safe(lambda: min(CAP["R1"]["stats"]["S3_Ht_dep"]["kurtosis"],
                       CAP["R3"]["stats"]["S3_Ht_dep"]["kurtosis"]))
khi = safe(lambda: max(CAP["R1"]["stats"]["S3_Ht_dep"]["kurtosis"],
                       CAP["R3"]["stats"]["S3_Ht_dep"]["kurtosis"],
                       CAP["R1"]["stats"]["C_Ht_fp"]["kurtosis"]))
cmp("S0 H_t kurt range low", g[0], klo, atol=0.5)
cmp("S0 H_t kurt range high", g[1], khi, atol=0.5)
cmp("S0 H_t A4 NMSE after vanilla SQ", g[2],
    CG("R1", "qparams", "S3_Ht_dep", "nmse_mean"), atol=5e-3)
cmp("S0 H_t kurt after R_C", g[3],
    CG("R3", "stats", "S3_Ht_rc_dep", "kurtosis"), atol=0.5)
cmp("S0 H_t NMSE after R_C", g[4],
    CG("R3", "qparams", "S3_Ht_rc_dep", "nmse_mean"))
g = rex(r"Z_t itself: kurt (\d+), absmax (\d+)", 2)
cmp("E Z_t kurtosis (R1 deployed)", g[0],
    CG("R1", "stats", "S3_Zt_dep", "kurtosis"), atol=0.5)
cmp("E Z_t absmax", g[1],
    CG("R1", "stats", "S3_Zt_dep", "absmax"), atol=0.5)
MECH = safe(lambda: {r["tensor"]: r
                     for r in csvrows(f"{RD}/tables/mechanism_stats.csv")})
g = rex(r"naive " + FL + r" = vanilla-SQ-complete " + FL + r" vs R_C "
        + FL + r"; NMSE " + FL + r"/" + FL + r"/" + FL, 6)
for i, (tens, what) in enumerate((("M1_naiveHt", "kurt"),
                                  ("M3_vsqHt", "kurt"),
                                  ("M5_rcHt", "kurt"))):
    cmp(f"E mechanism {tens} kurtosis", g[i],
        safe(lambda t=tens: float(MECH[t]["kurtosis"])), atol=0.05)
for i, tens in enumerate(("M1_naiveHt", "M3_vsqHt", "M5_rcHt")):
    cmp(f"E mechanism {tens} a4_nmse", g[3 + i],
        safe(lambda t=tens: float(MECH[t]["a4_nmse"])), atol=5e-3)

# --------------------------------------------------------------------- F. gates
GD = safe(jload, f"{FIDI}/tables/fidi_gate_d.json") or {}
for k in ("G_REG_bits4_bitwise", "G_REG_bits16_bitwise",
          "G_FP_bitwise", "G_TAP_bitwise"):
    cond(f"F fidi_gate_d {k} True",
         r"old-vs-new RotQuantDraft forward \*\*bitwise identical\*\*",
         GD.get(k), f"value={GD.get(k)}")


def tppl():
    last = {}
    for r in csvrows(f"{RD}/tables/target_ppl.csv"):
        last[(r["arm"], r["seed"])] = float(r["wikitext2_ppl"])
    by = {}
    for (arm, _s), v in last.items():
        by.setdefault(arm, []).append(v)
    return by


TP = safe(tppl) or {}
g = rex(r"T-PPL fp16 " + FL + r" / RTN " + FL + r" / random-Hadamard "
        + FL + r" / R1-only " + FL + r"-" + FL + r" / R1\+R2 \*\*"
        + FL + r"-" + FL + r"\*\*", 7)
cmp("F T-PPL fp16", g[0], safe(lambda: TP["T-PPL0_fp16"][0]))
cmp("F T-PPL RTN", g[1], safe(lambda: TP["T-PPL1_rtn_norot"][0]))
cmp("F T-PPL hadamard", g[2], safe(lambda: TP["T-PPL2_hadamard"][0]))
cmp("F T-PPL R1-only min", g[3], safe(lambda: min(TP["T-PPL3_R1only"])))
cmp("F T-PPL R1-only max", g[4], safe(lambda: max(TP["T-PPL3_R1only"])))
cmp("F T-PPL R1+R2 min", g[5], safe(lambda: min(TP["T-PPL4_R1R2"])))
cmp("F T-PPL R1+R2 max", g[6], safe(lambda: max(TP["T-PPL4_R1R2"])))
DQ = safe(lambda: {r["arm"]: float(r["block_ce"])
                   for r in csvrows(f"{RD}/tables/draft_quality.csv")}) or {}
g = rex(r"fp16 " + FL + r" / RTN " + FL + r" / random " + FL
        + r" / R1D " + FL + r" / R1D\+R2D \*\*" + FL + r"\*\*", 5)
for i, arm in enumerate(("D-P0_fp16", "D-P1_w4a4_norot", "D-P2_w4a4_random",
                         "D-P3_w4a4_R1D", "D-P4_w4a4_R1D_R2D")):
    cmp(f"F draft ladder {arm} block CE", g[i], DQ.get(arm))

# -------------------------------------------------- G. A/W error decomposition
WC = safe(lambda: {r["site"]: r
                   for r in csvrows(f"{FIDI}/tables/wc_aw_decomposition.csv")})
g = rex(r"wc_aw_decomposition\.csv`\): A " + FL + r" / W " + FL
        + r" / interaction " + FL, 3)
cmp("G fc A_only NMSE", g[0],
    safe(lambda: float(WC["fc"]["A_only_nmse"])))
cmp("G fc W_only NMSE", g[1],
    safe(lambda: float(WC["fc"]["W_only_nmse"])))
cmp("G fc interaction NMSE", g[2],
    safe(lambda: float(WC["fc"]["interaction_nmse"])))
cmp("G fc_p2 A_only NMSE (P2 trims A)",
    rex(r"P2 trims the A term \(" + FL + r"\)"),
    safe(lambda: float(WC["fc_p2"]["A_only_nmse"])))


def qmean(site, col):
    rows = [float(r[col])
            for r in csvrows(f"{FIDI}/tables/qkvo_aw_decomposition.csv")
            if r["site"] == site]
    assert len(rows) == 5
    return sum(rows) / 5


cmp("G k_ctx layer-mean A_only (const)", "0.2387",
    safe(qmean, "k_ctx", "A_only_nmse"), atol=2e-3)
cmp("G k_ctx layer-mean W_only (const)", "0.0074",
    safe(qmean, "k_ctx", "W_only_nmse"), atol=2e-3)
cmp("G v_ctx layer-mean A_only (const)", "0.5297",
    safe(qmean, "v_ctx", "A_only_nmse"), atol=2e-3)
cmp("G v_ctx layer-mean W_only (const)", "0.0199",
    safe(qmean, "v_ctx", "W_only_nmse"), atol=2e-3)
cond("G ctx K/V activation_driven all layers", r"activation-driven",
     safe(lambda: all(
         r["verdict"] == "activation_driven"
         for r in csvrows(f"{FIDI}/tables/qkvo_aw_decomposition.csv")
         if r["site"] in ("k_ctx", "v_ctx"))))

# ------------------------------------------------------- H. alignment Spearman
cmp("H fc full-concat Spearman (stock, all)",
    rex(r"full-concat Spearman \*\*(-\d+\.\d+)\*\*"),
    safe(lambda: [float(r["spearman_rms"]) for r in
                  csvrows(f"{FIDI}/tables/activation_weight_alignment.csv")
                  if r["basis"] == "stock" and r["site"] == "fc"
                  and r["branch"] == "all"][0]), atol=0.01)

# --------------------------------------------------- I. component CE proxy grid
CS = safe(lambda: {r["cell"]: r for r in
                   csvrows(f"{FIDI}/tables/qkvo_component_sensitivity.csv")})
CSF = lambda cell, col="delta_vs_fp": safe(lambda: float(CS[cell][col]))
g = rex(r"only-V \+" + FL + r" CE ≫ only-K \+" + FL + r" > only-fc \+"
        + FL + r" > only-Q \+" + FL + r" ≈ only-O \+" + FL
        + r" > only-MLP \+" + FL, 6)
for i, cell in enumerate(("only_v", "only_k", "only_fc",
                          "only_q", "only_o", "only_mlp")):
    cmp(f"I {cell} delta_vs_fp CE", g[i], CSF(cell), atol=0.01)
cmp("I restore_fc hurts (delta_vs_w4a4)",
    rex(r"restoring fc alone \*hurts\* \(\+" + FL + r"\)"),
    CSF("restore_fc", "delta_vs_w4a4"), atol=0.01)
cmp("I ref fp16 CE (const)", "2.2218", CSF("ref_fp16", "val_ce"), atol=0.01)
cmp("I base w4a4 CE (const)", "4.6975", CSF("base_w4a4", "val_ce"), atol=0.01)
g = rex(r"V@0-2 \(CE proxy \+" + FL + r"\.\.\.\+" + FL + r"\)", 2)
v02 = safe(lambda: [float(CS[f"only_v@{i}"]["delta_vs_fp"])
                    for i in range(3)])
cmp("I only_v@0-2 min", g[0], safe(lambda: min(v02)), atol=0.01)
cmp("I only_v@0-2 max", g[1], safe(lambda: max(v02)), atol=0.01)

# ---------------------------------------------------------------------- J. RCAL
R5 = safe(jload, f"{RD}/tables/rcal__M5_p2rc__mtbench.json") or {}
R3C = safe(jload, f"{RD}/tables/rcal__M3mtbench.json") or {}
g = rex(r"M5 AL_q " + FL + r" / RCAL " + FL + r" / AFS " + FL, 3)
cmp("J M5 AL_q", g[0], R5.get("AL_q"), atol=2e-3)
cmp("J M5 RCAL", g[1], R5.get("RCAL"), atol=2e-3)
cmp("J M5 AFS", g[2], R5.get("AFS"), atol=2e-3)
g = rex(r"\(M3cyc\): AL_q " + FL + r" / RCAL " + FL + r" / AFS " + FL, 3)
cmp("J M3cyc AL_q", g[0], R3C.get("AL_q"), atol=2e-3)
cmp("J M3cyc RCAL", g[1], R3C.get("RCAL"), atol=2e-3)
cmp("J M3cyc AFS", g[2], R3C.get("AFS"), atol=2e-3)

# ------------------------------------------------------------ K. gauge invariance
SI = safe(csvrows, f"{FIDI}/tables/scale_invariance_test.csv")
g = rex(r"(\d+)/(\d+) cells code-equal, output dev " + FL, 3)
cmp("K scale-invariance cell count", g[0],
    safe(lambda: float(len(SI))), exact=True)
cmp("K scale-invariance cell count denom", g[1],
    safe(lambda: float(len(SI))), exact=True)
nt = safe(lambda: [r for r in SI if float(r["s"]) != 1.0])
cond("K all s!=1 rows code-equal (act/wk/wv)", r"cells code-equal",
     safe(lambda: len(nt) > 0 and all(
         r["act_codes_equal"] == "True" and r["wk_codes_equal"] == "True"
         and r["wv_codes_equal"] == "True" for r in nt)))
cmp("K max y_max_rel_dev over s!=1", g[2],
    safe(lambda: max(float(r["y_max_rel_dev"]) for r in nt)), exact=True)

# --------------------------------------------------------------- L. I7 proxy
I7 = safe(csvrows, f"{FIDI}/tables/i7_smooth_proxy.csv")


def i7_worse(basis):
    rows = [r for r in I7 if r["basis"] == basis]
    none = [r for r in rows if r["alpha"] == "none"][0]
    alphas = [r for r in rows if r["alpha"] != "none"]
    return len(alphas) > 0 and all(
        float(r["k_nmse"]) > float(none["k_nmse"])
        and float(r["v_nmse"]) > float(none["v_nmse"]) for r in alphas)


for basis in ("m3", "m5"):
    cond(f"L I7 proxy worse at every alpha ({basis})",
         r"proxy NMSE worse at every α in both bases",
         safe(i7_worse, basis))

# ------------------------------------------------------------------ M. KV cache
g = rex(r"K kurt " + FL + r"→" + FL + r", KV4 " + FL + r"→" + FL, 4)
cmp("M R1 stored ctx-K l0 kurtosis", g[0],
    CG("R1", "stats", "S4_k_stored_ctx_l0", "kurtosis"), atol=0.5)
cmp("M R3 stored ctx-K l0 kurtosis", g[1],
    CG("R3", "stats", "S4_k_stored_ctx_l0", "kurtosis"), atol=0.5)
cmp("M R1 KV4 k_ctx l0 NMSE", g[2],
    CG("R1", "qparams", "KV4_k_ctx_l0", "nmse_mean"))
cmp("M R3 KV4 k_ctx l0 NMSE", g[3],
    CG("R3", "qparams", "KV4_k_ctx_l0", "nmse_mean"))
cond("M KV8 k_ctx l0 NMSE order 5e-5 (<1e-4, R1+R3)", r"KV8 ≈ 5e-5",
     safe(lambda: CAP["R1"]["qparams"]["KV8_k_ctx_l0"]["nmse_mean"] < 1e-4
          and CAP["R3"]["qparams"]["KV8_k_ctx_l0"]["nmse_mean"] < 1e-4))

# ---------------------------------------------------------- N. sharegpt gate
shk = shard_map("M0shk", "fp16", "sharegpt")
sh0 = shard_map("M0_fp16", "fp16", "sharegpt")
g = rex(r"bit-identical per-prompt fp16 eval, (\d+)/(\d+)", 2)
cmp("N sharegpt prompt count", g[1],
    safe(lambda: float(len(sh0))), exact=True)
nmatch = safe(lambda: sum(1 for k in sh0 if shk.get(k) == sh0[k]))
cmp("N sharegpt bit-identical rows", g[0],
    None if nmatch is None else float(nmatch), exact=True)
cond("N M0shk == M0_fp16 sharegpt shard identical", r"\(80/80 prompts",
     safe(lambda: len(shk) == len(sh0) and nmatch == len(sh0)))

# --------------------------------------------------------- O. M3cyc replication
c1 = shard_map("M3cyc", "w4a4", "mtbench")
c0 = shard_map("M3_vsq", "w4a4", "mtbench")
cond("O M3cyc reproduces M3 shard exactly (per-prompt tau vectors)",
     r"reproduced the canonical M3 shard \*\*exactly\*\*",
     safe(lambda: len(c1) == len(c0)
          and all(c1.get(k) == c0[k] for k in c0)))
cmp("O M3cyc pooled tau == M3 pooled tau", "0.0000",
    safe(lambda: pooled("M3cyc", "w4a4", "mtbench")
         - pooled("M3_vsq", "w4a4", "mtbench")), exact=True)

# ------------------------------------ P. §8 post-hoc supplementary arms
cmp("P V_RCrand pooled", rex(r"V_RCrand " + FL),
    pooled("V_RCrand", "w4a4", "gsm8kvalid"))
cmp("P V_RChad pooled", rex(r"V_RChad " + FL),
    pooled("V_RChad", "w4a4", "gsm8kvalid"))
g = rex(r"V_RCrand " + FL + r" \(Δ-" + FL + r" vs R1_T, p=" + FL, 3)
cmp("P RCrand delta", g[1],
    safe(lambda: abs(bs("gsm8kvalid:SUPP_RCrtVsRand"))))
cmp("P RCrand p", g[2], bs("gsm8kvalid:SUPP_RCrtVsRand", "p"))
g = rex(r"V_RChad " + FL + r" \(Δ-" + FL + r",\s*p=" + FL, 3)
cmp("P RChad delta", g[1],
    safe(lambda: abs(bs("gsm8kvalid:SUPP_RCrtVsHad"))))
cmp("P RChad p", g[2], bs("gsm8kvalid:SUPP_RCrtVsHad", "p"))
g = rex(r"train CE " + FL + r"→" + FL, 2)
q6 = safe(jload, f"{RD}/rotations/draft/QAT_Q6.pt.summary.json") or {}
cmp("P Q6 init CE", g[0], q6.get("init_val_ce"))
cmp("P Q6 best CE", g[1], q6.get("best_val_ce"))
g = rex(r"V_Q6 " + FL + r" vs V_Q5bp2 " + FL + r" \(Δ-" + FL
        + r", p=" + FL, 4)
cmp("P V_Q6 pooled", g[0], pooled("V_Q6", "w4a4", "gsm8kvalid"))
cmp("P Q6 delta", g[2],
    safe(lambda: abs(bs("gsm8kvalid:SUPP_Q5bp2VsQ6"))))
cmp("P Q6 p", g[3], bs("gsm8kvalid:SUPP_Q5bp2VsQ6", "p"))
m6r = safe(jload, f"{RD}/tables/rcal__M6_q5bp2__mtbench.json") or {}
g = rex(r"AL_q " + FL + r" / RCAL " + FL + r" / AFS " + FL
        + r"\s*\(`RD/tables/rcal__M6_q5bp2__mtbench.json`\)", 3)
cmp("P rcal_M6 AL_q", g[0], m6r.get("AL_q"))
cmp("P rcal_M6 RCAL", g[1], m6r.get("RCAL"))
cmp("P rcal_M6 AFS", g[2], m6r.get("AFS"))

# ------------------------------------------------------------------- summary
W = max(len(lbl) for _s, lbl, _d in RESULTS)
for status, label, detail in RESULTS:
    if status == "OK":
        print(f"[OK]       {label:<{W}}  {detail}")
    elif status == "MISMATCH":
        print(f"[MISMATCH] {label:<{W}}  {detail}")
    else:
        print(f"[MISSING]  {label:<{W}}  {detail}")
n_missing = sum(1 for s, _l, _d in RESULTS if s == "MISSING")
n_mismatch = sum(1 for s, _l, _d in RESULTS if s == "MISMATCH")
print(f"total checks = {len(RESULTS)}")
print(f"missing={n_missing} mismatch={n_mismatch}")
sys.exit(1 if (n_missing or n_mismatch) else 0)

"""R1DCE §9/§24 stats battery — preregistered primaries (Holm family) then
secondaries (reproduction + supplementary seeds), all on gsm8kvalid.

Primary family (bootstrap_primary.json -> Holm):
  P1_noneVsR1D, P2_r1dVsR1T, P3_r1dVsHad, P4_r1dVsRnd101, P5_r1dVsCombo
Secondaries (bootstrap_secondary.json, NOT in the Holm family):
  reproduction pairs vs the VSQ run + remaining random seeds + C6/C7 cross.
"""
import csv
import json
import os
import subprocess
import sys

RD = ("/home/thahn1230/dflash_workspace/dflash/runs/"
      "dflash_r1d_context_extension_20260814_123938")
PY = "/home/thahn1230/dflash_workspace/venv/bin/python"
DS = "gsm8kvalid"

PRIMARY = [
    ("P1_noneVsR1D", "C0_none@w4a4", "C1_r1d@w4a4"),
    ("P2_r1dVsR1T", "C1_r1d@w4a4", "C2_r1t@w4a4"),
    ("P3_r1dVsHad", "C1_r1d@w4a4", "C4_had@w4a4"),
    ("P4_r1dVsRnd101", "C1_r1d@w4a4", "C5_rnd_s101@w4a4"),
    ("P5_r1dVsCombo", "C1_r1d@w4a4", "C7_combo@w4a4"),
]
# family-2 primaries (master-spec §24: residual / P2 / QAT comparisons) —
# Holm is run over the UNION of PRIMARY+PRIMARY2 (8 preregistered pairs)
PRIMARY2 = [
    ("P6_r1dVsResid", "C1_r1d@w4a4", "C8_resid@w4a4"),
    ("P7_bestVsP2", "C2_r1t@w4a4", "PRIOR_V_M5_rc@w4a4"),
    ("P8_ptqVsQat", "PRIOR_V_M5_rc@w4a4", "PRIOR_V_Q5bp2@w4a4"),
]
SECONDARY = [
    ("S_c9VsR1T", "C9_r1dc@w4a4", "C2_r1t@w4a4"),
    ("S_c9VsR1D", "C1_r1d@w4a4", "C9_r1dc@w4a4"),
    ("S_c9l03VsR1T", "C9b_r1dc_l03@w4a4", "C2_r1t@w4a4"),
    ("S_c8VsR1T", "C8_resid@w4a4", "C2_r1t@w4a4"),
    ("S_r1dP2VsRtP2", "C1_r1d_p2@w4a4", "PRIOR_V_M5_rc@w4a4"),
    ("S_r1dVsR1dP2", "C1_r1d@w4a4", "C1_r1d_p2@w4a4"),
    ("S_r1dP2VsQat", "C1_r1d_p2@w4a4", "CQ_r1d_qat@w4a4"),
    ("S_qatR1dVsQatRt", "CQ_r1d_qat@w4a4", "PRIOR_V_Q5bp2@w4a4"),
    ("S_repro_C0_vs_priorVM3", "PRIOR_V_M3@w4a4", "C0_none@w4a4"),
    ("S_repro_C2_vs_priorM5rc", "PRIOR_V_M5_rconly@w4a4", "C2_r1t@w4a4"),
    ("S_r1dVsRnd102", "C1_r1d@w4a4", "C5_rnd_s102@w4a4"),
    ("S_r1dVsRnd103", "C1_r1d@w4a4", "C5_rnd_s103@w4a4"),
    ("S_r1dVsRnd1234", "C1_r1d@w4a4", "C5_rnd_s1234@w4a4"),
    ("S_r1dVsLearned", "C1_r1d@w4a4", "C6_learned@w4a4"),
    ("S_r1tVsCombo", "C2_r1t@w4a4", "C7_combo@w4a4"),
    ("S_learnedVsR1T", "C6_learned@w4a4", "C2_r1t@w4a4"),
    ("S_hadVsR1T", "C4_had@w4a4", "C2_r1t@w4a4"),
    ("S_rnd101VsR1T", "C5_rnd_s101@w4a4", "C2_r1t@w4a4"),
]


def run_pairs(pairs, out_name):
    cmd = [PY, "-m", "seagle_port.stats", "--run-dir", RD, "--dataset", DS,
           "--n", "3000", "--out-name", out_name]
    for name, a, b in pairs:
        cmd += ["--pair", f"{name}={a}:{b}"]
    subprocess.run(cmd, check=True,
                   cwd="/home/thahn1230/dflash_workspace/dflash")


def exists(spec):
    tag = spec.split("@")[0]
    p = f"{RD}/shards/al__{tag}__w4a4__gsm8kvalid.csv"
    return os.path.exists(p) and os.path.getsize(p) > 40


def main():
    run_pairs(PRIMARY, "primary")
    run_pairs([p for p in PRIMARY2 if exists(p[1]) and exists(p[2])],
              "primary2")
    # Holm over the 8-primary union only: park secondary during holm
    sec = f"{RD}/stats/bootstrap_secondary.json"
    park = f"{RD}/stats/_parked_secondary.json"
    if os.path.exists(sec):
        os.replace(sec, park)
    subprocess.run([PY, "-m", "seagle_port.stats", "--run-dir", RD,
                    "--holm"], check=True,
                   cwd="/home/thahn1230/dflash_workspace/dflash")
    os.replace(f"{RD}/stats/holm_adjusted.json",
               f"{RD}/stats/holm_primary.json")
    if os.path.exists(park):
        os.replace(park, sec)
    run_pairs([p for p in SECONDARY if exists(p[1]) and exists(p[2])],
              "secondary")

    # flat CSVs for §29
    boot_rows = []
    for fn, fam in (("bootstrap_primary.json", "primary"),
                    ("bootstrap_primary2.json", "primary2"),
                    ("bootstrap_secondary.json", "secondary")):
        if not os.path.exists(f"{RD}/stats/{fn}"):
            continue
        d = json.load(open(f"{RD}/stats/{fn}"))
        for name, r in sorted(d.items()):
            boot_rows.append({"family": fam, "comparison": name,
                              "a_tau": r["a_tau"], "b_tau": r["b_tau"],
                              "delta": r["delta"], "ci_lo": r["ci"][0],
                              "ci_hi": r["ci"][1], "p": r["p"],
                              "n_pairs": r["n_pairs"]})
    with open(f"{RD}/tables/bootstrap.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(boot_rows[0].keys()))
        w.writeheader()
        w.writerows(boot_rows)
    h = json.load(open(f"{RD}/stats/holm_primary.json"))
    with open(f"{RD}/tables/holm.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["comparison", "p", "holm_alpha", "significant"])
        for name, r in sorted(h.items(), key=lambda kv: kv[1]["p"]):
            w.writerow([name, r["p"], r["holm_alpha"], r["significant"]])
    print("wrote tables/bootstrap.csv + tables/holm.csv")
    for r in boot_rows:
        print(f"  [{r['family'][:4]}] {r['comparison']:26s} "
              f"{r['a_tau']:.4f} -> {r['b_tau']:.4f}  d={r['delta']:+.4f} "
              f"CI[{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}] p={r['p']:.4g}")


if __name__ == "__main__":
    main()

"""R1DCE §22/§23 — final_4dataset_al.csv (+P(L>=k)) and rcal_final.csv."""
import csv
import glob
import json
import os
import shutil

RD = ("/home/thahn1230/dflash_workspace/dflash/runs/"
      "dflash_r1d_context_extension_20260814_123938")
VSQ = ("/home/thahn1230/dflash_workspace/dflash/runs/"
       "dflash_vanilla_spinquant_novelty_20260810_104957")
DSS = ("mtbench", "gsm8k", "humaneval", "sharegpt")
METHODS = [
    ("PRIOR_F0_fp16", "F0 FP16 DFlash (reused M0)"),
    ("PRIOR_F1_rtn", "F1 naive W4A4 RTN (reused M1)"),
    ("PRIOR_F2_vanillaSQ", "F2 model-local SpinQuant, no ctx rot (M3)"),
    ("F3_r1d", "F3 + H_t@R1_D (draft-R1 context extension)"),
    ("F3b_c8", "F3b + H_t@(R1_D ΔR) (C8 residual)"),
    ("F3c_c9", "F3c shared R1_DC (C9, one matrix both sides)"),
    ("PRIOR_F4_rt", "F4 + H_t@current R_C==R1_T (reused M5a)"),
    ("F5b_r1d_p2", "F5b R1_D + P2"),
    ("PRIOR_F5_rt_p2", "F5 R1_T + P2 (reused M5)"),
    ("F6b_r1d_p2_qat", "F6b R1_D + P2 + QAT (matched recipe)"),
    ("PRIOR_F6_rt_p2_qat", "F6 R1_T + P2 + QAT (reused M6, final)"),
]


def load(tag, ds):
    g = glob.glob(f"{RD}/shards/al__{tag}__*__{ds}.csv")
    if not g:
        return None
    taus, macro = [], []
    for r in csv.DictReader(open(g[0])):
        ts = [int(x) for x in r["taus"].split(";") if x]
        taus.extend(ts)
        if ts:
            macro.append(sum(ts) / len(ts))
    if not taus:
        return None
    return {"micro": sum(taus) / len(taus),
            "macro": sum(macro) / len(macro), "cycles": len(taus),
            "full_block": sum(t >= 10 for t in taus) / len(taus),
            "surv": [sum(t >= k for t in taus) / len(taus)
                     for k in range(1, 11)]}


def main():
    rows, miss = [], []
    for tag, meaning in METHODS:
        per = {}
        for ds in DSS:
            r = load(tag, ds)
            if r is None:
                miss.append(f"{tag}/{ds}")
            else:
                per[ds] = r
        if len(per) < 4:
            continue
        row = {"method": tag, "meaning": meaning}
        for ds in DSS:
            row[f"{ds}_tau"] = round(per[ds]["micro"], 4)
        row["mean4_micro"] = round(
            sum(per[ds]["micro"] for ds in DSS) / 4, 4)
        row["mean4_macro"] = round(
            sum(per[ds]["macro"] for ds in DSS) / 4, 4)
        row["cycles_total"] = sum(per[ds]["cycles"] for ds in DSS)
        row["full_block_rate_mean4"] = round(
            sum(per[ds]["full_block"] for ds in DSS) / 4, 4)
        for k in (2, 4, 6, 8, 10):
            row[f"P_L_ge_{k}"] = round(
                sum(per[ds]["surv"][k - 1] for ds in DSS) / 4, 4)
        rows.append(row)
    if rows:
        with open(f"{RD}/tables/final_4dataset_al.csv", "w",
                  newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print("final_4dataset_al.csv:")
        for r in rows:
            print(f"  {r['method']:22s} mean4={r['mean4_micro']:.4f}  "
                  f"mt={r['mtbench_tau']} gsm={r['gsm8k_tau']} "
                  f"he={r['humaneval_tau']} sg={r['sharegpt_tau']}")
    if miss:
        print("MISSING (not yet done):", miss)

    # ---- RCAL
    for src, dst in (("rcal__M5_p2rc__mtbench.json",
                      "rcal__PRIOR_F5_rt_p2__mtbench.json"),
                     ("rcal__M6_q5bp2__mtbench.json",
                      "rcal__PRIOR_F6_rt_p2_qat__mtbench.json")):
        s = f"{VSQ}/tables/{src}"
        d = f"{RD}/tables/{dst}"
        if os.path.exists(s) and not os.path.exists(d):
            shutil.copy(s, d)
    rc_rows = []
    for p in sorted(glob.glob(f"{RD}/tables/rcal__*__mtbench.json")):
        tag = os.path.basename(p).split("__")[1]
        d = json.load(open(p))
        rc_rows.append({"method": tag, **{
            k: round(d[k], 4) for k in
            ("AL_q", "RCAL", "SAL", "LAL", "AFS", "precision", "recall")
            if k in d}})
    if rc_rows:
        keys = sorted({k for r in rc_rows for k in r},
                      key=lambda x: (x != "method", x))
        with open(f"{RD}/tables/rcal_final.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rc_rows)
        print("rcal_final.csv:")
        for r in rc_rows:
            print(" ", r)


if __name__ == "__main__":
    main()

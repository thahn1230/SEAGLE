"""R1DCE §29 table assembler (idempotent) — builds the composite tables
from shards + earlier artifacts.  Rerun freely as results land."""
import csv
import glob
import json
import os

RD = ("/home/thahn1230/dflash_workspace/dflash/runs/"
      "dflash_r1d_context_extension_20260814_123938")

ARM_MEANING = {
    "C0_none": ("none", "model-local SpinQuant only"),
    "C1_r1d": ("R1_D", "extend Draft R1 to H_t (PRIMARY)"),
    "C2_r1t": ("R1_T", "reuse Target R1 == current deployed R_C"),
    "C4_had": ("Hadamard", "training-free orthogonal (seed 1234)"),
    "C5_rnd_s101": ("random s101", "training-free orthogonal"),
    "C5_rnd_s102": ("random s102", "training-free orthogonal"),
    "C5_rnd_s103": ("random s103", "training-free orthogonal"),
    "C5_rnd_s1234": ("random s1234", "FIDI control matrix"),
    "C6_learned": ("learned R_C (DFST L0)", "cycle-replay-trained (diag.)"),
    "C7_combo": ("R1_D@R1_T combo", "C7b combined (== sequential, GATE-6)"),
    "C8_resid": ("R1_D@DeltaR", "C8 residual-trained context rotation"),
    "C9_r1dc": ("shared R1_DC", "C9 joint draft+ctx shared rotation"),
    "C9b_r1dc_l03": ("shared R1_DC l0.3", "C9 lambda_ctx=0.3 sweep arm"),
    "C1_r1d_p2": ("R1_D + P2", "R1_D ctx rotation + fc branch scales"),
    "PRIOR_V_M3": ("none (prior)", "VSQ-run V_M3 reproduction anchor"),
    "PRIOR_V_M5_rconly": ("R1_T (prior)", "VSQ-run V_M5_rconly anchor"),
    "PRIOR_V_M4a_p2": ("P2 alone (prior)", "M3+P2, no ctx rotation"),
    "PRIOR_V_M5_rc": ("R1_T + P2 (prior)", "M5 config"),
    "PRIOR_V_Q5bp2": ("R1_T + P2 + QAT (prior)", "M6 config"),
    "CQ_r1d_qat": ("R1_D + P2 + QAT", "Q5-recipe QAT on the R1_D basis"),
}
# candidate key in ht/kv tables for each arm
ARM_CAND = {
    "C0_none": "none", "C1_r1d": "R1_D", "C2_r1t": "R1_T_currentRC",
    "C4_had": "Hadamard", "C5_rnd_s101": "Random_s101",
    "C5_rnd_s102": "Random_s102", "C5_rnd_s103": "Random_s103",
    "C5_rnd_s1234": "Random_s1234", "C6_learned": "RC_L0_learned",
    "C7_combo": "R_combo_c7b", "C8_resid": "R1D_DeltaR",
    "C9_r1dc": "R1DC",
}


def shard_tau(path):
    tot = cyc = 0
    taus_per_prompt = []
    for r in csv.DictReader(open(path)):
        ts = [int(x) for x in r["taus"].split(";") if x]
        tot += sum(ts)
        cyc += len(ts)
        if ts:
            taus_per_prompt.append(sum(ts) / len(ts))
    micro = tot / max(cyc, 1)
    macro = sum(taus_per_prompt) / max(len(taus_per_prompt), 1)
    return micro, macro, cyc


def full_block_rate(path, B=10):
    n = full = 0
    for r in csv.DictReader(open(path)):
        for x in r["taus"].split(";"):
            if x:
                n += 1
                full += int(x) >= B
    return full / max(n, 1)


def main():
    # ---- validation_context_rotation_al.csv (§9 primary table)
    ht = {r["candidate"]: r for r in csv.DictReader(
        open(f"{RD}/tables/ht_a4_candidate_stats.csv"))} \
        if os.path.exists(f"{RD}/tables/ht_a4_candidate_stats.csv") else {}
    kv = {}
    if os.path.exists(f"{RD}/tables/ctx_kv_aw_decomposition.csv"):
        for r in csv.DictReader(
                open(f"{RD}/tables/ctx_kv_aw_decomposition.csv")):
            kv.setdefault((r["candidate"], r["proj"]), []).append(
                float(r["AW_nmse"]))
    base_micro = None
    rows = []
    for p in sorted(glob.glob(f"{RD}/shards/al__*__w4a4__gsm8kvalid.csv")):
        tag = os.path.basename(p).split("__")[1]
        micro, macro, cyc = shard_tau(p)
        if tag == "C0_none":
            base_micro = micro
        rows.append((tag, micro, macro, cyc, full_block_rate(p)))
    out = []
    for tag, micro, macro, cyc, fb in rows:
        cand = ARM_CAND.get(tag)
        m = ARM_MEANING.get(tag, (tag, ""))
        out.append({
            "arm": tag, "context_policy": m[0], "meaning": m[1],
            "tau_micro": round(micro, 4), "AL_macro": round(macro, 4),
            "cycles": cyc, "full_block_rate": round(fb, 4),
            "delta_vs_C0": round(micro - base_micro, 4)
            if base_micro else "",
            "ht_a4_nmse": ht.get(cand, {}).get("a4_nmse_mean", ""),
            "K_nmse": round(sum(kv[(cand, "K")]) / 5, 5)
            if (cand, "K") in kv else "",
            "V_nmse": round(sum(kv[(cand, "V")]) / 5, 5)
            if (cand, "V") in kv else ""})
    with open(f"{RD}/tables/validation_context_rotation_al.csv", "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    print("validation_context_rotation_al.csv:")
    for r in out:
        print(f"  {r['arm']:22s} tau={r['tau_micro']:.4f} "
              f"macro={r['AL_macro']:.4f} dC0={r['delta_vs_C0']} "
              f"htNMSE={r['ht_a4_nmse']} K={r['K_nmse']} V={r['V_nmse']}")

    # ---- sequential_vs_combined_parity.csv (§12)
    if os.path.exists(f"{RD}/tables/gates.json"):
        g6 = json.load(open(f"{RD}/tables/gates.json"))[
            "GATE-6 sequential == combined (C7a==C7b)"]
        with open(f"{RD}/tables/sequential_vs_combined_parity.csv", "w",
                  newline="") as f:
            w = csv.writer(f)
            w.writerow(["check", "value"])
            w.writerow(["pass", g6["pass"]])
            w.writerow(["detail", g6["detail"]])
            w.writerow(["conclusion",
                        "R1_D then R_C is ONE combined orthogonal rotation "
                        "at one quantization boundary, not two independent "
                        "corrections (no quant/nonlinearity between them)"])

    # ---- residual_rotation_stats.csv (§15)
    if os.path.exists(f"{RD}/tables/residual_rotation_stats.json"):
        s = json.load(open(f"{RD}/tables/residual_rotation_stats.json"))
        with open(f"{RD}/tables/residual_rotation_stats.csv", "w",
                  newline="") as f:
            w = csv.writer(f)
            w.writerow(["metric", "value"])
            for k, v in s.items():
                w.writerow([k, v])

    # ---- shared_r1dc_quality.csv (§17)
    rws = []
    for p in sorted(glob.glob(f"{RD}/rotations/R1DC_*.pt.summary.json")):
        s = json.load(open(p))
        s["ckpt"] = os.path.basename(p).replace(".summary.json", "")
        rws.append(s)
    if rws:
        keys = ["ckpt", "lambda_ctx", "lambda_draft", "best_step",
                "best_ce", "best_ctx", "R1DC_dist_from_R1D"]
        with open(f"{RD}/tables/shared_r1dc_quality.csv", "w",
                  newline="") as f:
            w = csv.writer(f)
            w.writerow(keys + ["init_ce", "init_ctx"])
            for s in rws:
                w.writerow([s.get(k) for k in keys]
                           + [s["init"]["ce"], s["init"]["ctx"]])

    # ---- p2_interaction.csv (§20)
    p2rows = []
    for tag in ("C0_none", "PRIOR_V_M4a_p2", "C1_r1d", "C1_r1d_p2",
                "CQ_r1d_qat", "C2_r1t", "PRIOR_V_M5_rc", "PRIOR_V_Q5bp2"):
        p = f"{RD}/shards/al__{tag}__w4a4__gsm8kvalid.csv"
        if os.path.exists(p) and os.path.getsize(p) > 40:
            micro, _, _ = shard_tau(p)
            p2rows.append({"arm": tag,
                           "meaning": ARM_MEANING.get(tag, (tag, ""))[0],
                           "tau_micro": round(micro, 4)})
    if p2rows:
        with open(f"{RD}/tables/p2_interaction.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["arm", "meaning", "tau_micro"])
            w.writeheader()
            w.writerows(p2rows)
        print("p2_interaction:", {r["arm"]: r["tau_micro"] for r in p2rows})


if __name__ == "__main__":
    main()

"""VSQ+FIDI preregistered statistics battery (P1-P6 + FIDI arms).

Thin orchestrator over seagle_port.stats: paired cluster bootstrap (n=3000,
seed 0) per dataset, results accumulated into stats/bootstrap_<ds>.json
(one json per dataset -> unique names, Holm dedupe-safe), then Holm step-down
over everything. Pairs whose shards are missing are skipped with a notice —
re-run after more shards land (idempotent accumulation, same names overwrite
same values).
"""
import os
import subprocess
import sys

RD = ("/home/thahn1230/dflash_workspace/dflash/runs/"
      "dflash_vanilla_spinquant_novelty_20260810_104957")
PY = sys.executable

TEST_DS = ("mtbench", "gsm8k", "humaneval", "sharegpt")
BATTERY = []
for ds in TEST_DS:
    BATTERY += [
        (ds, None, "P1_M0vM3", "M0_fp16@fp16", "M3_vsq@w4a4"),
        (ds, None, "P4a_M3vM5rc", "M3_vsq@w4a4", "M5_rconly@w4a4"),
        (ds, None, "P4b_M3vM5p2rc", "M3_vsq@w4a4", "M5_p2rc@w4a4"),
        (ds, None, "P3_M5rcVsM5p2rc", "M5_rconly@w4a4", "M5_p2rc@w4a4"),
        (ds, None, "P5_M5vM6", "M5_p2rc@w4a4", "M6_q5bp2@w4a4"),
        (ds, None, "GAP_M0vM6", "M0_fp16@fp16", "M6_q5bp2@w4a4"),
        (ds, None, "GAP_M0vM5", "M0_fp16@fp16", "M5_p2rc@w4a4"),
    ]
BATTERY += [
    ("mtbench", None, "P6_M2rawVsM3", "M2_vsqraw@w4a4", "M3_vsq@w4a4"),
    ("mtbench", 40, "S2SEED_M3vS2CHK", "M3_vsq@w4a4", "S2CHK@w4a4"),
    ("mtbench", None, "REP_M3vM3cyc", "M3_vsq@w4a4", "M3cyc@w4a4"),
]
VAL = [
    ("P2a_M3vP2", "V_M3", "V_M4a_p2"),
    ("P2b_M3vMP3", "V_M3", "V_M4b_mp3"),
    ("VRC_M3vRConly", "V_M3", "V_M5_rconly"),
    ("VP2RC_M3vP2RC", "V_M3", "V_M5_rc"),
    ("VQAT_M5rcVsQ5bp2", "V_M5_rc", "V_Q5bp2"),
    ("VQATrc_RConlyVsQ5b", "V_M5_rconly", "V_Q5b"),
    ("VQ2_M3vQ2", "V_M3", "V_Q2"),
    ("VQ3_M3vQ3", "V_M3", "V_Q3"),
    ("VG1_M3vG1", "V_M3", "V_G1"),
    ("VI7A_M3vI7m3", "V_M3", "V_I7m3"),
    ("VI7B_RConlyVsI7m5", "V_M5_rconly", "V_I7m5"),
    ("VRESTQ_M3vRestQ", "V_M3", "V_restq"),
    ("VRESTK_M3vRestK", "V_M3", "V_restk"),
    ("VRESTV_M3vRestV", "V_M3", "V_restv"),
    ("VRESTO_M3vRestO", "V_M3", "V_resto"),
    ("VONLYQ_M3vOnlyQ", "V_M3", "V_onlyq"),
    ("VONLYK_M3vOnlyK", "V_M3", "V_onlyk"),
    ("VONLYV_M3vOnlyV", "V_M3", "V_onlyv"),
    ("VONLYO_M3vOnlyO", "V_M3", "V_onlyo"),
    # post-hoc supplementary (audit Q3 / caveat-1 closure; selection frozen)
    ("SUPP_RCrtVsRand", "V_M5_rconly", "V_RCrand"),
    ("SUPP_RCrtVsHad", "V_M5_rconly", "V_RChad"),
    ("SUPP_Q5bp2VsQ6", "V_Q5bp2", "V_Q6"),
]
for name, a, b in VAL:
    BATTERY.append(("gsm8kvalid", None, name, f"{a}@w4a4", f"{b}@w4a4"))


def shard_path(spec, ds):
    tag, tgt = spec.split("@")
    return os.path.join(RD, "shards", f"al__{tag}__{tgt}__{ds}.csv")


def main():
    def ready(p):
        # shard files are created header-first at job start; require data rows
        try:
            with open(p) as f:
                return sum(1 for _ in f) > 1
        except OSError:
            return False

    ran = skipped = 0
    for ds, subset, name, sa, sb in BATTERY:
        if not (ready(shard_path(sa, ds)) and ready(shard_path(sb, ds))):
            print(f"[skip] {ds}:{name} (shard missing or header-only)")
            skipped += 1
            continue
        cmd = [PY, "-m", "seagle_port.stats", "--run-dir", RD,
               "--dataset", ds, "--pair", f"{name}={sa}:{sb}",
               "--n", "3000", "--out-name", ds]
        if subset:
            cmd += ["--subset-prompts", str(subset)]
        subprocess.run(cmd, check=True,
                       cwd="/home/thahn1230/dflash_workspace/dflash")
        ran += 1
    subprocess.run([PY, "-m", "seagle_port.stats", "--run-dir", RD,
                    "--holm"], check=True,
                   cwd="/home/thahn1230/dflash_workspace/dflash")
    print(f"[battery] ran {ran}, skipped {skipped}")


if __name__ == "__main__":
    main()

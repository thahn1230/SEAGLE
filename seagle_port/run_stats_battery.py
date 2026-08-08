"""Pre-registered comparison battery (§33) -> bootstrap + Holm.

Families:
  grid    : T16D16 vs each cell; draft effect per target; target effect per draft
  iface   : A1 vs A2 per target; A2 vs A3 (TR); A0 vs A2_TR (rotation cost)
  fcfix   : S16_fc vs each fix (mtbench-40 subset arms)
  fulldraft: naive vs P2/MP3/rot at T16; folded vs P2/MP3 at T4/T8
  rc      : RC0 vs RC1 / RC0 vs RC2 (4 ds) / RC2 vs D16 gap / RC2 vs RC2xT8
"""
import os
import subprocess
import sys

rd = sys.argv[1]
PY = sys.executable


def run(ds, pairs, subset=None, name=None):
    ok = []
    for nm, a, b in pairs:
        fa, fb = a.split("@"), b.split("@")
        pa = f"{rd}/shards/al__{fa[0]}__{fa[1]}__{ds}.csv"
        pb = f"{rd}/shards/al__{fb[0]}__{fb[1]}__{ds}.csv"
        if os.path.exists(pa) and os.path.exists(pb):
            ok.append((nm, a, b))
        else:
            print(f"[stats] SKIP {ds}:{nm}")
    if not ok:
        return
    cmd = [PY, "-m", "seagle_port.stats", "--run-dir", rd, "--dataset", ds]
    if subset:
        cmd += ["--subset-prompts", str(subset)]
    if name:
        cmd += ["--out-name", name]
    for nm, a, b in ok:
        cmd += ["--pair", f"{nm}={a}:{b}"]
    subprocess.run(cmd, check=True)


CELL = {("T16", "D16"): "A0_T16D16@fp16", ("T16", "D8"): "G_T16D8@fp16",
        ("T16", "D4"): "G_T16D4@fp16", ("T8", "D16"): "G_T8D16@w8a8",
        ("T8", "D8"): "G_T8D8@w8a8", ("T8", "D4"): "G_T8D4@w8a8",
        ("T4", "D16"): "G_T4D16@w4a4", ("T4", "D8"): "G_T4D8@w4a4",
        ("T4", "D4"): "G_T4D4@w4a4"}

for ds in ("mtbench", "gsm8k", "humaneval", "sharegpt"):
    pairs = []
    for (t, d), spec in CELL.items():
        if (t, d) != ("T16", "D16"):
            pairs.append((f"grid_T16D16_vs_{t}{d}", CELL[("T16", "D16")],
                          spec))
    for t in ("T16", "T8", "T4"):
        pairs.append((f"draft_{t}_D16vD8", CELL[(t, "D16")], CELL[(t, "D8")]))
        pairs.append((f"draft_{t}_D8vD4", CELL[(t, "D8")], CELL[(t, "D4")]))
    for d in ("D16", "D8", "D4"):
        pairs.append((f"target_{d}_T16vT8", CELL[("T16", d)],
                      CELL[("T8", d)]))
        pairs.append((f"target_{d}_T8vT4", CELL[("T8", d)], CELL[("T4", d)]))
    run(ds, pairs)

# mtbench-only families
mt = []
for T, tgt in (("TR", "rot_fp16"), ("T8", "w8a8"), ("T4", "w4a4")):
    mt.append((f"iface_A1vA2_{T}", f"A1_{T}@{tgt}", f"A2_{T}@{tgt}"))
mt += [("iface_A2vA3_TR", "A2_TR@rot_fp16", "A3_TR@rot_fp16"),
       ("iface_A0vA2_TR", "A0_T16D16@fp16", "A2_TR@rot_fp16"),
       ("full_T16_naive_v_p2", "G_T16D4@fp16", "G_T16D4_p2@fp16"),
       ("full_T16_naive_v_mp3", "G_T16D4@fp16", "G_T16D4_mp3@fp16"),
       ("full_T16_naive_v_rotH", "G_T16D4@fp16", "IF_T16D4_rotH@fp16"),
       ("full_T16_p2_v_rotHp2", "G_T16D4_p2@fp16", "IF_T16D4_rotH_p2@fp16"),
       ("full_T4_folded_v_p2", "G_T4D4@w4a4", "G_T4D4_p2@w4a4"),
       ("full_T4_folded_v_mp3", "G_T4D4@w4a4", "G_T4D4_mp3@w4a4"),
       ("full_T8_folded_v_p2", "G_T8D4@w8a8", "G_T8D4_p2@w8a8"),
       ("rc_RC0_v_RC1", "RC0@w4a4", "RC1@w4a4"),
       ("rc_RC0_v_RC2", "RC0@w4a4", "RC2@w4a4"),
       ("rc_RC2_v_D16gap", "RC2@w4a4", "G_T4D16@w4a4"),
       ("rc_RC0_v_G_T4D4p2", "G_T4D4_p2@w4a4", "RC0@w4a4"),
       ("rc_RC2xT8_v_T8D4p2", "G_T8D4_p2@w8a8", "RC2xT8@w8a8"),
       ("b8_T16_B10vB8", "A0_T16D16@fp16", "B8_T16D16@fp16"),
       ("b8_RC2_B10vB8", "RC2@w4a4", "B8_T4RC2@w4a4")]
run("mtbench", mt, name="mtbench_families")

fc = [("fc_naive_v_mp3", "S16_fc@fp16", "S16_fc_mp3@fp16"),
      ("fc_naive_v_p2", "S16_fc@fp16", "S16_fc_p2@fp16"),
      ("fc_naive_v_rot", "S16_fc@fp16", "S16_fc_rotH@fp16"),
      ("fc_rot_v_rotp2", "S16_fc_rotH@fp16", "S16_fc_rotH_p2@fp16"),
      ("fc_rotp2_v_rotp2mp3", "S16_fc_rotH_p2@fp16",
       "S16_fc_rotH_p2_mp3@fp16")]
run("mtbench", fc, subset=40, name="mtbench_fcfix")

# RC 4-dataset
for ds in ("gsm8k", "humaneval", "sharegpt"):
    run(ds, [("rc_RC0_v_RC2", "RC0@w4a4", "RC2@w4a4")],
        name=f"rc_{ds}")

subprocess.run([PY, "-m", "seagle_port.stats", "--run-dir", rd, "--holm"],
               check=True)
print("[battery] complete")

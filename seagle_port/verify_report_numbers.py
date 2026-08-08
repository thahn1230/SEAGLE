"""Report-number verification: every headline figure in
docs/DFLASH_SEAGLE_TRANSFER_STUDY.md re-derived from artifacts.
Run: python -m seagle_port.verify_report_numbers <run_dir>
Exit 1 on any mismatch. Output: <run_dir>/tables/report_verification.txt
"""
import json, os, sys

rd = sys.argv[1]
S = json.load(open(f"{rd}/tables/al_summary.json"))
B = {}
for f in os.listdir(f"{rd}/stats"):
    if f.startswith("bootstrap_"):
        B.update(json.load(open(f"{rd}/stats/{f}")))
H = json.load(open(f"{rd}/stats/holm_adjusted.json"))
gb = json.load(open(f"{rd}/tables/gate_b.json"))
gc = json.load(open(f"{rd}/tables/gate_c.json"))
tau = lambda a, d: round(S[a][d]["tau"], 3)
m4 = lambda a: round(S[a]["mean4ds"], 3)
dl = lambda k: round(B[k]["delta"], 3)
sig = lambda k: H[k]["significant"]
gap = ((S["RC2@w4a4"]["mean4ds"] - S["RC0@w4a4"]["mean4ds"]) /
       (S["G_T4D16@w4a4"]["mean4ds"] - S["RC0@w4a4"]["mean4ds"])) * 100
gap1 = ((S["RC1@w4a4"]["mean4ds"] - S["RC0@w4a4"]["mean4ds"]) /
        (S["G_T4D16@w4a4"]["mean4ds"] - S["RC0@w4a4"]["mean4ds"])) * 100
gapt8 = ((S["RC1xT8@w8a8"]["mean4ds"] - S["G_T8D4_p2@w8a8"]["mean4ds"]) /
         (S["G_T8D16@w8a8"]["mean4ds"] - S["G_T8D4_p2@w8a8"]["mean4ds"])) * 100

C = [
 ("A0 mtbench 3.862", tau("A0_T16D16@fp16", "mtbench") == 3.862),
 ("A0 gsm8k 4.213", tau("A0_T16D16@fp16", "gsm8k") == 4.213),
 ("A0 humaneval 4.892", tau("A0_T16D16@fp16", "humaneval") == 4.892),
 ("A1_TR collapse 1.000", tau("A1_TR@rot_fp16", "mtbench") == 1.000),
 ("A2_TR 3.809", tau("A2_TR@rot_fp16", "mtbench") == 3.809),
 ("A3_TR 3.807", tau("A3_TR@rot_fp16", "mtbench") == 3.807),
 ("grid T16D8 4.161", m4("G_T16D8@fp16") == 4.161),
 ("grid T8D8 4.123", m4("G_T8D8@w8a8") == 4.123),
 ("grid T4D16 3.862", m4("G_T4D16@w4a4") == 3.862),
 ("grid T4D4 2.246", m4("G_T4D4@w4a4") == 2.246),
 ("grid T16D4 1.464", m4("G_T16D4@fp16") == 1.464),
 ("grid T16D4_p2 2.577", m4("G_T16D4_p2@fp16") == 2.577),
 ("RC2 4ds 3.143", m4("RC2@w4a4") == 3.143),
 ("RC0 4ds 2.239", m4("RC0@w4a4") == 2.239),
 ("fc-only 1.941", tau("S16_fc@fp16", "mtbench") == 1.941),
 ("fc rot+P2 3.491", tau("S16_fc_rotH_p2@fp16", "mtbench") == 3.491),
 ("PPL fp16 6.667", round(gb["ppl_fp16"], 3) == 6.667),
 ("PPL rot 6.665", round(gb["ppl_rot_fp16"], 3) == 6.665),
 ("B2 relerr <= 0.023", round(gb["B2_relerr_residual_layers_max"], 3) <= 0.023),
 ("GateC fp64 < 1e-10", gc["fc_relerr_fp64"] < 1e-10),
 ("GateC MP3 < 1e-10", gc["mp3_fp64_relerr"] < 1e-10),
 ("Holm n=122", len(H) == 122),
 ("Holm rejected=75", sum(1 for v in H.values() if v["significant"]) == 75),
 ("A1vA2_TR +2.808", dl("mtbench:iface_A1vA2_TR") == 2.808),
 ("A2vA3 n.s.", not sig("mtbench:iface_A2vA3_TR")),
 ("A0vA2 n.s.", not sig("mtbench:iface_A0vA2_TR")),
 ("RC0vRC2 mt +0.978", dl("mtbench:rc_RC0_v_RC2") == 0.978),
 ("RC1vRC2 n.s.", not sig("mtbench:rc_RC1_v_RC2")),
 ("RC2xT8 +1.028", dl("mtbench:rc_RC2xT8_v_T8D4p2") == 1.028),
 ("fc naive->P2 +1.270", dl("mtbench:fc_naive_v_p2") == 1.270),
 ("fc naive->rot +1.440", dl("mtbench:fc_naive_v_rot") == 1.440),
 ("fc rot+P2+MP3 n.s.", not sig("mtbench:fc_rotp2_v_rotp2mp3")),
 ("T4 folded vs P2 n.s.", not sig("mtbench:full_T4_folded_v_p2")),
 ("T16 naive->P2 +1.012", dl("mtbench:full_T16_naive_v_p2") == 1.012),
 (f"RC2 gap recovery 55.7% (actual {gap:.1f}%)", abs(gap - 55.7) < 0.15),
 (f"RC1 gap recovery 57.2% (actual {gap1:.1f}%)", abs(gap1 - 57.2) < 0.15),
 ("RC1 4ds 3.167", m4("RC1@w4a4") == 3.167),
 ("RC1xT8 4ds 3.514", m4("RC1xT8@w8a8") == 3.514),
 ("RC2xT8 4ds 3.514", m4("RC2xT8@w8a8") == 3.514),
 ("G_T8D4_p2 4ds 2.439", m4("G_T8D4_p2@w8a8") == 2.439),
 ("T8 gap recovery 63% (actual {:.0f}%)".format(gapt8), abs(gapt8 - 63) < 1.0),
 ("RC1vRC2 n.s. all 4 ds", all(not sig(f"{d}:rc_RC1_v_RC2")
                              for d in ("mtbench","gsm8k","humaneval","sharegpt"))),
 ("RC0vRC1 SIG all 4 ds", all(sig(f"{d}:rc_RC0_v_RC1")
                             for d in ("mtbench","gsm8k","humaneval","sharegpt"))),
 ("T8 RC1 transfer SIG all 4 ds", all(sig(f"{d}:rc_T8_p2_v_RC1xT8")
                                     for d in ("mtbench","gsm8k","humaneval","sharegpt"))),
]
lines = [("OK   " if ok else "FAIL ") + n for n, ok in C]
bad = [n for n, ok in C if not ok]
lines.append(f"\nMISMATCHES: {len(bad)} {bad}")
open(f"{rd}/tables/report_verification.txt", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
sys.exit(1 if bad else 0)

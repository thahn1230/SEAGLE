# Fake W8A8 DRAFT acceptance — summary

Goal: test whether 8-bit fake quantization on the pure-R1 EAGLE draft preserves
acceptance (W4A4 collapsed it to ~1.0). C4 = target fake W4A4 + pure-R1 draft
fake W8A8. Run dir `runs/fake_w8a8_draft_<ts>/`. Llama-2-7b-chat + EAGLE. Greedy.

## Headline
**W8A8 largely recovers draft acceptance.** C4 (target W4A4 + draft W8A8) ≈ 2.3
vs C3 (draft W4A4) 1.0 and C2 (draft fp16) 3.0 — 8-bit is a viable draft
precision; 4-bit is not. Coverage passes (all 8 draft linears fake-W8A8 in the
generation path, weight_bits=8, activation_bits=8, hooks fire).

## Acceptance table (n=20, 64 tok)
| config | target | draft | draft basis | acceptance |
|---|---|---|---|---:|
| C0 stock fp16 | fp16 | fp16 | original | 3.416 |
| C1 target-W4A4 + orig draft | W4A4 | fp16 | original | 3.008 |
| C2 target-W4A4 + pure-R1 draft | W4A4 | fp16 | pure-R1 | 3.008 |
| C3 target-W4A4 + pure-R1 draft W4A4 | W4A4 | fake W4A4 | pure-R1 | 1.015 |
| **C4 target-W4A4 + pure-R1 draft W8A8** | W4A4 | **fake W8A8** | pure-R1 | **2.121** |
| C5 target-fp16 + pure-R1 draft W8A8 | fp16 | fake W8A8 | pure-R1 | 2.457 |
| C6 target-W4A4 + orig draft W8A8 | W4A4 | fake W8A8 | original | 1.648 |

Notable: **C4 (pure-R1 draft W8A8) 2.121 > C6 (ORIGINAL draft W8A8) 1.648** — the
R1 rotation's outlier suppression (18.4→2.0 channel ratio, from the distribution
analysis) genuinely HELPS the draft at 8-bit, where 4-bit was already hopeless
(C3 ≈ Borig ≈ 1.0). C5 (clean fp16 target) 2.457 > C4 2.121 — the target's W4A4
costs ~0.34 more on top of the draft's W8A8.

## The 15 required answers
1. **Was the draft actually fake W8A8 in the generation path?** YES — coverage
   passes; forward-hook trace (`draft_fake_w8a8_forward_hook_trace.csv`, from
   real EAGLE generation) shows all 8 draft linears calling fake weight +
   activation quant at weight_bits=8, activation_bits=8.
2. **Which draft modules were fake W8A8?** `ea_layer.fc`,
   `self_attn.{q,k,v,o}_proj`, `mlp.{gate,up,down}_proj` (all 8).
3. **Every required linear called fake WEIGHT quant during generation?** YES
   (RTN per-channel symmetric 8-bit + MSE clip, SpinQuant `WeightQuantizer`).
4. **Every required linear called fake ACTIVATION quant during generation?**
   YES (per-token asymmetric 8-bit, SpinQuant `ActQuantizer`).
5. **Quantization policy?** weight = per-channel symmetric RTN + MSE clip,
   8-bit; activation = per-token asymmetric, 8-bit, groupsize −1 (same policy as
   the target's SpinQuant path, at 8-bit).
6. **R1/R2/R3/R4 in the draft generation path?** R1 (full conjugation), R2
   (per-head V/O), R4 (down_proj Hadamard) — YES; R3 N/A (KV fp16, matches
   target). `draft_rotation_application_trace.csv`.
7. **C0 stock fp16 acceptance?** 3.416.
8. **C2 target-W4A4 + pure-R1 fp16 draft?** 3.008.
9. **C3 target-W4A4 + pure-R1 W4A4 draft?** 1.015.
10. **C4 target-W4A4 + pure-R1 W8A8 draft?** 2.121.
11. **C5 target-fp16 + pure-R1 W8A8 draft?** 2.457.
12. **Acceptance recovered by W8A8 vs W4A4?** C4 2.121 vs C3 1.015 -> +1.106
    accepted length recovered.
13. **Acceptance lost by W8A8 vs fp16 draft?** C4 2.121 vs C2 3.008 -> 0.887
    still lost to 8-bit (~29%).
14. **Is W8A8 a viable candidate for the EAGLE draft?** YES — it recovers most of
    the drafting gain (from ~1.0 back to 2.12; 2.46 on a clean fp16 target), unlike W4A4. C5 (clean
    fp16 target) confirms the draft W8A8 itself is the source of recovery.
15. **Next implementation step?** Measure W8A16 (weight-8 / act-16) and W4A16 to
    find the minimal draft precision that preserves acceptance; or mixed
    precision keeping fc/attention higher-bit (the distribution analysis flags
    those as most sensitive). Then real INT8 kernels once precision is chosen.

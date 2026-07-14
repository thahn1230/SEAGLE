# Fake W4A4 DRAFT acceptance — summary

Goal: quantize the pure-R1 EAGLE **draft** to fake W4A4 (in the real generation
path) and report accepted length. B4 = target fake W4A4 + draft fake W4A4.
Run dir `runs/fake_w4a4_draft_<ts>/`. Llama-2-7b-chat + EAGLE. Greedy.

## Headline
**B4 (draft fake W4A4) collapses acceptance to ~1.0.** Coverage PASSES (all 8
draft linears fake-quantized in generation, hooks fire), R1/R2/R4 applied, head
and PL verified — so this is NOT a plumbing/rotation bug. It is the GENUINE cost
of 4-bit quantizing the tiny single-layer EAGLE draft, PROVEN by a control: the
ORIGINAL un-conjugated draft on an fp16 target, quantized to 4-bit, ALSO
collapses (Borig ≈ 1.12). The EAGLE draft predicts FEATURES that must match the
target hidden closely; 4-bit noise (weight AND activation, each alone) destroys
that, so almost nothing is accepted.

## Acceptance table (n=20, 64 tok)
| config | target | draft | draft basis | mean acceptance |
|---|---|---|---|---:|
| B0 stock fp16 | fp16 | fp16 | original | **3.416** |
| B1 target-W4A4 + orig draft fp16 | fake W4A4 | fp16 | original | 3.008 |
| B2 prev [R1.T,I]/[R1.T,R1.T] | fake W4A4 | fp16 | prev-failed | 1.000 |
| B3 target-W4A4 + pure-R1 draft fp16 | fake W4A4 | fp16 | pure-R1 | 3.008 |
| **B4 target-W4A4 + pure-R1 draft W4A4** | fake W4A4 | **fake W4A4** | pure-R1 | **1.015** |
| B5 target-fp16 + pure-R1 draft W4A4 | fp16 | **fake W4A4** | pure-R1 | 1.010 |
| Borig orig un-conjugated draft + 4-bit | fp16 | fake W4A4 | original | 1.122 |

## Localization (n=2/32 — why B4 ≈ 1.0)
| config | what | acceptance |
|---|---|---:|
| B3 | draft fp16 (reference) | 3.26 |
| B4 | draft W4A4 (weight+act, R1+R2+R4) | 1.02 |
| B4w | draft weight-only 4-bit | 1.23 |
| B4a | draft act-only 4-bit | 1.57 |
| B4r1 | draft W4A4, R1-only (no R2/R4) | 1.03 |
| **Borig** | **ORIGINAL un-conjugated draft + 4-bit, fp16 target** | **1.12** |
| Borigw | original draft, weight-only 4-bit, fp16 target | 1.18 |

Reading: B4r1 = B4 (R2/R4 don't change it). B4w and B4a each collapse (weight
AND activation 4-bit are independently destructive). Borig (no R1, no rotation,
clean fp16 target) collapses too → the cause is 4-bit on the small draft, not
the R1/interface. B3 (identical rotation/head/PL, draft fp16) is 3.26, so the
plumbing is correct.

## The 13 required answers
1. **Was the draft actually fake W4A4 in the generation path?** YES — coverage
   passes; the forward-hook trace (`draft_fake_w4a4_forward_hook_trace.csv`,
   from real EAGLE generation) shows all 8 draft linears calling fake weight +
   activation quant.
2. **Which draft modules were fake W4A4?** `ea_layer.fc`,
   `self_attn.{q,k,v,o}_proj`, `mlp.{gate,up,down}_proj` (all 8 required).
3. **Every required linear called fake WEIGHT quant during generation?** YES
   (RTN per-channel symmetric 4-bit + MSE clip, SpinQuant `WeightQuantizer`;
   applied to the weight used in every generation forward).
4. **Every required linear called fake ACTIVATION quant during generation?**
   YES (per-token asymmetric 4-bit, SpinQuant `ActQuantizer`, `find_params`+
   quantize on every forward; hook `fake_activation_quant_called=True`).
5. **R1/R2/R3/R4 applied to the draft generation path?** R1 (full conjugation),
   R2 (per-head V/O weight conjugation), R4 (down_proj Hadamard fold + online
   Hadamard on the intermediate) — all in the generation path
   (`draft_rotation_application_trace.csv`, `generation_path_applied=true`).
6. **If R2/R3/R4 not applied, what's missing?** R3 is intentionally N/A for
   w4a4: KV stays 16-bit (`k_bits=16`), and SpinQuant adds R3 only when
   `k_bits<16` — the target itself applies {R1,R2,R4} and no R3 for w4a4, so the
   draft matches. R1/R2/R4 are all applied.
7. **B0 stock fp16 acceptance?** 3.416.
8. **B3 target-W4A4 + pure-R1 fp16 draft?** 3.008.
9. **B4 target-W4A4 + pure-R1 W4A4 draft?** 1.015 (collapses).
10. **B5 target-fp16 + pure-R1 W4A4 draft?** 1.010 (draft quant alone collapses,
    consistent with Borig 1.122).
11. **How much acceptance lost to draft fake W4A4?** ~all of it: B3 3.008 -> B4
    1.015 (drop 1.993; the speculative gain is essentially eliminated).
12. **Did exact greedy token equality fail due to fake-quant path dependence?**
    YES — fake-W4A4 greedy is execution-path-dependent (incremental vs tree),
    established previously (`path_determinism_w4a4.json`); exact match is not the
    success criterion here — acceptance length is.
13. **Next implementation step?** The single-layer 0.24B EAGLE draft is too
    quantization-sensitive for 4-bit (feature prediction, not argmax). Next:
    keep the draft at higher precision (W8A8 or W4A16 weight-only) and measure
    the precision-vs-acceptance tradeoff; or quantization-aware draft training
    so the draft learns to tolerate 4-bit. Do NOT ship a 4-bit draft — acceptance
    ~1.0 means no speculative speedup.

# Strict SEAGLE-RT — Basis Audit (pre-training, binding)

Branch `exp/eagle1-strict-seagle-rt-native-w4a4-target`, base b2eb71e,
scripts committed @57660af. All claims below are code-cited; the
machine-readable versions are `tables/seagle_rt_interface_contract.csv`
and `tables/seagle_rt_training_basis_contract.csv`.

## 1. The native tensor h_RT_native

Produced at `third_party/EAGLE/eagle/model/modeling_llama_kv.py:1074`
(`hidden_states = self.norm(hidden_states)`) inside the DEPLOYED W4A4
SpinQuant target built by `study.build_study_target(rotation='full',
rotation_type='learned_chat_w4a4kv16', quant='w4a4', seed=0)`;
consumed by the draft at `ea_model.py:132` (`outputs[0]`).

Because `fuse_layer_norms` zeroes gamma into the head and R1 is
orthogonal (commutes with RMS scaling):

    h_RT_native = a_t = RMSNorm0(x_R)  =  (h / gamma_f) @ R1_T

- shape (B,T,4096), dtype fp16, measured RMS = 1.0 (unit-RMS confirms
  gamma-exclusion empirically; `tables/bench_teacher.json`)
- tap is outside every activation quantizer (RMSNorm never wrapped;
  lm_head input pinned 16-bit) but is the true output of the fully
  fake-quantized W4A4 stack (all 32 layers W4 RTN + dyn per-token A4,
  online R4 on down_proj; KV16; R1 + 32 per-layer R2 from R.bin
  sha256 4b7e91d2a7531bb8...).

## 2. Restore operations — enumerated and FORBIDDEN (Gates D/E)

Runtime: `rotation_interface.unrotate_hidden` (rotation_interface.py:36
`matmul(h_rot, R1.t()) * gamma_f`); `causal_interface.
RestoredInterfaceCSAdapter.transform_hidden` (causal_interface.py:31);
`study.UnrotateAdapter.transform_hidden` (study.py:238); in-target
tails `tail_unfused.TailAdapter` T2/T3 and `w4a4_impl_fix.
UnfusedTailAdapter`.
Weight-folded equivalents: `W_first = [W_e | W_h·D_gamma·R1]`
(gamma_R1 fold) and `W_rec = [W_e | W_h·R1]`
(concat_selective_projection.py:63-85) — algebraically exact restores
for an original-basis draft; excluded for the strict draft.
Scale reparameterization: `embed_scale_alpha` (P3/GS/LS) — excluded
from strict training and from the RTN baselines (Gate L).
Training-side restore precedent that this study REPLACES:
`train_eagle_draft_int4_qat.py:333` labels `((h @ R1d.t()) * gd) @
Rint` — the C7 objective; strict labels use raw a_{t+1}.
There is NO scalar "interface scale" anywhere in the tree; the only
scale restorations are per-channel gamma_f and draft-only alpha, both
excluded (audit: exhaustive grep, workflow wp6hp7csi).
Enforcement: `scripts/strict_rt/parity_and_norestore_gates.py` gate NR
(runtime monkeypatch counters == 0 + static scan).

## 3. Training objective bases (Gate F/G)

Official v1 objective (validated reproduction, `docs/
EAGLE1_OFFICIAL_RECIPE_AUDIT.md`): single-step teacher-forced;
loss = 1.0·SmoothL1(masked) + 0.1·softCE(masked).
Strict substitution — one coordinate change, no new objective:

| term | prediction | label | basis |
|---|---|---|---|
| SmoothL1 | draft feature f_i | a_{i+1} (same tap, shifted) | native = native ✓ |
| softCE | log_softmax(H_f(f_i)) | softmax(H_f(a_{i+1})) | same head H_f ✓ |

H_f = deployed fused head `W_lm·diag(gamma_f)·R1` extracted from the
deployed build (`native_head.pt`, sha16 0335833c38f8187c), verified
algebraically against the pre-mutation stash (rel err < 5e-3
asserted at extraction). This is exactly the head `ea_model` passes to
`topK_genrate` at eval time — train/deploy scorer identity holds.

## 4. First vs recurrent path (spec §9)

FIRST (training row i): fc(concat[E_orig(token_{i+1}), a_i + noise]).
The embedding half stays ORIGINAL-basis: official init copies the
target's embedding from the ORIGINAL safetensors (cnets.py:486, frozen
at 494-495) — the deployed draft runtime likewise keeps its own
original-basis embedding (eagle_interface_audit.md). The deployed
target's in-memory embedding mutation (mean-centering/rotation) does
NOT propagate to the draft.
RECURRENT: official v1 has NO recurrent training rows. At eval the
draft recycles its own feature through the SAME shared fc
(cnets.py:592) — that feature lives in the native a-basis by the
regression construction, so first and recurrent inputs share one basis
and one projection, exactly the official architecture. No SEAGLE
alpha/D4P3/GS/LS/R5 anywhere in FP16 training (spec §9).

## 5. Noise-scale note (declared interpretation)

Official AddUniformNoise is ABSOLUTE-scale ((rand−.5)·0.2·512/L).
The native feature RMS (1.0 by construction) differs from the
original-basis h RMS; keeping the official absolute noise is the
letter of the recipe and is what we do. Declared here because its
RELATIVE strength differs across bases; changing it would be a new
recipe (forbidden).

## 6. Renaming of the previous "SEAGLE-RT"

The old label mapped to the original-interface from-scratch anchor
(a7c6ccc8) + W4A4 D4P3(alpha=32) deploy — its training consumed FP16
original-basis features (canonical_method_mapping.csv:4; SEAGLE_
PROJECTION_VISUALIZATION_AND_GPUHOURS.md:17-28 states no completed run
trains from scratch on rotated features). In all new documents it is
"Original-interface scratch reference" and serves as
RT-CONTROL-original-interface (P1 comparator). Historical artifacts
are not rewritten.

## 7. Nearest precedents (for interpretation, not evidence)

- Variant G (train_rotation_aware_draft.py): small-budget (≤2400 step)
  native-h_hat training on a rotate-only (quant OFF, random-Hadamard
  R.bin) teacher collapsed (1.38-1.53) — not comparable: wrong teacher,
  wrong rotation artifact, ~1/300 of the official budget.
- C7 (July study): full official recipe applied to the CONVERGED
  public draft on the int4 interface at LR 3e-5 was harmful (−0.65 vs
  PTQ) — a recipe/LR artifact on a converged model, and its labels
  passed through the forbidden restore; the strict study trains FROM
  SCRATCH where LR 3e-5 is the official regime.

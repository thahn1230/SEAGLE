# INITIAL B2-SPLIT AUDIT — eagle_spinquant_w4a4

Date: 2026-07-11. Scope: answers to the 7 "FIRST ACTION" questions of the B2
split-projection spec, verified against the actual vendored EAGLE-1 v1 source
and this project's existing (already-validated) adapters. GPU policy for this
study: `CUDA_DEVICE_ORDER=PCI_BUS_ID`, `CUDA_VISIBLE_DEVICES=6,7` (physical 6,7;
inside the process cuda:0 -> phys 6, cuda:1 -> phys 7).

Pinned third_party commits (verified):
- SafeAILab/EAGLE, branch **v1**, commit `4a9cf3a1f6cd4a294e6d30a4e7c77cba246d7ca5`
- facebookresearch/SpinQuant, branch main, commit `8f47aa3f00e8662caf1a484153920a07e5281c3a`
- Real kernels: QuaRot CUTLASS W4A4 build at `/data/thahn1230/quarot` (sm_89) +
  torch-native tinygemm W4A16 (`aten._weight_int4pack_mm`), both already
  integrated in `src/eagle_spinquant/realint4.py`.
- Models: target `meta-llama/Llama-2-7b-chat-hf`, draft `yuhuili/EAGLE-llama2-chat-7B`.

## 1. Concat order of the EAGLE projection input

`third_party/EAGLE/eagle/model/cnets.py:593`:

    hidden_states = self.fc(torch.cat((inputs_embeds, hidden_states), dim=-1))

So the order is **[token_embedding | target/draft feature]**:
- `W_proj[:, :4096]`  = `W_embedding` block (multiplies the token embedding)
- `W_proj[:, 4096:]`  = `W_feature` block (multiplies the hidden feature)
- `fc = nn.Linear(2D, D, bias=True)` for this checkpoint (bias exists; the
  existing conversion handles `fc.bias`).

(Also independently documented in `docs/eagle_interface_audit.md:49`.)

## 2. Tensor entering the projection on the FIRST forward

`ea_model.py:124-145`: the target runs `self.base_model.model(...)` and passes
`hidden_states = outputs[0]` to `ea_layer.topK_genrate(...)`;
`modeling_llama_kv.py:1074` shows `outputs[0]` is **post-final-norm**
(`hidden_states = self.norm(hidden_states)`). Per verification cycle,
`utils.py:466` passes `accept_hidden_state_new` (target post-norm hiddens of
accepted positions).

- Stock target: first-forward feature = `h = RMSNorm_gamma(x) = n(x)·diag(gamma_f)`
  (**gamma INCLUDED, original basis**).
- SpinQuant-fused target (this repo's standard `full` build): `model.norm.weight`
  is all-ones after fusion (gamma folded into downstream weights + head), so the
  exposed tensor is `a_t = n(x_R) = n(x)·R1` (**gamma EXCLUDED, rotated basis**)
  — exactly the spec's `a_t`. The fused LM head is `W_lm·diag(gamma_f)·R1`, so
  target logits remain exact.

## 3. Tensor entering the projection on RECURRENT forwards

`cnets.py:772-820` (`topK_genrate`): after the first `self(...)` call, the tree
loop feeds `out_hidden` — the draft decoder-layer **output feature f** — back as
`hidden_states` (via `repeat_hidden`). There is **no draft final norm** in this
loop; the head is applied raw (`head(out_hidden[0])`). The draft was trained so
f approximates the target's **gamma-included** post-norm feature h. In a fully
R1-conjugated draft the recurrent feature is `f_R = f·R1` — a complete rotated
feature that must NOT receive gamma_f again.

## 4. Are first-step and recurrent rows ever batched together?

**No.** Within one `topK_genrate` cycle: call #0 consumes ONLY target-originated
features (rows = accepted positions); calls #1..#depth consume ONLY
draft-originated `out_hidden` rows (tree nodes). Verified from the source flow
and from this project's prior runtime traces (`draft_rotation_application_trace`
/ `forward_hook_trace`: `_fc_idx==0 ⇔ input_hidden_source=target_h`, else
`draft_f_R`). A per-cycle fc-call counter reset at each wrapped `topK_genrate`
entry is therefore an EXACT dispatch signal (not a heuristic): `_fc_idx == 0` →
first path, `> 0` → recurrent path.

## 5. Does the AR-head output include final RMSNorm gamma?

Semantically yes: the recurrent feature f imitates `h = n(x)·diag(gamma_f)`
(gamma-included). Mechanically the draft never applies a final norm to the
recycled feature. So in the rotated draft the recurrent contract is
`h_d_R = h_d·R1` with `gamma_already_included = true`, and only the FIRST
(target-originated, fused-target) input `a_t = n(x)·R1` has
`gamma_already_included = false`.

## 6. Embedding / LM-head sharing

- `D_EMBED`: the draft has its **own** `ea_layer.embed_tokens` (a frozen copy
  loaded from the target checkpoint at init — separate storage; prior passes
  rotated it independently without touching the target). Runtime
  data_ptr audit is included in the new test suite.
- `D_LM_HEAD`: stock EAGLE-1 **aliases the target lm_head module** — it is
  passed as the `head` argument (`ea_model.py:145`). This repo's adapters
  substitute `adapter.head` inside the wrapped `topK_genrate`, giving the draft
  an ISOLATED head copy (isolated_components mode de facto); target
  verification always uses the target's own lm_head. This isolation is what
  allows "draft-only quantization" claims.

## 7. Does the current implementation incorrectly reuse ONE gamma-folded projection for every step?

**No.** The prior studies in this repo implement the correct taxonomy already,
with measured fp16 anchors (n=20-45 MT-bench prompts, greedy):

| prior variant | = spec concept | draft basis | fp16 acceptance |
|---|---|---|---:|
| `stock` | C00/FP00 baseline | original | 3.27-3.58 |
| `A` (UnrotateAdapter) | **Arch A-explicit** (`h=(a_t@R1ᵀ)*γ` runtime) | original | 3.37 |
| `B` (FoldAdapter, single fold everywhere) | **FP03 negative control** | original | **2.17** (degrades ✓) |
| `B2` (TwoPathAdapter: fold ONLY first call) | **Arch A-folded split** | original | 3.34-3.37 |
| `F_R_only` (AlgebraicDraftAdapter mode=r1) | **Arch B split** (fully rotated + fc_ext/fc) | R1-rotated | 3.35-3.37 |
| `F_R_gamma` (single-path S-basis) | inexact single-path control | S-basis | 2.48-2.50 |
| `naive` | **N1 negative control** (no correction) | original | 1.14 |
| `A_nogamma` | **FP04/N3** (gamma omitted) | original | degraded |
| real-INT4 target + `B2` | real-kernel split | original | 3.815 (matched A 3.814) |

Existing building blocks to REUSE (not reimplement):
- `rotation_aware.convert_draft_state(mode='r1')` → returns the recurrent fc
  (`out['fc.weight']` = R1-conjugated both halves) AND `extra['fc_ext']` = the
  FIRST-step fc (h-block `in_fold(W_h, R1, gamma)` = `W_h·diag(γ)·R1`, e-block
  pure R1, whole-output fold R1ᵀ) — precisely the spec's
  `W_feature_first = Rᵀ W_feature D_gamma R` after accounting for the output
  fold and [out,in] weight convention.
- `study.TwoPathAdapter` / `realint4.TimedTwoPathAdapter` — dispatch pattern
  (arm-at-topK-entry + first-call flag) proven in fake and REAL INT4 runs.
- `w4a4_impl_fix.UnfusedTailAdapter` — A-explicit target tail (h exposed).
- `fake_w4a4_draft.FakeW4A4Linear` + coverage/tracing — per-module fake W4A4.
- `realint4.QuarotW4A4Linear.from_linear` / `Int4TinygemmLinear.from_linear` —
  real packed kernels.
- `study.build_study_target`, `experiment.*`, MT-bench harness, PPL scripts.

## Gaps the new B2 study must add (the actual new work)

1. **Named split modules** `projection_first` / `projection_recurrent` as
   separate `nn.Linear`s (not a `.data` pointer swap) — required so the two
   projections can be **independently quantized and packed** with their own
   quant scales.
2. Dispatch trace (`feature_origin`, `projection_selected`,
   `gamma_already_included`, checksums) + `validate_b2_projection_dispatch.py`
   with the four failure assertions + state-reset checks.
3. Algebra/negative-control unit-test suite (9 files) + commutator diagnostic
   `C = diag(γ)R1 − R1·diag(γ)`.
4. **Per-projection quantization ablation** (first-only vs recurrent-only W4A4)
   — never measured before.
5. Q-matrix (Q00/Q10/Q01/Q11) under ONE harness with paired prompt-level
   bootstrap + TOST equivalence decisions, common-prefix + e2e regimes.
6. Real-W4A4 packing/dispatch validation for BOTH projection weights
   (fc shape [4096, 8192]) with separate scales.
7. Docs suite + figures + final report + reproducibility bundle.

## Environment

8× RTX 4090 24GB; GPUs 6,7 assigned to this study (0-5 untouched; GPU 2 busy
with another user's job). torch 2.6.0+cu124, transformers 4.51.3, python
3.12.4. Project is not yet a git repo → a repo is initialized (code only;
runs/outputs/third_party excluded) with branch `exp/eagle1-spinquant-b2-split`;
third_party pinned by the commit hashes above.

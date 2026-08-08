# §38 Initial Audit — DFlash × SEAGLE transfer (2026-08-07)

## 1. DFlash commit
- z-lab/dflash @ `94e4abc5e0c31b67bc1a9d30f1cc34ece28a8756` (main), work branch `exp/dflash-spinquant-seagle-transfer`
- Clone: `/home/thahn1230/dflash_workspace/dflash` (SEAGLE repo untouched, module-level port only)
- SEAGLE source: `/home/thahn1230/eagle_spinquant_w4a4` @ `0f6966a`
- Env: venv (torch 2.6.0+cu124 system-site, transformers 4.57.3, sdpa — flash_attn 미설치, correctness backend로 사용)

## 2. Selected checkpoints (Priority 1 CONFIRMED viable)
- Target: `meta-llama/Llama-3.1-8B-Instruct` (gated access OK, config: 4096 hidden / 32 layers / GQA 32:8 / head 128 / inter 14336 / vocab 128256 / tie_word_embeddings=false / rope_scaling llama3)
- Draft: `z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat` @ `d3af30d` — 1048.6M params, 5 Qwen3-style layers (q_norm/k_norm per-head-dim RMSNorm), GQA 32:8, inter 12288, `block_size=10`, `mask_token_id=128002`
- Transformers backend 공식 지원: benchmark.py:173 정규식 `llama.*3\.1.*8b.*instruct`. 논문 reference: τ=4.32 (gsm8k), 4.91 (humaneval), SGLang concurrency=1.
- Secondary (미사용): Qwen3-8B + z-lab/Qwen3-8B-DFlash-b16. 사유: SEAGLE SpinQuant infra는 LLaMA-family 검증됨(R4 14336=28×512 지원 확인), Llama-3.1이 rotation 재사용성·EAGLE 비교 대칭성 최상.

## 3. DFlash hidden-feature dataflow (model.py 실측)
```
target forward(output_hidden_states=True)
  → hidden_states[l+1] for l in target_layer_ids     # residual stream output of layer l
  → concat(dim=-1): [B, S, 5*4096=20480]             # extract_context_feature, model.py:39-45
  → fc: Linear(20480→4096, bias=False)               # model.py:317
  → hidden_norm: RMSNorm(4096)                       # model.py:318, applied model.py:334
  → H_t (target context), fed to EVERY draft layer
draft layer i (model.py:221-233):
  q       = q_norm(q_proj(H_d))                       # draft hidden only
  k_ctx   = k_proj(H_t);  k_noise = k_proj(H_d)       # SHARED k_proj, two distributions
  v_ctx   = v_proj(H_t);  v_noise = v_proj(H_d)       # SHARED v_proj
  k = k_norm(concat_seq(k_ctx, k_noise)); RoPE(q, k)  # k_norm+RoPE applied to ctx K too
  KV cache: ctx K/V persist across blocks (dflash_generate crop(start), model.py:120,139)
```
- Acceptance: linear block prefix, greedy token match `(block[1:] == posterior[:-1]).cumprod` (model.py:135); per-cycle tau = accepted+1 (EAGLE 관례 동일). Draft prediction: 1 target-sampled token + 9 masked positions parallel.
- Official AL metric (benchmark.py:127) = prompt-macro (prompt별 mean의 mean). Cycle-pooled tau도 병행 보고 (SEAGLE 비교용).

## 4. Selected target layer IDs
- `[1, 8, 15, 22, 29]` (0-indexed decoder layer outputs; index offset +1은 embedding output이 hidden_states[0]이기 때문), m=5.
- 3-H vs 5-H ablation (§23): **retraining 없이는 불가** — fc가 20480 입력 고정, 3-H checkpoint 미공개, training recipe 미공개 ("will open-source soon"). 대체: branch-ablation proxy (branch zero/mean 치환 sensitivity)로 답하고 한계 명시. [금지주장 §35 준수: full 3H/5H 비교는 future work]

## 5. W_c shape / block slicing
- `fc.weight ∈ R[4096, 20480]` (bf16, bias 없음). F.linear: `y = x @ W^T`.
- Block: `W_i = fc.weight[:, 4096*i : 4096*(i+1)]` (i=0..4, source layer 1,8,15,22,29 순).
- FP: `y = Σ_i H_i W_i^T`. Weight per-row scale은 row가 5 block 전체를 관통 → per-branch 조작이 weight grid에 유의미 (EAGLE W_first와 동형).

## 6. Shared embedding/head — CONFIRMED
- `noise_embedding = target.model.embed_tokens(block_ids)` (model.py:111)
- `draft_logits = target.lm_head(draft_out)` (model.py:112)
- tie_word_embeddings=false (양쪽 config): embed와 head는 **별도 행렬** — RD2 view 비용은 각각 128256×4096×2B ≈ 0.98GB, 둘 다 필요 시 ~1.96GB.

## 7. SpinQuant hidden basis
- SEAGLE 기존 R.bin(`learned_chat_w4a4kv16`)은 **Llama-2-7b-chat용 — 재사용 불가**.
- Llama-3.1용 신규 R 필요. Infra 존재 (config-driven, 감사 확인):
  - 구조: R1 global [4096²] (residual basis 회전) + R2 per-layer [128²] (V/O 내부, residual basis 불변) + R4 online Hadamard (down_proj 내부, 14336=28×512 지원 확인).
  - **모든 residual-stream hidden은 동일 R1**: H_i^rot = H_i·R1, 모든 소스 layer 공통 (R.bin에 R1 단일 키 — 구현으로 재검증 예정, Gate B에서 layer별 수치 확인).
  - 절차: (a) random-Hadamard R.bin 즉시 민팅 (`spinquant_bridge.py:213-234` 패턴) → Gate B/C; (b) Cayley-SGD 학습 R1/R2 (`scripts/10_optimize_rotation.py`, 신규 config, SpinQuant pin transformers 4.44.2 env) 병렬 실행 → W4A4 headline은 학습 R 사용.
- DFlash에는 EAGLE-vendored modeling 불필요 → SEAGLE 감사에서 나온 rope_scaling blocker 해당 없음 (HF 4.57 표준 LlamaForCausalLM이 llama3 rope 처리).

## 8. Exact interface-fold 수식 (row-vector, F.linear)
Rotated target: H_i^rot = H_i R1 (모든 i 동일 R1, orthogonal).
```
y_i = H_i W_i^T = (H_i^rot R1^T) W_i^T = H_i^rot (W_i R1)^T
⇒ fold: W_i' = W_i @ R1,  즉 fc.weight[:, i*4096:(i+1)*4096] ← W_i @ R1
```
- 검증 경로 (Gate C): explicit `H_i^rot @ R1.T` + stock W_c ↔ folded W_c, FP64/FP32/FP16, 비교점: fc out, hidden_norm out, K/V, draft logits, 생성 block, acceptance 시퀀스.
- hidden_norm은 fc **출력**에 작용 → fold는 norm과 무관 (입력측 회전만 처리).

## 9. MP3 수식
```
H_i' = m_i · H_i^rot          (branch scalar, activation side)
W_i' = (W_i R1) / m_i         (weight side)
y = Σ_i (m_i H_i^rot)((W_i R1)/m_i)^T = Σ_i H_i W_i^T   (FP 불변)
```
- Gauge: 전역 c는 per-token A4 scale + per-row W4 scale이 흡수 → 순수 gauge. 제거: `geomean(m_i)=1` ⇔ `mean(beta_i)=0`, m_i = D^{beta_i} (SEAGLE EP3-P 관례).
- Quantizer 정책 확인 필수 (Exp C 전): concat-before-quant(단일 per-token scale)인지 branch-before-quant인지 — SEAGLE 포팅 quantizer는 concat 단일 scale이 기본 → P2 arm 유효.
- Runtime 분류 예상: weight측 F0 (offline fold), activation branch scale은 target weight로 pre-fold **불가** (H_i는 target residual 내부값) → gather→scale→concat→quant→GEMM prologue fusion 평가, F2 예상 (측정 후 확정).

## 10. R_D shared-boundary 문제 수식
- Shared embed 출력은 target basis (rotated target: e·R_T). Draft 내부 basis를 R_D로 바꾸면 경계 변환 B = R_T^T R_D 필요:
```
input:  x_D = (e R_T) B = e R_D          (draft 진입 시)
output: h_out = (h_D) B^T                (lm_head 진입 전, head는 R_T basis 기대)
```
- Draft 내부 5개 layer는 표준 SpinQuant 논리로 B-conjugation fold 가능 (residual 내부 일관 변환). 그러나 **경계의 B/B^T는 residual identity path에 weight가 없고 embed/head가 target과 공유(수정 금지)라 fold 불가** → RD1: runtime 2×4096² GEMM/block; RD2: draft 전용 embed/head view ~1.96GB; RD3: 불가능성 formal 증명 작성 예정.
- R_C 대안: H_t' = H_t R_C (fc 출력측). ctx-branch만 통과 → embed/head 무관. 단 k_proj/v_proj가 ctx/noise 공유 → ctx 전용 folded K/V weight view 필요: (k+v) × 5 layers × 1024×4096×2B ≈ 84MB — RD2 대비 ~23× 저렴. hidden_norm(γ)과 R_C 순서 비가환 처리 포함.

## 11. Planned experiment arms
- **A** (target-only): A0 stock FP16 / A1 naive rotated→stock W_c (diagnostic) / A2 explicit unrotate / A3 folded W_c × target {FP16, W8A8, W4A4}, draft FP16
- **B**: 3×3 grid Target×Draft ∈ {FP16,W8A8,W4A4}². Draft 양자화 범위: fc + Q/K/V/O + MLP(gate/up/down). 유지: shared embed/head, RMSNorm(q_norm/k_norm/hidden_norm/input/post/final), RoPE, softmax, residual, KV cache bf16
- **C**: P2 (branch별 독립 A4 scale, quantizer 정책 확인 후)
- **D**: MP3 global(gauge 검증) + source-wise (beta grid, coarse→proxy→validation AL)
- **K**: K0 shared / K1 ctx 별도 A4 scale / K2 ctx migration scale + folded view
- **R**: RD0 (R_D=R_T) / RD1 explicit / RD2 views / RD3 증명; RC0-RC3
- **L**: L0 DFlash weighted CE (γ=5, B=10, paper 값) / L1 LK-style adaptive hybrid (TV/KL + w_k) / L2 soft prefix-survival (diagnostic)
- Ablations: block size B∈{8,10} (inference-only, trained B=10 명시); 3H/5H → branch-ablation proxy로 대체 (item 4)
- Datasets: mtbench80(2턴), gsm8k200, humaneval164, sharegpt80 (SEAGLE frozen set), +math500-100 (baseline만). Phase 2-4 iteration은 mtbench40 subset.

## 12. GPU allocation
- 8× RTX 4090 24GB 전부, persistent artifact-idempotent scheduler (SEAGLE `_pmg_scheduler.py` 포팅) + 3중 감시 + wakeup (standing rule).
- 1 arm = 1 GPU (target bf16 16G + draft 2G + hidden states < 24G; W4A4 fake-quant 동일). R1 Cayley 학습은 torchrun 4-8 GPU 단독 phase.

## 13. Estimated GPU-hours
| Phase | GPU-h |
|---|---|
| P1 gates + baseline (5 datasets) | ~15 |
| R1/R2 Cayley (Llama-3.1, w4a4 + w8a8 config) | ~40 |
| P2 interface arms | ~6 |
| P3 W_c/component sensitivity (~14 restores × subset) | ~18 |
| P4 MP3 search + validation | ~12 |
| P5 full-dataset grids (~10 arms × 4 ds) | ~55 |
| P6 R_C/R_D 학습 + eval | ~45 |
| P7 RCAL replay + runtime/folding | ~18 |
| slack/retry | ~30 |
| **Total** | **~240 GPU-h ≈ wall-clock ~1.5-2일 (8 GPU 포화)** |

## 14. Correctness gates (§32)
A upstream FP16 baseline 재현(mtbench subset, τ sanity vs paper 4.3-4.9 범위) / B SpinQuant target 단독 correctness (PPL + rotation invariance FP) / C explicit==folded (FP64/32/16, 6개 비교점) / D MP3 FP invariance / E fake quantizer code 변경 확인 / F sensitivity run에서 의도 component만 양자화 / G R_D/R_C orthogonality / H trainer forward == deployment forward (Gate-DFlash: same mask/block/W_c/KV/shared boundary) / I shared embed/head 미변경 (checksum) / J RCAL exact block replay (linear prefix 전용 신규) / K block_size/target_layer_ids 동일성 (모든 비교쌍)

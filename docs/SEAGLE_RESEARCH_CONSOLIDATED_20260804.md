# SEAGLE 연구 통합 정리 (2026-08-04)

EAGLE-1 tree speculative decoding × SpinQuant-계열 W4A4(-KV4) 양자화,
LLaMA-2-7B-chat, 8× RTX 4090. 이 문서는 지금까지의 모든 스터디의
**최종 verdict·핵심 수치·자산 위치**를 한 곳에 모은 인덱스다.
각 수치의 원본은 해당 스터디 보고서(`docs/`)와 run 디렉토리
(`runs/` → `/data/thahn1230/tldr_runs`)에 있다.

---

## 1. 한눈에 보는 결론 (frontier)

| 축 | 현재 최선 | 근거 |
|---|---|---|
| Draft 품질 (W4A4, weights frozen) | **ACC-LR** (acceptance-aware learned rotation): EP3-P 대비 mtbench dAL **+0.092 [+0.014, +0.163] SIG** | LRGF 스터디 |
| Draft 품질 (weights 학습 허용) | **LP3-QAT** (low-LR QAT, rotation/scale 전부 frozen, draft core만 STE 학습): +0.24~0.28 | PTQ-vs-QAT 스터디 |
| 고정 변환 baseline | **EP3-P** (pathwise scale migration m=D^β, β=0.40/0.45): naive projection 붕괴 tau 1.276 → 3.201 복구 | EP3-P/LRGF |
| KV cache | **KV4 공짜** (AL 손실 없음) | TLDR-KV4 |
| 실커널 | CUTLASS sm89 s4s4s32 **fused epilogue**: unfused 대비 bitwise 동일, c32 제거, T16D4 −44% ms/token | fused epilogue 스터디 |
| E2E break-even | **아직 미달**: T4D4-fused 71.3 tok/s vs T4D16 115.0 (cycle 42.8 ms, 필요 ≤26.5 ms) | fused epilogue 스터디 |

Quality-validated tau (fake-quant EP3-P 계열, projected tok/s 계산용):
T16D16 **3.5762** · T4D16 **3.2744** · T4D4 **3.0517** · T16D4 **2.9146**.

---

## 2. 연구 타임라인과 verdict

### 2.1 인터페이스/기초 (7월 초·중순)
- **Rotation 기초 스터디** (rotstudy/variants A·B·C, pure-R1):
  단일 경로 pure-R1 draft는 h_R tail 필요; fp16에서 R1은 gauge
  (품질 무영향 좌표변환). → `docs/spinquant_draft_pure_r1_*`,
  `docs/01_ROTATED_EAGLE1_ARCHITECTURE.md`.
- **TLDR-KV4**: KV4는 AL 손실 없이 공짜. ea_generate 1960-token cap
  주의. fp16-only 구성에서만 losslessness. →
  `eagle_tldr_kv4_review_bundle_20260718_132823.tar.gz`.
- **LK exact-path rotation**: weights frozen에서 독립 R_D가 공유 R_T를
  이김 (exact-path). P2-P3 audit: Conclusion C, weight imbalance. →
  `runs/eagle_lk_exactpath_draft_rotation_20260721_151703`.

### 2.2 학습 방법론 (7월 하순)
- **PTQ-vs-QAT**: TF-PTQ가 primary 승자; low-LR QAT(=LP3-QAT)는
  +0.24~0.28 sensitivity 이득; INT4는 lossless 아님 (~2.5% 격차).
  LP3-QAT freeze 계약: target/R_T·R1·R2·R4/alpha/embedding/head 전부
  frozen, draft core 235.9M 텐서만 ExactQuantizedRotationForward STE로
  학습, train_alpha=False, official EAGLE loss. →
  `docs/EAGLE1_OFFICIAL_FROM_SCRATCH_PTQ_VS_QAT_STUDY.md`.
- **Official from-scratch 재현**: FP16 재현 파이프라인 + phase-2
  adapter 준비. anchor: `checkpoints/eagle1_fresh_fp16_anchor/anchor.pt`
  (3.2G, sha256 동봉).

### 2.3 GQ / EP3-P / 시각화 (7/31–8/1)
- **GQ 스터디** (@d1cefba): quantizer recipe A/C/F/H 채택, B/D/E/G
  기각; deceptive AL gain 없음 확인; W4A4 타겟에서 AFS 0.95 도달 불가.
  → `docs/EAGLE1_GENERIC_QAT_PATHWISE_P3EXP_RCAL_STUDY.md`,
  `runs/eagle1_generic_qat_pathwise_p3exp_rcal_20260731_154652`.
- **EP3-P 시각화** (@e717f8e): m=D^β 스케일 미그레이션의 U-curve
  메커니즘 규명 (β 과대 시 hidden-side 손상), hidden-scaling은 gauge라
  추가 자유도 없음. →
  `runs/ep3p_projection_visualization_20260801_104655`.

### 2.4 R-EP3-P / LRGF (8/3)
- **R-EP3-P** (@1cc5ac4): projection-국소 rotation은 local NMSE −40%
  이지만 AL/RCAL 전이 **0** (Conclusion E) — NMSE proxy 전이 실패
  재확인. cross-mixing은 local-only. shared > pathwise. →
  `docs/EAGLE1_ROTATED_EP3P_PROJECTION_STUDY.md`.
- **LRGF** (@e8bab9e): 4가지 질문 정리.
  - Q-A: projection 단독 naive 양자화가 최대 민감도 (tau 3.268→1.276);
    EP3-P가 3.201로 복구, projection 문제 소진 (+0.00). 병목은 **AR
    decoder로 이동** (FP16 복원 +0.128 SIG).
  - Q-B: **ACC-LR**이 최초의 frozen-weight EP3-P 초과: val dAL +0.183
    [+0.072,+0.304], mtbench dAL **+0.092 [+0.014,+0.163] SIG**.
    EAGLE-objective LR은 FIXED-R와 동률; NMSE-LR 전이 실패 재현.
  - Q-C: cross-branch mixing은 블록 ≥2면 충분 (b2…b8192 0.0002 이내);
    32-ch Cayley 블록이면 됨.
  - Q-D: EP3-P scale은 **완전 foldable**; cross-branch rotation은
    unfoldable-but-fusable — Triton 융합커널 (concat+scale+rot+A4)
    integer-code parity ≥0.99999, 0.046 ms.
  → `docs/EAGLE1_LEARNED_ROTATION_GRANULARITY_FOLDING_STUDY.md`.

### 2.5 실 INT4 커널 (8/3)
- **SEAGLE real-INT4 E2E** (@077b0f1, @8b74819): CUTLASS 3.4.1
  s4s4s32 (m16n8k64 IMMA, SASS 검증), 850 TOPS @M512.
  RealInt4Linear 전 모델 swap −70% 메모리. 실커널 granularity:
  per-row(token) SYM A4 (absmax/7) + per-out-channel SYM W4 RTN —
  fake-quant 계약(asym A4 + MSE-clip W4 + rotation folds)과 다르므로
  실커널 tau는 품질지표 아님 (validated tau로 projection).
  → `docs/SEAGLE_INT4_TENSORCORE_E2E_STUDY.md`,
  `runs/seagle_int4_e2e_20260803_204811`.
- **Fused epilogue** (@11a3f64): epilogue visitor로
  `fp16(acc_i32·s_a[m]·s_w[n]+b[n])` in-kernel. 14/14 shape bitwise
  동일, c32 [M,N] int32 중간버퍼 제거 (12→4 MiB 증가분), 생성 토큰
  해시 동일. E2E: T16D4 44.3→24.7 ms/token (−44%), T4D4 −5%.
  Break-even 미달 (남은 격차: unfused act-quant launch, small-M
  occupancy, ~9 ms 미분류, tau 3.05 vs 3.27).
  → `runs/seagle_int4_fused_epilogue_20260803_213742/final_report.md`.
- **Draft를 양자화해야 하는 근거** (@d6fa209, 측정 기반):
  - 메모리: draft 702→365 MiB (linear만 −75%); 절약 337 MiB ≈ KV
    640 tokens.
  - Linear M-sweep: INT4가 M=1에서 2.9×, M≥256에서 2.6–4.2× 빠름;
    tree-M에선 동률, M=64만 +6%.
  - Cycle breakdown: int4 draft 5.51 ≈ fp16 draft 5.41 ms (fused).
  - Amdahl: verify가 빨라질수록 fp16 draft 점유율 18%→44% — draft가
    미래 병목.
  - 반대 증거 명시: 현재 tau 열세, tree E2E 순손실.
  → `docs/DRAFT_QUANTIZATION_CASE.md`,
  `runs/seagle_int4_fused_epilogue_20260803_213742/why_draft_quant/`.
- **SpecMQuant latency-share** (별도 스터디): verify 가속 시 draft
  점유율 36→60%; QQQ draft는 g128 target과만 페어링.

---

## 3. 자산 지도

### 3.1 코드
- `src/eagle_spinquant/` — 브리지/스터디 코어, `real_int4_linear.py`
  (fused/unfused 백엔드 스위치 `SEAGLE_INT4_EPILOGUE`),
  `projection_rotation.py` (Structured/Learned rotation),
  `concat_selective_projection.py` (EP3-P/rotation adapter).
- `kernels/w4a4_cutlass_sm89/` — 실 INT4 커널 패키지: `w4a4_sm89.h/.cu`
  (unfused), `epilogue_visitor_row_col_scale.h` +
  `gemm_with_epilogue_visitor.h` + `w4a4_fused.cu` (fused),
  `torch_binding.cpp` + `build_torch_ext.py` (JIT;
  CUDA_HOME=/usr/local/cuda-12.8 강제), `benchmark*.cu`.
- `scripts/` — 스터디 하네스 전체 (train_eagle_learned_rotation.py,
  benchmark_seagle_int4_e2e.py, benchmark_fused_linear_sweep.py 등).
- `third_party/EAGLE`(v1 브랜치)·`third_party/SpinQuant` — 수정 금지.

### 3.2 데이터/모델
- `checkpoints/eagle1_fresh_fp16_anchor/anchor.pt` — FP16 재현 anchor.
- `runs/` → `/data/thahn1230/tldr_runs` (심링크): 스터디별 run 디렉토리
  (각각 tables/, figures/, logs/, checkpoint 포함). 대형:
  official_fromscratch 74G, GQ 56G, ptq_vs_qat_al 21G, LK 8.4G.
- `outputs/` → `/data/thahn1230/tldr_outputs`.
- 포인터 파일: `runs/{GQ,EP3PVIZ,REP3P,LRGF,SEAGLE_INT4,SEAGLE_FUSED,
  LK,LRAS,OF,PQ,TLDR,…}_RUN_DIR`.

### 3.3 번들 (repo 루트, timestamped가 canonical)
`*_latest.tar.gz`는 전부 timestamped 파일의 심링크 (2026-08-04 정리에서
바이트 동일 사본 6개를 심링크로 교체 — `docs/cleanup_20260804/` 참조).
sha256은 각 `.sha256` 파일 및
`docs/cleanup_20260804/PRE_CLEANUP_CHECKSUMS.sha256`.

### 3.4 스냅샷/정리 기록
`docs/cleanup_20260804/`: PRE_CLEANUP_GIT_STATUS.txt,
PRE_CLEANUP_FILE_MANIFEST.tsv (5684행: path/size/mtime/git_status/
sha256/classification/action/reason), PRE_CLEANUP_LARGE_FILES.tsv,
PRE_CLEANUP_CHECKSUMS.sha256, PRE_CLEANUP_TREE.txt,
CLEANUP_ACTIONS_LOG.md (실행된 정리 액션).

---

## 4. 방법론 교훈 (반복 확인된 것)

1. **NMSE/local proxy는 AL로 전이 안 됨** — R-EP3-P(−40% NMSE, AL 0),
   LRGF NMSE-LR 재현. acceptance-aware surrogate(soft prefix
   survival)만 전이함.
2. **R1은 fp16에서 gauge** — 품질을 바꾸려면 양자화 경계에서의
   변환(EP3-P scale, learned rotation)이어야 함.
3. **실커널 tau ≠ 품질** — 실커널은 RTN 계약이라 fake-quant 계약과
   다름. 속도는 실커널, 품질은 validated tau로 분리 보고.
4. **fused vs unfused는 bitwise로 검증** — 같은 quantizer codes에
   같은 산술이면 fp16 결과까지 byte-exact 가능.
5. **병목은 이동한다** — projection(구조적, EP3-P로 해결) → AR
   decoder(현 병목, +0.128 SIG headroom).

---

## 5. 다음 단계: draft rotation matrix 학습

진입점은 `docs/NEXT_PHASE_DRAFT_ROTATION.md` 참조 (학습 코드·데이터·
평가 프로토콜·frontier 수치·시작 커맨드 정리).

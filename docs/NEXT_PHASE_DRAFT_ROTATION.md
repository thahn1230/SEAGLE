# 다음 단계: Draft Rotation Matrix 학습 (진입 가이드)

작성 2026-08-04. 전체 맥락은
`docs/SEAGLE_RESEARCH_CONSOLIDATED_20260804.md` 참조.

## 1. 출발점 (검증된 frontier)

- **ACC-LR** (acceptance-aware learned rotation, weights frozen):
  EP3-P 대비 mtbench dAL **+0.092 [+0.014, +0.163] SIG** —
  frozen-weight로 EP3-P를 이긴 유일한 방법 (LRGF Q-B).
- 학습 objective는 **acceptance surrogate** (soft prefix survival)만
  전이함. NMSE-LR·EAGLE-objective-LR은 실패/동률 (LRGF·R-EP3-P에서
  반복 확인). 새 rotation 학습도 반드시 이 surrogate 기반으로.
- Granularity: **32-ch Cayley 블록**이면 충분 (block ≥2에서 local
  gain 포화, b2…b8192 0.0002 이내 — LRGF Q-C).
- Deploy 계약: scale은 fold, rotation은 **fuse** (Triton
  concat+scale+rot+A4 커널, integer-code parity ≥0.99999, 0.046 ms —
  LRGF Q-D). 별도 커널 비용으로 rotation 후보를 기각하지 말 것.

## 2. 재사용할 코드

| 파일 | 역할 |
|---|---|
| `scripts/train_eagle_learned_rotation.py` | 학습 하네스: RotCore(ExactQuantizedRotationForward, rot_f/rot_r), teacher precompute, frozen-check (rot_ 접두어 제외), objective 스위치 |
| `src/eagle_spinquant/projection_rotation.py` | StructuredRotation(FWHT butterfly) / LearnedRotation(blockwise Cayley·Givens·Householder) / build_rotation factory (learned_ckpt 로드, rot.to(device)) |
| `src/eagle_spinquant/concat_selective_projection.py` | 배포 adapter: proj_rot_first/rec, W fold(double 산술), 복원 down_proj의 online R4 유지 |
| `scripts/eval_eagle_learned_rotation.py` | 평가 (LRGF_GPUS env로 GPU pool 지정) |
| `scripts/build_eagle_fused_transform_quant_kernel.py` | Triton 융합 커널 + parity 체크 |
| `scripts/benchmark_seagle_int4_e2e.py` | 실커널 E2E (validated tau projection 포함) |

## 3. 데이터/평가 프로토콜 (기존 계약 유지)

- 학습 corpus: 기존 tokenized cache (int16 저장 — 로드 후 `.long()`
  필수; 12400행 한도, LRGF는 rows[:256] 사용).
- 평가: RCAL 프로토콜 (lockstep FP16 reference replay, LCP, SAL/LAL/
  AFS) + mtbench AL, prompt-cluster bootstrap 3000. validation 먼저,
  mtbench 확인은 최종 arm만.
- 품질 기준선: EP3-P tau 3.201 (FP16-draft ceiling 3.2678), ACC-LR
  +0.092. 이걸 못 넘으면 승리 주장 금지.
- 실커널 tau는 품질지표 아님 (RTN 계약) — projected tok/s는
  validated tau (T16D16 3.5762 / T4D16 3.2744 / T4D4 3.0517 /
  T16D4 2.9146)로 계산.

## 4. 알려진 함정 (이번 프로그램에서 실제로 밟은 것)

1. rotation 파라미터 attach 후 frozen-assert가 걸림 — `rot_f`/`rot_`
   접두어 제외 처리 이미 반영됨.
2. nn.Module class-attribute가 등록된 submodule을 가림 (`rot=None`
   trap) — TracedCore는 `proj_rot`로 개명됨.
3. LearnedRotation 파라미터 device — factory가 `rot.to(device)`,
   fold가 W를 device로 이동. 새 코드도 같은 규약.
4. down_proj 복원 시 online R4 Hadamard 유지 필수 (안 하면 tau 2.13
   급 붕괴 시그니처).
5. 배포 후 반드시 FP16 gauge 검사: quant OFF에서 stock EAGLE greedy
   토큰과 일치해야 함.
6. GPU: 공유 서버 — 시작 전 `nvidia-smi` 확인, 타 사용자 프로세스
   금지, 장기 job은 `setsid nohup … > log 2>&1 &` + liveness 확인.
   CUDA ext 빌드는 `CUDA_HOME=/usr/local/cuda-12.8` 강제
   (`build_torch_ext.py`가 처리).

## 5. 시작 커맨드 (예시)

```bash
cd ~/eagle_spinquant_w4a4
# 새 실험 브랜치
git checkout -b exp/draft-rotation-learning-v2
# 학습 (objective=acc 가 검증된 유일 전이 objective)
setsid nohup python scripts/train_eagle_learned_rotation.py \
  --objective acc --block 32 --param cayley \
  --run-dir runs/draft_rotation_v2_$(date +%Y%m%d_%H%M%S) \
  > runs/draft_rotation_v2_train.log 2>&1 &
```

체크포인트·rotation ckpt는 run-dir 아래에만 쓰고, 완료 시 timestamped
bundle + `.sha256` + `_latest` symlink 규약을 따를 것.

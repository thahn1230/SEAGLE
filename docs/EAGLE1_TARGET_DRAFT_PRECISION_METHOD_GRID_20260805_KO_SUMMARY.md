# Target/Draft 정밀도 × W4A4 Draft 방법 그리드 — 한국어 요약 (2026-08-07)

상세: `EAGLE1_TARGET_DRAFT_PRECISION_METHOD_GRID_20260805.md`.
전부 fake-quant, official micro tau, 4개 데이터셋, paired bootstrap
3000 + Holm. 표기: [확보]=통계 유의+검증, [provisional]=조건부,
[미확인]=미결.

## RQ1 — 3×3 정밀도 그리드 (mtbench / [4-데이터셋 평균])

| T\D | FP16 | W8A8 | W4A4(naive) |
|---|---|---|---|
| FP16 | 3.576 [3.80] | 2.110 [2.17] | 1.044 [1.05] |
| W8A8 | 3.538 [3.78] | 3.367 [3.53] | 1.269 [1.28] |
| W4A4 | 3.274 [3.58] | 2.895 [3.18] | 1.252 [1.27] |

1. **Draft 축이 지배** — naive W4A4 draft는 어느 타깃에서도 붕괴 [확보]
2. **W8A8 타깃은 거의 공짜** (T8D16 −0.03 vs T16D16) [확보]
3. **양자화 draft는 회전 타깃 아래서 훨씬 강함** — 같은 D8이
   T16에서 2.17, T8에서 3.53. 원인은 정밀도 매칭이 아니라
   **interface** (a_t·gamma_R1 fold 기저가 양자화 친화적) [확보]
4. 대각선(정밀도 일치) 특별 우위 없음 [확보]

## RQ2 — 8개 방법 × 3 타깃 (4-데이터셋 평균 tau)

| 방법 | T16 | T8 | T4 |
|---|---|---|---|
| naive PTQ / generic QAT | 1.05 / 1.57 | 1.28 / 1.83 | 1.27 / 1.80 |
| EP3-G PTQ / QAT | 3.11 / 3.32 | 3.58 / 3.43 | 3.33 / 3.45 |
| EP3-P PTQ / QAT | 3.11 / 3.30 | 3.58 / 3.44 | 3.34 / 3.45 |
| **EP3-P+R_D PTQ** / QAT | 3.31 / 3.30 | **3.69** / 3.45 | **3.53** / 3.47 |

- **Generic QAT는 구조적 실패가 아니라도 P3 대비 ×2 격차** —
  동일 budget/공유 LR(1e-5)에서도 1.6-1.8 수준 [확보]
- **EP3-G ≈ EP3-P**: 전 타깃 통계적 동률 (Holm 후 n.s.).
  calibrated 전역 스케일 하나면 충분; T16은 calibration이 아예
  동일값으로 수렴 [확보]
- **R_D는 PTQ 위 최고 업그레이드**: +0.13~0.20 SIG, 전 타깃 [확보]
- **QAT는 타깃 의존**: T16 +0.2 도움 / T8은 전 P3-arm에서 손해
  (PTQ 승) / T4는 ep3g·ep3p만 도움, R_D-PTQ는 못 넘음 [확보]
- **R_D 이득은 QAT 후 소멸** (rd-QAT ≈ ep3p-QAT) — 회전과
  weight-학습이 같은 basin. 하나만 선택: 학습-불필요 R_D-PTQ
  또는 EP3-P QAT [확보]
- Transfer: T4-학습 R_D를 T8에 이식해도 3.66 (matched 3.69) —
  놀랍게 전이되나 matched가 안전 기본값 [확보]

## RCAL (T4 × 8 방법, mtbench)

RCAL 순위 = AL 순위 (verifier drift만으로 이긴 방법 없음, deceptive
플래그 0건). P3-계열 AL의 ~35-42%는 SAL(양자화 verifier 관용),
AFS 0.83-0.86. 최고 절대 RCAL: R_D-QAT 1.816 [확보].

## Runtime / Folding

- **방법 간 runtime 차이 ≈ 0 측정** — draft 사이클 25.9-27.5 ms로
  동일, PostProjectionR1 0.60-0.63 ms/cycle 전 방법 공통 [확보]
- Fold 상태: R_T·γ_f·EP3-G m·EP3-P weight측·R_D·R2 = 완전 fold(F0);
  EP3-P m_rec만 단일 공유 테이블에 co-fold 불가 → fused e-slice
  multiply(F2) 또는 dual table(+250MiB, F1); R4는 온라인
  Hadamard(F2/F3); QAT는 runtime 연산자 없음 [확보]
- 절대 ms/token은 fake-quant 지배 — 실커널 지연 주장 금지

## 최종 권장 (Q14)

**W8A8 타깃 + W4A4 draft (EP3-P + R_D, PTQ)** — 평균 AL 3.69,
방법 추가 runtime 0, 학습 불필요. 타깃이 W4A4여야 하면 같은 방법
(3.53). QAT는 T16 타깃이거나 calibration 불가 시에만 [확보/
provisional — 실커널 미검증].

## 미결/한계

target-quality 절대치 측정 실패(256-token chat 형식 — pass@1 전부 0)
[미확인]; RCAL은 T4×mtbench만; 중도 버그 2건(T8-stock restore 누락,
C3+R_D teacher) 수정·격리 완료 — 오염 데이터 quarantine/ 보관.

# SEAGLE → DFlash 이식 연구 — 한국어 요약 (2026-08-08)

상세: `DFLASH_SEAGLE_TRANSFER_STUDY.md`. 전부 fake-quant, official tau
(cycle-pooled), 4 데이터셋, paired bootstrap 3000 + Holm (122개 비교, 75개
기각). 표기: [확보]/[provisional]/[미확인].

## 한 줄 결론

SEAGLE 원리는 EAGLE 전용이 아니다 — "target-hidden interface-aware
quantization"으로 일반화된다. 단, EAGLE 처방의 1:1 이식이 아니라 DFlash
구조에 맞는 새 처방이 나왔다: **W_c fold (무손실) + P2 (동적 branch scale)
+ context 회전 R_C = R_T 재사용 (학습 불필요, +84MB view)**.

## RQ별 답

1. **R_T가 인터페이스를 깨는가** — 깬다, 치명적으로: naive τ=1.000 (전
   타깃), explicit/folded 3.73-3.81 완전 회복, fold==explicit n.s. [확보]
2. **EAGLE식 W_c 불균형?** — 아니다. branch RMS 비 1.11×뿐. 지배 요인은
   **outlier** (absmax 322/RMS 1.15, A4 zero-code 94.6%). R1 회전이 이걸
   고침 (NMSE 0.054→0.020). Weight-block 불균형(2.33×)은 실재 — MP3 대상.
   W_c는 압도적 단일 병목: fc 단독 W4A4 τ 1.94 vs 타 컴포넌트 3.0-3.5.
   [확보]
3. **P2/MP3 (fc-only, T16)**: naive 1.94 → MP3 2.36 → P2 3.21 → rot 3.38
   → rot+P2 3.49 (+MP3 n.s.). 정적 MP3는 최약체 — 동적 outlier에 정적
   scale은 역부족. 회전 타깃에선 folded 인터페이스가 이미 회전 기저라
   P2/MP3 추가 이득 소멸. [확보]
4. **R_D**: shared embed/head 하에서 zero-overhead 불가 **증명** (경계
   2회 변환 비가환원; RD2 view 1.96GB). 학습 실험은 미수행 — R_C가 1/23
   메모리로 학습-회전 이득을 달성해 우선순위 탈락. [확보(증명)/미확인(AL)]
5. **R_C (context-only)**: T4 + 전체-linear W4A4 draft + P2에서 (양자화
   범위: fc+Q/K/V/O+MLP; embed/head/norm/RoPE/KV는 bf16) RC0 2.09 →
   RC2(학습) 3.07 (mtbench, +0.98 SIG; **4-ds 2.24→3.14**). fp16-draft
   상한(4-ds 3.86) 대비 gap **55.7%** 회수 (mtbench 기준 55.3%).
   **RC1 (R_T 재사용, 학습 0) 4-ds 3.167 — 학습판 RC2 3.143과 전
   데이터셋 통계적 동률 (Holm 후 n.s.), 수치는 오히려 근소 우위.**
   RC1의 gap 회수율 57.2%. T8 전이 (동일 조건 folded+P2, 4-ds):
   무회전 2.439 → RC1xT8 **3.514** / RC2xT8 3.514 (+1.01~1.22 SIG,
   RC1 vs RC2 n.s.) — T8 상한(4.141) 대비 gap 63% 회수, 전이가 더 유리.
   ※ 그리드 T8/D4 셀 2.465는 P2 없는 arm이라 직접 비교 불가. 구조적 발견: 어떤 draft-측 rebasing도 ctx 전용 K/V view
   (84MB) 필요 (k/v_proj가 norm 다른 두 branch 공유). [확보]
6. **Persistent ctx K/V** — 제2 인터페이스 병목: H_t는 outlier 분포 단일체
   (A4 zero 83%), ctx NMSE가 draft branch 대비 최대 10×, scalar migration
   무효 (s_opt≈1.0) → 처방도 회전. [확보]
7. **3×3 그리드 (4-ds 평균)**: T16/T8/T4 × D16 4.20/4.14/3.86, D8
   4.16/4.12/3.85 (**전 타깃 근사-공짜 — EAGLE D8@T16 붕괴와 대조**), D4
   1.46(naive)/2.47/2.25(folded); +P2+R_C(RC1) T8 3.514 / T4 3.167. W8A8 타깃 −0.06, W4A4 타깃
   −0.34. 회전 타깃 아래 quantized draft 강세 = SEAGLE 인터페이스 효과
   재현. [확보]

## RCAL (T4, mtbench)

RCAL 순위 = AL 순위, deceptive 0건. RC2: AL_q 2.07 / RCAL 1.86 / AFS 0.92
— R_C 이득의 ~85%가 reference-consistent. [확보]

## Runtime / Folding

fake-quant 지배 (T4 verify 104ms vs fp16 21.7ms) — 절대 latency 주장 금지.
실측 인터페이스 비용 소액 (embed restore 0.03-0.16ms, head +0.1ms).
Fold 분류: R1→W_c F0 / MP3 weight F0·act F2 / R_C F0(+84MB view) /
embed·head restore F1(+각 0.98GB view 대안) / R_D 경계 NF(증명) / R4 F2.

## 권장 배포

8-bit로 충분하면 **T8+D8** (4-ds 4.12, 공짜). W4A4-linear draft 필요 시
**folded 인터페이스 + P2 + R_C=R_T 재사용**: 학습 0, 추가 메모리 84MB, F0.
측정 4-ds tau: **T4 3.167 / T8 3.514** (무회전 2.239 / 2.439 대비).
학습판 R_C는 전 데이터셋 통계적 동률이라 학습 이득 없음. MP3 비추천, R_D 비추천 (DFlash 한정).

## 한계

3H/5H ablation 불가 (training recipe 미공개; branch-deletion proxy로
대체) / R_D AL 미측정 / L1·L2 objective 미실행 (RC1≈RC2로 가치 하락) /
위치별 KL·TV, 실커널, target-quality 절대치 [미확인] / 사고 기록: 114 job
실패 0, 밤새 큐 소진 후 미충전 공백 1회 (drain-watcher로 해결).

## 부록 A — 미결 항목 마감 (2026-08-08 저녁)

- **L1 objective**: 3.062 — R_T 재사용(3.098)도 L0 학습(3.072)도 못 넘음.
  삼중 null → 학습-불필요 권장 최종 확정. [확보]
- **3H/5H proxy (소스 ablation, AL 실측)**: FP16에선 깊은 소스가 지배
  (H_29 −0.84), 양자화+RC1에선 중-심층(H_15/H_22)이 −0.54/−0.53으로 더
  중요해짐. 소스 축소가 유리하다는 증거 없음 (H_8만 근사-무료). [확보/proxy]
- **RC1 RCAL 4-dataset**: AFS 0.91-0.93 균일, SAL ~10.6%, deceptive 0건 —
  권장 recipe의 충실성 전 데이터셋 검증. [확보]
- **B=16**: 이득 없음 (RC1 −0.17 n.s., fp16 동률) — B=10 유지. [확보]
- 재현성: RC1 재실행이 4-ds tau를 소수 4자리까지 동일 재현. [확보]

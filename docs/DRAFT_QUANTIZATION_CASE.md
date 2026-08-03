# Draft도 양자화해야 하는가 — 측정 근거 정리 (defensible claims only)

모든 수치는 이 레포 실측 (4090/SM89, CUTLASS s4s4s32 IMMA 검증 커널,
`runs/seagle_int4_e2e_*` + `runs/seagle_int4_fused_epilogue_*`).
fused/unfused 커널은 비트동일이 검증되어 있으므로 아래 결론은 fusion
여부와 무관하게 성립한다.

## 표 1 — 메모리 (측정): draft를 fp16으로 두는 비용

| 구성요소 | FP16 | INT4 | 절감 |
|---|---|---|---|
| Target (32 layers) | 12.6 GiB | ~3.2 GiB | −75% |
| **Draft (1 layer + fusion proj)** | **702 MiB** | **365 MiB** | **−48%** (embed table 250MiB는 계약상 fp16 유지; linear부만은 452→115MiB, −75%) |
| 전체 모델 (target+draft 스왑) | 13.30 GiB | 3.93 GiB | −70% |

Draft를 fp16으로 남기면 **337 MiB**를 계속 점유 — LLaMA-2-7B fp16 KV
기준 **약 640 토큰 분량의 KV cache** 또는 그만큼의 배치 여유를 잃는다.
T4D16 배포의 실측 잔여 fp16 덩어리 중 draft가 최대 항목.

## 표 2 — 선형 latency by M (측정, **unfused 포함** — fusion 없이도 성립)

full linear (quant+GEMM+dequant) vs fp16 cuBLAS, 4096→4096, 중앙값 ms:

| M | INT4 unfused | INT4 fused | FP16 | unfused/FP16 | 해당 EAGLE 구간 |
|---|---|---|---|---|---|
| **1** | **0.0144** | 0.0189 | 0.0423 | **0.34×** | chain-draft / 단발 스텝 |
| 16 | 0.0147 | 0.0162 | 0.0171 | 0.86× | 소형 트리 |
| 26 | 0.0211 | 0.0207 | 0.0231 | 0.91× | mc_sim 트리 draft/verify |
| 64 | 0.0217 | 0.0210 | 0.0205 | 1.06× | 대형 트리 |
| 256 | 0.0264 | 0.0278 | 0.0690 | **0.38×** | 배치 verify |
| **512** | **0.0541** | 0.0516 | 0.2296 | **0.24×** | prefill/배치 |

**Fusion 없이도**: M=1에서 2.9×, M≥256에서 2.6–4.2× 빠름; 트리 구간
(16–26)도 이미 fp16보다 빠르거나 동률. 열세 지점은 M=64 (+6%) 하나.
(K=8192 fusion projection: M=64에서 unfused 0.0478 vs fp16 0.0390 —
fused 0.0262로 역전, 이 shape만 fusion이 필요.)

## 표 3 — Amdahl 상한 (측정 기반): draft를 fp16으로 두면 생기는 천장

T4D16-fused 실측 사이클 28.5 ms = draft(fp16) 5.25 + verify(int4)
22.28 + 기타 1.0. Verify가 커널 성숙으로 계속 빨라질 때, fp16 draft
5.25 ms는 고정 하한으로 남는다:

| verify가 지금의 → | 사이클 | fp16-draft 점유율 | draft를 0.6×로 양자화 시 |
|---|---|---|---|
| 1.0× (현재) | 28.5 ms | 18% | −7% cycle |
| 0.5× | 17.4 ms | 30% | −12% |
| 0.25× | 11.8 ms | 45% | −18% |

표 2의 M=1/M≥256 실측 비율(0.24–0.34×)이 보여주듯 draft-linear의
0.6× 사이클화는 커널-한계 내에 있다 (트리 M=16–26 구간은 quant-런치
융합이 전제 — measured가 아닌 projected).

## 정직한 반대편 (숨기지 않음)

- **현 트리-draft E2E 실측에서는 draft int4가 latency 순손실** (draft/
  cycle 5.4→7.8 ms): M=10–26 반복 소형 GEMM + 미융합 activation-quant
  런치 고정비 때문. 손익분기 미달 (T4D4 71.3 vs T4D16 115.0 proj tok/s).
- **품질**: validated tau 3.05(T4D4) < 3.27(T4D16) — 7% 수락률 열세.
- 따라서 "draft 양자화가 **지금 당장 트리-모드 속도**에 낫다"는 주장은
  측정이 지지하지 않는다. 측정이 지지하는 claim은:
  1) **메모리/배포**: 즉시 −48% (측정),
  2) **chain-mode·배치·prefill 구간 latency**: fusion 없이도 승 (측정),
  3) **장기 상한**: verify가 빨라질수록 fp16 draft가 병목으로 승격
     (Amdahl, 측정 기반 산식),
  4) 트리 구간 승리는 quant-런치 융합 후 도달 가능 (projected, 미검증).

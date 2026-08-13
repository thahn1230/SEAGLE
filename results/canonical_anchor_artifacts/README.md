# Canonical anchor artifacts (원 서버 → cross-server 절대 스케일 통일용)

- `learned_chat_w4a4kv16_R.bin` — 원 서버 canonical target R1_T
  (sha256 4b7e91d2…). 배치 위치: `outputs/rotations/learned_chat_w4a4kv16/R.bin`
- `RD_HYB_s2.pt` — canonical R5 (sha256 20c03f00…). 배치 위치:
  `runs/eagle1_draft_residual_rotation_ep3p_20260804_165317/rotations/RD_HYB_s2.pt`

이 두 파일을 위 경로에 배치하면 PTQ 계열 arm(GS/GS+R5/GS+R6/R5R6)의
배포 타깃·회전이 원 서버와 동일 함수가 된다. 이후 남는 차이는 GPU/cuBLAS
라운딩 수준(±0.01~0.05)뿐이며, 그래도 paired 비교는 같은 서버 shard끼리만.

주의: QAT anchor(`anchor.pt`, 3.2GB, sha a7c6ccc8…)는 용량 문제로 미포함 —
QAT arm의 절대 스케일 통일이 필요하면 별도 전송 요청할 것.

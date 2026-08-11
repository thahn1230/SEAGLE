# HANDOFF — DFlash × SEAGLE 연구 인계 문서 (2026-08-11)

이 브랜치(`exp/dflash-vanilla-spinquant-novelty-grid`)를 pull한 서버의
Claude가 **즉시 이어서 진행**하기 위한 완전한 컨텍스트. 원 세션은 사용자
요청으로 VSQ 스터디 중반에 일시정지됨 (@ec1fd18 체크포인트).

## 0. 저장소 구성 (3개 스터디 브랜치, 모두 이 repo)

| branch | 스터디 | 상태 |
|---|---|---|
| `exp/dflash-spinquant-seagle-transfer` | DFST: SEAGLE→DFlash 이식 (본편) | **완료** @cd1771a — 보고서 docs/DFLASH_SEAGLE_TRANSFER_STUDY.md(+KO) |
| `exp/dflash-context-kv-distribution-audit` | DKVA: ctx-K/V 분포 메커니즘 audit | **완료** @32ffba0 — docs/DFLASH_CONTEXT_KV_DISTRIBUTION_AUDIT.md |
| `exp/dflash-vanilla-spinquant-novelty-grid` | VSQ: vanilla-SpinQuant novelty grid | **일시정지** @ec1fd18 — 이 문서의 §3 재개 절차 |

베이스: z-lab/dflash @94e4abc (upstream). 모든 신규 코드는 `seagle_port/`.

## 1. 환경 셋업 (필수 순서)

```bash
# 1) 이 repo를 아무 곳에나 clone (예: ~/dflash_workspace/dflash)
# 2) SEAGLE(EAGLE-side) repo가 반드시 필요 — SpinQuant 코드를 sys.path로 참조:
git clone git@github.com:thahn1230/SEAGLE.git /home/<user>/eagle_spinquant_w4a4
#    경로가 다르면 seagle_port/__init__.py 의 SEAGLE_ROOT 수정 (현재
#    /home/thahn1230/eagle_spinquant_w4a4 하드코딩)
# 3) venv: 시스템 torch 재사용 + transformers만 교체
python -m venv --system-site-packages venv   # dflash 상위 디렉토리에
./venv/bin/pip install "transformers==4.57.3" "tokenizers>=0.22" loguru rich datasets
#    필요 pkg: torch>=2.6+cu, fast_hadamard_transform (SpinQuant R4용)
# 4) HF 로그인 (meta-llama/Llama-3.1-8B-Instruct gated 접근 필요)
# 5) 모델: 자동 다운로드 (target 16GB + draft 2GB)
```

경로 하드코딩 주의: seagle_port 곳곳에
`/home/thahn1230/dflash_workspace/...` 절대경로 (rotations 경로 등) —
`grep -rn "/home/thahn1230" seagle_port/ | grep -v \.pyc` 로 확인 후 일괄
치환하거나 동일 경로 구조 사용 권장.

## 2. 포함된 자산 (assets/ — git 내 커밋됨, 각 <100MB)

- `assets/rotations/llama31_w4a4kv16_s{0,1,2}/R.bin` — **W4A4-전용 학습
  SpinQuant target rotation 3-seed** (공식 optimize_rotation, w4a4kv16;
  sha: ca9f1769/f433a88b/96bf25e3). **체인은 s1 사용** (wikitext PPL
  8.7961; s2 8.7897로 명목 최소지만 Δ0.007=노이즈 — 보고서에 명시할 것)
- `assets/rotations/llama31_w16a4kv16/R.bin` — 구 rotation (DFST/DKVA
  스터디용; W4A4 baseline으로는 부적격)
- `assets/rotations/llama31_hadamard/R.bin` — 진단용 random-Hadamard
- `assets/draft_rotations/R1D_s1r1.pt{,.best}` — **draft R1_D/R2_D 학습본
  best seed** (SpecForge-parity forward, val CE 7.09→4.35)
- `assets/draft_rotations/R1Donly_s1r0.pt{,.best}` — R2_D 제외 ablation
- `assets/dfst_rc/RC1_reuseRT.pt`, `RC_L0.pt` — DFST 스터디 R_C ckpt
- 원 경로 매핑: 코드가 기대하는 위치는
  `/home/thahn1230/dflash_workspace/outputs/rotations/...` 와
  `runs/<VSQ run>/rotations/draft/...` — clone 후:
  ```bash
  mkdir -p ../outputs && cp -r assets/rotations ../outputs/
  RD=$(cat runs/VSQ_RUN_DIR)
  mkdir -p $RD/rotations/draft && cp assets/draft_rotations/* $RD/rotations/draft/
  PREV=runs/dflash_seagle_transfer_20260807_180238
  mkdir -p $PREV/rotations && cp assets/dfst_rc/* $PREV/rotations/
  ```

재생성 필요 (git 미포함, 스크립트로 복원):
- 학습 corpus: `python -m seagle_port.vsq_build_corpus --out <RD>/tables/train_corpus.jsonl` (~30분)
- hidden 캐시: `python -m seagle_port.vsq_cache_hidden --corpus ... --target-mode w4a4 --rbin <s1 R.bin> --out-dir <RD>/hcache_w4a4_s1` (~30분, 3.7GB)
- QAT pilot ckpt (2.4GB×4)는 미포함 — summary json은 커밋됨

## 3. VSQ 스터디 재개 절차 (여기부터가 남은 일)

전체 spec: 사용자의 "vanilla SpinQuant novelty grid" + "SpecForge 업데이트"
지시문 2건 — 요약이 `runs/<VSQ>/manifests/PROGRESS_NOTE.md`와 아래에 있음.

**확보된 결과 (게이트 전부 PASS, test-clean):**
- T-PPL: fp16 7.44 / RTN 189.9 / Had 10.89 / R1 10.2-10.7 / **R1+R2 8.79-8.85** (Gate C ×3 seeds)
- Draft ladder (block CE): fp16 2.41 / RTN 7.23 / random 4.83 / R1D 4.96 / **R1D+R2D 4.47** (Gate E; 정직: R1D 단독 < random)
- **AL 4-ds mean**: M0 4.202 / M1 ~1.7 / M2 RAW **1.000** / M3 CORRECT **1.836** / M5 R_C-only 3.383 / **M5 P2+R_C 3.676** (잔여 gap 78% 회복)
- Validation(gsm8kvalid): M3 2.017 / +P2 1.902 / +MP3 1.923 / +RC 3.213 / +P2+RC 3.407 / G1(R1_D:=R1_T) 1.734 / G1+RC 3.276 / R_D=I 1.134
- **§20 counterfactual (논문 핵심)**: H_t kurtosis naive 69.5 = **vanilla-SQ 완비 69.4** vs R_C 3.0; A4 NMSE 0.417/0.417/0.019 — W_c fold가 R1_T를 대수적으로 상쇄 → fused-context 루프는 model-local SpinQuant closure 밖 (tables/mechanism_stats.csv)
- RCAL M5: AL_q 2.440 / RCAL 2.191 / AFS 0.915 (tables/rcal__M5_p2rc__mtbench.json)
- QAT LR pilot (Q2, SGD-mom, 100 steps): 3e-3/1e-2 무반응, 3e-2→6.36, **1e-1→5.44** — 재개 시 **3e-1 추가 후 선택**

**재개 시 할 일 (순서):**
1. 스케줄러 기동: `nohup bash <workspace>/launch_vsq_sched.sh &` (경로 수정 필요; **단일 인스턴스만** — double-scheduler가 GPU 이중배정 사고 이력)
2. 킬된 3개 재큐 (fresh id, 부분 artifact 먼저 삭제):
   `rcal_M3_vsq` (cycles/cyc__M3_vsq__mtbench.jsonl replay), `S2CHK__mtbench` (s2 rotation 동등성), `HP2` (ctx A8 대조)
3. QAT pilot에 lr 3e-1 1개 추가 → best LR로 **Q2/Q3/Q5 본학습** (`vsq_train_qat.py`, 400 steps 동일예산, arm별 init만 상이) → 각 ckpt 배포-AL 평가 (eval_al에 QAT-ckpt 로드 경로는 미구현 — RotQuantDraft state_dict 로드 flag 추가 필요, ~20줄)
4. 통계 배터리: 사전등록 P1(M0vM3)/P2(M3vM4)/P3(M4vM5)/P4(M3vM5)/P5(M5vM6-analog)/P6(target-only) — `seagle_port/stats.py` 재사용 (paired bootstrap 3000 + Holm; **여러 json 중복 이름 dedupe 필수**)
5. FIG-A~H (`analyze.py` 패턴 참조), `verify_dflash_spinquant_novelty.py` (missing=0/mismatch=0), 보고서 + `docs/DFLASH_SPINQUANT_NOVELTY_ADVERSARIAL_AUDIT.md` (§35 10문), bundle, commit
6. Novelty 판정: 현 증거로 **CASE D** (M4 정적처방 무효 변형: "SEAGLE-direct는 VSQ 위에서 기여 없음, DFlash-특이 R_C가 지배") + CASE E 기각 (G1 실측) — 최종 표·통계로 확정할 것

## 4. 운영 규칙 (원 세션 표준 — 그대로 적용 권장)

- GPU 8대 상시 포화: `seagle_port/scheduler.py` (artifact-idempotent 큐;
  `SCHED_GPUS` env로 GPU 제외 가능) + 유휴/STALL/GAVE-UP watchdog + 주기 wakeup
- **함정 목록** (전부 실제로 밟았던 것): ① 부분-shard가 SKIP-ARTIFACT로
  오인 → 재큐 전 artifact 삭제 ② 스케줄러 재시작 시 gaveup 상태 소실 →
  done.json 수동 패치 ③ double-scheduler → 기동 전 반드시 pkill (self-match
  주의: pgrep 패턴이 자기 셸과 일치하면 exit 144) ④ `&&…&` 체인에서 변수
  소실 → launcher는 스크립트 파일로 ⑤ cwd 리셋 → 항상 절대경로
  ⑥ eval 시 rotation 재계산 OOM → `freeze_for_eval()` 필수
- 프로토콜: greedy, max_new 1024, B=10, 4-ds = mtbench80(2턴)/gsm8k200/
  humaneval164/sharegpt80 (frozen checksums은 manifests/), 4-ds mean =
  dataset-level tau 평균 (pooling 금지), 방법 선택은 gsm8kvalid만

## 5. 참조 문서

- `runs/<VSQ run>/manifests/PROGRESS_NOTE.md` — 상세 타임라인+재개 플레이북
- `runs/<VSQ run>/tables/` — 전 결과 (target_ppl.csv, draft_quality.csv,
  mechanism_stats.csv, rcal_*, specforge_parity.json, 스냅샷/감사표)
- DFST/DKVA 브랜치의 docs/ — 완결 보고서 2+1건
- upstream 참조 pin: SpecForge @87e8cf4b, DeepSpec @005e03b8
  (manifests/upstream_refs.json; refs/ clone은 미포함 — 재clone)

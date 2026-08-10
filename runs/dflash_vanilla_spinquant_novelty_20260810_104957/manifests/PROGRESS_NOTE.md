# VSQ study progress (2026-08-10)
- Target W4A4KV16 rotations: seeds 0,1 training (official SpinQuant, GPUs 0-7,
  ~70/100 @ last check), out: /home/thahn1230/dflash_workspace/outputs/rotations/llama31_w4a4kv16_s{0,1}
  seed2 pending GPU availability. OLD llama31_w16a4kv16 R.bin = NOT valid W4A4 SpinQuant baseline (wrong objective).
- SpecForge @87e8cf4b, DeepSpec @005e03b8 pinned in /home/thahn1230/dflash_workspace/refs/ (read-only).
- specforge_parity.json: PASS (ported seagle_port/vsq_specforge_port.py bitwise-matches OnlineDFlashModel
  anchors/mask(4D)/noise/positions/labels/weights/loss; intra-block causal only under sliding_window — ours full bidir).
- No draft rotation jobs existed at update time; new draft trainer must build on vsq_specforge_port
  (Operation A: rotations only R1_D/R2_D Cayley + STE W4A4; Operation B: QAT weights Q2/Q3/Q5).
- Upstream has NO Llama-3.1 DFlash config and NO QAT -> label ours "SpecForge-semantics DFlash W4A4 QAT".
- Next: rotation done -> launch seed2 + T-PPL gates (T-PPL0..4, wikitext-8x2048 protocol, gate_b infra reusable);
  then draft R1_D/R2_D trainer; then M0-M8/G/HP/Q families per §10-§20 of update.

## 2026-08-10 저녁 update
- Target W4A4KV16 rotations: s0 DONE (sha ca9f1769), s1 DONE (sha f433a88b), s2 RUNNING (GPU 4-7).
- T-PPL gate jobs queued (tppl_s0/s1/s2; vsq_target_ppl.py — arms fp16/RTN-norot/hadamard/R1only/R1+R2,
  wikitext 16x2048, R1-only minted as R1only.bin; GATE C criterion R1R2 < RTN and < hadamard).
- spinquant_target.py: added w4a4_norot mode + install_act_quant(online_had=) flag.
- seagle_port/vsq_draft_rot.py: RotQuantDraft — draft R1_D/R2_D rotated-quantized forward.
  KEY: γ-fusion forces ctx-specific K/V views as MANDATORY CORRECTNESS (input_ln γ vs hidden_norm γ
  cannot co-fuse into shared k/v); boundaries = e@R1_D (in), bare-norm @ D_γf·R1_D^T (out, shared head).
  Gate-D basis PASS (R=I bits16 == stock, rel 5.8e-7); Gate-D rotation-invariance PASS
  (random R1/R2(+R4): rel 1.5e-6).
- NEXT: (1) read T-PPL results -> select seed (Gate C); (2) draft rotation trainer script
  (Adam on cayley r1/r2 of RotQuantDraft, specforge_block_forward loss γ=5, corpus from
  prev RCCAP gsm8kcalib trajectories + sharegpt-calib gen, H1 = online W4A4 target hidden);
  (3) M-arm eval integration (RotQuantDraft eval mode into eval_al via --draft-transform or new flag);
  (4) Q-family QAT (same forward, weights trainable) after Q1 sanity.
GPU0 exclusion active (SCHED_GPUS=1-7). User request 18:13. GPU 0 re-allowed by user 18:2x — full 8-GPU operation restored.

## == PAUSED BY USER (2026-08-11) ==
State at pause: 59 jobs done, 0 failed. Killed while running (REQUEUE ON RESUME):
  rcal_M3_vsq, S2CHK__mtbench, HP2_rcVSa8__mtbench  (+ QAT pilot x4 never started)
RESUME PLAYBOOK:
 1. bash /home/thahn1230/dflash_workspace/launch_vsq_sched.sh  (single instance!)
 2. Re-append killed jobs to scheduler/queue.jsonl with fresh ids (v2 suffix),
    delete their partial artifacts first (rcal jsonl / S2CHK / HP2 shards if header-only).
 3. Re-arm: saturation monitor (idle/stall/gave-up watchdog) + cron 20min wakeup.
 4. Then: QAT pilot x4 -> pick LR -> Q2/Q3/Q5 mains (vsq_train_qat) -> QAT deployed-AL eval
    -> stats battery (P1-P6+Holm) -> FIG-A..H -> verify_dflash_spinquant_novelty.py
    -> reports (incl. DFLASH_SPINQUANT_NOVELTY_ADVERSARIAL_AUDIT.md) -> bundle -> commit.
KEY RESULTS SO FAR (all in tables/ + logs):
  T-PPL gate: fp16 7.437 / RTN 189.9 / Had 10.89 / R1 10.17-10.68 / R1R2 8.79-8.85 (3 seeds PASS; s2 nominal best, s1 used in chain, delta 0.007 = noise)
  Draft ladder: fp16 CE 2.41 / RTN 7.23 / random 4.83 / R1D 4.96 / R1D+R2D 4.47 (Gate E PASS)
  AL 4-ds mean: M0 4.202 / M1 ~1.7 / M2 RAW 1.000 / M3 1.836 / M5 rc-only 3.383 / M5 p2+rc 3.676
  Validation: M3 2.017 / +P2 1.902 / +MP3 1.923 / +RC 3.213 / +P2+RC 3.407 / G1 1.734 / G1+RC 3.276 / R_D=I 1.134
  RCAL M5: AL_q 2.440 RCAL 2.191 AFS 0.915
  MECH (paper core): H_t kurtosis M1 69.5 = M3 69.4 (vanilla SQ does NOT close the ctx loop; fc fold cancels the rotation) vs M5 R_C 3.0; A4 NMSE 0.417/0.417/0.019

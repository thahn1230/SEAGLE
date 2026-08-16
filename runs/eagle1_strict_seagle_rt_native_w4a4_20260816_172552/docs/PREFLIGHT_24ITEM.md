# Strict SEAGLE-RT — Section-36 pre-flight (24 items)

Printed BEFORE full-training launch. Sources: grounding audit
(workflow wp6hp7csi, 7 agents, 951k tokens), measured benches, gates.

1. **Target model**: meta-llama/Llama-2-7b-chat-hf, pinned rev
   f5db02db..., local snapshot /data/thahn1230/hf_cache/.../f5db02db...
2. **Target W4A4 quantization config**: deployed eval build =
   study.build_study_target(rotation='full',
   rotation_type='learned_chat_w4a4kv16', quant='w4a4', seed=0):
   weights RTN 4-bit per-out-channel symmetric + MSE clip (grid 100,
   maxshrink 0.8; lm_head/embed excluded); activations dynamic
   per-token asymmetric 4-bit, groupsize -1, clip 1.0 (o_proj input
   gs=128; lm_head input 16-bit); KV16; online R4 Hadamard on
   down_proj input.
3. **R1_T/R2_T hashes**: outputs/rotations/learned_chat_w4a4kv16/R.bin
   sha256 4b7e91d2a7531bb8ccde55d35299d3bde38b0f849ebcca951c9320e42a
   48fa6e — contains R1 [4096,4096] fp32 + 32× per-layer self_attn.R2
   [128,128] fp32. (Backup 0a2d1997 NOT used.)
4. **Native hidden tensor source line**:
   third_party/EAGLE/eagle/model/modeling_llama_kv.py:1074
   `hidden_states = self.norm(hidden_states)` → last_hidden_state,
   consumed at ea_model.py:132; norm.weight==ones asserted.
5. **Native hidden mathematical definition**: h_RT_native = a_t =
   RMSNorm0(x_R) = (h/γ_f)@R1_T; shape (B,T,4096) fp16, measured
   RMS 1.0 (γ-excluded confirmed).
6. **R1_T.T restore absent**: Gate NR runtime counters == 0
   (unrotate_hidden / RestoredInterfaceCSAdapter / UnrotateAdapter)
   + comment-stripped static scan clean (tables/parity_gate.json).
7. **Old interface scale restore absent**: no scalar interface scale
   exists in-tree (exhaustive audit); γ_f multiply and
   embed_scale_alpha both excluded by the same gate; γ carried ONLY
   by the fused head (item 15).
8. **Fresh draft initialization**: official cnets.Model(EConfig
   llama_2_chat_7B_config.json, load_emb=True, path=<ORIGINAL target
   safetensors>, bias=True) after torch.manual_seed(seed=0);
   PyTorch-default kaiming-uniform for layers[0]+fc; embedding
   copied+frozen (sha logged at launch); NO checkpoint init of any
   kind (public draft / T0 / anchors / QAT ckpts all unused).
9. **Official training recipe source**: validated
   scripts/train_eagle1_official_fp16.py (reproduced public draft,
   gate PASS docs/EAGLE1_OFFICIAL_FROM_SCRATCH_PTQ_VS_QAT_STUDY.md);
   strict fork scripts/strict_rt/train_eagle1_strict_rt.py, TC
   constants asserted identical.
10. **Dataset size**: official ShareGPT 68k coverage → 66,890 usable
    conversations (1,110 role-alternation skips), 99,303,654 tokens,
    split 63,545/3,345 (95/5) — counts IDENTICAL to the validated
    reproduction. Token cache official_tok_68k_v1.pt (regenerated,
    335 MB).
11. **Epochs**: 21 (official range(num_epochs+1) off-by-one).
12. **Expected total steps**: ~41.7k optimizer steps at world 8
    (63,545/(4·8)=1,985 per epoch × 21; the validated 43.9k was the
    same schedule at mixed world 8/7/6/8).
13. **Optimizer/LR/scheduler**: AdamW lr 3e-5, betas (0.9,0.95),
    weight_decay PyTorch-default 0.01; value-clip 0.5; linear warmup
    2000 over FIXED 800,000-step horizon; bf16 autocast; bs1×accum4
    per rank (official effective per-rank 4).
14. **Feature-loss definition**: SmoothL1(masked token-mean) between
    draft feature f_i and teacher a_{i+1}, weight v_w=1.0; +
    p_w=0.1 soft-CE (item 15). Masked-row form (validated
    math-identical).
15. **Feature-label native-basis transformation**: NONE — labels are
    the raw native a_{i+1} (same tap, shifted). Soft-CE teacher and
    prediction both scored by the DEPLOYED fused head
    W_lm·diag(γ_f)·R1 (native_head.pt sha16 0335833c38f8187c,
    algebra-verified rel<5e-3). tables/
    seagle_rt_training_basis_contract.csv: all rows MATCH.
16. **First-path equation**: f = fc(concat[E_orig(token_{i+1}),
    a_i + u]) with official noise u=(rand−.5)·0.2·512/L (input only);
    embedding half original-basis (official), hidden half native.
17. **Recurrent-path equation**: official v1 training has no
    recurrent rows; at eval the draft recycles its own f (native
    basis by construction) through the SAME shared fc (cnets.py:592),
    scored by base_model.lm_head = the fused head. No folds.
18. **Full-training GPU allocation**: 8× RTX 4090 24GB, one DDP rank
    per GPU with per-rank single-GPU visibility (launcher
    _launch_ddp.sh; avoids the SpinQuant-build cuda:0 leak measured
    at 7×384 MiB). Teacher: offline 495 GiB bit-exact cache (65.3%
    tokens; validation-first admission) + online deployed W4A4
    forward for the uncached remainder (hybrid; Gate P1 128/128
    bit-exact, Gate P2 grad-exact).
19. **Expected wall time**: 2.802 s/step × 41,706 steps ≈ 32.5 h wall (world-8 single-visibility,
    hybrid teacher, grad-ckpt on, sync-every = validated semantics;
    scaling measured w1 0.953 / w2 1.517 / w4 2.097; see
    PREFLIGHT_SPEED_20ITEM.md).
20. **Expected GPU-hours**: ≈ 260 GPU-h training (32.5 h × 8) + measured so far: cache
    generation 4.66 GPU-h, benches/gates ≈ 3 GPU-h, tokenize CPU-only.
21. **Checkpoint-selection rule**: FINAL checkpoint (the validated
    official-reproduction rule; official v1 has no best-ckpt
    selection). Best-val saved for reference only; c4-calib pool
    reserved for Stage-B selection tasks; final 4 datasets never used
    for selection (Gate I).
22. **Final four-dataset manifests**: mtbench:80 (sha 82137e5372f5ff7a),
    gsm8k:200 (6a94d85419e8cd66), sharegpt:80 (96060988a4674227),
    humaneval:164 (fd806a46bdb75c16) — regenerated + verified via
    write_or_verify_manifest + Gate F no-overlap before Stage A.
23. **W8A8/W4A4 follow-up plan**: single selected ckpt → arms
    SRT_W8A8_RTN / SRT_W8A8_SQ / SRT_W4A4_RTN / SRT_W4A4_HAD /
    SRT_W4A4_SQ (Cayley R_D=R_init·C(A) only; FullRotation+QR
    forbidden) / optional SRT_RESCUE; pure-RTN baselines carry NO
    alpha/GS/LS/D4P3/R5/R2/R4 (Gate L); component sensitivity +
    projection stats if W4A4 drops; RCAL for headline arms;
    all-GPU parallel dispatch per amendment §12-14.
24. **CASE A-E decision criteria**: preregistered in
    configs/preregistration.md (P1-P7 bootstrap 10k + Holm;
    comparable = P1 n.s. on ≥3/4 datasets AND |Δmean4|≤0.06;
    collapse = retention<60%; material = retention<90% w/ P4 sig).

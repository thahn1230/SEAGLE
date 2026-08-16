# Strict SEAGLE-RT — speed-amendment §20 printout (measured)

1. online W4A4 teacher: 0.376 s/conv (4,142 tok/s, GPU0 bench, 24
   convs mean len 1558) = 2.4× the FP16 teacher (0.157 s/conv).
   Online-mode step cost ≈ 4×0.376 = 1.50 s/step teacher-only.
2. cached-teacher: bit-exact mmap reads; measured data_wait 14.7% of a
   3.6 s step at w8 (≈0.5 s incl. the online 35% remainder); pure
   cached reads are fully hidden by the 2-deep prefetch at w1-w4.
3. optimized cached-loader: pinned-memory + threaded prefetch +
   non_blocking H2D (CachedTeacher); same numbers as (2) — the loader
   is not the bottleneck (NVMe ~reads 380 MB/step aggregate).
4. speedup ratio (teacher compute eliminated): 21 epochs × cached
   fraction 65.3% → teacher recompute cut from 21 full passes to
   1 pass + 21×34.7% ≈ 8.3 passes-equivalent (2.5× teacher-compute
   reduction; disk-bound ceiling per §17: full cache 757.6 GiB >
   528.7 GiB free after the user-approved 391 GB prune).
5. cache generation wall time: 35.2 min on 8 GPUs = 4.66 GPU-h
   (43,705 convs / 64.88 M tokens, exactly once each).
6. cache size: 495.0 GiB in 128 parts (~4 GiB each), fp16 raw,
   manifest-merged, per-conv sha16.
7. disk throughput: /data NVMe sustained the 8-writer generation at
   ~240 MB/s aggregate write; training reads ~380 MB/step aggregate
   (well under device limits); /data 34 GiB free after cache.
8. 8-GPU DDP s/step: measured ladder (hybrid teacher):
   w1 0.953 / w2 1.517 / w4 2.097 (ckpt-on, sync-every);
   w8 ckpt-off no_sync single-visibility 3.646 (NCCL probe: SHM
   confirmed; the short un-checkpointed backward simply cannot hide
   the all-reduce, so comm is exposed);
   w8 ckpt-on single-visibility (slimmed teacher, sync-every =
   validated semantics): 2.802 s/step, peak 20.06 GiB, no OOM
   ← FINAL launch mode.
   Bottleneck: no-P2P RTX-4090 allreduce of 940 MB fp32 grads;
   no_sync (1 allreduce/step) adopted only where static_graph allows.
9. predicted full-schedule wall time: 21 epochs = 41,706 steps ×
   2.802 s/step = 32.5 h (+ val ~75 s / 1000 steps, ckpt
   ~15 s / 2000 steps ≈ +1%).
10. predicted total study wall time: training + Stage A (4 evals ≈
    1 h wall parallel) + Stage B (rotation training + ladder evals
    ≈ 5-7 h) + RCAL/stats/report ≈ 42-45 h (train 32.5-33 + Stage A ~1.5 + Stage B ~6-7 + stats ~2).
11. optimizations enabled: offline bit-exact fp16 feature cache
    (validation-first admission); hybrid online remainder through the
    exact deployed build; threaded pinned prefetch; per-step (not
    per-micro-batch) allreduce where static_graph permits; staggered
    teacher builds (transient 24 GB peaks); expandable_segments;
    checkpoints on /data. NOT enabled (bench-rejected or
    numerics-risk): torch.compile, TF32, fused AdamW, bf16 comm
    hooks, grad-ckpt-off at w8 (NCCL visibility trade-off).
12. parity checks: Gate P1 cache↔online 128/128 BIT-EXACT
    (max_abs 0.0); Gate P2 one full training step loss+grads diff
    exactly 0.0; Gate NR restore-op counters 0 + static scan clean
    (tables/parity_gate.json).
13. scientific definition unchanged: fresh init, official recipe/
    constants asserted, 21 epochs, W4A4 SpinQuant teacher, native
    rotated feature both input and label, loss/dataset/selection rule
    identical (docs/PREFLIGHT_24ITEM.md items 8-17, 21).

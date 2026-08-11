# Tensor locations (basis-audit-verified)
| item | value |
|---|---|
| target hidden layer IDs | [1, 8, 15, 22, 29] (H1..H5) |
| hidden size / concat width | 4096 / 20480 |
| W_c shape | fc.weight [4096, 20480] (dflash/model.py:317) |
| pre-R1 H_i | rot-target hidden_states[l+1] @ R1_T^T (exact orthogonal unrotation; basis audit S1) |
| post-R1 H_i | rot_fp16/w4a4 target output.hidden_states[l+1] (deployed basis) |
| stock W_c | DFlashDraftModel.fc.weight (z-lab ckpt) |
| folded W_c | interfaces.fold_wc: per-block Wc_i @ R1_T (fp64 staging) |
| pre-R_C H_t | RotQuantDraft forward tap S3_Ht_dep (bare-rms, ORIGINAL basis — fold cancels R1_T) |
| post-R_C H_t | tap S3_Ht_rc_dep = deployed `Ht = Ht @ rc_matrix_buf` (class A, runtime) |
| draft layers | 5 |
| sample seed / plot rows | 0 / 256 pooled (evenly-spaced, pre-registered rule) |

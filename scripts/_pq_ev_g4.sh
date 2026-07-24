set -uo pipefail
cd /home/thahn1230/eagle_spinquant_w4a4
python scripts/eval_eagle_acceptance_length.py --run-dir runs/eagle_ptq_vs_qat_al_20260723_143129 --n-prompts 80 --datasets mtbench --target fp16 --draft-cfg rot --ckpt runs/eagle_lk_exactpath_draft_rotation_20260721_151703/rotations/LK2_HYBRID_s2.pt --tag C11_fp16_rot
python scripts/eval_eagle_acceptance_length.py --run-dir runs/eagle_ptq_vs_qat_al_20260723_143129 --n-prompts 80 --datasets mtbench --target int4 --draft-cfg p2 --tag N3b_int4_p2
echo [pq_ev_g4] DONE

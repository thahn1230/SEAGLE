set -uo pipefail
cd /home/thahn1230/eagle_spinquant_w4a4
python scripts/eval_eagle_acceptance_length.py --run-dir runs/eagle_ptq_vs_qat_al_20260723_143129 --n-prompts 80 --datasets mtbench --target fp16 --draft-cfg d4p3 --tag C2_fp16_d4p3
python scripts/eval_eagle_acceptance_length.py --run-dir runs/eagle_ptq_vs_qat_al_20260723_143129 --n-prompts 80 --datasets mtbench --target fp16 --draft-cfg p2 --tag N3_fp16_p2
echo [pq_ev_g1] DONE

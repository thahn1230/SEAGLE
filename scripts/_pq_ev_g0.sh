set -uo pipefail
cd /home/thahn1230/eagle_spinquant_w4a4
python scripts/eval_eagle_acceptance_length.py --run-dir runs/eagle_ptq_vs_qat_al_20260723_143129 --n-prompts 80 --datasets mtbench --target fp16 --draft-cfg stock --tag C1_fp16_stock
python scripts/eval_eagle_acceptance_length.py --run-dir runs/eagle_ptq_vs_qat_al_20260723_143129 --n-prompts 80 --datasets mtbench --target fp16 --draft-cfg naive_w4a4 --tag N1_fp16_naive
echo [pq_ev_g0] DONE

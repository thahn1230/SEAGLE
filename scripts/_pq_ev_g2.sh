set -uo pipefail
cd /home/thahn1230/eagle_spinquant_w4a4
python scripts/eval_eagle_acceptance_length.py --run-dir runs/eagle_ptq_vs_qat_al_20260723_143129 --n-prompts 80 --datasets mtbench --target int4 --draft-cfg stock --tag C4_int4_stockrestored
python scripts/eval_eagle_acceptance_length.py --run-dir runs/eagle_ptq_vs_qat_al_20260723_143129 --n-prompts 80 --datasets mtbench --target int4 --draft-cfg naive_w4a4 --tag N2_int4_naive
echo [pq_ev_g2] DONE

set -uo pipefail
cd /home/thahn1230/eagle_spinquant_w4a4
python scripts/eval_eagle_acceptance_length.py --run-dir runs/eagle_ptq_vs_qat_al_20260723_143129 --n-prompts 20 --datasets c4 --pool calib --target fp16 --draft-cfg d4p3 --alpha 16 --tag CAL_fp16_a16
python scripts/eval_eagle_acceptance_length.py --run-dir runs/eagle_ptq_vs_qat_al_20260723_143129 --n-prompts 20 --datasets c4 --pool calib --target fp16 --draft-cfg d4p3 --alpha 22.627417 --tag CAL_fp16_a22.627417
python scripts/eval_eagle_acceptance_length.py --run-dir runs/eagle_ptq_vs_qat_al_20260723_143129 --n-prompts 20 --datasets c4 --pool calib --target fp16 --draft-cfg d4p3 --alpha 32 --tag CAL_fp16_a32
python scripts/eval_eagle_acceptance_length.py --run-dir runs/eagle_ptq_vs_qat_al_20260723_143129 --n-prompts 20 --datasets c4 --pool calib --target fp16 --draft-cfg d4p3 --alpha 45.254834 --tag CAL_fp16_a45.254834
python scripts/eval_eagle_acceptance_length.py --run-dir runs/eagle_ptq_vs_qat_al_20260723_143129 --n-prompts 20 --datasets c4 --pool calib --target fp16 --draft-cfg d4p3 --alpha 64 --tag CAL_fp16_a64
python scripts/eval_eagle_acceptance_length.py --run-dir runs/eagle_ptq_vs_qat_al_20260723_143129 --n-prompts 20 --datasets c4 --pool calib --target int4 --draft-cfg d4p3 --alpha 16 --tag CAL_int4_a16
python scripts/eval_eagle_acceptance_length.py --run-dir runs/eagle_ptq_vs_qat_al_20260723_143129 --n-prompts 20 --datasets c4 --pool calib --target int4 --draft-cfg d4p3 --alpha 22.627417 --tag CAL_int4_a22.627417
python scripts/eval_eagle_acceptance_length.py --run-dir runs/eagle_ptq_vs_qat_al_20260723_143129 --n-prompts 20 --datasets c4 --pool calib --target int4 --draft-cfg d4p3 --alpha 32 --tag CAL_int4_a32
python scripts/eval_eagle_acceptance_length.py --run-dir runs/eagle_ptq_vs_qat_al_20260723_143129 --n-prompts 20 --datasets c4 --pool calib --target int4 --draft-cfg d4p3 --alpha 45.254834 --tag CAL_int4_a45.254834
python scripts/eval_eagle_acceptance_length.py --run-dir runs/eagle_ptq_vs_qat_al_20260723_143129 --n-prompts 20 --datasets c4 --pool calib --target int4 --draft-cfg d4p3 --alpha 64 --tag CAL_int4_a64
echo [pq_ev_g5] DONE

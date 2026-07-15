set -euo pipefail
cd /home/thahn1230/eagle_spinquant_w4a4
BASE=/home/thahn1230/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9
OFF_R=/data/thahn1230/spinquant_official_rotations/LLaMA-2-7B/7B_W4A4KV16_lr_1.5_seed_0/R.bin
Q="--w_bits 4 --a_bits 4 --k_bits 16 --v_bits 16 --w_clip --a_asym --k_asym --v_asym --k_groupsize 128 --v_groupsize 128 --rotate --optimized_rotation_path $OFF_R"
bash scripts/_run_official_ptq.sh "$BASE" BASE_03b_gptq_actorder $Q --act_order
echo "[gptqDiag] ALL DONE"

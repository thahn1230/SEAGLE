set -euo pipefail
cd /home/thahn1230/eagle_spinquant_w4a4
CHAT=/home/thahn1230/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-chat-hf/snapshots/f5db02db724555f92da89c216ac04704f23d4590
LR=/data/thahn1230/spinquant_rotations/chat_w4a4kv16/R.bin
# wait for training to write R.bin (and the trainer to exit)
until [ -f "$LR" ]; do sleep 120; done
while pgrep -f "[o]ptimize_rotation.py" >/dev/null; do sleep 60; done
Q4="--w_bits 4 --a_bits 4 --k_bits 16 --v_bits 16 --w_clip --a_asym --k_asym --v_asym --k_groupsize 128 --v_groupsize 128 --rotate --optimized_rotation_path $LR"
Q8="--w_bits 8 --a_bits 8 --k_bits 16 --v_bits 16 --w_clip --a_asym --k_asym --v_asym --k_groupsize 128 --v_groupsize 128 --rotate --optimized_rotation_path $LR"
bash scripts/_run_official_ptq.sh "$CHAT" CHAT_03_learned_rtn_w4a4  $Q4 --w_rtn
bash scripts/_run_official_ptq.sh "$CHAT" CHAT_04_learned_gptq_w4a4 $Q4
bash scripts/_run_official_ptq.sh "$CHAT" CHAT_05_learned_rtn_w8a8  $Q8 --w_rtn
echo "[chatTrio] ALL DONE"

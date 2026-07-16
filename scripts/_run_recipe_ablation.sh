set -euo pipefail
cd /home/thahn1230/eagle_spinquant_w4a4
CHAT=/home/thahn1230/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-chat-hf/snapshots/f5db02db724555f92da89c216ac04704f23d4590
LR=/data/thahn1230/spinquant_rotations/chat_w4a4kv16/R.bin
B="--k_bits 16 --v_bits 16 --k_asym --v_asym --k_groupsize 128 --v_groupsize 128 --rotate --optimized_rotation_path $LR --w_rtn"
bash scripts/_run_official_ptq.sh "$CHAT" abl_w8a16 $B --w_bits 8  --a_bits 16 --w_clip --a_asym
bash scripts/_run_official_ptq.sh "$CHAT" abl_w16a8 $B --w_bits 16 --a_bits 8  --w_clip --a_asym
bash scripts/_run_official_ptq.sh "$CHAT" abl_w4a16 $B --w_bits 4  --a_bits 16 --w_clip --a_asym
bash scripts/_run_official_ptq.sh "$CHAT" abl_w16a4 $B --w_bits 16 --a_bits 4  --w_clip --a_asym
bash scripts/_run_official_ptq.sh "$CHAT" abl_w4a4_noclip $B --w_bits 4 --a_bits 4 --a_asym
bash scripts/_run_official_ptq.sh "$CHAT" abl_w4a4_asym   $B --w_bits 4 --a_bits 4 --w_clip
echo "[ablRecipe] ALL DONE"

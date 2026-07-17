set -euo pipefail
cd /home/thahn1230/eagle_spinquant_w4a4
CHAT=/home/thahn1230/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-chat-hf/snapshots/f5db02db724555f92da89c216ac04704f23d4590
LR=outputs/rotations/learned_chat_w4a4kv16/R.bin
RH=$(python -c "
import sys; sys.path.insert(0,'src')
from eagle_spinquant import experiment, study
p=experiment.resolve_paths(experiment.load_config(None))
print(study.r_bin_path('random_hadamard',0,p['target_path'],experiment.load_config(None).get('paths',{}).get('rotations_root')))")
Q="--k_bits 16 --v_bits 16 --k_asym --v_asym --k_groupsize 128 --v_groupsize 128 --rotate --w_rtn --w_clip --a_asym"
bash scripts/_run_official_ptq.sh "$CHAT" tldr_chat_fp16 --w_bits 16 --a_bits 16 --k_bits 16 --v_bits 16
bash scripts/_run_official_ptq.sh "$CHAT" tldr_chat_learned_w8a8 $Q --w_bits 8 --a_bits 8 --optimized_rotation_path "$LR"
bash scripts/_run_official_ptq.sh "$CHAT" tldr_chat_learned_w4a4 $Q --w_bits 4 --a_bits 4 --optimized_rotation_path "$LR"
bash scripts/_run_official_ptq.sh "$CHAT" tldr_chat_rh0_w4a4     $Q --w_bits 4 --a_bits 4 --optimized_rotation_path "$RH"
echo "[tldrB1] ALL DONE"

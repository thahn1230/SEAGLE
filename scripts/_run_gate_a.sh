set -euo pipefail
cd /home/thahn1230/eagle_spinquant_w4a4
BASE=$(python -c "from huggingface_hub import snapshot_download; print(snapshot_download('meta-llama/Llama-2-7b-hf'))")
CHAT=$(python -c "
import sys; sys.path.insert(0,'src')
from eagle_spinquant import experiment
print(experiment.resolve_paths(experiment.load_config(None))['target_path'])")
echo "base=$BASE"; echo "chat=$CHAT"
bash scripts/_run_official_ptq.sh "$BASE" base_fp16_official --w_bits 16 --a_bits 16 --k_bits 16 --v_bits 16
bash scripts/_run_official_ptq.sh "$BASE" base_fp16_official_rerun --w_bits 16 --a_bits 16 --k_bits 16 --v_bits 16
bash scripts/_run_official_ptq.sh "$CHAT" chat_fp16_official --w_bits 16 --a_bits 16 --k_bits 16 --v_bits 16
echo "[gateA] ALL DONE"

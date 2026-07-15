set -euo pipefail
cd /home/thahn1230/eagle_spinquant_w4a4
BASE=/home/thahn1230/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9
CHAT=/home/thahn1230/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-chat-hf/snapshots/f5db02db724555f92da89c216ac04704f23d4590
# pre-generate all seed R.bins (CPU-light; auto via r_bin_path)
python - <<'PY'
import sys; sys.path.insert(0,'src')
from eagle_spinquant import experiment, study
p=experiment.resolve_paths(experiment.load_config(None))
rr=experiment.load_config(None).get('paths',{}).get('rotations_root')
for s in range(10):
    print(s, study.r_bin_path('random_hadamard', s, p['target_path'], rr))
PY
Q="--w_bits 4 --a_bits 4 --k_bits 16 --v_bits 16 --w_clip --a_asym --k_asym --v_asym --k_groupsize 128 --v_groupsize 128 --rotate --w_rtn"
for s in 0 1 2 3 4 5 6 7 8 9; do
  R=$(python -c "
import sys; sys.path.insert(0,'src')
from eagle_spinquant import experiment, study
p=experiment.resolve_paths(experiment.load_config(None))
print(study.r_bin_path('random_hadamard',$s,p['target_path'],experiment.load_config(None).get('paths',{}).get('rotations_root')))")
    bash scripts/_run_official_ptq.sh "$CHAT" sweep_chat_seed${s} $Q --optimized_rotation_path "$R"
done
echo "[sweepChat] ALL DONE"

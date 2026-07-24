set -uo pipefail
cd /home/thahn1230/eagle_spinquant_w4a4
while pgrep -f "_lk_scr_[g]0.sh" >/dev/null; do sleep 60; done
python scripts/build_target_generated_lk_corpus.py --teacher t4kv4 --mode greedy --n-windows 5000 --tag med_t4kv4_greedy --run-dir runs/eagle_lk_exactpath_draft_rotation_20260721_151703 --device cuda:0 --seed 1
echo "[medcorpus] DONE"

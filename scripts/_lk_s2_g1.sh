set -uo pipefail
cd /home/thahn1230/eagle_spinquant_w4a4
while [ ! -f runs/eagle_lk_exactpath_draft_rotation_20260721_151703/manifests/lkcorpus__med_t4kv4_greedy.json ]; do sleep 60; done
while pgrep -f "_lk_scr_[g]1.sh" >/dev/null; do sleep 60; done
while pgrep -f "_lk_pilot_[g]1.sh" >/dev/null; do sleep 60; done
while pgrep -f "_lk_conf_[g]1.sh" >/dev/null; do sleep 60; done
while pgrep -f "_lk_confb_[g]1.sh" >/dev/null; do sleep 60; done
python scripts/train_eagle_lk_rotation.py --run-dir runs/eagle_lk_exactpath_draft_rotation_20260721_151703 --corpus runs/eagle_lk_exactpath_draft_rotation_20260721_151703/manifests/lkcorpus__med_t4kv4_greedy.json --batch 32 --accum 1 --steps 3000 --eval-every 500 --teacher t4kv4 --objective hybrid --rot residual --seed 1 --out runs/eagle_lk_exactpath_draft_rotation_20260721_151703/rotations/LK2_HYBRID_s1.pt
python scripts/train_eagle_lk_rotation.py --run-dir runs/eagle_lk_exactpath_draft_rotation_20260721_151703 --corpus runs/eagle_lk_exactpath_draft_rotation_20260721_151703/manifests/lkcorpus__med_t4kv4_greedy.json --batch 32 --accum 1 --steps 3000 --eval-every 500 --teacher t4kv4 --objective tv --rot residual --seed 0 --out runs/eagle_lk_exactpath_draft_rotation_20260721_151703/rotations/LK2_TV_s0.pt
echo "[s2_g1] DONE"

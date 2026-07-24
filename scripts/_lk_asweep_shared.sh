set -uo pipefail
cd /home/thahn1230/eagle_spinquant_w4a4
python scripts/eval_eagle_lk_tree.py --lk-ckpt shared --tag ASWEEP_shared_a8 --target t4kv4 --datasets mtbench --n-prompts 20 --alpha-override 8 --run-dir runs/eagle_lk_exactpath_draft_rotation_20260721_151703
python scripts/eval_eagle_lk_tree.py --lk-ckpt shared --tag ASWEEP_shared_a11.3137 --target t4kv4 --datasets mtbench --n-prompts 20 --alpha-override 11.3137 --run-dir runs/eagle_lk_exactpath_draft_rotation_20260721_151703
python scripts/eval_eagle_lk_tree.py --lk-ckpt shared --tag ASWEEP_shared_a16 --target t4kv4 --datasets mtbench --n-prompts 20 --alpha-override 16 --run-dir runs/eagle_lk_exactpath_draft_rotation_20260721_151703
python scripts/eval_eagle_lk_tree.py --lk-ckpt shared --tag ASWEEP_shared_a22.6274 --target t4kv4 --datasets mtbench --n-prompts 20 --alpha-override 22.6274 --run-dir runs/eagle_lk_exactpath_draft_rotation_20260721_151703
python scripts/eval_eagle_lk_tree.py --lk-ckpt shared --tag ASWEEP_shared_a32 --target t4kv4 --datasets mtbench --n-prompts 20 --alpha-override 32 --run-dir runs/eagle_lk_exactpath_draft_rotation_20260721_151703
python scripts/eval_eagle_lk_tree.py --lk-ckpt shared --tag ASWEEP_shared_a45.2548 --target t4kv4 --datasets mtbench --n-prompts 20 --alpha-override 45.2548 --run-dir runs/eagle_lk_exactpath_draft_rotation_20260721_151703
python scripts/eval_eagle_lk_tree.py --lk-ckpt shared --tag ASWEEP_shared_a64 --target t4kv4 --datasets mtbench --n-prompts 20 --alpha-override 64 --run-dir runs/eagle_lk_exactpath_draft_rotation_20260721_151703
python scripts/eval_eagle_lk_tree.py --lk-ckpt shared --tag ASWEEP_shared_a90.5097 --target t4kv4 --datasets mtbench --n-prompts 20 --alpha-override 90.5097 --run-dir runs/eagle_lk_exactpath_draft_rotation_20260721_151703
python scripts/eval_eagle_lk_tree.py --lk-ckpt shared --tag ASWEEP_shared_a128 --target t4kv4 --datasets mtbench --n-prompts 20 --alpha-override 128 --run-dir runs/eagle_lk_exactpath_draft_rotation_20260721_151703
echo [asweep_shared] DONE

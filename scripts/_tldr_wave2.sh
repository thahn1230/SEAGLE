set -euo pipefail
cd /home/thahn1230/eagle_spinquant_w4a4
O=runs/eagle_target_logit_draft_rotation_w4a4kv4_20260717_170003/rotations
BEST=deployKL+rank+feature+self
# specialization (G5/G7): per-target full-objective rotations
python scripts/train_eagle_draft_rotation.py --cache $O/traincache__t8__wiki-c4-sharegpt-gsm8k-code.pt    --out $O/DROT_EAGLE_T8.pt    --objective $BEST --steps 400 --device cuda:0
python scripts/train_eagle_draft_rotation.py --cache $O/traincache__t4__wiki-c4-sharegpt-gsm8k-code.pt    --out $O/DROT_EAGLE_T4.pt    --objective $BEST --steps 400 --device cuda:0
python scripts/train_eagle_draft_rotation.py --cache $O/traincache__t4kv4__wiki-c4-sharegpt-gsm8k-code.pt --out $O/DROT_EAGLE_T4KV4.pt --objective $BEST --steps 400 --device cuda:0
# mixed-target universal (G6/G7): concat caches
python - <<'PY'
import torch, os
O="runs/eagle_target_logit_draft_rotation_w4a4kv4_20260717_170003/rotations"
ws=[]
for t in ("t8","t4","t4kv4"):
    c=torch.load(f"{O}/traincache__{t}__wiki-c4-sharegpt-gsm8k-code.pt", map_location="cpu", weights_only=False)
    ws += c["windows"][:350]
meta=c["meta"]; meta["target"]="mixed"; meta["n"]=len(ws)
torch.save(dict(windows=ws, meta=meta), f"{O}/traincache__mixed.pt")
print("mixed cache", len(ws))
PY
python scripts/train_eagle_draft_rotation.py --cache $O/traincache__mixed.pt --out $O/DROT_EAGLE_MIXED.pt --objective $BEST --steps 500 --device cuda:0
# G2 random-init + wiki-only (overfit control) on t4
python scripts/train_eagle_draft_rotation.py --cache $O/traincache__t4__wiki-c4-sharegpt-gsm8k-code.pt --out $O/DROT_T4_randominit.pt --objective $BEST --init random --steps 400 --device cuda:0
python scripts/train_eagle_draft_rotation.py --cache $O/traincache__t4__wiki.pt --out $O/DROT_T4_wikionly.pt --objective $BEST --steps 400 --device cuda:0
echo "[wave2] ALL DONE"

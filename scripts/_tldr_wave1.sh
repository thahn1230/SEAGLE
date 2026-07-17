set -euo pipefail
cd /home/thahn1230/eagle_spinquant_w4a4
C=runs/eagle_target_logit_draft_rotation_w4a4kv4_20260717_170003/rotations/traincache__t4__wiki-c4-sharegpt-gsm8k-code.pt
FC=runs/eagle_target_logit_draft_rotation_w4a4kv4_20260717_170003/rotations/traincache__fp16__wiki-c4-sharegpt-gsm8k-code.pt
O=runs/eagle_target_logit_draft_rotation_w4a4kv4_20260717_170003/rotations
for OBJ in selfrecon deployKL deployKL+targetCE deployKL+rank deployKL+feature deployKL+rank+feature deployKL+rank+feature+self; do
  python scripts/train_eagle_draft_rotation.py --cache $C --out $O/abl__t4__${OBJ//+/_}.pt --objective $OBJ --steps 400 --device cuda:0
done
python scripts/train_eagle_draft_rotation.py --cache $C --fp16-cache $FC --out $O/abl__t4__full_dual.pt --objective deployKL+rank+feature+self+fptarget --steps 400 --device cuda:0
# gamma + tau sweeps on the primary objective
for G in 0.5 0.85 1.0; do
  python scripts/train_eagle_draft_rotation.py --cache $C --out $O/abl__t4__full_gamma${G}.pt --objective deployKL+rank+feature+self --gamma $G --steps 400 --device cuda:0
done
for TAU in 1.0 4.0; do
  python scripts/train_eagle_draft_rotation.py --cache $C --out $O/abl__t4__full_tau${TAU}.pt --objective deployKL+rank+feature+self --tau $TAU --steps 400 --device cuda:0
done
echo "[wave1] ALL DONE"

#!/bin/bash
# R_D EP3-P confirmatory stage. Usage: _rd_confirm.sh <FINALIST_TAG>
# 1. RANDPERT control: random skew matched in geodesic norm to the
#    finalist, c4 calib 20 AL.
# 2. Finalist mtbench 80 AL (official tau).
# 3. RCAL captures (mtbench 40, lockstep 2-GPU) for BASE_EP3P,
#    finalist, LKXFER + metrics + paired bootstrap 3000.
set -uo pipefail
cd /home/thahn1230/eagle_spinquant_w4a4
FIN=$1
RD=$(cat runs/RDROT_RUN_DIR)
LKRD=$(cat runs/LK_RUN_DIR)
AF=$(python3 -c "print(4096**0.40)")
AR=$(python3 -c "print(4096**0.45)")

# ---- RANDPERT generation (matched ||A||_F, seed 123)
python3 - "$RD" "$FIN" <<'EOF'
import sys, torch
rd, fin = sys.argv[1], sys.argv[2]
ck = torch.load(f"{rd}/rotations/{fin}.pt", map_location="cpu",
                weights_only=False)
R_D = ck["R_D"].double()
import os, glob, json
sys.path.insert(0, "src")
from eagle_spinquant import experiment, study
cfg = experiment.load_config(None)
paths = experiment.resolve_paths(cfg)
rr = cfg.get("paths", {}).get("rotations_root")
R = torch.load(study.r_bin_path("learned_chat_w4a4kv16", 0,
                                paths["target_path"], rr),
               map_location="cpu", weights_only=False)
R_T = R["R1"].double()
Q = R_T.t() @ R_D
I = torch.eye(Q.shape[0], dtype=torch.float64)
A_fin = 2.0 * torch.linalg.solve((I + Q).t(), (Q - I).t()).t()
nrm = float(torch.linalg.norm(A_fin))
g = torch.Generator().manual_seed(123)
W = torch.randn(Q.shape, generator=g, dtype=torch.float64)
A_r = W - W.t()
A_r *= nrm / float(torch.linalg.norm(A_r))
Qr = torch.linalg.solve(I - A_r / 2, I + A_r / 2)
R_rand = R_T @ Qr
oe = float(torch.linalg.norm(R_rand @ R_rand.t() - I))
assert oe < 1e-8, oe
torch.save(dict(R_D=R_rand.float(), alpha=ck.get("alpha"),
                alpha_rec=ck.get("alpha_rec"),
                meta=dict(control="randpert", matched_to=fin,
                          A_fro=nrm, seed=123)),
           f"{rd}/rotations/RANDPERT.pt")
print(f"[randpert] ||A||_F={nrm:.4f} orth_err={oe:.2e} saved")
EOF

# ---- parallel: finalist mtbench AL (GPU 1) + RANDPERT c4 (GPU 2)
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
  python scripts/eval_eagle_acceptance_length.py --target int4 \
  --draft-cfg rot_ep3p --ckpt "$RD/rotations/$FIN.pt" \
  --alpha "$AF" --alpha-rec "$AR" --tag "$FIN" \
  --datasets mtbench --pool eval --n-prompts 80 --run-dir "$RD" \
  > "$RD/logs/confirm_${FIN}_mtbench.log" 2>&1 &
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
  python scripts/eval_eagle_acceptance_length.py --target int4 \
  --draft-cfg rot_ep3p --ckpt "$RD/rotations/RANDPERT.pt" \
  --alpha "$AF" --alpha-rec "$AR" --tag RANDPERT \
  --datasets c4 --pool calib --n-prompts 20 --run-dir "$RD" \
  > "$RD/logs/confirm_RANDPERT_c4.log" 2>&1 &

# ---- RCAL captures: 2 GPUs each; (3,4) (5,6) sequential pairs then
#      finalist on (3,4) after BASE
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3,4 \
  python scripts/capture_eagle_proposal_cycles.py --target int4 \
  --draft-cfg d4p3_deploy --alpha "$AF" --alpha-rec "$AR" \
  --tag VBASE_EP3P --datasets mtbench --n-prompts 40 \
  --run-dir "$RD" --ref-device cuda:1 \
  > "$RD/logs/cap_BASE.log" 2>&1 &
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=5,6 \
  python scripts/capture_eagle_proposal_cycles.py --target int4 \
  --draft-cfg rot_ep3p --ckpt "$LKRD/rotations/LK2_HYBRID_s2.pt" \
  --alpha "$AF" --alpha-rec "$AR" \
  --tag VLKXFER --datasets mtbench --n-prompts 40 \
  --run-dir "$RD" --ref-device cuda:1 \
  > "$RD/logs/cap_LKXFER.log" 2>&1 &
wait

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3,4 \
  python scripts/capture_eagle_proposal_cycles.py --target int4 \
  --draft-cfg rot_ep3p --ckpt "$RD/rotations/$FIN.pt" \
  --alpha "$AF" --alpha-rec "$AR" \
  --tag "V$FIN" --datasets mtbench --n-prompts 40 \
  --run-dir "$RD" --ref-device cuda:1 \
  > "$RD/logs/cap_FIN.log" 2>&1
python scripts/compute_eagle_rcal_metrics.py --run-dir "$RD" \
  > "$RD/logs/rcal_metrics.log" 2>&1
python scripts/bootstrap_eagle_rcal.py --run-dir "$RD" \
  --tags "VBASE_EP3P,V$FIN,VLKXFER" \
  --pairs "VBASE_EP3P:V$FIN,VBASE_EP3P:VLKXFER" \
  --dataset mtbench --reps 3000 \
  > "$RD/logs/rcal_bootstrap.log" 2>&1
echo "[confirm] DONE"

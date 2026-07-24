cd /home/thahn1230/eagle_spinquant_w4a4
while pgrep -f "scripts/_pq_ev_g[0-9].sh" > /dev/null; do sleep 60; done
RD=$(cat runs/PQ_RUN_DIR)
env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 TMPDIR=/data/thahn1230/tmp PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/train_eagle_draft_int4_qat.py --arm C3 --seed 0 --run-dir $RD --steps 30 --val-every 30 --tag SMOKE_C3 > outputs/pq_smoke_c3.log 2>&1
env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 TMPDIR=/data/thahn1230/tmp PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/train_eagle_draft_int4_qat.py --arm C7 --seed 0 --run-dir $RD --steps 8 --val-every 8 --tag SMOKE_C7 > outputs/pq_smoke_c7.log 2>&1 &
env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 TMPDIR=/data/thahn1230/tmp python scripts/check_qat_deploy_parity.py --run-dir $RD --mode both --alpha 45.254834 > outputs/pq_gateD.log 2>&1 &
wait
echo SMOKE_CHAIN_DONE

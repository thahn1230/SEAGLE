cd /home/thahn1230/eagle_spinquant_w4a4
while pgrep -f "tag C7lr_s0_qat" > /dev/null; do sleep 60; done
RD=$(cat runs/PQ_RUN_DIR)
env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 TMPDIR=/data/thahn1230/tmp PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/train_eagle_draft_int4_qat.py --run-dir $RD --alpha 45.254834 --lr 3e-6 --warmup 200 --steps 3000 --arm C7 --seed 2 --tag C7lr_s2 > outputs/pq_c7lr_s2.log 2>&1
EV="python scripts/eval_eagle_acceptance_length.py --run-dir $RD --n-prompts 80 --datasets mtbench"
env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 TMPDIR=/data/thahn1230/tmp $EV --target int4 --draft-cfg d4p3_deploy --alpha 45.254834 --draft-sd $RD/ckpts/C7lr_s2_last.pt --tag C7lr_s2_qat >> outputs/pq_c7lr_s2.log 2>&1
echo C7LR_S2_DONE

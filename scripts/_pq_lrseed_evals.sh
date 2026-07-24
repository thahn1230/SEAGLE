cd /home/thahn1230/eagle_spinquant_w4a4
RD=$(cat runs/PQ_RUN_DIR)
EV="python scripts/eval_eagle_acceptance_length.py --run-dir $RD --n-prompts 80 --datasets mtbench"
until [ -f $RD/manifests/train_C3lr_s1.json ]; do sleep 120; done
env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 TMPDIR=/data/thahn1230/tmp $EV --target fp16 --draft-cfg d4p3_deploy --alpha 45.254834 --draft-sd $RD/ckpts/C3lr_s1_last.pt --tag C3lr_s1_qat > outputs/pq_ev_c3lr_s1.log 2>&1
until [ -f $RD/manifests/train_C3lr_s2.json ]; do sleep 120; done
env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 TMPDIR=/data/thahn1230/tmp $EV --target fp16 --draft-cfg d4p3_deploy --alpha 45.254834 --draft-sd $RD/ckpts/C3lr_s2_last.pt --tag C3lr_s2_qat > outputs/pq_ev_c3lr_s2.log 2>&1
until [ -f $RD/manifests/train_C7lr_s1.json ]; do sleep 120; done
env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 TMPDIR=/data/thahn1230/tmp $EV --target int4 --draft-cfg d4p3_deploy --alpha 45.254834 --draft-sd $RD/ckpts/C7lr_s1_last.pt --tag C7lr_s1_qat > outputs/pq_ev_c7lr_s1.log 2>&1
echo LRSEED_EVALS_DONE

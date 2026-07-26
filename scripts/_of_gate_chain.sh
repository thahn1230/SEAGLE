cd /home/thahn1230/eagle_spinquant_w4a4
RD=$(cat runs/OF_RUN_DIR)
until [ -f $RD/ckpts/fp16_fresh_final.pt ]; do sleep 300; done
sleep 30
# reproduction-gate evals: public C0 + fresh C1, mtbench first (2 GPUs)
env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 TMPDIR=/data/thahn1230/tmp python scripts/eval_eagle1_ptq_vs_qat.py --cell C0 --target fp16 --run-dir $RD --tag OF_C0_public > outputs/of_gate_c0.log 2>&1 &
env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 TMPDIR=/data/thahn1230/tmp python scripts/eval_eagle1_ptq_vs_qat.py --cell C1 --target fp16 --run-dir $RD --sd $RD/ckpts/fp16_fresh_final.pt --tag OF_C1_fresh > outputs/of_gate_c1.log 2>&1 &
# cross-domain for both, remaining GPUs
env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 TMPDIR=/data/thahn1230/tmp python scripts/eval_eagle1_ptq_vs_qat.py --cell C0 --target fp16 --run-dir $RD --datasets sharegpt:80,c4:200 --tag OF_C0_public > outputs/of_gate_c0x1.log 2>&1 &
env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 TMPDIR=/data/thahn1230/tmp python scripts/eval_eagle1_ptq_vs_qat.py --cell C0 --target fp16 --run-dir $RD --datasets gsm8k:200,humaneval:164 --tag OF_C0_public > outputs/of_gate_c0x2.log 2>&1 &
env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 TMPDIR=/data/thahn1230/tmp python scripts/eval_eagle1_ptq_vs_qat.py --cell C1 --target fp16 --run-dir $RD --sd $RD/ckpts/fp16_fresh_final.pt --datasets sharegpt:80,c4:200 --tag OF_C1_fresh > outputs/of_gate_c1x1.log 2>&1 &
env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=5 TMPDIR=/data/thahn1230/tmp python scripts/eval_eagle1_ptq_vs_qat.py --cell C1 --target fp16 --run-dir $RD --sd $RD/ckpts/fp16_fresh_final.pt --datasets gsm8k:200,humaneval:164 --tag OF_C1_fresh > outputs/of_gate_c1x2.log 2>&1 &
wait
grep -h "tau=" outputs/of_gate_c*.log
# anchor freeze + alpha stats (gate verdict computed by the session)
mkdir -p checkpoints/eagle1_fresh_fp16_anchor
cp $RD/ckpts/fp16_fresh_final.pt checkpoints/eagle1_fresh_fp16_anchor/anchor.pt
sha256sum checkpoints/eagle1_fresh_fp16_anchor/anchor.pt > checkpoints/eagle1_fresh_fp16_anchor/anchor.sha256
env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 TMPDIR=/data/thahn1230/tmp python scripts/calibrate_eagle1_p3_alpha.py --stage stats --anchor checkpoints/eagle1_fresh_fp16_anchor/anchor.pt --run-dir $RD > outputs/of_alpha_stats.log 2>&1
echo GATE_CHAIN_DONE

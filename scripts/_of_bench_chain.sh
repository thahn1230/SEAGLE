cd /home/thahn1230/eagle_spinquant_w4a4
RD=$(cat runs/OF_RUN_DIR)
for W in 1 2 4 8; do
  env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((W-1))) TMPDIR=/data/thahn1230/tmp PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True torchrun --nproc_per_node=$W --master_port=29617 scripts/train_eagle1_official_fp16.py --run-dir $RD --bench 120 > outputs/of_bench_w$W.log 2>&1
done
grep -h "\[bench\]" outputs/of_bench_w*.log
echo BENCH_CHAIN_DONE

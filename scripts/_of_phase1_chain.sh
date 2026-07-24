cd /home/thahn1230/eagle_spinquant_w4a4
RD=$(cat runs/OF_RUN_DIR)
until grep -q "cached" outputs/of_tokenize.log 2>/dev/null; do sleep 60; done
# 8-GPU deterministic audit shards
for s in 0 1 2 3 4 5 6 7; do
  env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$s TMPDIR=/data/thahn1230/tmp python scripts/generate_eagle1_official_training_data.py --mode audit-shard --shard $s --run-dir $RD > outputs/of_audit_s$s.log 2>&1 &
done
wait
python scripts/generate_eagle1_official_training_data.py --mode validate --run-dir $RD > outputs/of_validate.log 2>&1
env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 TMPDIR=/data/thahn1230/tmp python scripts/generate_eagle1_official_training_data.py --mode verify-fused --run-dir $RD > outputs/of_verify_fused.log 2>&1
# DDP benchmark: 1/2/4/8 GPUs x 120 steps (pre-registered 2B)
for W in 1 2 4 8; do
  env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((W-1))) TMPDIR=/data/thahn1230/tmp PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True torchrun --nproc_per_node=$W --master_port=29617 scripts/train_eagle1_official_fp16.py --run-dir $RD --bench 120 > outputs/of_bench_w$W.log 2>&1
done
grep -h "\[bench\]" outputs/of_bench_w*.log
echo PHASE1_CHAIN_DONE

RD=$(cat /home/thahn1230/eagle_spinquant_w4a4/runs/OF_RUN_DIR)
while pgrep -f "train_eagle1_official_fp16.*--run-dir" > /dev/null; do
  echo "$(date +%s),$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | tr '\n' ';')" >> /home/thahn1230/eagle_spinquant_w4a4/$RD/gpu_logs/full_run_gpu.csv
  sleep 120
done

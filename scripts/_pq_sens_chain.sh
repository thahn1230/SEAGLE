cd /home/thahn1230/eagle_spinquant_w4a4
until grep -q "FINISHED" outputs/pq_sched.log 2>/dev/null; do sleep 120; done
RD=$(cat runs/PQ_RUN_DIR)
env TMPDIR=/data/thahn1230/tmp python scripts/schedule_eagle_ptq_qat_jobs.py --run-dir $RD --alpha-fp16 45.254834 --alpha-int4 45.254834 --with-sensitivity > outputs/pq_sched2.log 2>&1
echo SENS_ROUND_DONE

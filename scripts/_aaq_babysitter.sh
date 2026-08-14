#!/bin/bash
# AA-QAT standing GPU assigner (2026-08-13). SINGLE OWNER of all 8 GPUs
# once the legacy phase-5a/5b dispatchers exit. Polls every 30s; for each
# GPU <2GB it atomically pops one job from backlog_jobs.txt (flock) and
# launches it pinned. Feed it by APPENDING jobs; stop with rm STOPFILE.
RUN=/home/thahn1230/SEAGLE/runs/eagle1_acceptance_aware_qat_20260812_120000
BL=$RUN/configs/backlog_jobs.txt
AS=$RUN/configs/backlog_assigned.txt
LK=$RUN/configs/backlog.lock
STOPFILE=$RUN/configs/babysitter.on
# PER-GPU gates (fix 2026-08-13: a blanket all-pids gate idled released
# GPUs while an unrelated dispatcher finished elsewhere). Format:
# "gpu:pid gpu:pid ..." — that GPU is skipped while pid is alive.
GATES="$@"
touch $STOPFILE
cd /home/thahn1230/SEAGLE
while [ -f $STOPFILE ]; do
  if true; then
    for g in 0 1 2 3 4 5 6 7; do
      skip=0
      for gp in $GATES; do
        [ "${gp%%:*}" = "$g" ] && [ -d /proc/${gp##*:} ] && skip=1
      done
      [ $skip -eq 1 ] && continue
      # pid ledger beats VRAM: a just-launched job builds its model on
      # CPU for minutes with <2GB claimed — never double-assign a GPU
      # whose assigned process is still alive
      PF=$RUN/configs/gpu$g.pid
      if [ -f $PF ] && [ -d /proc/$(cat $PF) ]; then continue; fi
      M=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g 2>/dev/null || echo 99999)
      if [ "$M" -lt 2000 ]; then
        JOB=$(flock $LK -c "head -1 $BL; sed -i '1d' $BL")
        [ -z "$JOB" ] && break
        CUDA_VISIBLE_DEVICES=$g nohup bash -c "$JOB" > /dev/null 2>&1 &
        echo $! > $PF
        echo "$(date +%H:%M:%S) gpu$g pid $! :: $(echo "$JOB" | grep -oP '(?<=--tag )\S+|(?<=--out )\S+' | head -1)" >> $AS
      fi
    done
  fi
  sleep 10
done
echo "$(date) babysitter stopped" >> $AS

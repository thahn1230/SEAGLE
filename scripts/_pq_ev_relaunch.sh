cd /home/thahn1230/eagle_spinquant_w4a4
while pgrep -f "scripts/_pq_ev_g[0-9].sh" > /dev/null; do sleep 60; done
for g in 0 1 2 3 4 5; do
  setsid env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$g TMPDIR=/data/thahn1230/tmp nohup bash scripts/_pq_ev_g$g.sh >> outputs/pq_ev_g$g.log 2>&1 < /dev/null &
done
echo RELAUNCHED

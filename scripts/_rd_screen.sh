#!/bin/bash
# R_D screening: wait for each trained arm's checkpoint, then run the
# held-out c4 calib-pool 20-prompt AL eval (official tau) with the
# rot_ep3p deploy. One eval per free GPU from the pool 1-7.
set -uo pipefail
cd /home/thahn1230/eagle_spinquant_w4a4
RD=$(cat runs/RDROT_RUN_DIR)
AF=$(python3 -c "print(4096**0.40)")
AR=$(python3 -c "print(4096**0.45)")
ARMS="RD_HYB_s0 RD_HYB_s1 RD_HYB_s2 RD_ACCS_s0 RD_ACCS_s1 RD_ACCS_s2 RD_AUXG_s0 RD_EXPT_s0"

for a in $ARMS; do
  while [ ! -f "$RD/rotations/$a.pt" ]; do sleep 120; done
done
# all checkpoints present -> training GPUs are free; fan out
g=1
for a in $ARMS; do
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$g \
    python scripts/eval_eagle_acceptance_length.py --target int4 \
    --draft-cfg rot_ep3p --ckpt "$RD/rotations/$a.pt" \
    --alpha "$AF" --alpha-rec "$AR" --tag "$a" \
    --datasets c4 --pool calib --n-prompts 20 --run-dir "$RD" \
    > "$RD/logs/screen_$a.log" 2>&1 &
  g=$((g % 7 + 1))
  if [ "$g" = "1" ]; then wait; fi
done
wait
python3 - <<'EOF'
import csv, glob, json, os
rd = open("runs/RDROT_RUN_DIR").read().strip()
out = {}
for p in glob.glob(os.path.join(rd, "shards", "al__*__int4__c4__calib.csv")):
    tag = os.path.basename(p).split("__")[1]
    ts = [t for r in csv.DictReader(open(p))
          for t in json.loads(r["acceptance_list"])]
    out[tag] = round(sum(ts) / max(len(ts), 1), 4)
path = os.path.join(rd, "tables", "screen_heldout.json")
json.dump(dict(sorted(out.items(), key=lambda kv: -kv[1])),
          open(path, "w"), indent=1)
print("[screen]", json.dumps(out, indent=1))
EOF
echo "[screen] DONE"

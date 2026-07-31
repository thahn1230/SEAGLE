cd /home/thahn1230/eagle_spinquant_w4a4
RD=$(cat runs/GQ_RUN_DIR)
AN=checkpoints/eagle1_fresh_fp16_anchor/anchor.pt
# wait pilots (8 manifests)
until [ $(ls $RD/manifests/train_GQp_* 2>/dev/null | wc -l) -ge 8 ]; do sleep 180; done
# pilot calib evals, 8 GPUs
G=0
for V in T0 T1; do TGT=$([ $V = T0 ] && echo fp16 || echo int4)
for LR in 3e-07 1e-06 3e-06 1e-05; do
env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$G TMPDIR=/data/thahn1230/tmp python scripts/eval_eagle_acceptance_length.py --run-dir $RD --target $TGT --draft-cfg naive_w4a4 --draft-sd $RD/ckpts/GQp_${V}_lr${LR}_last.pt --datasets c4:20 --pool calib --tag GQPEV_${V}_lr$LR > outputs/gq_pev_${V}_$LR.log 2>&1 &
G=$((G+1)); done; done
wait
# LR select per variant by calib tau
python - <<'PYEOF'
import csv, glob, json, os
rd = open("runs/GQ_RUN_DIR").read().strip()
out = {}
for v in ("T0", "T1"):
    tgt = "fp16" if v == "T0" else "int4"
    rows = []
    for lr in ("3e-07", "1e-06", "3e-06", "1e-05"):
        p = f"{rd}/shards/al__GQPEV_{v}_lr{lr}__{tgt}__c4__calib.csv"
        if not os.path.exists(p): continue
        ts = [t for r in csv.DictReader(open(p)) for t in json.loads(r["acceptance_list"])]
        rows.append(dict(lr=float(lr), calib_tau=round(sum(ts)/max(len(ts),1),4)))
    if rows:
        out[v] = dict(grid=rows, selected_lr=max(rows, key=lambda r: r["calib_tau"])["lr"])
json.dump(out, open(f"{rd}/tables/gqat_lr_pilot.json","w"), indent=1)
print(out)
PYEOF
# 6 GQAT seeds, 6 GPUs
G=0
for V in T0 T1; do
LR=$(python scripts/_get_json.py $RD/tables/gqat_lr_pilot.json $V selected_lr)
for S in 0 1 2; do
env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$G TMPDIR=/data/thahn1230/tmp PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/train_eagle1_generic_qat.py --variant $V --anchor $AN --lr $LR --steps 3000 --seed $S --save-ckpt-every 500 --run-dir $RD --tag GQAT_${V}_s$S > outputs/gq_seed_${V}_s$S.log 2>&1 &
G=$((G+1)); done; done
# meanwhile p3exp capture+search on GPUs 6,7 (legacy alphas: A=45.254834, C=32.0)
env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 TMPDIR=/data/thahn1230/tmp python scripts/calibrate_eagle1_p3exp.py --stage capture --target fp16 --anchor $AN --alpha 45.254834 --run-dir $RD > outputs/gq_p3exp_cap_fp16.log 2>&1 && env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 python scripts/calibrate_eagle1_p3exp.py --stage search --target fp16 --run-dir $RD > outputs/gq_p3exp_search_fp16.log 2>&1 &
env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=7 TMPDIR=/data/thahn1230/tmp python scripts/calibrate_eagle1_p3exp.py --stage capture --target int4 --anchor $AN --alpha 32.0 --run-dir $RD > outputs/gq_p3exp_cap_int4.log 2>&1 && env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=7 python scripts/calibrate_eagle1_p3exp.py --stage search --target int4 --run-dir $RD > outputs/gq_p3exp_search_int4.log 2>&1 &
wait
echo GQ_PHASE2_DONE

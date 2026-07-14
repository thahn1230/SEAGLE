CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6,7 scripts/run_target_draft_3x3_matrix.py --run-dir runs/bwal_final_20260715_0249 --group stock --device cuda:0 --num-prompts 80 --max-new-tokens 128
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6,7 scripts/run_target_draft_3x3_matrix.py --run-dir runs/bwal_final_20260715_0249 --group t8 --device cuda:0 --num-prompts 80 --max-new-tokens 128
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6,7 scripts/run_target_draft_3x3_matrix.py --run-dir runs/bwal_final_20260715_0249 --group t4 --device cuda:0 --num-prompts 80 --max-new-tokens 128

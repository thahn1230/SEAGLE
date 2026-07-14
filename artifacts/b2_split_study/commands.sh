# consolidated commands
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6,7 scripts/validate_b2_fp_equivalence.py --device cuda:0 --num-prompts 8 --max-new-tokens 48
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6,7 scripts/run_b2_acceptance_matrix.py --run-dir runs/b2_matrix_20260714_1834 --group stock --device cuda:0 --num-prompts 20 --max-new-tokens 64
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6,7 scripts/run_b2_acceptance_matrix.py --run-dir runs/b2_matrix_20260714_1834 --group rot --device cuda:1 --num-prompts 20 --max-new-tokens 64
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6,7 scripts/run_b2_acceptance_matrix.py --run-dir runs/b2_matrix_20260714_1834 --group quant --device cuda:0 --num-prompts 20 --max-new-tokens 64
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6,7 scripts/run_b2_acceptance_matrix.py --run-dir runs/b2_matrix_20260714_1834 --group quant_w4a16 --device cuda:1 --num-prompts 20 --max-new-tokens 64
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6,7 scripts/run_b2_acceptance_matrix.py --run-dir runs/b2_matrix_20260714_1834 --group quant_kv4 --device cuda:0 --num-prompts 20 --max-new-tokens 64

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6,7 scripts/run_target_draft_3x3_matrix.py --run-dir runs/bwal_final_20260715_0249 --group stock --device cuda:0 --num-prompts 80 --max-new-tokens 128
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6,7 scripts/run_target_draft_3x3_matrix.py --run-dir runs/bwal_final_20260715_0249 --group t8 --device cuda:0 --num-prompts 80 --max-new-tokens 128
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6,7 scripts/run_target_draft_3x3_matrix.py --run-dir runs/bwal_final_20260715_0249 --group t4 --device cuda:0 --num-prompts 80 --max-new-tokens 128
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6,7 scripts/run_component_precision_ablation.py --run-dir runs/bwal_comp_final_20260715_0522 --group draft --device cuda:0 --num-prompts 20 --max-new-tokens 64
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6,7 scripts/run_component_precision_ablation.py --run-dir runs/bwal_comp_final_20260715_0522 --group branch --device cuda:0 --num-prompts 20 --max-new-tokens 64
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6,7 scripts/run_component_precision_ablation.py --run-dir runs/bwal_comp_final_20260715_0522 --group thead --device cuda:0 --num-prompts 20 --max-new-tokens 64
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6,7 scripts/run_component_precision_ablation.py --run-dir runs/bwal_comp_final_20260715_0522 --group tembed --device cuda:0 --num-prompts 20 --max-new-tokens 64
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6,7 scripts/run_component_precision_ablation.py --run-dir runs/bwal_comp_final_20260715_0522 --group tbody --device cuda:0 --num-prompts 20 --max-new-tokens 64
# gates + supporting runs (serial, physical GPU 6, CVD=6,7)
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6,7 python scripts/run_acceptance_equivalence_forensics.py --device cuda:0 --num-prompts 20 --max-new-tokens 64
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6,7 python scripts/run_fixed_tree_target_grader.py --device cuda:0 --num-prompts 12
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6,7 python scripts/run_bwal_verifier_consistency.py cuda:0
python scripts/capture_bwal_distributions.py --weights
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6,7 python scripts/capture_bwal_distributions.py --acts --num-prompts 4 --device cuda:0
python scripts/analyze_bitwidth_al_component_causality.py --run-dir runs/bwal_final_20260715_0249
python scripts/analyze_component_ablations.py --run-dir runs/bwal_comp_final_20260715_0522
python scripts/plot_bitwidth_al_component_causality.py --run-dir runs/bwal_final_20260715_0249 --comp-dir runs/bwal_comp_final_20260715_0522

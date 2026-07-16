CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6,7 scripts/run_target_draft_3x3_matrix.py --run-dir runs/bwal_corrected_20260716_1434 --group t8 --device cuda:0 --num-prompts 80 --max-new-tokens 128 --rotation-kind learned_chat_w4a4kv16
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6,7 scripts/run_target_draft_3x3_matrix.py --run-dir runs/bwal_corrected_20260716_1434 --group t4 --device cuda:0 --num-prompts 80 --max-new-tokens 128 --rotation-kind learned_chat_w4a4kv16
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6,7 scripts/run_target_draft_3x3_matrix.py --run-dir runs/bwal_corrected_20260716_1434 --group stock --device cuda:0 --num-prompts 80 --max-new-tokens 128 --rotation-kind learned_chat_w4a4kv16
# official evals (wrapper: scripts/_run_official_ptq.sh, pinned venv torchrun)
bash scripts/_run_gate_a.sh
bash scripts/_run_base_trio.sh
bash scripts/_run_gptq_diag.sh ; bash scripts/_run_gptq_diag2.sh
bash scripts/_run_seed_sweep_base.sh   # GPU 4
bash scripts/_run_seed_sweep_chat.sh   # GPU 5
# chat rotation training (GPUs 0-3, official global batch 8 via NPROC=4 ACCUM=2)
CVD_OVERRIDE=0,1,2,3 NPROC=4 ACCUM=2 bash scripts/_run_optimize_rotation.sh <chat_path> /data/thahn1230/spinquant_rotations/chat_w4a4kv16 chat_optrot_w4a4 4 4 16
bash scripts/_run_chat_learned_trio.sh   # CHAT_03/04/05
bash scripts/_run_recipe_ablation.sh     # GPU 4
# parity
python scripts/run_evaluator_parity.py --base <base> --chat <chat>
python scripts/dump_pipeline_state.py --pipeline official --rbin <rh0 R.bin>   # pinned venv
python scripts/dump_pipeline_state.py --pipeline inprocess --rbin <rh0 R.bin>
python scripts/compare_pipeline_dumps.py
python scripts/run_inprocess_official_ppl.py
python scripts/audit_current_ppl_contract.py --device cuda:0
# corrected EAGLE matrix + grader + VC (learned rotation)
bash scripts/_run_corr_stock.sh ; bash scripts/_run_corr_t8.sh ; bash scripts/_run_corr_t4.sh
python scripts/analyze_bitwidth_al_component_causality.py --run-dir runs/bwal_corrected_20260716_1434
python scripts/run_fixed_tree_target_grader.py --device cuda:0 --num-prompts 12 --rotation-kind learned_chat_w4a4kv16 --out-suffix _learned
python scripts/run_bwal_verifier_consistency.py cuda:0 learned_chat_w4a4kv16 _learned

# scripts/

Numbered pipeline scripts. Run from the project root (`eagle_spinquant_w4a4/`), e.g.
`python scripts/00_env_check.py`. Scripts read defaults from
`configs/default_experiment.yaml` unless flags override them.

All scripts are implemented and run. Run from the project root.

| script | purpose |
|---|---|
| `00_env_check.py` | GPU / package / repo / HF-auth + math-sanity check → `results/env_check.json`. Run first. |
| `01_discover_model_intersection.py` | Verifies `configs/model_candidates.yaml` against both repos (arch, hidden/vocab, cache); `--write` updates the YAML. |
| `10_optimize_rotation.py` | Wraps SpinQuant `optimize_rotation.py` (torchrun); saves learned R1/R2 to `outputs/rotations/<tag>/R.bin`. Heavy. |
| `11_eval_spinquant_ppl.py` | Wraps SpinQuant `ptq.py`; wikitext-2 PPL for fp16 / W4A16 / W4A4 / W4A4KV4 / learned-R → `results/ppl_*`. |
| `20_eval_eagle1_baseline.py` | EAGLE-1 FP16 baseline: MT-bench speedup + acceptance → `results/eagle_fp_baseline*`. |
| `30_capture_rotated_hidden.py` | Captures draft-input tensor shapes → `docs/03` + `runs/debug_hidden_capture/`. |
| `31_eval_rotated_target_unrotate_draft.py` | Variant A (unrotate) vs naive on the rotated target → `results/unrotate_interface_*`. |
| `32_eval_draft_conjugated_rotation.py` | Variant B (fc-fold) + conjugation identity check → `results/conjugated_rotation_*`, `docs/04`. |
| `40_train_rotated_eagle_draft.py` | Variant C: retrain the draft on quantized features → `outputs/draft_ckpts/`. |
| `50_run_batch_sweep.py` | Final method × quant sweep (fp16/W4A4/W4A4KV4 × A/B/C/naive) → `results/final_sweep_*`. Run one setting per process. |
| `60_aggregate_results.py` | Collects all run JSONLs → `results/results.{jsonl,csv}` with provenance. |

Tests: `tests/test_math_sanity.py` (download-free), `tests/test_fp_equivalence.py`
(rotation port gate). Reproduction: `results/repro_commands.sh`. Results discussion:
`docs/05_FINAL_EXPERIMENT_REPORT.md`.

Conventions for new scripts:
- argparse + `--config configs/default_experiment.yaml`
- write artifacts under `outputs/<run_name>/`, one JSONL line per measurement
- never hardcode GPU ids; respect `CUDA_VISIBLE_DEVICES`
- prefer 1 GPU for 7B-class work; multi-GPU only for rotation optimization if
  memory-bound (see audit doc) or for >7B models

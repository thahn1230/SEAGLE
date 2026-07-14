cd /home/thahn1230/eagle_spinquant_w4a4

TS=$(date +%Y%m%d_%H%M%S)

tar -czf "eagle_spinquant_b2_review_bundle_${TS}.tar.gz" \
  docs/B2_FINAL_REPORT.md \
  docs/B2_SPLIT_BASIS_CONTRACT.md \
  docs/EAGLE1_PROJECTION_LAYOUT.md \
  docs/EAGLE1_AR_FEATURE_SEMANTICS.md \
  docs/B2_IMPLEMENTATION.md \
  docs/B2_EXPERIMENT_PROTOCOL.md \
  artifacts/b2_split_study \
  configs/b2_split_study.yaml \
  scripts/validate_b2_fp_equivalence.py \
  scripts/validate_b2_projection_dispatch.py \
  scripts/validate_b2_real_w4a4.py \
  scripts/run_b2_acceptance_matrix.py \
  scripts/run_b2_component_ablation.py \
  scripts/analyze_b2_results.py \
  scripts/plot_b2_results.py \
  scripts/build_b2_final_report.py \
  src/eagle_spinquant \
  tests \
  2>/tmp/b2_tar_warnings.txt

echo "Bundle:"
realpath "eagle_spinquant_b2_review_bundle_${TS}.tar.gz"

echo "Size:"
du -h "eagle_spinquant_b2_review_bundle_${TS}.tar.gz"

echo "Warnings:"
cat /tmp/b2_tar_warnings.txt
# qanchor bundle manifest (2026-08-15)
Large tensors excluded from the bundle (>100MB git cap); server paths + sha16:
  d290220cd41e012b  runs/eagle1_qat_qanchor_causal_20260814/ckpts/anchor_gsr5_ptq.pt
  08cb92f65322473c  runs/eagle1_qat_qanchor_causal_20260814/ckpts/expected_c0.pt
  ba21175e72a32770  runs/eagle1_qat_qanchor_causal_20260814/ckpts/codes__HB_B_hybrid_b0.005_s0_st3000.pt
  ac3ce2200f1a7a07  runs/eagle1_qat_qanchor_causal_20260814/ckpts/codes__HB_B_conv_b0.02_s0_st3000.pt
  56bd6fc11d95337b  /data/thahn1230/aaq_ckpts/HB_B_hybrid_b0.005_s0.pt.step3000.pt
  7ed8c27457e2f971  /data/thahn1230/aaq_ckpts/HB_B_conv_b0.02_s0.pt.step3000.pt
  510cd8f1e2091a0a  /data/thahn1230/aaq_ckpts/DP_B_conv_beta1_s2.pt.step0600.pt
  d6b3f5ca4809c2b2  /data/thahn1230/aaq_ckpts/DP_B_hybrid_beta0.1_s2.pt.step0600.pt
Regenerate codes: scripts/capture_adapter_anchor.py --codes-only --draft-sd <ckpt>
Anchor capture: scripts/capture_adapter_anchor.py (public draft, RD_HYB_s2, GS alpha)

# RCAL Related-Work and Novelty Audit (study §28)

Web search performed 2026-07-31 over the spec's term list ("reference
consistent acceptance", "acceptance fidelity", "quantized verifier
drift", "quantized target speculative decoding", "lossless speculative
decoding quantized verifier", "verifier disagreement", "false/spurious
acceptance", "target drift"). Findings:

## Closest related work found

- **QSpec** (arXiv:2410.11305): one weight-quantized model toggling
  W4A4 drafting / W4A16 verification. Related problem space (quantized
  verification) but measures end task quality and speed — no paired
  same-proposal replay metric against an FP16 reference verifier.
- **QuantSpec** (arXiv:2502.10424): self-speculative decoding with
  hierarchical quantized KV cache; tracks acceptance-rate degradation
  from quantized components, mitigates with full-precision buffers.
  Acceptance rate w.r.t. its own (quantized) verifier only.
- **ML-SpecQD** (arXiv:2503.13565): quantized drafts in multi-level
  speculation; acceptance measured against the deployed verifier.
- Practitioner reports (FP8 targets + low-precision drafts) mention
  "measurable drift" between quantized-verifier and full-precision
  behavior, without a formal per-cycle decomposition metric.
- Lossless-verification theory (e.g. Hierarchical Speculative Decoding,
  arXiv:2601.05724) proves distribution-lossless acceptance w.r.t. the
  DEPLOYED target; it does not treat a quantized deployed target vs an
  FP16 reference.
- Acceptance-collapse / robustness lines (arXiv:2605.14005,
  2607.21804) study accepted-length reduction, not reference
  consistency.

## Novelty positioning (as supported by this search)

Not claimed novel: the observations that quantized verifiers drift, or
that quantization lowers acceptance — both appear in prior work.

Claimed (search found no prior instance):
1. the **paired same-proposal, same-prefix replay protocol** — replaying
   the deployed system's exact proposal trees under an FP16 reference
   verifier with lockstep KV along the deployed trajectory;
2. the **LCP-based decomposition** AL_q = RCAL + SAL and
   AL_0 = RCAL + LAL with precision/recall/AFS on accepted proposal
   tokens, including tree-branch divergence handling
   (R_RC < min(R_q, R_0));
3. using this to classify AL gains as faithful vs
   **verifier-drift-driven ("deceptive")** per method.

The NAME "RCAL" did not appear in the searched sources. If later
reviewing surfaces a closer metric, RCAL should cite and reposition.

Caveats stated everywhere RCAL is reported: RCAL measures fidelity to a
chosen FP16 reference verifier — NOT ground-truth task correctness,
semantic quality, or throughput; it is always reported alongside
deployed AL, task metrics, target fidelity, and measured wall-clock.

Sources:
- [QSpec](https://arxiv.org/html/2410.11305)
- [QuantSpec](https://arxiv.org/html/2502.10424v1)
- [ML-SpecQD](https://arxiv.org/html/2503.13565v1)
- [Hierarchical Speculative Decoding](https://arxiv.org/abs/2601.05724)
- [Mistletoe acceptance-collapse](https://arxiv.org/pdf/2605.14005)
- [Adversarial acceptance collapse](https://arxiv.org/html/2607.21804)
- [Lynx progressive speculative quantization](https://arxiv.org/pdf/2607.01831)
- [Lossless speculative decoding overview](https://www.emergentmind.com/topics/lossless-speculative-decoding)

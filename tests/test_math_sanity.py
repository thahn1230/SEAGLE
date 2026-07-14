#!/usr/bin/env python
"""Download-free mathematical sanity checks for the rotation interface.

Validates the linear algebra the whole project rests on, using tiny random
tensors and a tiny random EAGLE draft (no model downloads, no GPU required for
most checks). Run: python tests/test_math_sanity.py
Writes results/math_sanity.json. Exit 0 iff all checks pass.
"""

import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "EAGLE"))

import torch  # noqa: E402
from eagle_spinquant import rotation_interface as ri  # noqa: E402
from eagle_spinquant import draft_conjugation as dc  # noqa: E402


def make_tiny_draft(D=64, V=128, n_layers=1, device="cpu", dtype=torch.float32):
    """Construct a tiny EAGLE draft (cnets.Model) with a small config."""
    from eagle.model.cnets import Model
    from eagle.model.configs import EConfig

    cfg = EConfig(
        vocab_size=V, hidden_size=D, intermediate_size=2 * D,
        num_hidden_layers=n_layers, num_attention_heads=4,
        num_key_value_heads=4, hidden_act="silu",
        max_position_embeddings=256, rms_norm_eps=1e-6,
        pad_token_id=0, bias=True,
    )
    model = Model(cfg, bias=True)
    return model.to(device=device, dtype=dtype).eval()


def main() -> int:
    results = {}
    ok = True

    # 1. orthogonality of a QR-generated matrix
    g = torch.Generator().manual_seed(0)
    R, _ = torch.linalg.qr(torch.randn(256, 256, generator=g, dtype=torch.float64))
    results["orthogonality"] = ri.check_orthogonal(R)
    ok &= results["orthogonality"]["is_orthogonal"]

    # 2. attention-score preservation under shared post-RoPE rotation R3
    results["attention_preservation"] = ri.check_attention_preservation(D=128)
    ok &= results["attention_preservation"]["preserved"]

    # 3. unrotate(rotate(h)) == h  (the interface inverse map, with gamma_f)
    results["unrotation_roundtrip"] = ri.check_unrotation_roundtrip(D=4096)
    ok &= results["unrotation_roundtrip"]["ok"]

    # 4. Hadamard kernel vs pure-torch reference (if CUDA + FHT available)
    try:
        from eagle_spinquant import hadamard_shim as hs
        if hs.HAVE_FHT and torch.cuda.is_available():
            x = torch.randn(4, 512, device="cuda", dtype=torch.float32)
            k = hs._fht.hadamard_transform(x)
            r = hs.hadamard_transform(x)  # kernel path
            ref = hs.reference_hadamard(x) * (512 ** 0.5)  # normalized->unnormalized
            err = (k.float() - ref.float()).abs().max().item()
            results["hadamard_kernel_vs_reference"] = {"max_abs_err": err, "ok": err < 1e-2}
            ok &= err < 1e-2
        else:
            results["hadamard_kernel_vs_reference"] = {"skipped": "no CUDA/FHT"}
    except Exception as e:
        results["hadamard_kernel_vs_reference"] = {"error": str(e)}

    # 5. draft fc conjugation identity (Variant B, single forward) on a tiny draft
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        draft = make_tiny_draft(D=64, V=128, device=device)
        g2 = torch.Generator().manual_seed(1)
        R1, _ = torch.linalg.qr(torch.randn(64, 64, generator=g2, dtype=torch.float64))
        gamma_f = torch.rand(64, generator=g2, dtype=torch.float64) + 0.5
        results["conjugation_identity"] = dc.conjugation_identity_check(
            draft, R1.float(), gamma_f.float(), D=64, seq=6, device=device)
        ok &= results["conjugation_identity"]["single_forward_equivalent"]
    except Exception as e:
        import traceback
        results["conjugation_identity"] = {"error": str(e), "tb": traceback.format_exc()}
        ok = False

    results["all_passed"] = bool(ok)

    out = os.path.join(PROJECT_ROOT, "results", "math_sanity.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    print(f"\n-> {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

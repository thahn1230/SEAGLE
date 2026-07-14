"""Shared fixtures for the B2 split-projection test suite.

Pure-CPU fp64: random orthogonal R1 (QR), random positive gamma, and a TINY
real EAGLE-1 cnets.Model (the actual vendored class, hidden=64) so the chain
tests exercise the genuine decoder-layer code path, not a re-implementation.
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "EAGLE"))

import torch  # noqa: E402

torch.manual_seed(0)

D_TINY = 64


def rand_orthogonal(n, seed=0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(n, n, generator=g, dtype=dtype)
    Q, _ = torch.linalg.qr(A)
    return Q


def rand_gamma(n, seed=1, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    return 0.5 + torch.rand(n, generator=g, dtype=dtype)   # positive, non-trivial


def rms_normalize(x, eps=1e-6):
    return x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def tiny_config():
    from transformers import LlamaConfig
    return LlamaConfig(hidden_size=D_TINY, intermediate_size=128,
                       num_hidden_layers=1, num_attention_heads=4,
                       num_key_value_heads=4, vocab_size=97,
                       max_position_embeddings=256, pad_token_id=0,
                       hidden_act="silu", rms_norm_eps=1e-6)


def tiny_model(seed=2, dtype=torch.float64):
    """Genuine vendored cnets.Model, tiny, fp64, deterministic weights."""
    from eagle.model.cnets import Model
    torch.manual_seed(seed)
    m = Model(tiny_config(), load_emb=False, path=None, bias=True).to(dtype)
    m.eval()
    m.reset()          # tree_mask=None
    return m


def rel_l2(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return float((a - b).norm() / (b.norm() + 1e-30))


@torch.no_grad()
def run_chain(model, first_hidden, ids_list, fc_weights_per_step, fc_bias=None):
    """Drive the REAL cnets.Model forward as a feature chain: step 0 consumes
    `first_hidden`; step k>0 consumes the previous step's output feature.
    fc_weights_per_step[k] is loaded into model.fc before step k (data swap,
    same mechanism the adapters use). Returns list of output features."""
    outs = []
    h = first_hidden
    for k, ids in enumerate(ids_list):
        model.fc.weight.data = fc_weights_per_step[k].to(model.fc.weight.dtype)
        if fc_bias is not None:
            model.fc.bias.data = fc_bias.to(model.fc.bias.dtype)
        h = model(h, input_ids=ids)
        outs.append(h)
    return outs


def tiny_setup(depth=4, T=3, seed=3):
    """Returns everything needed for original-vs-rotated chain comparisons on
    the tiny model: (model, sd, R1, gamma, n_t, h_t, a_t, ids_list)."""
    from eagle_spinquant import rotation_aware as ra   # noqa: F401 (import check)
    m = tiny_model()
    sd = {k: v.detach().clone() for k, v in m.state_dict().items()}
    R1 = rand_orthogonal(D_TINY, seed=10)
    gamma = rand_gamma(D_TINY, seed=11)
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(1, T, D_TINY, generator=g, dtype=torch.float64)
    n_t = rms_normalize(x)                       # unit-RMS, NO gamma
    h_t = n_t * gamma                            # gamma-included original feature
    a_t = n_t @ R1                               # fused-tail exposure
    ids_list = [torch.randint(1, 97, (1, T), generator=g) for _ in range(depth)]
    return m, sd, R1, gamma, n_t, h_t, a_t, ids_list

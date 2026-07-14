"""Concat-order proof (spec §4): fc input is cat((inputs_embeds, hidden), -1)
— [:, :D] multiplies the TOKEN EMBEDDING, [:, D:] multiplies the FEATURE.
Verified (a) in the vendored source text and (b) at runtime on the real tiny
cnets.Model by zeroing each weight half and checking which input the output
stops depending on."""

import os
import re

import torch

from b2_common import D_TINY, PROJECT_ROOT, tiny_setup


def test_source_concat_order():
    src = open(os.path.join(PROJECT_ROOT, "third_party", "EAGLE", "eagle",
                            "model", "cnets.py")).read()
    assert re.search(
        r"self\.fc\(torch\.cat\(\(inputs_embeds,\s*hidden_states\)", src), \
        "cnets concat order changed — re-derive the weight split!"


@torch.no_grad()
def test_runtime_slice_sensitivity():
    m, sd, R1, gamma, n_t, h_t, a_t, ids = tiny_setup(depth=1)
    m.load_state_dict(sd)
    h1 = torch.randn_like(h_t)
    h2 = torch.randn_like(h_t)
    i1 = ids[0]
    i2 = (i1 + 7) % 96 + 1

    # zero FEATURE half [:, D:] -> output must ignore hidden, follow ids
    W = sd["fc.weight"].clone(); W[:, D_TINY:] = 0
    m.fc.weight.data = W
    assert torch.allclose(m(h1, input_ids=i1), m(h2, input_ids=i1)), \
        "[:, D:] is not the feature block"
    assert not torch.allclose(m(h1, input_ids=i1), m(h1, input_ids=i2)), \
        "[:, :D] is not the embedding block"

    # zero EMBEDDING half [:, :D] -> output must ignore ids, follow hidden
    W = sd["fc.weight"].clone(); W[:, :D_TINY] = 0
    m.fc.weight.data = W
    assert torch.allclose(m(h1, input_ids=i1), m(h1, input_ids=i2)), \
        "[:, :D] is not the embedding block"
    assert not torch.allclose(m(h1, input_ids=i1), m(h2, input_ids=i1)), \
        "[:, D:] is not the feature block"


if __name__ == "__main__":
    test_source_concat_order()
    test_runtime_slice_sensitivity()
    print("OK")

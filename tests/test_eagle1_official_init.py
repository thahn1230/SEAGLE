"""Official fresh-init contract (audit §1): embed copied from target and
frozen; decoder/fc PyTorch-default init (NOT initializer_range normal);
head frozen; no input_layernorm on layer 0."""
import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "third_party", "EAGLE"))

CFG = os.path.join(ROOT, "third_party/EAGLE/eagle/train/"
                   "llama_2_chat_7B_config.json")


def _target_path():
    from eagle_spinquant import experiment
    cfg = experiment.load_config(None)
    return experiment.resolve_paths(cfg)["target_path"]


@pytest.fixture(scope="module")
def fresh():
    from eagle.model.cnets import Model
    from eagle.model.configs import EConfig
    torch.manual_seed(0)
    return Model(EConfig.from_pretrained(CFG), load_emb=True,
                 path=_target_path(), bias=True)


def test_embed_copied_and_frozen(fresh):
    assert not fresh.embed_tokens.weight.requires_grad
    from safetensors import safe_open
    import json as js
    tp = _target_path()
    idx = js.load(open(os.path.join(tp, "model.safetensors.index.json")))
    with safe_open(os.path.join(
            tp, idx["weight_map"]["model.embed_tokens.weight"]),
            framework="pt", device="cpu") as f:
        ref = f.get_slice("model.embed_tokens.weight")[:, :4096].float()
    assert torch.equal(fresh.embed_tokens.weight.data, ref)


def test_fresh_layers_are_default_init_not_pretrained(fresh):
    w = fresh.fc.weight.data
    # PyTorch default kaiming-uniform bound for fan_in=8192:
    bound = (1 / (8192 ** 0.5)) * (3 ** 0.5)
    assert float(w.abs().max()) <= bound * 1.001
    # a normal(0, 0.02) init would exceed this bound with prob ~1
    q = fresh.layers[0].self_attn.q_proj.weight.data
    qbound = (1 / (4096 ** 0.5)) * (3 ** 0.5)
    assert float(q.abs().max()) <= qbound * 1.001
    # and it must NOT equal the public trained draft's values: default
    # uniform bound rules out the trained weight scale statistics
    assert float(w.std()) < 0.011      # uniform(-b,b) std = b/sqrt(3)


def test_trainable_scope(fresh):
    trainable = {n for n, p in fresh.named_parameters()
                 if p.requires_grad}
    assert "embed_tokens.weight" not in trainable
    assert "fc.weight" in trainable and "fc.bias" in trainable
    assert any(n.startswith("layers.0.") for n in trainable)


def test_layer0_has_no_input_layernorm(fresh):
    assert not hasattr(fresh.layers[0], "input_layernorm") or \
        fresh.layers[0].input_layernorm is None or \
        "input_layernorm" not in dict(fresh.layers[0].named_children())


def test_gradient_checkpointing_default_on(fresh):
    assert fresh.gradient_checkpointing is True

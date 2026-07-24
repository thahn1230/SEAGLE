"""Official loss formula parity (audit §1): the study trainer's
compute_loss must equal the official main.py computation on the same
tensors: vloss = masked token-mean SmoothL1; ploss = masked soft-CE via
the frozen head; loss = 1.0*vloss + 0.1*ploss."""
import importlib.util
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))


def _load_trainer():
    spec = importlib.util.spec_from_file_location(
        "oft", os.path.join(ROOT, "scripts",
                            "train_eagle1_official_fp16.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def official_reference(predict, target_h, loss_mask, head_w):
    """Verbatim official main.py lines 357-367 (fp32)."""
    crit = torch.nn.SmoothL1Loss(reduction="none")
    target_head = predict.new_tensor(0)  # placeholder
    target_head = torch.nn.functional.linear(target_h, head_w)
    target_p = torch.nn.Softmax(dim=2)(target_head)
    out_head = torch.nn.functional.linear(predict, head_w)
    out_logp = torch.nn.LogSoftmax(dim=2)(out_head)
    lm = loss_mask[:, :, None]
    plogp = target_p * out_logp
    ploss = -torch.sum(torch.sum(lm * plogp, 2)) / (loss_mask.sum()
                                                    + 1e-5)
    vloss = crit(predict, target_h)
    vloss = torch.sum(torch.mean(lm * vloss, 2)) / (loss_mask.sum()
                                                    + 1e-5)
    return 1.0 * vloss + 0.1 * ploss, vloss, ploss


class _Head(torch.nn.Module):
    def __init__(self, w):
        super().__init__()
        self.weight = torch.nn.Parameter(w.clone(), requires_grad=False)

    def forward(self, x):
        return torch.nn.functional.linear(x.float(),
                                          self.weight.float())


class _Identity(torch.nn.Module):
    def forward(self, hidden, input_ids=None, attention_mask=None):
        return hidden


def test_loss_matches_official_reference():
    m = _load_trainer()
    g = torch.Generator().manual_seed(0)
    B, T, D, V = 2, 12, 16, 40
    predict = torch.randn(B, T, D, generator=g)
    target_h = torch.randn(B, T, D, generator=g)
    lm = (torch.rand(B, T, generator=g) > 0.4).float()
    head_w = torch.randn(V, D, generator=g) * 0.2
    ref, ref_v, ref_p = official_reference(predict, target_h, lm, head_w)

    crit = torch.nn.SmoothL1Loss(reduction="none")
    batch = (predict, torch.zeros(B, T, dtype=torch.long), target_h,
             lm, torch.ones(B, T, dtype=torch.long))
    loss, vloss, ploss, _, _ = m.compute_loss(
        _Identity(), _Head(head_w.half()), batch, crit)
    assert abs(float(vloss) - float(ref_v)) < 2e-3
    assert abs(float(ploss) - float(ref_p)) < 2e-2   # fp16 head cast
    assert abs(float(loss) - float(ref)) < 2e-2


def test_coefficients_are_official():
    m = _load_trainer()
    assert m.TC["v_w"] == 1.0 and m.TC["p_w"] == 0.1
    assert m.TC["lr"] == 3e-5 and m.TC["num_warmup_steps"] == 2000
    assert m.TC["total_steps"] == 800000 and m.TC["bs"] == 4
    assert m.TC["noise_std"] == 0.2 and m.TC["max_len"] == 2048
    assert m.TC["b1"] == 0.9 and m.TC["b2"] == 0.95
    assert m.TC["grad_clip"] == 0.5

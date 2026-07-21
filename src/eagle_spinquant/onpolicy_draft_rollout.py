"""On-policy / free-running rollout policies (study spec section 11.2).

At each draft depth the trainer either feeds the teacher trajectory token
(teacher-forced) or the draft's OWN choice (on-policy; discrete token is
stop-gradiented — `ExactQATRotatedDraft.forward_chain` detaches `chosen`).
The deployed target is then run on the visited prefix to obtain full-vocab
p for the LK loss at that state (the trainer does this with the online
teacher; see train_eagle_lk_rotation.py).

Curriculum (spec): first 30% teacher-forced, middle 40% mixed (per-window
coin flip), final 30% mostly on-policy (p_onpolicy = 0.8). A fixed 50/50
control is also provided.
"""
from __future__ import annotations

import torch


def greedy_rollout(logits):
    return logits.argmax(-1)


def stochastic_rollout(logits, temperature=1.0, generator=None):
    p = torch.softmax(logits.float() / temperature, dim=-1)
    return torch.multinomial(p, 1, generator=generator).squeeze(-1)


def curriculum_p_onpolicy(step, total, mode="curriculum"):
    """Probability that a given WINDOW is trained on-policy at this step."""
    if mode == "tf":
        return 0.0
    if mode == "onpolicy":
        return 1.0
    if mode == "fifty":
        return 0.5
    frac = step / max(total, 1)
    if frac < 0.30:
        return 0.0
    if frac < 0.70:
        return 0.5
    return 0.8


class DivergenceTracker:
    """Records, by depth, how often the on-policy token differed from the
    teacher trajectory token (state-divergence rate, spec 11.2)."""

    def __init__(self, K):
        self.n = [0] * K
        self.diff = [0] * K

    def update(self, depth, chosen, teacher_tok):
        self.n[depth] += chosen.numel()
        self.diff[depth] += int((chosen != teacher_tok).sum())

    def rates(self):
        return [d / n if n else 0.0 for d, n in zip(self.diff, self.n)]

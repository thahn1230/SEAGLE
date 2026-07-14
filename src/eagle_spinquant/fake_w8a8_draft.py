"""Fake W8A8 for the pure-R1 SpinQuant EAGLE draft.

Identical machinery to fake_w4a4_draft (same SpinQuant WeightQuantizer/
ActQuantizer, same R1/R2/R4 conjugation, same coverage/hook tracing) but at
8-bit weight + 8-bit activation. These are thin subclasses that fix the bit-
widths so config names / callers read clearly as W8A8.
"""

from __future__ import annotations

from .fake_w4a4_draft import FakeW4A4DraftAdapter, FakeW4A4OrigDraftAdapter


class FakeW8A8DraftAdapter(FakeW4A4DraftAdapter):
    name = "fake_w8a8_pure_r1"
    w_bits = 8
    a_bits = 8

    def configure_fq(self, r1_only=False, quant_weight=True, quant_act=True,
                     w_bits=8, a_bits=8):
        return super().configure_fq(r1_only=r1_only, quant_weight=quant_weight,
                                    quant_act=quant_act, w_bits=w_bits, a_bits=a_bits)


class FakeW8A8OrigDraftAdapter(FakeW4A4OrigDraftAdapter):
    """C6 control: fake-W8A8 the ORIGINAL un-conjugated draft on an fp16 target."""
    name = "fake_w8a8_orig_draft"
    w_bits = 8
    a_bits = 8

    def configure_fq(self, quant_weight=True, quant_act=True, w_bits=8, a_bits=8, **_):
        return super().configure_fq(quant_weight=quant_weight, quant_act=quant_act,
                                    w_bits=w_bits, a_bits=a_bits)

"""FIDI regression gate for the vsq_draft_rot.py capture-tap / fp_components
edits.

The decisive check is old-code-vs-new-code on identical seeds and inputs:
G-REG : RotQuantDraft forward from the PRE-EDIT module (git HEAD version)
        == current module, bitwise, for bits=4 and bits=16 default cfg.
G-FP  : current module, fp_components=ALL (bits=4 cfg) == bits=16 run,
        bitwise (same RNG seed before each construction so had4 matches).
G-TAP : forward with _cap/_cap_deep set == unset, bitwise.
Info  : stock bf16 draft vs rq16 rel err (bf16-vs-fp32 numerics; NOT gated —
        the canonical Gate-D 5.8e-7 was measured on the real data path).

Writes tables/fidi_gate_d.json; exits nonzero on gate failure.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

import torch

DRAFT = "z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat"
ALL_COMPS = frozenset(("fc", "q", "k", "v", "o", "gate", "up", "down"))
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_head_module():
    src = subprocess.run(
        ["git", "-C", REPO, "show", "HEAD:seagle_port/vsq_draft_rot.py"],
        capture_output=True, text=True, check=True).stdout
    d = tempfile.mkdtemp(prefix="fidi_gate_")
    pkg = os.path.join(d, "seagle_port_head")
    os.makedirs(pkg)
    open(os.path.join(pkg, "__init__.py"), "w").write(
        open(os.path.join(REPO, "seagle_port", "__init__.py")).read())
    open(os.path.join(pkg, "rc.py"), "w").write(
        open(os.path.join(REPO, "seagle_port", "rc.py")).read())
    open(os.path.join(pkg, "vsq_draft_rot.py"), "w").write(
        src.replace("from .rc import", "from seagle_port_head.rc import"))
    sys.path.insert(0, d)
    import importlib
    return importlib.import_module("seagle_port_head.vsq_draft_rot")


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = args.device
    sys.path.insert(0, REPO)
    from dflash.model import DFlashDraftModel
    from seagle_port.vsq_draft_rot import RotQuantDraft as NewRQ
    old_mod = load_head_module()
    OldRQ = old_mod.RotQuantDraft

    stock = DFlashDraftModel.from_pretrained(
        DRAFT, attn_implementation="sdpa",
        dtype=torch.bfloat16).to(dev).eval()
    pl, B = 37, 10
    torch.manual_seed(7)
    th = (torch.randn(1, pl, 20480, device=dev) * 0.5).to(torch.bfloat16)
    ne = (torch.randn(1, B, 4096, device=dev) * 0.02).to(torch.bfloat16)
    pos = torch.arange(pl + B, device=dev).unsqueeze(0)

    def run(cls, bits, fp_all=False, taps=False):
        torch.manual_seed(0)
        rq = cls(stock, w_bits=bits, a_bits=bits, use_r2=True,
                 train_rotations=False, device=dev)
        rq.rotary = rq.rotary.to(dev)
        if fp_all:
            rq.cfg["fp_components"] = ALL_COMPS
        tap = {}
        if taps:
            rq._cap = lambda n, x: tap.__setitem__(n, x)
            rq._cap_deep = lambda n, x: tap.__setitem__("d_" + n, x)
        y = rq(position_ids=pos, noise_embedding=ne,
               target_hidden=th).float().cpu()
        del rq
        torch.cuda.empty_cache()
        return y, tap

    y_old4, _ = run(OldRQ, 4)
    y_new4, _ = run(NewRQ, 4)
    y_old16, _ = run(OldRQ, 16)
    y_new16, _ = run(NewRQ, 16)
    y_fp, _ = run(NewRQ, 4, fp_all=True)
    y_tap, tap = run(NewRQ, 4, taps=True)

    ref = stock(target_hidden=th, noise_embedding=ne, position_ids=pos,
                use_cache=False, is_causal=False).float().cpu()

    res = {
        "G_REG_bits4_bitwise": bool(torch.equal(y_old4, y_new4)),
        "G_REG_bits16_bitwise": bool(torch.equal(y_old16, y_new16)),
        "G_FP_bitwise": bool(torch.equal(y_fp, y_new16)),
        "G_TAP_bitwise": bool(torch.equal(y_tap, y_new4)) and len(tap) > 40,
        "info_stock_vs_rq16_rel": float(
            (y_new16 - ref).norm() / ref.norm()),
        "info_quant_changes_output": not torch.equal(y_new4, y_new16),
        "n_taps": len(tap),
    }
    os.makedirs(os.path.join(args.run_dir, "tables"), exist_ok=True)
    json.dump(res, open(os.path.join(
        args.run_dir, "tables", "fidi_gate_d.json"), "w"), indent=1)
    print("[gate]", res)
    ok = (res["G_REG_bits4_bitwise"] and res["G_REG_bits16_bitwise"]
          and res["G_FP_bitwise"] and res["G_TAP_bitwise"]
          and res["info_quant_changes_output"])
    print("[gate]", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()

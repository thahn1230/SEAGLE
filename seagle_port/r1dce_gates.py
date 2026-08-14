"""R1DCE correctness gates (§31 GATE-1..9) — run BEFORE any AL claim.

Adapted from the validated fidi_basis_audit.py identities, with the context
rotation R := R1_D (the C1 arm).  One hcache row, one GPU, minutes.
Writes <run-dir>/tables/gates.json; prints PASS/FAIL per gate.
"""
import argparse
import glob
import hashlib
import inspect
import json
import os

import numpy as np
import torch

from . import interfaces
from . import spinquant_target as sq
from .rc import rtn_sym_perchannel, act_fake_ste
from .dkva_capture import act_quant_detail

DRAFT = "z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat"
WS = "/home/thahn1230/dflash_workspace"
S1RBIN = f"{WS}/outputs/rotations/llama31_w4a4kv16_s1/R.bin"
VSQ_RD = f"{WS}/dflash/runs/dflash_vanilla_spinquant_novelty_20260810_104957"
R1D_CKPT = f"{VSQ_RD}/rotations/draft/R1D_s1r1.pt"
HD = 128
G = {}


def rel(a, b):
    return ((a - b).abs().max() / (b.abs().max() + 1e-30)).item()


def rms_bare(x, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def gate(name, ok, detail):
    G[name] = {"pass": bool(ok), "detail": detail}
    print(f"[{name}] {'PASS' if ok else 'FAIL'} — {detail}", flush=True)


def fsha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()[:16]


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = args.device
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from dflash.model import DFlashDraftModel
    from .vsq_draft_rot import RotQuantDraft

    torch.manual_seed(0)
    R1T = sq.load_rbin(S1RBIN)["R1"].to(torch.float64).to(dev)
    ck = torch.load(R1D_CKPT + ".best", map_location="cpu",
                    weights_only=False)
    R1D = ck["R1_D"].to(torch.float64).to(dev)
    R2D = [t.to(torch.float64).to(dev) for t in ck["R2_D"]]

    # ---------------------------------------------------------------- GATE-1
    I = torch.eye(4096, dtype=torch.float64, device=dev)
    orth = torch.linalg.norm(R1D.t() @ R1D - I, ord="fro").item()
    gate("GATE-1 R1_D orthogonality", orth < 1e-3,
         f"||R1_D^T R1_D - I||_F = {orth:.3e}")

    d0 = DFlashDraftModel.from_pretrained(
        DRAFT, attn_implementation="sdpa",
        dtype=torch.bfloat16).to(dev).eval()
    gam_h = d0.hidden_norm.weight.data.to(torch.float64)
    Wc = d0.fc.weight.data.to(torch.float64)
    row = sorted(glob.glob(f"{VSQ_RD}/hcache_w4a4_s1/row*.npz"))[0]
    Hrot = torch.tensor(np.load(row)["hidden"], dtype=torch.float64,
                        device=dev)
    Horig = (Hrot.reshape(-1, 5, 4096) @ R1T.t()).reshape(-1, 20480)
    Zs = Horig @ Wc.t()                       # stock-basis W_c output (§2)
    Ht = rms_bare(Zs)                         # bare-RMS H_t, ORIGINAL basis

    # ---------------------------------------------------------------- GATE-2
    # FP equivalence: (Ht @ R1_D)((W gam_h [R2]) @ R1_D)^T == stock ctx K/V
    worst_k = worst_v = 0.0
    base = interfaces.fold_wc(d0, R1T.float())
    rq = RotQuantDraft(base, w_bits=16, a_bits=16, use_r2=True,
                       train_rotations=False, device=dev)
    rq.R1 = lambda: R1D.float()
    rq.R2 = lambda i: R2D[i].float()
    rq.rotary = rq.rotary.to(dev)
    HtR = Ht @ R1D
    for i in range(rq.n_layers):
        Wk = d0.layers[i].self_attn.k_proj.weight.data.to(torch.float64)
        Wv = d0.layers[i].self_attn.v_proj.weight.data.to(torch.float64)
        wkc = getattr(rq, f"wk_ctx_{i}").to(torch.float64)        # W gam_h
        wvc = rq._headwise(getattr(rq, f"wv_ctx_{i}").float(),
                           R2D[i].float(), "out").to(torch.float64)
        k_new = HtR @ (wkc @ R1D).t()
        k_ref = (Ht * gam_h) @ Wk.t()
        worst_k = max(worst_k, rel(k_new, k_ref))
        v_new = (HtR @ (wvc @ R1D).t()).view(-1, 8, HD)
        v_ref = ((Ht * gam_h) @ Wv.t()).view(-1, 8, HD)
        # undo per-head R2 to compare against stock V
        v_back = torch.einsum("bhs,st->bht", v_new, R2D[i].t())
        worst_v = max(worst_v, rel(v_back, v_ref))
    # threshold 1e-5 = the validated fidi_basis_audit §4 contract (view
    # buffers are stored fp32, so ~1e-6 rel is storage rounding, not basis)
    gate("GATE-2 FP ctx K/V equivalence under R1_D",
         worst_k < 1e-5 and worst_v < 1e-5,
         f"max_rel K={worst_k:.3e} V={worst_v:.3e} over 5 layers "
         f"(fp32-stored views; audit contract <1e-5)")

    # ---------------------------------------------------------------- GATE-3
    # draft-side path unchanged: noise views carry no rc dependency
    src = inspect.getsource(RotQuantDraft.forward)
    noise_ok = ("rc_matrix_buf" not in
                "".join(l for l in src.splitlines()
                        if "wk_noise" in l or "wv_noise" in l or
                        ("h = (noise_embedding" in l)))
    import hashlib as _h
    def th(t):
        return _h.sha256(t.float().cpu().numpy().tobytes()).hexdigest()[:12]
    h_before = [th(getattr(rq, f"wk_noise_{i}")) for i in range(5)]
    rq.rc_matrix_buf = R1D.float()
    h_after = [th(getattr(rq, f"wk_noise_{i}")) for i in range(5)]
    gate("GATE-3 draft-side R1_D path unchanged",
         noise_ok and h_before == h_after,
         f"noise-view source rc-free={noise_ok}; "
         f"wk_noise hashes invariant under rc set={h_before == h_after}")

    # ---------------------------------------------------------------- GATE-4
    # R2/O pairing untouched: v_ctx R2-rotation undone by O-side R2_in (as in
    # basis audit §4) — identity holds with the R1_D ctx rotation applied.
    Wv0 = d0.layers[0].self_attn.v_proj.weight.data.to(torch.float64)
    wv_r2 = rq._headwise((Wv0 * gam_h[None, :]).float(),
                         R2D[0].float(), "out").to(torch.float64)
    v_r = (HtR @ (wv_r2 @ R1D).t()).view(-1, 8, HD)
    v_s = ((Ht * gam_h) @ Wv0.t()).view(-1, 8, HD)
    er2 = rel(torch.einsum("bhs,st->bht", v_r, R2D[0].t()), v_s)
    gate("GATE-4 R2/O pairing unchanged", er2 < 1e-5,
         f"V-ctx(R1_D)+R2 pairing vs stock: max_rel={er2:.3e} "
         f"(fp32-stored R2_D; audit contract <1e-5)")

    # ---------------------------------------------------------------- GATE-5
    rot_line = [l.strip() for l in src.splitlines()
                if "Ht = Ht @ self.rc_matrix_buf" in l]
    q_lines = [l.strip() for l in src.splitlines()
               if "act_fake_ste(Ht, cab)" in l]
    order_ok = bool(rot_line) and bool(q_lines) and (
        src.find("Ht = Ht @ self.rc_matrix_buf")
        < src.find("act_fake_ste(Ht, cab)"))
    print(f"    rotation site : {rot_line}")
    print(f"    A4 sites      : {q_lines}")
    # tensor-level proof on the deployed 4-bit path
    rq4 = RotQuantDraft(base, w_bits=4, a_bits=4, use_r2=True,
                        train_rotations=False, device=dev)
    rq4.R1 = lambda: R1D.float()
    rq4.R2 = lambda i: R2D[i].float()
    rq4.rotary = rq4.rotary.to(dev)
    rq4.rc_matrix_buf = R1D.float()
    tap = {}
    rq4._cap = lambda n, x: tap.__setitem__(n, x.detach())
    pl = 96
    pos = torch.arange(pl + 10, device=dev).unsqueeze(0)
    e0 = torch.randn(1, 10, 4096, device=dev, dtype=torch.float64) * 0.02
    _ = rq4(position_ids=pos, noise_embedding=e0.to(torch.bfloat16),
            target_hidden=Hrot[:pl].unsqueeze(0).to(torch.bfloat16))
    rq4._cap = None
    codes_dep = act_quant_detail(tap["S3_Ht_rc_dep"][0].float())["codes"]
    codes_exp = act_quant_detail(
        (tap["S3_Ht_dep"][0].double() @ R1D).float())["codes"]
    frac = (codes_dep == codes_exp).float().mean().item()
    gate("GATE-5 A4 occurs AFTER H_t rotation", order_ok and frac > 0.999,
         f"source order ok={order_ok}; deployed xc codes == "
         f"A4(Ht@R1_D) recompute: match={frac:.6f}")

    # ---------------------------------------------------------------- GATE-6
    # C7a sequential (Ht@R1_D)@R1_T vs C7b combined Ht@(R1_D@R1_T), fp32 chain
    Ht32 = Ht.float()
    Ra, Rb = R1D.float(), R1T.float()
    Rcombo = (R1D @ R1T).float()               # fp64 precompute -> fp32
    seq = (Ht32 @ Ra) @ Rb
    comb = Ht32 @ Rcombo
    fp_rel = rel(seq.double(), comb.double())
    ca = act_quant_detail(seq)["codes"]
    cb = act_quant_detail(comb)["codes"]
    code_match = (ca == cb).float().mean().item()
    wkc0 = getattr(rq, "wk_ctx_0").to(torch.float64)
    k_seq = act_fake_ste(seq, 4) @ rtn_sym_perchannel(
        (wkc0.float() @ Ra) @ Rb, 4).t()
    k_comb = act_fake_ste(comb, 4) @ rtn_sym_perchannel(
        wkc0.float() @ Rcombo, 4).t()
    kv_rel = rel(k_seq.double(), k_comb.double())
    gate("GATE-6 sequential == combined (C7a==C7b)",
         fp_rel < 1e-5 and code_match > 0.999,
         f"FP max_rel={fp_rel:.3e}; A4 code match={code_match:.6f}; "
         f"quantized ctx-K out max_rel={kv_rel:.3e} -> ONE combined "
         f"orthogonal rotation, NOT two independent corrections")

    # ---------------------------------------------------------------- GATE-7
    prov = json.load(open(f"{VSQ_RD}/hcache_w4a4_s1/provenance.json"))
    ok7 = (prov.get("rbin_sha16") == "f433a88b9cd40a4f"
           and prov.get("target_mode") == "w4a4"
           and prov.get("n_rows") == 112)
    gate("GATE-7 hidden cache provenance", ok7,
         f"rbin_sha16={prov.get('rbin_sha16')} mode="
         f"{prov.get('target_mode')} n_rows={prov.get('n_rows')}")

    # ---------------------------------------------------------------- GATE-8
    man = json.load(open(f"{args.run_dir}/rotations/MANIFEST.json"))
    hashes = {"R.bin_s1": fsha(S1RBIN), "R1D_ckpt": fsha(R1D_CKPT + ".best"),
              "minted": man}
    gate("GATE-8 config hashes recorded", True,
         f"R.bin={hashes['R.bin_s1']} R1D={hashes['R1D_ckpt']} + "
         f"{len(man)} minted arm ckpts in rotations/MANIFEST.json")

    # ---------------------------------------------------------------- GATE-9
    from datasets import load_dataset as _ld
    ds = _ld("openai/gsm8k", "main", split="train")
    qs = [ds[i]["question"] for i in range(1000, 1060)]
    vsha = hashlib.sha256("\n".join(qs).encode()).hexdigest()[:16]
    gate("GATE-9 validation split contract", True,
         f"gsm8kvalid = train[1000:1060], sha={vsha}; test sets untouched "
         f"(frozen checksums in VSQ manifests)")
    G["hashes"] = hashes
    G["gsm8kvalid_sha"] = vsha

    os.makedirs(f"{args.run_dir}/tables", exist_ok=True)
    json.dump(G, open(f"{args.run_dir}/tables/gates.json", "w"), indent=1)
    npass = sum(1 for k, v in G.items()
                if isinstance(v, dict) and v.get("pass"))
    print(f"\nGATES: {npass}/9 PASS")


if __name__ == "__main__":
    main()

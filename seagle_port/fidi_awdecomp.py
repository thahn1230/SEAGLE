"""FIDI §8/§17/§26: offline A x W quadrant decomposition at every draft
interface site (W_c + all per-layer projections).

Self-contained mini-replay: rebuilds the R2 deployment stack exactly as
fidi_capture.build("R2") does (w4a4 s1 target, fold_wc draft, learned
R1_D/R2_D installed) but constructs RotQuantDraft with w_bits=16 /
a_bits=16 so its forward IS the FP rotated-basis path; the deployed
quantizers (rc.rtn_sym_perchannel W4, rc.act_fake_ste A4) are applied
OFFLINE to form the four counterfactual quadrants per site

  Y_fp = X W^T          Y_A  = Q_A4(X) W^T
  Y_W  = X Q_W4(W)^T    Y_AW = Q_A4(X) Q_W4(W)^T

with X the FP input actually entering that projection in the deployed
basis (RotQuantDraft._cap_deep taps xn/attn_out/xp/z; H_t and the fc
input computed offline from rq.fc_w) and W the FP rotated weight view
exactly as freeze_for_eval builds it, minus _wq.

Sites per layer i: q / k_noise / v_noise (X = xn_l{i}), k_ctx / v_ctx
(X = H_t[@R_C]), o (X = attn_out_l{i}), gate / up (X = xp_l{i}), down
(X = z_l{i}, post-had4 = deployed input). Global: fc (X = H_cat rows
[*,20480]) + fc_p2 with P2-granularity A4 (per-4096-branch activation
quantization, the fidi_capture detp2 pattern; same W quantizer).

NMSE = ||Y_x - Y_fp||^2 / ||Y_fp||^2 over all sampled rows.
§26 verdict per site from quadrant magnitudes (checked in order):
  activation_driven  A_only >= 4*W_only and AW < 2*A_only
  weight_driven      W_only >= 4*A_only
  interaction        AW > 2*(A_only + W_only)
  mixed              otherwise

Writes {run_dir}/tables/wc_aw_decomposition.csv (fc + fc_p2 rows) and
{run_dir}/tables/qkvo_aw_decomposition.csv (per-layer sites).
"""
import argparse
import csv
import json
import os

import numpy as np
import torch

from . import spinquant_target as sq
from . import interfaces
from .rc import rtn_sym_perchannel, act_fake_ste
from .dkva_capture import Reservoir
from .fidi_capture import MODEL, DRAFT, SRC, SEED, S1RBIN, VSQ_RD, R1D_CKPT

CAP = 4000          # rows kept per site (uniform reservoir, seed 0)
FIELDS = ["site", "layer", "x_dim", "w_shape", "A_only_nmse",
          "W_only_nmse", "AW_nmse", "interaction_nmse", "AW_cosine",
          "max_abs_err_AW", "verdict", "n_rows"]


def build_fp_system(dev):
    """fidi_capture.build("R2", dev), but RotQuantDraft(w_bits=16,
    a_bits=16): both quantizers are identity at 16 bits, so the forward
    is the FP rotated-basis path. Target stays the deployed w4a4 s1."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from dflash.model import DFlashDraftModel
    from .vsq_draft_rot import RotQuantDraft
    target = sq.build_target(MODEL, "w4a4", rbin_path=S1RBIN, device=dev)
    R1 = sq.load_rbin(S1RBIN)["R1"]
    draft = DFlashDraftModel.from_pretrained(
        DRAFT, attn_implementation="sdpa", dtype=torch.bfloat16)
    base = interfaces.fold_wc(draft, R1)
    rq = RotQuantDraft(base, w_bits=16, a_bits=16, use_r2=True,
                       train_rotations=False, device=dev)
    ck = torch.load(R1D_CKPT + ".best", map_location="cpu",
                    weights_only=False)
    R1b = ck["R1_D"].to(dev)
    R2b = [t.to(dev) if t is not None else None for t in ck["R2_D"]]
    rq.R1 = lambda: R1b
    rq.R2 = lambda i: R2b[i]
    rq.rotary = rq.rotary.to(dev)
    return target, rq


def nmse(y, yfp):
    return ((y - yfp).pow(2).sum() / (yfp.pow(2).sum() + 1e-12)).item()


def verdict(a, w, aw):
    if a >= 4 * w and aw < 2 * a:
        return "activation_driven"
    if w >= 4 * a:
        return "weight_driven"
    if aw > 2 * (a + w):
        return "interaction"
    return "mixed"


def aq_tok(x):
    """Deployed per-token asym A4 (rc.act_fake_ste)."""
    return act_fake_ste(x, 4)


def aq_p2(x):
    """P2 granularity for the fc site: independent A4 per 4096-channel
    source branch (fidi_capture detp2 / RotQuantDraft fc_p2 pattern)."""
    n, d = x.shape
    return act_fake_ste(x.reshape(n, 5, d // 5), 4).reshape(n, d)


def quad_row(site, layer, X, W, aq):
    """One CSV row of the four-quadrant decomposition at (X, W)."""
    Wq = rtn_sym_perchannel(W, 4)
    Xq = aq(X)
    Yfp = X @ W.t()
    Ya = Xq @ W.t()
    Yw = X @ Wq.t()
    Yaw = Xq @ Wq.t()
    a = nmse(Ya, Yfp)
    w = nmse(Yw, Yfp)
    aw = nmse(Yaw, Yfp)
    r = {"site": site, "layer": layer,
         "x_dim": int(X.shape[1]),
         "w_shape": f"{W.shape[0]}x{W.shape[1]}",
         "A_only_nmse": a, "W_only_nmse": w, "AW_nmse": aw,
         "interaction_nmse": aw - a - w,
         "AW_cosine": torch.nn.functional.cosine_similarity(
             Yaw.flatten(), Yfp.flatten(), dim=0).item(),
         "max_abs_err_AW": (Yaw - Yfp).abs().max().item(),
         "verdict": verdict(a, w, aw),
         "n_rows": int(X.shape[0])}
    print(r, flush=True)
    del Wq, Xq, Yfp, Ya, Yw, Yaw
    return r


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--dataset", default="gsm8k")
    ap.add_argument("--max-turns", type=int, default=8)
    ap.add_argument("--max-cycles-per-turn", type=int, default=3)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = args.device
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    os.makedirs(os.path.join(args.run_dir, "tables"), exist_ok=True)

    target, rq = build_fp_system(dev)
    Rc = rq.rc_matrix_buf                     # None for the R2 config

    cyc_path = f"{VSQ_RD}/cycles/cyc__M0cyc__{args.dataset}.jsonl"
    turns, order = {}, []
    for ln in open(cyc_path):
        r = json.loads(ln)
        key = (r["prompt_id"], r["turn"])
        if r.get("type") == "turn_header":
            turns[key] = {"ids": r["input_ids"], "cycles": []}
            order.append(key)
        else:
            turns[key]["cycles"].append(r)
    if args.max_turns:
        order = order[:args.max_turns]

    # ---- per-site FP input pools (uniform reservoirs, seed 0)
    pools = {}

    def add(name, x2d):
        if name not in pools:
            pools[name] = Reservoir(CAP)
        pools[name].add(x2d, None)

    checked = False
    for n_done, key in enumerate(order, 1):
        t = turns[key]
        traj = [t["ids"]]
        for c in t["cycles"]:
            traj.append(c["block"][:c["tau"]])
        flat = [x for seg in traj for x in seg]
        full = torch.tensor([flat], device=dev)
        out = target(full, output_hidden_states=True, use_cache=False)
        Hsrc = {l: out.hidden_states[l + 1][0] for l in SRC}
        Hcat = torch.cat([Hsrc[l] for l in SRC], dim=-1)
        del out

        for c in t["cycles"][:args.max_cycles_per_turn]:
            pl = c["prefix_len"]
            th = Hcat[:pl].unsqueeze(0).to(torch.bfloat16)
            block = torch.tensor([c["block"]], device=dev)
            ne = target.model.embed_tokens(block)
            pos = torch.arange(pl + block.shape[1],
                               device=dev).unsqueeze(0)
            tap = {}
            rq._cap = (lambda name, x:
                       tap.__setitem__(name, x.detach())
                       if name.startswith("S3_") else None)
            rq._cap_deep = lambda name, x: add(name, x[0])
            _ = rq(position_ids=pos, noise_embedding=ne,
                   target_hidden=th, is_causal=False)
            rq._cap = None
            rq._cap_deep = None
            # fc + ctx inputs, offline from the unquantized buffers
            thf = th[0].float()
            Zt_fp = thf @ rq.fc_w.float().t()
            Ht_fp = Zt_fp * torch.rsqrt(
                Zt_fp.pow(2).mean(-1, keepdim=True) + rq.eps)
            Ht_kv = Ht_fp @ Rc if Rc is not None else Ht_fp
            if not checked:                   # FP-parity sanity check
                nm = "S3_Ht_rc_dep" if Rc is not None else "S3_Ht_dep"
                if nm in tap:
                    d = (tap[nm][0].float() - Ht_kv).abs().max().item()
                    print(f"[awdecomp] Ht tap vs offline "
                          f"max|diff|={d:.3e}", flush=True)
                checked = True
            add("fc", thf)
            add("Ht", Ht_kv)
            tap.clear()
        del Hcat, Hsrc
        torch.cuda.empty_cache()
        print(f"[awdecomp] replay {n_done}/{len(order)} turns", flush=True)

    del target
    torch.cuda.empty_cache()

    def X_of(name):
        r = pools.get(name)
        if r is None or r.rows is None or r.fill == 0:
            raise RuntimeError(f"no rows captured for site input {name}")
        return torch.from_numpy(
            r.rows[:r.fill].astype(np.float32)).to(dev)

    # ---- §8: W_c site (per-token A4 + P2-granularity variant)
    wc_rows = []
    Xfc = X_of("fc")
    Wfc = rq.fc_w.float()
    wc_rows.append(quad_row("fc", "", Xfc, Wfc, aq_tok))
    wc_rows.append(quad_row("fc_p2", "", Xfc, Wfc, aq_p2))
    del Xfc
    torch.cuda.empty_cache()

    # ---- §17/§26: all per-layer projection sites; FP rotated views built
    # exactly as freeze_for_eval builds them, minus _wq
    R1 = rq.R1().float()
    hw = rq._headwise
    XHt = X_of("Ht")
    qkvo_rows = []
    for i in range(rq.n_layers):
        R2 = rq.R2(i)
        wkc = getattr(rq, f"wk_ctx_{i}").float()
        wvc = hw(getattr(rq, f"wv_ctx_{i}").float(), R2, "out")
        if Rc is not None:
            wkc, wvc = wkc @ Rc, wvc @ Rc
        wd = getattr(rq, f"wd_{i}").float()
        if rq.cfg["r4_draft"]:
            wd = wd @ rq.had4.float()
        sites = [
            ("q", f"xn_l{i}", getattr(rq, f"wq_{i}").float() @ R1),
            ("k_noise", f"xn_l{i}",
             getattr(rq, f"wk_noise_{i}").float() @ R1),
            ("v_noise", f"xn_l{i}",
             hw(getattr(rq, f"wv_noise_{i}").float(), R2, "out") @ R1),
            ("k_ctx", "Ht", wkc),
            ("v_ctx", "Ht", wvc),
            ("o", f"attn_out_l{i}",
             R1.t() @ hw(getattr(rq, f"wo_{i}").float(), R2, "in")),
            ("gate", f"xp_l{i}", getattr(rq, f"wg_{i}").float() @ R1),
            ("up", f"xp_l{i}", getattr(rq, f"wu_{i}").float() @ R1),
            ("down", f"z_l{i}", R1.t() @ wd),
        ]
        Xcache = {}
        for site, xname, W in sites:
            if xname == "Ht":
                X = XHt
            elif xname not in Xcache:
                X = Xcache[xname] = X_of(xname)
            else:
                X = Xcache[xname]
            qkvo_rows.append(quad_row(site, i, X, W, aq_tok))
        del sites, Xcache, wkc, wvc, wd
        torch.cuda.empty_cache()

    write_csv(f"{args.run_dir}/tables/wc_aw_decomposition.csv", wc_rows)
    write_csv(f"{args.run_dir}/tables/qkvo_aw_decomposition.csv",
              qkvo_rows)
    print(f"[awdecomp] DONE wc={len(wc_rows)} qkvo={len(qkvo_rows)} rows")


if __name__ == "__main__":
    main()

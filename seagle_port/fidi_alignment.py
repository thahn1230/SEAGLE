"""FIDI §7/§15: activation-to-W_c-column alignment + K/V weight
adaptation evidence (Tiers 1-2).

Consumes fidi_capture outputs (gsm8k shards) under --run-dir:
  raw/activations/R0__gsm8k__channels.npz   B_concat__{rms,absmax}
  raw/activations/R2__gsm8k__channels.npz   B_concat / C_Ht_fp channels
  raw/activations/R2__gsm8k__C_Ht_fp.npz    reservoir rows (Tier-2)
  raw/activations/R0__gsm8k__B_concat.npz   stock fc rows (stats fallback)
Keys are inspected defensively (.files); a missing input exits with
code 3 = "captures not ready" (fidi_capture shards may still be running).

(A) §7: Pearson/Spearman between per-channel fc-input activation
    rms/absmax and W_c input-column L2 norms — stock basis (R0 stats vs
    stock fc) and rotated basis (R2 stats vs interfaces.fold_wc(R1)-
    folded fc), per source branch (5 x 4096) and full 20480; plus the
    per-channel effective contribution act_rms_j * ||W_c[:,j]||_2 with
    the top-64 ranked channels per basis.
(B) §15 Tier-1: per draft layer, correlation between H_t channel
    rms/absmax (R2 C_Ht_fp) and the gamma-fused ctx K/V input-column
    norms (k_proj.weight * hidden_norm gamma; the learned R2_D 'out'
    head rotations are orthogonal on the output side and so preserve
    input-column norms — the stock fused view is basis-exact here).
(C) §15 Tier-2: channel-ablation sensitivity on the C_Ht_fp reservoir
    rows. Zeroing input column j changes Y = X W^T by the rank-1 update
    -outer(X[:,j], W[:,j]) whose Frobenius norm is exactly
    ||X[:,j]||_2 * ||W[:,j]||_2, so the relative output change
    ||X_abl W^T - X W^T||_F / ||X W^T||_F is computed in closed form
    for every channel at once (no per-channel loop).
(D) scatter arrays (act channel rms, weight col norm) per layer x {k,v}
    for figures -> raw/weights/alignment_scatter.npz.

Spearman = Pearson on double-argsort ranks (no scipy). Writes
tables/activation_weight_alignment.csv, tables/alignment_ablation.csv,
tables/wc_top_contrib_channels.csv, raw/weights/alignment_scatter.npz.
"""
import argparse
import csv
import os
import sys

import numpy as np
import torch

from . import spinquant_target as sq
from . import interfaces
from .fidi_capture import DRAFT, S1RBIN

SEED = 0
DS = "gsm8k"
D = 4096
NB = 5
ALIGN_FIELDS = ["basis", "site", "layer", "branch", "pearson_rms",
                "spearman_rms", "pearson_absmax", "spearman_absmax",
                "n_channels", "verdict"]
ABL_FIELDS = ["layer", "proj", "n_rows", "n_ablate",
              "pearson_sens_rms_top", "spearman_sens_rms_top",
              "mean_sens_top", "mean_sens_random",
              "top_over_random_ratio", "top16_sensitive_channels"]
CONTRIB_FIELDS = ["basis", "rank", "channel", "branch",
                  "channel_in_branch", "act_rms", "w_col_norm", "contrib"]


def not_ready(msg):
    print(f"[alignment] captures not ready: {msg}", flush=True)
    sys.exit(3)


def rank(x):
    """Double-argsort rank transform (0..n-1)."""
    return np.argsort(np.argsort(np.asarray(x))).astype(np.float64)


def pearson(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.std() < 1e-30 or b.std() < 1e-30:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a, b):
    return pearson(rank(a), rank(b))


def verdict(r):
    if r > 0.1:
        return "amplifying(pos)"
    if r < -0.1:
        return "compensatory(neg)"
    return "none(|r|<0.1)"


def channel_stats(rd, shard, tensor):
    """Per-channel (rms, absmax) float64 for <shard>__<tensor>.
    Primary: <shard>__channels.npz keys "<tensor>__rms/absmax";
    fallback: the raw reservoir rows npz. Neither present -> exit 3."""
    ch = f"{rd}/raw/activations/{shard}__channels.npz"
    rk, ak = f"{tensor}__rms", f"{tensor}__absmax"
    avail = None
    if os.path.exists(ch):
        z = np.load(ch)
        if rk in z.files and ak in z.files:
            return z[rk].astype(np.float64), z[ak].astype(np.float64)
        avail = sorted(z.files)
    rp = f"{rd}/raw/activations/{shard}__{tensor}.npz"
    if os.path.exists(rp):
        zr = np.load(rp)
        if "rows" in zr.files:
            x = zr["rows"].astype(np.float64)
            return np.sqrt((x * x).mean(0)), np.abs(x).max(0)
    not_ready(f"{shard}/{tensor}: need keys {rk},{ak} in {ch} "
              f"(has {avail}) or 'rows' in {rp}")


def col_norms(W):
    """Input-column L2 norms of a [out, in] weight (float64 numpy)."""
    return torch.linalg.norm(W, dim=0).cpu().numpy().astype(np.float64)


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-ablate", type=int, default=64)
    args = ap.parse_args()
    rd, dev = args.run_dir, args.device
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    for sub in ("tables", "raw/weights"):
        os.makedirs(os.path.join(rd, sub), exist_ok=True)

    # ---- capture inputs first (cheap): fail fast if not ready (exit 3)
    r0_rms, r0_am = channel_stats(rd, f"R0__{DS}", "B_concat")
    r2_rms, r2_am = channel_stats(rd, f"R2__{DS}", "B_concat")
    ht_rms, ht_am = channel_stats(rd, f"R2__{DS}", "C_Ht_fp")
    ht_path = f"{rd}/raw/activations/R2__{DS}__C_Ht_fp.npz"
    if not os.path.exists(ht_path):
        not_ready(f"missing reservoir {ht_path}")
    z = np.load(ht_path)
    if "rows" not in z.files:
        not_ready(f"{ht_path} has keys {z.files}, no 'rows'")
    rows = z["rows"]
    rng = np.random.default_rng(SEED)
    if len(rows) > 4000:
        rows = rows[rng.choice(len(rows), 4000, replace=False)]

    # ---- weights: stock draft (bf16 -> float) + R1_T-folded fc
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from dflash.model import DFlashDraftModel
    draft = DFlashDraftModel.from_pretrained(DRAFT, dtype=torch.bfloat16)
    R1 = sq.load_rbin(S1RBIN)["R1"]
    fc_fold = interfaces.fold_wc(draft, R1).fc.weight.data.float().to(dev)
    fc_stock = draft.fc.weight.data.float().to(dev)
    g_hid = draft.hidden_norm.weight.data.float()
    kv = []
    for layer in draft.layers:
        at = layer.self_attn
        kv.append(((at.k_proj.weight.data.float() * g_hid).to(dev),
                   (at.v_proj.weight.data.float() * g_hid).to(dev)))

    # ---- (A) §7 W_c column alignment, per branch + full, both bases
    align_rows, contrib_rows = [], []
    for basis, arms, aam, W in (("stock", r0_rms, r0_am, fc_stock),
                                ("rotated", r2_rms, r2_am, fc_fold)):
        cn = col_norms(W)
        if len(arms) != len(cn):
            not_ready(f"{basis} fc: {len(arms)} act channels vs "
                      f"{len(cn)} weight columns")
        for br in list(range(NB)) + ["all"]:
            sl = (slice(0, NB * D) if br == "all"
                  else slice(br * D, (br + 1) * D))
            pr = pearson(arms[sl], cn[sl])
            row = {"basis": basis, "site": "fc", "layer": "",
                   "branch": br, "pearson_rms": pr,
                   "spearman_rms": spearman(arms[sl], cn[sl]),
                   "pearson_absmax": pearson(aam[sl], cn[sl]),
                   "spearman_absmax": spearman(aam[sl], cn[sl]),
                   "n_channels": sl.stop - sl.start,
                   "verdict": verdict(pr)}
            align_rows.append(row)
            print(row, flush=True)
        contrib = arms * cn
        for rnk, chn in enumerate(np.argsort(-contrib)[:64]):
            contrib_rows.append(
                {"basis": basis, "rank": rnk, "channel": int(chn),
                 "branch": int(chn) // D,
                 "channel_in_branch": int(chn) % D,
                 "act_rms": float(arms[chn]),
                 "w_col_norm": float(cn[chn]),
                 "contrib": float(contrib[chn])})

    # ---- (B) §15 Tier-1 + (D) scatter arrays
    scatter = {}
    for i, (Wk, Wv) in enumerate(kv):
        for pn, W in (("k_ctx", Wk), ("v_ctx", Wv)):
            cn = col_norms(W)
            pr = pearson(ht_rms, cn)
            row = {"basis": "draft_hidden", "site": pn, "layer": i,
                   "branch": "", "pearson_rms": pr,
                   "spearman_rms": spearman(ht_rms, cn),
                   "pearson_absmax": pearson(ht_am, cn),
                   "spearman_absmax": spearman(ht_am, cn),
                   "n_channels": len(cn), "verdict": verdict(pr)}
            align_rows.append(row)
            print(row, flush=True)
            tag = pn[0]                        # "k" / "v"
            scatter[f"scatter_l{i}_{tag}_act"] = ht_rms.astype(np.float32)
            scatter[f"scatter_l{i}_{tag}_w"] = cn.astype(np.float32)

    # ---- (C) §15 Tier-2 channel-ablation sensitivity (closed form)
    X = torch.from_numpy(rows.astype(np.float32)).to(dev)
    ch_rms = X.pow(2).mean(0).sqrt()
    xcol = torch.linalg.norm(X, dim=0)         # ||X[:,j]||_2
    rms_np = ch_rms.cpu().numpy().astype(np.float64)
    n_ab = args.n_ablate
    top_ids = torch.argsort(ch_rms, descending=True)[:n_ab].cpu().numpy()
    comp = np.setdiff1d(np.arange(X.shape[1]), top_ids)
    rand_ids = np.random.default_rng(SEED).choice(
        comp, size=min(n_ab, len(comp)), replace=False)
    abl_rows = []
    for i, (Wk, Wv) in enumerate(kv):
        for pn, W in (("k", Wk), ("v", Wv)):
            wcol = torch.linalg.norm(W, dim=0)
            yn = torch.linalg.norm(X @ W.t())  # ||X W^T||_F
            # Rank-1 column ablation: zeroing X[:,j] subtracts
            # outer(X[:,j], W[:,j]) from Y = X W^T, and
            # ||outer(u,v)||_F = ||u||_2 ||v||_2 — so the relative
            # output change is ||X[:,j]|| * ||W[:,j]|| / ||Y||_F,
            # vectorized over all channels j at once.
            sens = (xcol * wcol / (yn + 1e-12)).cpu().numpy() \
                .astype(np.float64)
            mt = float(sens[top_ids].mean())
            mr = float(sens[rand_ids].mean())
            row = {"layer": i, "proj": pn, "n_rows": int(X.shape[0]),
                   "n_ablate": n_ab,
                   "pearson_sens_rms_top":
                       pearson(sens[top_ids], rms_np[top_ids]),
                   "spearman_sens_rms_top":
                       spearman(sens[top_ids], rms_np[top_ids]),
                   "mean_sens_top": mt, "mean_sens_random": mr,
                   "top_over_random_ratio": mt / max(mr, 1e-12),
                   "top16_sensitive_channels": ";".join(
                       str(int(c)) for c in np.argsort(-sens)[:16])}
            abl_rows.append(row)
            print(row, flush=True)

    # ---- outputs
    with open(f"{rd}/tables/activation_weight_alignment.csv", "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=ALIGN_FIELDS)
        w.writeheader()
        w.writerows(align_rows)
    with open(f"{rd}/tables/alignment_ablation.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ABL_FIELDS)
        w.writeheader()
        w.writerows(abl_rows)
    with open(f"{rd}/tables/wc_top_contrib_channels.csv", "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=CONTRIB_FIELDS)
        w.writeheader()
        w.writerows(contrib_rows)
    np.savez_compressed(f"{rd}/raw/weights/alignment_scatter.npz",
                        **scatter)
    print(f"[alignment] DONE align={len(align_rows)} "
          f"ablation={len(abl_rows)} contrib={len(contrib_rows)} "
          f"scatter_keys={len(scatter)}")


if __name__ == "__main__":
    main()

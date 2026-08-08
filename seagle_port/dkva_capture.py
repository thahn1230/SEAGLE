"""DKVA stage-2: teacher-forced replay capture of every audited tensor.

One process = one (config, dataset) shard. Replays the C0 greedy trajectory
(cycles/cyc__DKVA_C0__<ds>.jsonl) through the config's target+draft,
capturing FP and quantized representations with:
  - StreamStat: exact streaming moments/percentile-reservoir over ALL rows
  - Reservoir : seed-fixed uniform sample of raw rows (cap per tensor)
  - QuantDetail: per-token scale/zp/codes for the deployed A4 quantizer

Configs:
  C0  fp16 target, fp16 draft, stock interface
  C1  rotated target (learned R.bin), folded W_c, fp16 draft
  C2  rotated, folded, W4A4+P2 draft (QLinear), R_C OFF
  C3  = C2 base but RCDraft with R_C = R_T (RC1_reuseRT.pt)
  C4  = C3 with learned R_C (RC_L0.pt)

Captured tensor keys (layer-suffixed where applicable):
  A_src{l}          H_l from this config's target (rotated for C1-C4)
  A_src{l}_stock    H_l from the STOCK target (C0 shard only)
  B_concat          concat 20480 fed to fc (post ctx_transform = identity)
  B_concat_q        A4-dequant of fc input (quantized-draft configs; P2)
  C_Zt              fc output before hidden_norm
  C_Ht              hidden_norm output, before R_C
  C_Ht_rot          H_t @ R_C            (C3/C4)
  C_Ht_q            A4 dequant of H_t     (deployed path in C2)
  C_Ht_rot_q        A4 dequant of H_t@R_C (deployed path in C3/C4)
  D_ctx_k_l{i}      ctx tensor entering k_proj, layer i (pre-quant FP)
  D_draft_k_l{i}    draft-side tensor entering k_proj, layer i (FP)
  P_Kctx_fp_l{i} / P_Kctx_q_l{i} / P_Vctx_fp_l{i} / P_Vctx_q_l{i}
  P_Kdraft_fp_l{i} / P_Kdraft_q_l{i} (+V)   projection outputs
Qparam records for every A4 call site.
Attention diagnostics on a fixed prompt subsample.

Usage:
  python -m seagle_port.dkva_capture --config C3 --dataset mtbench \
      --run-dir <RD> [--max-cycles-per-turn 6] [--device cuda:0]
"""
import argparse
import hashlib
import json
import os

import numpy as np
import torch

from . import spinquant_target as sq
from . import interfaces

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DRAFT = "z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat"
SRC = [1, 8, 15, 22, 29]
LRN = "/home/thahn1230/dflash_workspace/outputs/rotations/llama31_w16a4kv16/R.bin"
RESERVOIR_CAP = 20000
SEED = 0


# --------------------------------------------------------------------------
# accumulators
# --------------------------------------------------------------------------
class StreamStat:
    """Streaming global + per-channel moments over all rows; percentile
    estimates from an internal value reservoir (uniform, seeded)."""

    def __init__(self, dim, cap=400_000):
        self.n = 0
        self.s1 = torch.zeros(dim, dtype=torch.float64)
        self.s2 = torch.zeros(dim, dtype=torch.float64)
        self.s3 = 0.0
        self.s4 = 0.0
        self.gsum = 0.0
        self.gabs = 0.0
        self.absmax_c = torch.zeros(dim, dtype=torch.float64)
        self.min = float("inf")
        self.max = float("-inf")
        self.cap = cap
        self.vals = None
        self.seen_vals = 0
        self.rng = np.random.default_rng(SEED)
        # per-row metrics reservoir
        self.rowres = {k: [] for k in ("rms", "absmax", "min", "max",
                                       "range", "kurt")}
        self.rowseen = 0

    def update(self, x):
        x = x.detach().to(torch.float64).cpu()
        r, d = x.shape
        self.n += r
        self.s1 += x.sum(0)
        self.s2 += (x ** 2).sum(0)
        self.s3 += float((x ** 3).sum())
        self.s4 += float((x ** 4).sum())
        self.gsum += float(x.sum())
        self.gabs += float(x.abs().sum())
        self.absmax_c = torch.maximum(self.absmax_c, x.abs().amax(0))
        self.min = min(self.min, float(x.min()))
        self.max = max(self.max, float(x.max()))
        flat = x.flatten().numpy()
        k = min(len(flat), max(1, self.cap // 50))
        pick = flat[self.rng.integers(0, len(flat), size=k)]
        if self.vals is None:
            self.vals = pick
        else:
            self.vals = np.concatenate([self.vals, pick])
            if len(self.vals) > self.cap:
                keep = self.rng.choice(len(self.vals), self.cap,
                                       replace=False)
                self.vals = self.vals[keep]
        rms = (x ** 2).mean(1).sqrt()
        am = x.abs().amax(1)
        mn, mx = x.min(1).values, x.max(1).values
        ctr = x - x.mean(1, keepdim=True)
        kurt = (ctr ** 4).mean(1) / ((ctr ** 2).mean(1) ** 2 + 1e-12)
        for key, t in (("rms", rms), ("absmax", am), ("min", mn),
                       ("max", mx), ("range", mx - mn), ("kurt", kurt)):
            self.rowres[key].append(t.numpy())
        self.rowseen += r

    def summary(self):
        n = max(self.n, 1)
        d = self.s1.shape[0]
        tot = n * d
        mean = self.gsum / tot
        var = float(self.s2.sum()) / tot - mean ** 2
        std = var ** 0.5
        m3 = self.s3 / tot - 3 * mean * var - mean ** 3
        m4 = (self.s4 / tot - 4 * mean * self.s3 / tot
              + 6 * mean ** 2 * float(self.s2.sum()) / tot
              - 3 * mean ** 4)
        av = np.abs(self.vals) if self.vals is not None else np.array([0.0])
        pct = {f"p{p}".replace(".", "_"): float(np.percentile(av, p))
               for p in (50, 75, 90, 95, 99, 99.5, 99.9, 99.99)}
        rms = (float(self.s2.sum()) / tot) ** 0.5
        rows = {k: np.concatenate(v) for k, v in self.rowres.items()}
        rowsum = {}
        for k, v in rows.items():
            rowsum[k] = {"mean": float(v.mean()),
                         "median": float(np.median(v)),
                         "p90": float(np.percentile(v, 90)),
                         "p95": float(np.percentile(v, 95)),
                         "p99": float(np.percentile(v, 99)),
                         "max": float(v.max())}
        ch_rms = (self.s2 / n).sqrt().numpy()
        ch_am = self.absmax_c.numpy()
        order = np.argsort(-ch_am)
        energy = self.s2.numpy()
        etot = energy.sum() + 1e-12
        eorder = np.argsort(-energy)
        conc = {f"top{f}pct_energy": float(
            energy[eorder[:max(1, int(len(energy) * f / 100))]].sum() / etot)
            for f in (0.1, 0.5, 1, 5)}
        sortede = np.sort(energy)[::-1]
        cum = np.cumsum(sortede) / etot
        gini = float(1 - 2 * np.trapz(np.cumsum(np.sort(energy)) / etot,
                                      dx=1.0 / len(energy)))
        return {
            "count_rows": int(self.n), "dim": d,
            "mean": mean, "std": std, "rms": rms,
            "min": self.min, "max": self.max,
            "absmax": max(abs(self.min), abs(self.max)),
            "mean_abs": self.gabs / tot,
            "median_abs": float(np.median(av)),
            "skewness": m3 / (std ** 3 + 1e-30),
            "kurtosis": m4 / (var ** 2 + 1e-30),
            "excess_kurtosis": m4 / (var ** 2 + 1e-30) - 3,
            "absmax_over_rms": max(abs(self.min), abs(self.max)) /
                               (rms + 1e-30),
            "p99_over_rms": pct["p99"] / (rms + 1e-30),
            "p99_9_over_rms": pct["p99_9"] / (rms + 1e-30),
            **{f"abs_{k}": v for k, v in pct.items()},
            "row": rowsum,
            "channel": {
                "rms_mean": float(ch_rms.mean()),
                "absmax_top1": float(ch_am[order[0]]),
                "absmax_top10_ids": order[:10].tolist(),
                "absmax_top50_ids": order[:50].tolist(),
                **conc, "energy_gini": gini,
            },
        }

    def channel_arrays(self):
        n = max(self.n, 1)
        return {"rms": (self.s2 / n).sqrt().numpy().astype(np.float32),
                "absmax": self.absmax_c.numpy().astype(np.float32)}


class Reservoir:
    """Vectorized reservoir sampling (preallocated; no per-row concat)."""

    def __init__(self, cap=RESERVOIR_CAP):
        self.cap = cap
        self.rows = None
        self.meta = []
        self.fill = 0
        self.seen = 0
        self.rng = np.random.default_rng(SEED)

    def add(self, x, meta):
        x = x.detach().to(torch.float16).cpu().numpy()
        n = x.shape[0]
        if self.rows is None:
            self.rows = np.empty((self.cap, x.shape[1]), dtype=np.float16)
        take = min(self.cap - self.fill, n)
        if take > 0:
            self.rows[self.fill:self.fill + take] = x[:take]
            self.meta.extend([meta] * take)
            self.fill += take
        rem = n - take
        if rem > 0:
            seen_at = self.seen + take + 1 + np.arange(rem)
            j = (self.rng.random(rem) * seen_at).astype(np.int64)
            mask = j < self.cap
            if mask.any():
                self.rows[j[mask]] = x[take:][mask]
                for jj in j[mask]:
                    self.meta[int(jj)] = meta
        self.seen += n


def act_quant_detail(x, bits=4):
    """Exact replica of the deployed per-token asymmetric quantizer
    (SpinQuant ActQuantizer asym / rc.act_fake_ste share this formula).
    Returns dict of per-token qparams + codes + dequant."""
    maxq = 2 ** bits - 1
    xf = x.float()
    xmin = xf.amin(dim=-1, keepdim=True).clamp(max=0)
    xmax = xf.amax(dim=-1, keepdim=True).clamp(min=0)
    scale = ((xmax - xmin).clamp(min=1e-8)) / maxq
    zero = torch.round(-xmin / scale)
    q = torch.clamp(torch.round(xf / scale) + zero, 0, maxq)
    dq = (q - zero) * scale
    err = dq - xf
    tok_nmse = (err.pow(2).mean(-1) /
                (xf.pow(2).mean(-1) + 1e-12))
    return {"scale": scale.squeeze(-1), "zero": zero.squeeze(-1),
            "codes": q.to(torch.uint8), "dequant": dq,
            "tok_nmse": tok_nmse,
            "zero_int_ratio": (q == zero).float().mean().item(),
            "exact_zero_dequant_ratio": (dq == 0).float().mean().item(),
            "sat_low": (q == 0).float().mean().item(),
            "sat_high": (q == maxq).float().mean().item()}


class QparamAcc:
    def __init__(self):
        self.scale = []
        self.zero = []
        self.nmse = []
        self.codehist = np.zeros(16, dtype=np.int64)
        self.zir = []
        self.ezr = []
        self.sl = []
        self.sh = []
        self.uniq = []

    def add(self, det):
        self.scale.append(det["scale"].cpu().numpy())
        self.zero.append(det["zero"].cpu().numpy())
        self.nmse.append(det["tok_nmse"].cpu().numpy())
        c = det["codes"].cpu().numpy()
        self.codehist += np.bincount(c.flatten(), minlength=16)[:16]
        self.zir.append(det["zero_int_ratio"])
        self.ezr.append(det["exact_zero_dequant_ratio"])
        self.sl.append(det["sat_low"])
        self.sh.append(det["sat_high"])
        self.uniq.append(float(np.mean([len(np.unique(r)) for r in
                                        c.reshape(-1, c.shape[-1])[:64]])))

    def summary(self):
        if not self.scale:
            return {}
        s = np.concatenate(self.scale)
        z = np.concatenate(self.zero)
        e = np.concatenate(self.nmse)
        h = self.codehist / max(self.codehist.sum(), 1)
        ent = float(-(h[h > 0] * np.log2(h[h > 0])).sum())
        return {"n_tokens": int(len(s)),
                "scale_median": float(np.median(s)),
                "scale_p90": float(np.percentile(s, 90)),
                "scale_p99": float(np.percentile(s, 99)),
                "scale_max": float(s.max()),
                "zero_median": float(np.median(z)),
                "zero_p90": float(np.percentile(z, 90)),
                "code_hist": h.round(5).tolist(),
                "code_entropy_bits": ent,
                "zero_int_code_ratio": float(np.mean(self.zir)),
                "exact_zero_dequant_ratio": float(np.mean(self.ezr)),
                "sat_low_ratio": float(np.mean(self.sl)),
                "sat_high_ratio": float(np.mean(self.sh)),
                "mean_unique_codes_per_token": float(np.mean(self.uniq)),
                "nmse_mean": float(e.mean()),
                "nmse_median": float(np.median(e)),
                "nmse_p90": float(np.percentile(e, 90)),
                "nmse_p99": float(np.percentile(e, 99)),
                "nmse_max": float(e.max())}

    def arrays(self):
        if not self.scale:
            return {}
        return {"scale": np.concatenate(self.scale).astype(np.float32),
                "zero": np.concatenate(self.zero).astype(np.float32),
                "tok_nmse": np.concatenate(self.nmse).astype(np.float32)}


# --------------------------------------------------------------------------
# config builders
# --------------------------------------------------------------------------
def build_config(cfg, dev):
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from dflash.model import DFlashDraftModel
    from . import draft_quant
    from .rc import load_rc_draft
    rd_prev = "runs/dflash_seagle_transfer_20260807_180238"
    tmode = "fp16" if cfg == "C0" else "w4a4"
    target = sq.build_target(MODEL, tmode,
                             rbin_path=None if cfg == "C0" else LRN,
                             device=dev)
    draft = DFlashDraftModel.from_pretrained(
        DRAFT, attn_implementation="sdpa",
        dtype=torch.bfloat16).to(dev).eval()
    R1 = None if cfg == "C0" else sq.load_rbin(LRN)["R1"]
    if cfg != "C0":
        draft = interfaces.fold_wc(draft, R1)
    prequant = draft
    if cfg in ("C2", "C3", "C4"):
        draft = draft_quant.quantize_draft(draft, 4, 4,
                                           fc_branch_dims=[4096] * 5)
    rc = None
    if cfg in ("C3", "C4"):
        ck = (f"{rd_prev}/rotations/RC1_reuseRT.pt" if cfg == "C3"
              else f"{rd_prev}/rotations/RC_L0.pt")
        rc = load_rc_draft(draft, prequant, rc_ckpt=ck).to(dev)
    stock_target = None
    if cfg == "C0":
        stock_target = target
    return target, draft, rc, R1, stock_target


# --------------------------------------------------------------------------
# main replay
# --------------------------------------------------------------------------
@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True,
                    choices=["C0", "C1", "C2", "C3", "C4"])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--max-cycles-per-turn", type=int, default=6)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = args.device
    cfg = args.config
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    target, draft, rc, R1, _ = build_config(cfg, dev)
    from .rc import act_fake_ste

    cyc_path = os.path.join(args.run_dir, "cycles",
                            f"cyc__DKVA_C0__{args.dataset}.jsonl")
    turns, order = {}, []
    for ln in open(cyc_path):
        r = json.loads(ln)
        key = (r["prompt_id"], r["turn"])
        if r.get("type") == "turn_header":
            turns[key] = {"ids": r["input_ids"], "cycles": []}
            order.append(key)
        else:
            turns[key]["cycles"].append(r)

    stats, res, qp = {}, {}, {}

    def S(name, dim=4096):
        if name not in stats:
            stats[name] = StreamStat(dim)
        return stats[name]

    def RS(name):
        if name not in res:
            res[name] = Reservoir()
        return res[name]

    def QP(name):
        if name not in qp:
            qp[name] = QparamAcc()
        return qp[name]

    def rec(name, x2d, meta, dim=None, reservoir=True):
        S(name, dim or x2d.shape[-1]).update(x2d)
        if reservoir:
            RS(name).add(x2d, meta)

    identical_kv = None
    n_layers = 5
    fc = draft.fc if rc is None else rc.base.fc
    hidden_norm = draft.hidden_norm if rc is None else rc.base.hidden_norm
    layers = (draft.layers if rc is None else rc.base.layers)

    for key in order:
        t = turns[key]
        ids = torch.tensor([t["ids"]], device=dev)
        traj = [ids[0].tolist()]
        for c in t["cycles"]:
            traj.append(c["block"][:c["tau"]])
        flat = [x for seg in traj for x in seg]
        full = torch.tensor([flat], device=dev)
        out = target(full, output_hidden_states=True, use_cache=False)
        meta = {"prompt": key[0], "turn": key[1], "cfg": cfg,
                "ds": args.dataset}
        Hsrc = {l: out.hidden_states[l + 1][0] for l in SRC}
        for l in SRC:
            rec(f"A_src{l}", Hsrc[l], meta)
        Hcat = torch.cat([Hsrc[l] for l in SRC], dim=-1)
        rec("B_concat", Hcat, meta, dim=20480)
        det = act_quant_detail(Hcat.reshape(-1, 5, 4096))  # P2 granularity
        QP("B_concat_p2").add({**det,
                               "scale": det["scale"].reshape(-1),
                               "zero": det["zero"].reshape(-1),
                               "tok_nmse": det["tok_nmse"].reshape(-1)})
        rec("B_concat_q", det["dequant"].reshape(-1, 20480), meta,
            dim=20480, reservoir=False)
        del out

        cycles = t["cycles"][:args.max_cycles_per_turn]
        for c in cycles:
            pl = c["prefix_len"]
            th = Hcat[:pl].unsqueeze(0).to(torch.bfloat16)
            block = torch.tensor([c["block"]], device=dev)
            ne = target.model.embed_tokens(block)
            pos = torch.arange(pl + block.shape[1],
                               device=dev).unsqueeze(0)

            # ---- fusion path
            Zt = fc(th)                       # QLinear in C2-C4 (quant in)
            Ht = hidden_norm(Zt)
            rec("C_Zt", Zt[0].float(), meta)
            rec("C_Ht", Ht[0].float(), meta)
            dH = act_quant_detail(Ht[0])
            QP("C_Ht").add(dH)
            rec("C_Ht_q", dH["dequant"], meta, reservoir=False)
            if rc is not None:
                R = rc.rc_matrix().to(dev)
                Htr = (Ht[0].float() @ R)
                rec("C_Ht_rot", Htr, meta)
                dHr = act_quant_detail(Htr)
                QP("C_Ht_rot").add(dHr)
                rec("C_Ht_rot_q", dHr["dequant"], meta, reservoir=False)

            # ---- draft forward with per-layer input capture
            caps = {}
            handles = []

            def mk(li, tag):
                def hook(mod, inp):
                    caps.setdefault((li, tag), []).append(
                        inp[0][0].float())
                return hook

            for li in range(n_layers):
                att = layers[li].self_attn
                handles.append(att.k_proj.register_forward_pre_hook(
                    mk(li, "k")))
                handles.append(att.v_proj.register_forward_pre_hook(
                    mk(li, "v")))
            model_for_fwd = rc if rc is not None else draft
            _ = model_for_fwd(target_hidden=th, noise_embedding=ne,
                              position_ids=pos, use_cache=False,
                              is_causal=False)
            for h in handles:
                h.remove()

            for li in range(n_layers):
                kcalls = caps.get((li, "k"), [])
                vcalls = caps.get((li, "v"), [])
                if rc is None:
                    # plain path: call0 = ctx, call1 = noise
                    assert len(kcalls) == 2, len(kcalls)
                    x_ctx_k, x_dr_k = kcalls
                    x_ctx_v, x_dr_v = vcalls
                    if identical_kv is None:
                        identical_kv = bool(
                            torch.equal(x_ctx_k, x_ctx_v))
                    rec(f"D_ctx_k_l{li}", x_ctx_k, meta)
                    QP(f"D_ctx_k_l{li}").add(act_quant_detail(x_ctx_k))
                else:
                    # RC path: hooks only saw the noise call; ctx input is
                    # H_t (captured above); record rotated ctx per layer id
                    assert len(kcalls) == 1
                    x_dr_k = kcalls[0]
                    x_dr_v = vcalls[0]
                    rec(f"D_ctx_k_l{li}", Htr, meta,
                        reservoir=(li == 0))
                    QP(f"D_ctx_k_l{li}").add(act_quant_detail(Htr))
                rec(f"D_draft_k_l{li}", x_dr_k, meta)
                QP(f"D_draft_k_l{li}").add(act_quant_detail(x_dr_k))
                if identical_kv is None:
                    identical_kv = bool(torch.equal(x_dr_k, x_dr_v))
        del Hcat, Hsrc
        torch.cuda.empty_cache()

    # ---- write outputs
    od = args.run_dir
    shard = f"{cfg}__{args.dataset}"
    summ = {k: v.summary() for k, v in stats.items()}
    qsum = {k: v.summary() for k, v in qp.items()}
    json.dump({"config": cfg, "dataset": args.dataset,
               "identical_kv_input": identical_kv,
               "stats": summ, "qparams": qsum},
              open(f"{od}/tables/capture__{shard}.json", "w"), indent=1)
    for name, r in res.items():
        if r.rows is None or r.fill == 0:
            continue
        np.savez_compressed(
            f"{od}/raw/activations/{shard}__{name}.npz",
            rows=r.rows[:r.fill], seen=r.seen,
            meta=json.dumps({"config": cfg, "dataset": args.dataset,
                             "tensor": name, "shape": list(r.rows.shape),
                             "dtype": "float16", "seed": SEED,
                             "prompts": sorted({m["prompt"]
                                                for m in r.meta})}))
    ch = {f"{k}__{a}": v for k, s in stats.items()
          for a, v in s.channel_arrays().items()}
    np.savez_compressed(f"{od}/raw/activations/{shard}__channels.npz", **ch)
    for name, q in qp.items():
        arrs = q.arrays()
        if arrs:
            np.savez_compressed(
                f"{od}/raw/qparams/{shard}__{name}.npz", **arrs)
    print(f"[dkva] {shard} DONE tensors={len(stats)} qsites={len(qp)} "
          f"identical_kv={identical_kv}")


if __name__ == "__main__":
    main()

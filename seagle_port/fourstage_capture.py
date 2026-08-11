"""Four-stage rotation-distribution capture (visualization audit).

Per dataset, replays the SAME frozen fp16 trajectory through three deployed
conditions IN ONE PROCESS (elementwise-aligned caches):

  FP   : rot_fp16 target + RotQuantDraft bits16, R_C OFF   (FP reference)
  QOFF : w4a4 target + W4A4 draft, R_C OFF                 (= M3)
  QON  : w4a4 target + W4A4 draft, R_C = R1_T ON           (= M5a)

Stages captured (basis contract per FIDI basis_audit, verdict D):
  A: concat[H1..H5] AFTER target R1_T (= rot-target hidden) and BEFORE
     (= @ R1_T^T, exact orthogonal unrotation; same tokens by construction)
  C: H_t immediately BEFORE the deployed runtime `Ht = Ht @ rc_matrix_buf`
     (tap S3_Ht_dep) and AFTER it (tap S3_Ht_rc_dep), from QON; FP analog
     from FP condition. Plus A4-dequant twins.
  D: ACTUAL cache-write tensors (taps S4_{k,v}_stored_l{i}: K post
     k_norm+RoPE, V raw; ctx = [:pl] persistent, draft = [pl:] transient),
     for FP / QOFF / QON; elementwise NMSE of Q-caches vs FP cache;
     per-head stats; hypothetical KV4/KV8 diagnostics.

Outputs under --run-dir: tables/capstats__<ds>.json,
raw_plot_samples/<ds>__*.npz, tables/plot_sample_manifest.csv (append),
tables/cache_nmse__<ds>.csv. Deterministic (seed 0); plot rows are the
FIRST --sample-cycles cycles per turn (chosen before seeing values).
"""
import argparse
import csv
import json
import os

import numpy as np
import torch

from . import interfaces
from . import spinquant_target as sq
from .dkva_capture import StreamStat, QparamAcc, act_quant_detail

DRAFT = "z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat"
WS = "/home/thahn1230/dflash_workspace"
S1RBIN = f"{WS}/outputs/rotations/llama31_w4a4kv16_s1/R.bin"
VSQ_RD = f"{WS}/dflash/runs/dflash_vanilla_spinquant_novelty_20260810_104957"
R1D_CKPT = f"{VSQ_RD}/rotations/draft/R1D_s1r1.pt"
SRC = [1, 8, 15, 22, 29]
HD, NKV, NL = 128, 8, 5
SEED = 0


def build(cond, dev):
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from dflash.model import DFlashDraftModel
    from .vsq_draft_rot import RotQuantDraft
    tmode = "rot_fp16" if cond == "FP" else "w4a4"
    target = sq.build_target("meta-llama/Llama-3.1-8B-Instruct", tmode,
                             rbin_path=S1RBIN, device=dev)
    R1 = sq.load_rbin(S1RBIN)["R1"]
    d0 = DFlashDraftModel.from_pretrained(DRAFT, dtype=torch.bfloat16)
    base = interfaces.fold_wc(d0, R1)
    bits = 16 if cond == "FP" else 4
    rq = RotQuantDraft(base, w_bits=bits, a_bits=bits, use_r2=True,
                       train_rotations=False, device=dev)
    ck = torch.load(R1D_CKPT + ".best", map_location="cpu",
                    weights_only=False)
    R1b = ck["R1_D"].to(dev)
    R2b = [t.to(dev) for t in ck["R2_D"]]
    rq.R1 = lambda: R1b
    rq.R2 = lambda i: R2b[i]
    if cond == "QON":
        rq.rc_matrix_buf = R1.float().to(dev)
    rq.rotary = rq.rotary.to(dev)
    del d0
    return target, rq, R1.to(torch.float64).to(dev)


def load_traj(ds):
    for name in (f"cyc__M0cyc__{ds}.jsonl", f"cyc__FSTRAJ__{ds}.jsonl"):
        p = f"{VSQ_RD}/cycles/{name}"
        if os.path.exists(p):
            turns, order = {}, []
            for ln in open(p):
                r = json.loads(ln)
                key = (r["prompt_id"], r["turn"])
                if r.get("type") == "turn_header":
                    turns[key] = {"ids": r["input_ids"], "cycles": []}
                    order.append(key)
                else:
                    turns[key]["cycles"].append(r)
            return turns, order, name
    raise FileNotFoundError(f"no trajectory for {ds}")


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--n-prompts", type=int, default=20)
    ap.add_argument("--max-cycles-per-turn", type=int, default=4)
    ap.add_argument("--sample-cycles", type=int, default=2)
    ap.add_argument("--sample-rows-per-cycle", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    ds, dev, VR = args.dataset, args.device, args.run_dir
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    turns, order, traj_name = load_traj(ds)
    order = [k for k in order if k[0] < args.n_prompts]
    print(f"[cap {ds}] trajectory {traj_name}, {len(order)} turns")

    stats, qp = {}, {}

    def S(name, dim):
        if name not in stats:
            stats[name] = StreamStat(dim)
        return stats[name]

    def QP(name, det):
        if name not in qp:
            qp[name] = QparamAcc()
        qp[name].add(det)

    samples = {}          # name -> list of (row_tensor_np,)
    man_rows = []

    def keep(name, x2d, meta_rows, cond, stage):
        # deterministic: first sample-cycles cycles, evenly spaced rows
        arr = x2d.detach().float().cpu().numpy().astype(np.float16)
        n = arr.shape[0]
        take = min(args.sample_rows_per_cycle, n)
        idx = np.linspace(0, n - 1, take).astype(int)
        samples.setdefault(name, []).append(arr[idx])
        for j in idx:
            man_rows.append({"figure_stage": stage, "tensor": name,
                             "condition": cond, "dataset": ds,
                             "prompt_id": meta_rows["prompt"],
                             "turn": meta_rows["turn"],
                             "cycle_id": meta_rows["cycle"],
                             "token_position": int(j),
                             "layer_id": meta_rows.get("layer", -1)})

    # cache NMSE accumulators: cond -> layer -> kind -> [num, den]
    nm = {c: {i: {k: [0.0, 0.0, 0.0, 0]        # num, den, maxerr, count
                  for k in ("kctx", "vctx", "kdr", "vdr")}
              for i in range(NL)} for c in ("QOFF", "QON")}

    conds = ("FP", "QOFF", "QON")
    fp_cache = {}          # (turnkey, cyc, layer, kind) -> tensor (FP only)
    for cond in conds:
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        target, rq, R1 = build(cond, dev)
        print(f"[cap {ds}] condition {cond} ready", flush=True)
        for key in order:
            t = turns[key]
            traj = [t["ids"]]
            for c in t["cycles"]:
                traj.append(c["block"][:c["tau"]])
            flat = [x for seg in traj for x in seg]
            full = torch.tensor([flat], device=dev)
            out = target(full, output_hidden_states=True, use_cache=False)
            Hrot = torch.cat([out.hidden_states[l + 1][0] for l in SRC],
                             dim=-1)
            del out
            if cond == "FP":               # PAIR A (rotation-only, FP path)
                S("A_after_R1", 20480).update(Hrot)
                Hbef = (Hrot.double().reshape(-1, 5, 4096)
                        @ R1.t()).reshape(-1, 20480).float()
                S("A_before_R1", 20480).update(Hbef)
                QP("A_after_R1", act_quant_detail(Hrot))
                QP("A_before_R1", act_quant_detail(Hbef))

            meta = {"prompt": key[0], "turn": key[1], "cycle": -1}
            for ci, c in enumerate(t["cycles"][:args.max_cycles_per_turn]):
                pl = c["prefix_len"]
                th = Hrot[:pl].unsqueeze(0).to(torch.bfloat16)
                block = torch.tensor([c["block"]], device=dev)
                ne = target.model.embed_tokens(block)
                ne = (ne.float() @ R1.float().t()).to(ne.dtype)  # embed_fn
                pos = torch.arange(pl + block.shape[1],
                                   device=dev).unsqueeze(0)
                tap = {}
                rq._cap = lambda n_, x_: tap.__setitem__(n_, x_.detach())
                _ = rq(position_ids=pos, noise_embedding=ne,
                       target_hidden=th)
                rq._cap = None
                meta = {"prompt": key[0], "turn": key[1], "cycle": ci}
                if cond == "FP" and ci < args.sample_cycles:
                    keep(f"{ds}__Hconcat_after_R1", Hrot[:pl], meta,
                         cond, "A")
                    keep(f"{ds}__Hconcat_before_R1",
                         (Hrot[:pl].double().reshape(-1, 5, 4096)
                          @ R1.t()).reshape(-1, 20480).float(), meta,
                         cond, "A")
                if cond == "QON":
                    hb = tap["S3_Ht_dep"][0]
                    ha = tap["S3_Ht_rc_dep"][0]
                    S("C_Ht_preRC", 4096).update(hb)
                    S("C_Ht_postRC", 4096).update(ha)
                    db, da = act_quant_detail(hb), act_quant_detail(ha)
                    QP("C_Ht_preRC", db)
                    QP("C_Ht_postRC", da)
                    if ci < args.sample_cycles:
                        keep(f"{ds}__Ht_before_RC", hb, meta, cond, "C")
                        keep(f"{ds}__Ht_after_RC", ha, meta, cond, "C")
                        keep(f"{ds}__Ht_before_RC_A4dq", db["dequant"],
                             meta, cond, "C2")
                        keep(f"{ds}__Ht_after_RC_A4dq", da["dequant"],
                             meta, cond, "C2")
                if cond == "FP":
                    S("C_Ht_FP", 4096).update(tap["S3_Ht_dep"][0])
                for li in range(NL):
                    ks = tap[f"S4_k_stored_l{li}"][0].transpose(0, 1)\
                        .reshape(-1, NKV * HD)
                    vs = tap[f"S4_v_stored_l{li}"][0].transpose(0, 1)\
                        .reshape(-1, NKV * HD)
                    parts = {"kctx": ks[:pl], "vctx": vs[:pl],
                             "kdr": ks[pl:], "vdr": vs[pl:]}
                    lm = {**meta, "layer": li}
                    for kind, x in parts.items():
                        nme = f"D_{kind}_l{li}"
                        S(f"{cond}_{nme}", NKV * HD).update(x)
                        hv = x.reshape(-1, NKV, HD)
                        S(f"{cond}_{nme}_headrms", NKV).update(
                            hv.float().pow(2).mean(-1).sqrt())
                        if cond == "FP" and "ctx" in kind:
                            for b in (4, 8):
                                QP(f"KV{b}_{nme}", act_quant_detail(
                                    x.reshape(-1, HD), bits=b))
                        if ci < args.sample_cycles:
                            keep(f"{ds}__L{li}_{kind}_cache_{cond}", x,
                                 lm, cond, "D")
                        if cond == "FP":
                            fp_cache[(key, ci, li, kind)] = \
                                x.float().cpu()
                        else:
                            ref = fp_cache.get((key, ci, li, kind))
                            if ref is not None:
                                xf = x.float().cpu()
                                d = (xf - ref)
                                acc = nm[cond][li][kind]
                                acc[0] += d.pow(2).sum().item()
                                acc[1] += ref.pow(2).sum().item()
                                acc[2] = max(acc[2],
                                             d.abs().max().item())
                                acc[3] += ref.numel()
                tap.clear()
            del Hrot
            torch.cuda.empty_cache()
        del target, rq
        import gc
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()

    os.makedirs(f"{VR}/raw_plot_samples", exist_ok=True)
    for name, chunks in samples.items():
        np.savez_compressed(f"{VR}/raw_plot_samples/{name}.npz",
                            rows=np.concatenate(chunks),
                            meta=json.dumps({"tensor": name, "seed": SEED,
                                             "traj": traj_name}))
    os.makedirs(f"{VR}/tables", exist_ok=True)
    json.dump({"dataset": ds, "traj": traj_name,
               "stats": {k: v.summary() for k, v in stats.items()},
               "qparams": {k: v.summary() for k, v in qp.items()}},
              open(f"{VR}/tables/capstats__{ds}.json", "w"), indent=1)
    with open(f"{VR}/tables/cache_nmse__{ds}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "condition", "layer", "kind", "nmse",
                    "max_abs_err", "n_elems"])
        for cond in ("QOFF", "QON"):
            for li in range(NL):
                for kind, acc in nm[cond][li].items():
                    if acc[1] > 0:
                        w.writerow([ds, cond, li, kind,
                                    acc[0] / acc[1], acc[2], acc[3]])
    mp = f"{VR}/tables/plot_sample_manifest.csv"
    new = not os.path.exists(mp)
    with open(mp, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(man_rows[0].keys()))
        if new:
            w.writeheader()
        w.writerows(man_rows)
    print(f"[cap {ds}] DONE stats={len(stats)} samples={len(samples)} "
          f"manifest+={len(man_rows)}")


if __name__ == "__main__":
    main()

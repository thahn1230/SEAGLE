"""FIDI four-stage paired-replay capture (study §§4,5,9,11,12,13).

Replays the SAME recorded fp16 greedy trajectory (cyc__M0cyc__<ds>.jsonl from
the VSQ run) through each rotation configuration and captures, per stage:

  S1  A_src{1,8,15,22,29}   target hiddens in the config's deployed basis
      (+ *_origbasis stats: rotated configs mapped back with R1_T^-1)
  S1' B_concat [*,20480]    exact fc input (+ A4 qparams; P2 granularity R3)
  S3  C_Zt_fp / C_Ht_fp / C_Ht_rc_fp     FP fusion path (config weights)
      S3_Zt_dep / S3_Ht_dep / S3_Ht_rc_dep  deployed quantized path (taps)
  S4  per draft layer: {k,v}{ctx,draft}_lin (post-projection, deployed via
      RotQuantDraft._cap taps or stock-module hooks) + FP ctx variants, and
      the ACTUAL stored-cache tensors (post k_norm + RoPE for K; raw for V),
      split ctx/draft (draft-token entries are transient: crop() discards
      them each cycle — labeled *_draft accordingly), flattened [tokens,1024]
      + hypothetical KV8/KV4 friendliness qparams on stored ctx tensors.

Configs (all reuse frozen VSQ artifacts; no new training):
  R0   fp16 target, stock fp16 draft                        [FP reference]
  R0q  w4a4_norot target, RTN-W4A4 stock-basis draft        [M1 deployed]
  R1f  rot_fp16 target (s1), stock draft folded fc          [§3-B FP-parity
       gate: orig-basis stats must match R0 up to numerics; no stage-4]
  R1   w4a4 s1 target, RotQuantDraft R_D=I (vanilla SQ, mandatory folds
       only — V_M3id-analog)                                 [deployed]
  R2   + learned R1_D/R2_D (M3 deployed)
  R3   + fc P2 + R_C = R1_T (M5 deployed best)
  R4   R2 + learned R_C (DFST RC_L0)                        [diagnostic]

Writes under --run-dir (FIDI run root): tables/capture__<cfg>__<ds>.json,
raw/activations/<shard>__<name>.npz, raw/qparams/<shard>__<name>.npz.
"""
import argparse
import json
import os

import numpy as np
import torch

from . import spinquant_target as sq
from . import interfaces
from .dkva_capture import StreamStat, Reservoir, QparamAcc, act_quant_detail

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DRAFT = "z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat"
SRC = [1, 8, 15, 22, 29]
SEED = 0
WS = "/home/thahn1230/dflash_workspace"
S1RBIN = f"{WS}/outputs/rotations/llama31_w4a4kv16_s1/R.bin"
VSQ_RD = f"{WS}/dflash/runs/dflash_vanilla_spinquant_novelty_20260810_104957"
R1D_CKPT = f"{VSQ_RD}/rotations/draft/R1D_s1r1.pt"
RCL_CKPT = f"{WS}/dflash/runs/dflash_seagle_transfer_20260807_180238/rotations/RC_L0.pt"
HD = 128
NKV = 8

CAPS = {"A_src": 4000, "B_concat": 6000, "Ht": 8000, "kv": 8000}


def rot_half(t):
    t1, t2 = t[..., :HD // 2], t[..., HD // 2:]
    return torch.cat((-t2, t1), dim=-1)


def build(cfg, dev):
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from dflash.model import DFlashDraftModel
    tmode = {"R0": "fp16", "R0q": "w4a4_norot", "R1f": "rot_fp16"}.get(cfg, "w4a4")
    rbin = None if cfg in ("R0", "R0q") else S1RBIN
    target = sq.build_target(MODEL, tmode, rbin_path=rbin, device=dev)
    R1 = sq.load_rbin(S1RBIN)["R1"] if rbin else None
    draft = DFlashDraftModel.from_pretrained(
        DRAFT, attn_implementation="sdpa", dtype=torch.bfloat16)
    rq = None
    if cfg in ("R0", "R0q", "R1f"):
        if cfg == "R1f":
            draft = interfaces.fold_wc(draft, R1)
        draft = draft.to(dev).eval()
        if cfg == "R0q":
            from . import draft_quant
            draft = draft_quant.quantize_draft(draft, 4, 4)
    else:
        from .vsq_draft_rot import RotQuantDraft
        base = interfaces.fold_wc(draft, R1)
        rq = RotQuantDraft(base, w_bits=4, a_bits=4, use_r2=True,
                           train_rotations=False, device=dev)
        if cfg in ("R2", "R3", "R4"):
            ck = torch.load(R1D_CKPT + ".best", map_location="cpu",
                            weights_only=False)
            R1b = ck["R1_D"].to(dev)
            R2b = [t.to(dev) if t is not None else None for t in ck["R2_D"]]
            rq.R1 = lambda: R1b
            rq.R2 = lambda i: R2b[i]
        if cfg == "R3":
            rq.cfg["fc_p2"] = True
            rq.rc_matrix_buf = R1.float().to(dev)
        if cfg == "R4":
            rq.rc_matrix_buf = torch.load(
                RCL_CKPT, weights_only=False)["R_C"].float().to(dev)
        rq.rotary = rq.rotary.to(dev)
        draft = None
    return target, draft, rq, R1


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True,
                    choices=["R0", "R0q", "R1f", "R1", "R2", "R3", "R4"])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--max-cycles-per-turn", type=int, default=6)
    ap.add_argument("--max-turns", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev, cfg = args.device, args.config
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    for sub in ("tables", "raw/activations", "raw/qparams"):
        os.makedirs(os.path.join(args.run_dir, sub), exist_ok=True)

    target, draft, rq, R1 = build(cfg, dev)
    R1d = R1.float().to(dev) if R1 is not None else None

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

    stats, res, qp = {}, {}, {}

    def S(name, dim):
        if name not in stats:
            stats[name] = StreamStat(dim)
        return stats[name]

    def rec(name, x2d, meta, cap_key=None):
        x2d = x2d.float()
        S(name, x2d.shape[-1]).update(x2d)
        if cap_key:
            if name not in res:
                res[name] = Reservoir(CAPS[cap_key])
            res[name].add(x2d, meta)

    def QP(name, det):
        if name not in qp:
            qp[name] = QparamAcc()
        qp[name].add(det)

    # FP-side fusion/KV views for RotQuant configs (unquantized buffers)
    if rq is not None:
        fc_fp = rq.fc_w.float()
        Rc = rq.rc_matrix_buf
        kv_fp = []
        for i in range(rq.n_layers):
            R2i = rq.R2(i)
            wk = getattr(rq, f"wk_ctx_{i}").float()
            wv = rq._headwise(getattr(rq, f"wv_ctx_{i}").float(), R2i, "out")
            if Rc is not None:
                wk, wv = wk @ Rc, wv @ Rc
            kv_fp.append((wk, wv))
        kn = [getattr(rq, f"kn_{i}").float() for i in range(rq.n_layers)]

    n_done = 0
    for key in order:
        t = turns[key]
        traj = [t["ids"]]
        for c in t["cycles"]:
            traj.append(c["block"][:c["tau"]])
        flat = [x for seg in traj for x in seg]
        full = torch.tensor([flat], device=dev)
        out = target(full, output_hidden_states=True, use_cache=False)
        meta = {"prompt": key[0], "turn": key[1], "cfg": cfg,
                "ds": args.dataset}
        Hsrc = {l: out.hidden_states[l + 1][0] for l in SRC}
        for l in SRC:
            rec(f"A_src{l}", Hsrc[l], meta, cap_key="A_src")
            QP(f"A_src{l}", act_quant_detail(Hsrc[l]))
            if R1d is not None:      # deployed basis -> original basis view
                rec(f"A_src{l}_origbasis", Hsrc[l].float() @ R1d.t(), meta)
        Hcat = torch.cat([Hsrc[l] for l in SRC], dim=-1)
        rec("B_concat", Hcat, meta, cap_key="B_concat")
        QP("B_concat", act_quant_detail(Hcat))
        detp2 = act_quant_detail(Hcat.reshape(-1, 5, 4096))
        QP("B_concat_p2", {**detp2, "scale": detp2["scale"].reshape(-1),
                           "zero": detp2["zero"].reshape(-1),
                           "tok_nmse": detp2["tok_nmse"].reshape(-1)})
        del out

        for c in t["cycles"][:args.max_cycles_per_turn]:
            pl = c["prefix_len"]
            th = Hcat[:pl].unsqueeze(0).to(torch.bfloat16)
            block = torch.tensor([c["block"]], device=dev)
            ne = target.model.embed_tokens(block)
            pos = torch.arange(pl + block.shape[1], device=dev).unsqueeze(0)

            if rq is None:
                # ---- stock-module path (R0 / R0q / R1f)
                fc_m, hn_m = draft.fc, draft.hidden_norm
                Zt = fc_m(th)                       # QLinear quant in R0q
                Ht = hn_m(Zt)
                rec("C_Zt_fp", Zt[0], meta)
                rec("C_Ht_fp", Ht[0], meta, cap_key="Ht")
                QP("C_Ht", act_quant_detail(Ht[0]))
                if cfg == "R1f":
                    continue                        # parity stats only
                caps, handles = {}, []

                def mk(li, tag):
                    def hook(mod, inp, o):
                        caps.setdefault((li, tag), []).append(
                            (inp[0][0].float(), o[0].float()))
                    return hook
                for li in range(5):
                    at = draft.layers[li].self_attn
                    handles.append(at.k_proj.register_forward_hook(mk(li, "k")))
                    handles.append(at.v_proj.register_forward_hook(mk(li, "v")))
                _ = draft(target_hidden=th, noise_embedding=ne,
                          position_ids=pos, use_cache=False, is_causal=False)
                for h in handles:
                    h.remove()
                cos, sin = draft.rotary_emb(ne, pos)
                cos, sin = cos.float(), sin.float()
                for li in range(5):
                    (xin_c, ko), (xin_d, kdo) = caps[(li, "k")]
                    (_, vo), (_, vdo) = caps[(li, "v")]
                    rec(f"S4_kctx_lin_l{li}", ko, meta, cap_key="kv")
                    rec(f"S4_vctx_lin_l{li}", vo, meta, cap_key="kv")
                    rec(f"S4_kdraft_lin_l{li}", kdo, meta)
                    rec(f"S4_vdraft_lin_l{li}", vdo, meta)
                    k = torch.cat([ko, kdo]).view(1, -1, NKV, HD)
                    k = draft.layers[li].self_attn.k_norm(k).transpose(1, 2)
                    k = k * cos.unsqueeze(1) + rot_half(k) * sin.unsqueeze(1)
                    v = torch.cat([vo, vdo]).view(1, -1, NKV, HD).transpose(1, 2)
                    ks = k[0].transpose(0, 1).reshape(-1, NKV * HD)
                    vs = v[0].transpose(0, 1).reshape(-1, NKV * HD)
                    rec(f"S4_k_stored_ctx_l{li}", ks[:pl], meta, cap_key="kv")
                    rec(f"S4_k_stored_draft_l{li}", ks[pl:], meta)
                    rec(f"S4_v_stored_ctx_l{li}", vs[:pl], meta, cap_key="kv")
                    rec(f"S4_v_stored_draft_l{li}", vs[pl:], meta)
                    for bits in (4, 8):
                        QP(f"KV{bits}_k_ctx_l{li}", act_quant_detail(
                            ks[:pl].reshape(-1, HD), bits=bits))
                        QP(f"KV{bits}_v_ctx_l{li}", act_quant_detail(
                            vs[:pl].reshape(-1, HD), bits=bits))
            else:
                # ---- deployed RotQuantDraft path with taps (R1/R2/R3/R4)
                tap = {}
                rq._cap = lambda name, x: tap.__setitem__(name, x.detach())
                _ = rq(position_ids=pos, noise_embedding=ne,
                       target_hidden=th, is_causal=False)
                rq._cap = None
                # FP fusion path from unquantized buffers
                Zt_fp = th[0].float() @ fc_fp.t()
                Ht_fp = Zt_fp * torch.rsqrt(
                    Zt_fp.pow(2).mean(-1, keepdim=True) + rq.eps)
                rec("C_Zt_fp", Zt_fp, meta)
                rec("C_Ht_fp", Ht_fp, meta, cap_key="Ht")
                QP("C_Ht_fp", act_quant_detail(Ht_fp))
                Ht_fp_kv = Ht_fp @ Rc if Rc is not None else Ht_fp
                if Rc is not None:
                    rec("C_Ht_rc_fp", Ht_fp_kv, meta, cap_key="Ht")
                    QP("C_Ht_rc_fp", act_quant_detail(Ht_fp_kv))
                rec("S3_Zt_dep", tap["S3_Zt_dep"][0], meta)
                rec("S3_Ht_dep", tap["S3_Ht_dep"][0], meta, cap_key="Ht")
                QP("S3_Ht_dep", act_quant_detail(tap["S3_Ht_dep"][0]))
                if "S3_Ht_rc_dep" in tap:
                    rec("S3_Ht_rc_dep", tap["S3_Ht_rc_dep"][0], meta,
                        cap_key="Ht")
                    QP("S3_Ht_rc_dep", act_quant_detail(tap["S3_Ht_rc_dep"][0]))
                cos, sin = rq.rotary(ne, pos)
                cos, sin = cos.float(), sin.float()
                for li in range(rq.n_layers):
                    for nm in ("kctx", "vctx", "kdraft", "vdraft"):
                        rec(f"S4_{nm}_lin_l{li}",
                            tap[f"S4_{nm}_lin_l{li}"][0], meta,
                            cap_key="kv" if "ctx" in nm else None)
                    # FP ctx projections + FP stored-K replica
                    kc_fp = Ht_fp_kv @ kv_fp[li][0].t()
                    vc_fp = Ht_fp_kv @ kv_fp[li][1].t()
                    rec(f"S4_kctx_lin_fp_l{li}", kc_fp, meta, cap_key="kv")
                    rec(f"S4_vctx_lin_fp_l{li}", vc_fp, meta)
                    kf = kc_fp.view(1, -1, NKV, HD)
                    kf = kf * torch.rsqrt(kf.pow(2).mean(-1, keepdim=True)
                                          + 1e-6) * kn[li]
                    kf = kf.transpose(1, 2)
                    kf = (kf * cos[:, :pl].unsqueeze(1)
                          + rot_half(kf) * sin[:, :pl].unsqueeze(1))
                    rec(f"S4_k_stored_ctx_fp_l{li}",
                        kf[0].transpose(0, 1).reshape(-1, NKV * HD), meta)
                    ks = tap[f"S4_k_stored_l{li}"][0].transpose(0, 1)
                    vs = tap[f"S4_v_stored_l{li}"][0].transpose(0, 1)
                    ks = ks.reshape(-1, NKV * HD)
                    vs = vs.reshape(-1, NKV * HD)
                    rec(f"S4_k_stored_ctx_l{li}", ks[:pl], meta, cap_key="kv")
                    rec(f"S4_k_stored_draft_l{li}", ks[pl:], meta)
                    rec(f"S4_v_stored_ctx_l{li}", vs[:pl], meta, cap_key="kv")
                    rec(f"S4_v_stored_draft_l{li}", vs[pl:], meta)
                    for bits in (4, 8):
                        QP(f"KV{bits}_k_ctx_l{li}", act_quant_detail(
                            ks[:pl].reshape(-1, HD), bits=bits))
                        QP(f"KV{bits}_v_ctx_l{li}", act_quant_detail(
                            vs[:pl].reshape(-1, HD), bits=bits))
                tap.clear()
        del Hcat, Hsrc
        torch.cuda.empty_cache()
        n_done += 1
        if n_done % 10 == 0:
            print(f"[fidi] {cfg} {args.dataset} {n_done}/{len(order)} turns",
                  flush=True)

    shard = f"{cfg}__{args.dataset}"
    json.dump({"config": cfg, "dataset": args.dataset,
               "trajectory": os.path.basename(cyc_path),
               "stats": {k: v.summary() for k, v in stats.items()},
               "qparams": {k: v.summary() for k, v in qp.items()}},
              open(f"{args.run_dir}/tables/capture__{shard}.json", "w"),
              indent=1)
    for name, r in res.items():
        if r.rows is None or r.fill == 0:
            continue
        np.savez_compressed(
            f"{args.run_dir}/raw/activations/{shard}__{name}.npz",
            rows=r.rows[:r.fill], seen=r.seen,
            meta=json.dumps({"config": cfg, "dataset": args.dataset,
                             "tensor": name, "seed": SEED}))
    ch = {f"{k}__{a}": v for k, s in stats.items()
          for a, v in s.channel_arrays().items()}
    np.savez_compressed(f"{args.run_dir}/raw/activations/{shard}__channels.npz",
                        **ch)
    for name, q in qp.items():
        arrs = q.arrays()
        if arrs:
            np.savez_compressed(
                f"{args.run_dir}/raw/qparams/{shard}__{name}.npz", **arrs)
    print(f"[fidi] {shard} DONE tensors={len(stats)} qsites={len(qp)}")


if __name__ == "__main__":
    main()

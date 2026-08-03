#!/usr/bin/env python
"""Layer-wise / depth-wise quantization error tracing (study §5).

FP16-bits vs W4A4 runs of the validated ExactQuantizedRotationForward
core on identical cached rows, per method:
  naive / ep3p / sharedq (EP3-P + fixed cross Q) / lp3qat.
Sites: proj_out, q, k, v, attn_out, resid_attn, mlp_in, gate_up_mid,
down_out, final_hidden, logits, rec_proj_out d1-4. Records NMSE /
cosine / amplification vs previous site / logits top-1.
Writes tables/error_flow.json + plot_data/error_flow.npz.
"""
import argparse, json, os, sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "third_party", "EAGLE"))

from eagle_spinquant.exact_quantized_rotation_forward import (
    ExactQuantizedRotationForward, rope_cos_sin, rotate_half)
from eagle_spinquant.projection_rotation import StructuredRotation

D = 4096
NH, HD = 32, 128
RMS_EPS = 1e-5


class TracedCore(ExactQuantizedRotationForward):
    proj_rot = None   # class default; instance shim self.rot untouched

    def _maybe_rot(self, z):
        if self.proj_rot is not None:
            return self.proj_rot.apply(z.float()).to(z.dtype)
        return z

    def traced_forward(self, tok_ids, feat_seq, K=4):
        qw, tw = self.quantized_weights(False)
        if self.proj_rot is not None:
            tw = dict(tw)
            for kk in ("W_first", "W_rec"):
                tw[kk] = self.proj_rot.apply(tw[kk].float()).to(tw[kk].dtype)
                qw[kk] = self._qw(tw[kk])
        R = tw["R"]
        a = self.alpha_exact
        B, T = feat_seq.shape[0], feat_seq.shape[1]
        E = self.E[tok_ids] * a
        z = torch.cat([E[:, 1:T + 1].half(), feat_seq.half()], -1)
        z = self._maybe_rot(z)
        sites = {}
        y = self._proj(z, qw["W_first"], R)
        sites["proj_out"] = y
        x = y
        pos = torch.arange(0, T, device=y.device)
        qh = F.linear(self._qa(x), qw["q"]).view(B, T, NH, HD).transpose(1, 2)
        kh = F.linear(self._qa(x), qw["k"]).view(B, T, NH, HD).transpose(1, 2)
        vh = F.linear(self._qa(x), qw["v"]).view(B, T, NH, HD).transpose(1, 2)
        sites["q"] = qh; sites["k"] = kh; sites["v"] = vh
        cos, sin = rope_cos_sin(pos, dim=HD, device=x.device)
        cos = cos[None, None].to(x.dtype)
        sin = sin[None, None].to(x.dtype)
        qh = qh * cos + rotate_half(qh) * sin
        kh = kh * cos + rotate_half(kh) * sin
        att = (qh @ kh.transpose(-1, -2)) / HD ** 0.5
        causal = torch.full((T, T), torch.finfo(att.dtype).min,
                            device=x.device, dtype=att.dtype).triu(1)
        att = (att + causal).float().softmax(-1).to(x.dtype)
        o = (att @ vh).transpose(1, 2).reshape(B, T, D)
        sites["attn_out"] = o
        x = x + F.linear(self._qa(o), qw["o"])
        sites["resid_attn"] = x
        h32 = x.float()
        var = h32.pow(2).mean(-1, keepdim=True)
        h2 = (h32 * torch.rsqrt(var + RMS_EPS)).to(x.dtype)
        sites["mlp_in"] = h2
        gt = F.linear(self._qa(h2), qw["gate"])
        up = F.linear(self._qa(h2), qw["up"])
        mid = F.silu(gt) * up
        sites["gate_up_mid"] = mid
        if self.had_K is not None:
            from eagle_spinquant import spinquant_bridge as sb
            sb.add_spinquant_to_syspath()
            from utils import hadamard_utils
            ms = mid.shape
            mid = hadamard_utils.matmul_hadU_cuda(
                mid.reshape(-1, ms[-1]), self.had_K,
                self.had_KK).reshape(ms)
        dn = F.linear(self._qa(mid), qw["down"])
        sites["down_out"] = dn
        x = x + dn
        sites["final_hidden"] = x
        h_last = x[:, -1:]
        lg = F.linear(h_last.squeeze(1).half(), qw["head"]).float()
        sites["logits"] = lg
        for k in range(1, K + 1):
            tok = lg.argmax(-1)
            e_k = (self.E[tok] * a).half().unsqueeze(1)
            z = torch.cat([e_k, h_last.half()], -1)
            z = self._maybe_rot(z)
            y = self._proj(z, qw["W_rec"], R)
            sites[f"rec_proj_out_d{k}"] = y
            h_last = y
            lg = F.linear(h_last.squeeze(1).half(), qw["head"]).float()
        return sites


def nmse(a, b):
    a = a.double(); b = b.double()
    return float(((a - b) ** 2).sum() / ((b ** 2).sum() + 1e-30))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--n-rows", type=int, default=24)
    args = ap.parse_args()
    rd = args.run_dir
    dev = "cuda:0"
    from eagle_spinquant import experiment, study
    cfg = experiment.load_config(None)
    paths = experiment.resolve_paths(cfg)
    rr = cfg.get("paths", {}).get("rotations_root")
    model, stash, _ = study.build_study_target(
        paths["target_path"], paths["draft_path"],
        cfg["model"]["target"], "full", "learned_chat_w4a4kv16",
        "w4a4", 0, device=dev, rotations_root=rr)
    cache = torch.load(os.path.join(rd, "tensors", "teacher_cache.pt"),
                       map_location="cpu", weights_only=False)
    rows = cache["rows"][: args.n_rows]
    R1 = stash["R1"].float()
    gamma = stash["gamma_f"].float()
    W_lm = stash["lm_head_weight"].float()

    class _FR(torch.nn.Module):
        def __init__(self, R):
            super().__init__()
            self.register_buffer("Rb", R)

        def R(self):
            return self.Rb

    qat = os.path.join(
        ROOT, "runs",
        "eagle1_official_fromscratch_ptq_vs_qat_20260724_213243",
        "ckpts", "Q1_s0_best.pt")

    def build(alpha, sd_override=None, bits=4):
        sd = ({k: v.detach().cpu()
               for k, v in model.ea_layer.state_dict().items()}
              if sd_override is None else sd_override)
        return TracedCore(sd, R1, gamma, W_lm, _FR(R1.to(dev)),
                          alpha_init=alpha, train_alpha=False,
                          w_bits=bits, a_bits=bits, draft_kv_bits=16,
                          train_draft_core=False, r2_seed=0,
                          device=dev, first_fold_R=R1).to(dev).eval()

    qsd = torch.load(qat, map_location="cpu", weights_only=False)
    qsd = qsd.get("draft_state_dict", qsd)
    methods = dict(
        naive=dict(alpha=1.0),
        ep3p=dict(alpha=float(D ** 0.40)),
        sharedq=dict(alpha=float(D ** 0.38),
                     rot=dict(family="cross", block=8192, seed=12,
                              interleave_chunk=1)),
        lp3qat=dict(alpha=32.0, sd=qsd))
    ref_cache = {}
    out = {m: {} for m in methods}
    rowstats = {m: [] for m in methods}
    SITES = None
    for mname, mc in methods.items():
        fp = build(mc["alpha"], mc.get("sd"), bits=16)
        qz = build(mc["alpha"], mc.get("sd"), bits=4)
        if mc.get("rot"):
            r = StructuredRotation(mc["rot"], device=dev)
            fp.proj_rot = r; qz.proj_rot = r
        agg = {}
        for r0 in rows:
            ids = r0["input_ids"][None].long().to(dev)
            key = int(ids.sum())
            if key not in ref_cache:
                h = model.base_model.model(
                    input_ids=ids).last_hidden_state.float()
                ref_cache[key] = h[:, :-1]
            feat = ref_cache[key]
            s_fp = fp.traced_forward(ids, feat)
            s_qz = qz.traced_forward(ids, feat)
            prev_n = None
            per_row = {}
            for site in s_fp:
                n = nmse(s_qz[site].float(), s_fp[site].float())
                c = float(F.cosine_similarity(
                    s_qz[site].float().flatten(),
                    s_fp[site].float().flatten(), dim=0))
                d = agg.setdefault(site, dict(nmse=[], cos=[], amp=[],
                                              top1=[]))
                d["nmse"].append(n); d["cos"].append(c)
                if prev_n and prev_n > 1e-12:
                    d["amp"].append(n / prev_n)
                if site == "logits":
                    d["top1"].append(float(
                        (s_qz[site].argmax(-1)
                         == s_fp[site].argmax(-1)).float().mean()))
                else:
                    prev_n = n
                per_row[site] = n
            rowstats[mname].append(per_row)
        SITES = list(agg)
        out[mname] = {site: dict(
            nmse=float(np.mean(v["nmse"])),
            cos=float(np.mean(v["cos"])),
            amp=(float(np.mean(v["amp"])) if v["amp"] else None),
            top1=(float(np.mean(v["top1"])) if v["top1"] else None))
            for site, v in agg.items()}
        del fp, qz
        torch.cuda.empty_cache()
        print(f"[trace] {mname}: proj "
              f"{out[mname]['proj_out']['nmse']:.4f} -> final "
              f"{out[mname]['final_hidden']['nmse']:.4f} -> logits "
              f"top1 {out[mname]['logits']['top1']:.4f}", flush=True)
    os.makedirs(os.path.join(rd, "plot_data"), exist_ok=True)
    np.savez_compressed(
        os.path.join(rd, "plot_data", "error_flow.npz"),
        sites=np.array(SITES),
        **{f"{m}_rows": np.array([[rw[s] for s in SITES]
                                  for rw in rowstats[m]])
           for m in methods})
    json.dump(out, open(os.path.join(
        rd, "tables", "error_flow.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())

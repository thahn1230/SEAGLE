#!/usr/bin/env python
"""Does scaling the HIDDEN slice too buy anything? (user question)

Three experiments on the measured tensors, official W4/A4 quantizers:

A. GAUGE CHECK — slice-wise (m_e, m_h) with m_h != 1 vs the single-m
   EP3-P at the same ratio m_e/m_h. Claim: per-token-asym A4 and
   per-out-channel-sym W4 are both equivariant under a uniform rescale
   of the whole token / whole weight row, so only the RATIO matters:
   a second slice-wide factor is a gauge direction, not a new knob.

B. SLICE-PAIR GRID — (m_e, m_h) over a small grid; output NMSE should
   be constant along constant-ratio diagonals and minimized at the
   already-chosen ratio.

C. PER-CHANNEL MIGRATION (SmoothQuant-style, genuinely larger space):
   s_i = |X|_max,i^alpha / |W|_max,i^(1-alpha) per input channel i
   (X' = X / s, W'[:,i] = W[:,i] * s_i, product preserved).
   Fit s on EVEN token rows, evaluate on ODD rows (held-out) to avoid
   in-sample flattery. Also a 64-channel-group variant.

Writes tables/hidden_scaling_extensions_<path>.json (+ printout).
"""
import argparse, json, os, sys

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "EAGLE"))

D = 4096


def nmse(y, ref):
    y = y.double(); ref = ref.double()
    return float(((y - ref) ** 2).sum() / ((ref ** 2).sum() + 1e-30))


@torch.no_grad()
def quant_out(X, W, bias, fq):
    Wq = fq._weight_fake_quant(W.half(), 4).float()
    aq = fq._act_quantizer(4)
    aq.find_params(X.half())
    Xq = aq(X.half()).float()
    aq.free()
    Y = Xq @ Wq.t()
    return Y + (bias if bias is not None else 0)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--path", default="first",
                    choices=["first", "recurrent"])
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    rd = args.run_dir
    dev = args.device if torch.cuda.is_available() else "cpu"
    from eagle_spinquant import fake_w4a4_draft as fq
    blob = torch.load(os.path.join(rd, "plot_data",
                                   "ep3p_tensors.pt"),
                      map_location="cpu", weights_only=False)
    man = json.load(open(os.path.join(
        rd, "metadata", "collection_manifest.json")))
    if args.path == "first":
        X = blob["X_first_raw"].float().to(dev)
        m_star = man["m_first"]
    else:
        X = torch.cat([blob[f"X_rec{k}_raw"]
                       for k in (1, 2, 3, 4)]).float().to(dev)
        m_star = man["m_rec"]
    W0 = blob["W_before"].float().to(dev)
    bias = blob.get("bias")
    b_t = bias.float().to(dev) if bias is not None else None
    Yref = X @ W0.t() + (b_t if b_t is not None else 0)
    fit, ev = X[0::2], X[1::2]
    Yref_ev = ev @ W0.t() + (b_t if b_t is not None else 0)

    def slicewise(me, mh, Xe=None):
        Xe = X if Xe is None else Xe
        Xa = Xe.clone()
        Xa[:, :D] *= me
        Xa[:, D:] *= mh
        Wa = W0.clone()
        Wa[:, :D] /= me
        Wa[:, D:] /= mh
        return Xa, Wa

    out = dict(path=args.path, m_star=m_star)

    # A. gauge check
    Xa, Wa = slicewise(m_star, 1.0)
    base = nmse(quant_out(Xa, Wa, b_t, fq), Yref)
    gauge = {}
    for c in (0.5, 2.0, 10.0):
        Xa, Wa = slicewise(m_star * c, c)
        gauge[f"c={c}"] = nmse(quant_out(Xa, Wa, b_t, fq), Yref)
    out["gauge_check"] = dict(single_m=base, scaled_pairs=gauge)
    print(f"[A gauge] single m*: {base:.4f} | (c*m*, c): {gauge}")

    # B. slice-pair grid
    grid = {}
    for me in (12.13, 27.86, 42.22, 64.0):
        for mh in (0.5, 1.0, 2.0, 4.0):
            Xa, Wa = slicewise(me, mh)
            grid[f"me={me},mh={mh}"] = round(nmse(
                quant_out(Xa, Wa, b_t, fq), Yref), 4)
    out["slice_pair_grid"] = grid
    print("[B grid]", json.dumps(grid, indent=1))

    # C. per-channel migration (fit on even rows, eval on odd rows)
    ref_ratio = None
    xa_max = fit.abs().amax(0).clamp_min(1e-8)
    w_max = W0.abs().amax(0).clamp_min(1e-8)
    perch = {}
    for alpha in (0.3, 0.5, 0.7, 0.9):
        s = xa_max.pow(alpha) / w_max.pow(1 - alpha)
        s = s.clamp_min(1e-6)
        Xa = ev / s
        Wa = W0 * s
        perch[f"alpha={alpha}"] = round(nmse(
            quant_out(Xa, Wa, b_t, fq), Yref_ev), 4)
        # 64-channel-group variant
        sg = s.reshape(-1, 64).mean(1).repeat_interleave(64)
        Xg = ev / sg
        Wg = W0 * sg
        perch[f"alpha={alpha}_g64"] = round(nmse(
            quant_out(Xg, Wg, b_t, fq), Yref_ev), 4)
    # references on the SAME eval split
    Xa, Wa = slicewise(m_star, 1.0, ev)
    perch["ep3p_slicewise_evalsplit"] = round(nmse(
        quant_out(Xa, Wa, b_t, fq), Yref_ev), 4)
    perch["naive_evalsplit"] = round(nmse(
        quant_out(ev, W0, b_t, fq), Yref_ev), 4)
    # per-channel ON TOP of the chosen slice-wise m (hybrid)
    for alpha in (0.5, 0.7):
        Xa_f, Wa_f = slicewise(m_star, 1.0, fit)
        xam = Xa_f.abs().amax(0).clamp_min(1e-8)
        wam = Wa_f.abs().amax(0).clamp_min(1e-8)
        s = (xam.pow(alpha) / wam.pow(1 - alpha)).clamp_min(1e-6)
        Xa_e, Wa_e = slicewise(m_star, 1.0, ev)
        perch[f"hybrid_ep3p+alpha={alpha}"] = round(nmse(
            quant_out(Xa_e / s, Wa_e * s, b_t, fq), Yref_ev), 4)
    out["per_channel"] = perch
    print("[C per-channel]", json.dumps(perch, indent=1))

    os.makedirs(os.path.join(rd, "tables"), exist_ok=True)
    json.dump(out, open(os.path.join(
        rd, "tables",
        f"hidden_scaling_extensions_{args.path}.json"), "w"),
        indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""WHY the migration factor has an interior optimum (U-curve mechanism).

From the measured tensors (plot_data/ep3p_tensors.pt), sweep m over the
exponent grid and, at every m, quantize with the OFFICIAL W4/A4
quantizers, then measure where the error actually comes from:

  panel 1  output NMSE total / e-branch / h-branch vs m  (the U-curve)
  panel 2  weight side: W_e zero-code rate + W_e W4 NMSE vs m
           (W4 is per-OUTPUT-CHANNEL: one scale per row [W_e/m | W_h];
            as m grows, W_e/m sinks below the row's step -> code 0)
  panel 3  activation side: A4 e-slice vs h-slice NMSE vs m
           (A4 is per-TOKEN: one scale across all 8192 channels;
            as m grows, m*e widens the token range -> h gets coarser)
  panel 4  |W_e| histograms at small/optimal/large m against the
           median W4 row step: the annihilation picture

Writes quantized_effect/ucurve_mechanism_<path>.png (+CSV/NPZ).
"""
import argparse, csv, json, os, sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "EAGLE"))

D = 4096
DPI = 300


def nmse(y, ref):
    y = y.double(); ref = ref.double()
    return float(((y - ref) ** 2).sum() / ((ref ** 2).sum() + 1e-30))


@torch.no_grad()
def sweep(X_raw, W0, bias, betas, dev, fq):
    rows = []
    Xb = X_raw.float().to(dev)
    W0d = W0.float().to(dev)
    Yref = Xb @ W0d.t()
    if bias is not None:
        Yref = Yref + bias.float().to(dev)
    ref_e = Xb[:, :D] @ W0d[:, :D].t()
    ref_h = Yref - ref_e - (bias.float().to(dev)
                            if bias is not None else 0)
    for b in betas:
        m = float(D ** b)
        Xa = Xb.clone(); Xa[:, :D] *= m
        Wa = W0d.clone(); Wa[:, :D] /= m
        Wq = fq._weight_fake_quant(Wa.half(), 4).float()
        aq = fq._act_quantizer(4)
        aq.find_params(Xa.half())
        Xq = aq(Xa.half()).float()
        aq.free()
        y_e = Xq[:, :D] @ Wq[:, :D].t()
        y_h = Xq[:, D:] @ Wq[:, D:].t()
        Y = y_e + y_h + (bias.float().to(dev)
                         if bias is not None else 0)
        # weight-side mechanism
        we_zero = float((Wq[:, :D] == 0).float().mean())
        wh_zero = float((Wq[:, D:] == 0).float().mean())
        we_nmse = nmse(Wq[:, :D] * m, W0d[:, :D])
        wh_nmse = nmse(Wq[:, D:], W0d[:, D:])
        # activation-side mechanism (compare in RAW units)
        a4_e = nmse(Xq[:, :D] / m, Xb[:, :D])
        a4_h = nmse(Xq[:, D:], Xb[:, D:])
        rows.append(dict(
            beta=b, m=m,
            out_nmse=nmse(Y, Yref), out_e=nmse(y_e, ref_e),
            out_h=nmse(y_h, ref_h),
            we_zero_rate=we_zero, wh_zero_rate=wh_zero,
            we_w4_nmse=we_nmse, wh_w4_nmse=wh_nmse,
            a4_e_nmse=a4_e, a4_h_nmse=a4_h))
        print(f"[sweep] beta={b:.2f} m={m:8.2f} out={rows[-1]['out_nmse']:.4f} "
              f"we_zero={we_zero:.3f} a4_h={a4_h:.4f}", flush=True)
    return rows


@torch.no_grad()
def we_hist_data(W0, ms, dev, fq):
    """|W_e/m| distribution vs that config's median W4 row step."""
    out = []
    W0d = W0.float().to(dev)
    for m in ms:
        Wa = W0d.clone(); Wa[:, :D] /= m
        # official W4: per-row sym; recover per-row step from dequant
        Wq = fq._weight_fake_quant(Wa.half(), 4).float()
        step = (Wq.abs().amax(dim=1) / 7.0).clamp_min(1e-12)
        vals = Wa[:, :D].abs().flatten()
        vals = vals[::37][:400000].cpu().numpy()   # thin, deterministic
        out.append(dict(m=m, vals=vals,
                        med_step=float(step.median()),
                        zero_rate=float((Wq[:, :D] == 0)
                                        .float().mean())))
    return out


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
        X = blob["X_first_raw"]
        m_star = man["m_first"]; b_star = man["beta_first"]
    else:
        X = torch.cat([blob[f"X_rec{k}_raw"] for k in (1, 2, 3, 4)])
        m_star = man["m_rec"]; b_star = man["beta_rec"]
    W0, bias = blob["W_before"], blob.get("bias")
    betas = [round(0.05 * i, 2) for i in range(15)]      # 0.00..0.70
    rows = sweep(X, W0, bias, betas, dev, fq)
    ms = [r["m"] for r in rows]
    hist = we_hist_data(W0, [D ** 0.2, m_star, D ** 0.6], dev, fq)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    ax = axes[0, 0]
    ax.plot(ms, [r["out_nmse"] for r in rows], "k-o", ms=4,
            label="TOTAL output NMSE")
    ax.plot(ms, [r["out_e"] for r in rows], "-o", ms=3,
            color="tab:blue", label="e-branch output NMSE")
    ax.plot(ms, [r["out_h"] for r in rows], "-o", ms=3,
            color="tab:red", label="h-branch output NMSE")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.axvline(m_star, color="g", ls="--",
               label=f"chosen m = {m_star:.1f} (beta {b_star})")
    ax.set_xlabel("migration factor m (log)")
    ax.set_ylabel("output NMSE (log)")
    ax.set_title("1. the U-curve: e-branch falls, h-branch rises "
                 "-> interior optimum")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(ms, [r["we_zero_rate"] for r in rows], "-o", ms=4,
            color="tab:blue", label="W_e zero-code rate")
    ax.plot(ms, [r["we_w4_nmse"] for r in rows], "-o", ms=4,
            color="tab:cyan", label="W_e W4 NMSE")
    ax.plot(ms, [r["wh_w4_nmse"] for r in rows], "-o", ms=3,
            color="tab:red", label="W_h W4 NMSE")
    ax.axvline(m_star, color="g", ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("migration factor m (log)")
    ax.set_ylabel("rate / NMSE")
    ax.set_title("2. weight side: W4 row scale is set by W_h; "
                 "W_e/m sinks below the step -> quantized to 0")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(ms, [r["a4_e_nmse"] for r in rows], "-o", ms=4,
            color="tab:blue", label="A4 e-slice NMSE (raw units)")
    ax.plot(ms, [r["a4_h_nmse"] for r in rows], "-o", ms=4,
            color="tab:red", label="A4 h-slice NMSE")
    ax.axvline(m_star, color="g", ls="--")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("migration factor m (log)")
    ax.set_ylabel("A4 reconstruction NMSE (log)")
    ax.set_title("3. activation side: one per-token scale over 8192 "
                 "ch; m*e widens the range -> h coarsens (see-saw)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    colors = ["tab:blue", "g", "tab:red"]
    for h, c in zip(hist, colors):
        ax.hist(np.log10(h["vals"] + 1e-12), bins=200, density=True,
                histtype="step", color=c,
                label=f"|W_e/m|, m={h['m']:.1f} "
                      f"(zeroed {h['zero_rate'] * 100:.0f}%)")
        ax.axvline(np.log10(h["med_step"] / 2), color=c, ls=":",
                   lw=1.5)
    ax.set_xlabel("log10 |W_e / m|   (dotted: that config's median "
                  "W4 half-step = rounding threshold)")
    ax.set_ylabel("density")
    ax.set_title("4. annihilation picture: mass left of the dotted "
                 "line rounds to code 0")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle(
        f"Why m has an interior optimum ({args.path} path, measured "
        "tensors, official W4/A4 quantizers): migration re-splits "
        "distortion between the weight quantizer (hurts as m grows) "
        "and the activation quantizer (helps e, then hurts h)",
        fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(rd, "quantized_effect",
                       f"ucurve_mechanism_{args.path}.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    with open(os.path.join(
            rd, "tables",
            f"ucurve_mechanism_{args.path}.csv"), "w",
            newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    np.savez_compressed(
        os.path.join(rd, "plot_data",
                     f"ucurve_mechanism_{args.path}.npz"),
        **{k: np.array([r[k] for r in rows]) for k in rows[0]})
    json.dump(dict(figure=f"ucurve_mechanism_{args.path}",
                   path=args.path, m_star=m_star, beta_star=b_star,
                   betas=betas, source="plot_data/ep3p_tensors.pt",
                   quantizers="official W4 per-out-channel sym "
                              "MSE-clip / A4 per-token asym"),
              open(os.path.join(
                  rd, "plot_data",
                  f"ucurve_mechanism_{args.path}_metadata.json"),
                  "w"), indent=1)
    print(f"[ucurve] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

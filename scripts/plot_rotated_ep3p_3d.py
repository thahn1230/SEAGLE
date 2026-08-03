#!/usr/bin/env python
"""R-EP3-P required 3D figures + heatmaps (spec §16).

original / EP3-P / R-EP3-P activations and weights as ONE complete
matrix per figure, single shared normalization per comparison set;
boundary at D for original/EP3-P; rotated figures labeled "rotated
projection-input coordinates" (never mislabeled e/h); contribution
decomposition figures for the mixed (cross) first path.
Reuses the proven Fig machinery from plot_ep3p_projection_3d.py.
"""
import argparse, importlib.util, json, os, sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "third_party", "EAGLE"))
from eagle_spinquant.projection_rotation import (StructuredRotation,
                                                 transform_xw)

spec = importlib.util.spec_from_file_location(
    "vz", os.path.join(ROOT, "scripts", "plot_ep3p_projection_3d.py"))
vz = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vz)

D = 4096
GA = 16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    rd = args.run_dir
    for sub in ("plots", "heatmaps", "plot_data", "log_scale",
                "activations", "weights", "outputs",
                "quantized_effect", "summaries"):
        os.makedirs(os.path.join(rd, sub), exist_ok=True)
    from eagle_spinquant import fake_w4a4_draft as fq
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    tens = torch.load(os.path.join(rd, "tensors", "calib_int4.pt"),
                      map_location="cpu", weights_only=False)
    W0 = tens["W"].float().to(dev)
    bias = tens.get("bias")
    b_t = bias.float().to(dev) if bias is not None else None
    s5 = json.load(open(os.path.join(rd, "candidates",
                                     "s5_mtbench.json")))
    best = s5["best"]
    F = vz.Fig(rd, dict(study="rotated_ep3p", best=str(best)))
    xs_act = np.arange(2 * D // GA) * GA + GA / 2
    xs_w = xs_act
    xs_out = np.arange(D // GA) * GA + GA / 2
    contrib = np.load(os.path.join(rd, "tables",
                                   "rep3p_contributions.npz"))

    def block(path, key, ep3p_beta, depth_tag=None):
        cfg = best["first" if path == "first" else "rec"]
        rot = StructuredRotation(
            {k: cfg[k] for k in ("family", "block", "seed")},
            device=dev)
        ident = StructuredRotation(dict(family="identity"))
        X = tens[key].float().to(dev)
        m_e = float(D ** ep3p_beta)
        m_r = float(D ** cfg["beta"])
        Xo, Wo = X, W0
        Xe, We = transform_xw(X, W0, m_e, ident, "SR")
        Xr, Wr = transform_xw(X, W0, m_r, rot, cfg["order"])
        Yref = X @ W0.t() + (b_t if b_t is not None else 0)

        def qout(Xt, Wt):
            Wq = fq._weight_fake_quant(Wt.half(), 4).float()
            aq = fq._act_quantizer(4)
            aq.find_params(Xt.half())
            Xq = aq(Xt.half()).float()
            aq.free()
            return Xq @ Wq.t() + (b_t if b_t is not None else 0)
        tag = depth_tag or path
        Ga = [vz.group_rows(np.abs(Z.cpu().numpy()), GA, "meanabs")
              for Z in (Xo, Xe, Xr)]
        anorm = vz.colors.Normalize(0, float(max(g.max()
                                                 for g in Ga)))
        rot_note = (f"{cfg['family']} b{cfg['block']} "
                    f"s{cfg['seed']} beta={cfg['beta']}")
        for G, name, ttl, bnd in (
                (Ga[0], f"{tag}_activation_original_3d",
                 "original input", D),
                (Ga[1], f"{tag}_activation_ep3p_3d",
                 f"EP3-P input (e x {m_e:.1f})", D),
                (Ga[2], f"{tag}_activation_rotated_ep3p_3d",
                 f"R-EP3-P input ({rot_note})",
                 None if cfg["family"] in ("full", "cross")
                 else D)):
            xl = ("rotated projection-input coordinate"
                  if bnd is None else "projection input channel")
            F.surf("activations", name, G, xl, "token index",
                   f"abs activation (meanabs, group {GA})",
                   xvals=xs_act, boundary=bnd, norm=anorm,
                   title=f"{tag}: {ttl}", log_companion=True)
            F.heat("heatmaps", name.replace("_3d", "_heatmap"), G,
                   xl, "token index", "abs activation",
                   xvals=xs_act, boundary=bnd, norm=anorm,
                   title=f"{tag}: {ttl}")
        Gw = [vz.group2d(np.abs(Z.cpu().numpy()), GA, GA, "meanabs")
              for Z in (Wo, We, Wr)]
        wnorm = vz.colors.Normalize(0, float(max(g.max()
                                                 for g in Gw)))
        for G, name, ttl, bnd in (
                (Gw[0], f"{tag}_weight_original_3d",
                 "original weight", D),
                (Gw[1], f"{tag}_weight_ep3p_3d",
                 f"EP3-P weight (W_e / {m_e:.1f})", D),
                (Gw[2], f"{tag}_weight_rotated_ep3p_3d",
                 f"R-EP3-P weight ({rot_note})",
                 None if cfg["family"] in ("full", "cross")
                 else D)):
            xl = ("rotated inner coordinate" if bnd is None
                  else "inner/input channel")
            F.surf("weights", name, G, xl, "outer/output channel",
                   f"abs weight (meanabs, {GA}x{GA})", xvals=xs_w,
                   boundary=bnd, norm=wnorm, title=f"{tag}: {ttl}",
                   region_labels=("Embedding-side input",
                                  "Hidden-side input"),
                   log_companion=True)
        Ee = (qout(Xe, We) - Yref).abs()
        Er = (qout(Xr, Wr) - Yref).abs()
        Ge = vz.group_rows(Ee.cpu().numpy(), GA, "meanabs")
        Gr = vz.group_rows(Er.cpu().numpy(), GA, "meanabs")
        qn = vz.colors.Normalize(0, float(max(Ge.max(), Gr.max())))
        for G, name, lab in (
                (Ge, f"{tag}_w4a4_output_error_ep3p_3d", "EP3-P"),
                (Gr, f"{tag}_w4a4_output_error_rotated_ep3p_3d",
                 "R-EP3-P")):
            F.surf("outputs", name, G, "output channel",
                   "token index", "abs(Y_q - Y_fp) (mean in group)",
                   xvals=xs_out, norm=qn,
                   title=f"{tag}: W4A4 output error, {lab} "
                         "(shared normalization)")

    block("first", "X_first", 0.40)
    for k in (1, 2, 3, 4):
        block("rec", f"X_rec{k}", 0.45,
              depth_tag=f"recurrent_depth{k}")
    block("rec", "X_rec_all", 0.45, depth_tag="recurrent")

    # contribution decomposition (mixed first path; rec is dual =>
    # contributions stay separated, plotted for contrast)
    for path in ("first", "rec"):
        pre = "first" if path == "first" else "recurrent"
        ce = contrib[f"{path}_contrib_e_rms"]
        ch = contrib[f"{path}_contrib_h_rms"]
        fr = contrib[f"{path}_contrib_e_frac"]
        g = GA
        Z = np.stack([ce.reshape(-1, g).mean(1),
                      ch.reshape(-1, g).mean(1)])
        F.surf("plots", f"{pre}_rotated_embedding_contribution_3d",
               Z[:1], "rotated projection-input coordinate",
               "(single row)", "embedding-contribution RMS",
               xvals=xs_act,
               title=f"{pre}: embedding-energy contribution per "
                     "rotated coordinate")
        F.surf("plots", f"{pre}_rotated_hidden_contribution_3d",
               Z[1:], "rotated projection-input coordinate",
               "(single row)", "hidden-contribution RMS",
               xvals=xs_act,
               title=f"{pre}: hidden-energy contribution per "
                     "rotated coordinate")
        F.heat("heatmaps",
               f"{pre}_rotated_contribution_fraction_heatmap",
               fr.reshape(1, -1), "rotated coordinate", "",
               "embedding contribution fraction",
               title=f"{pre}: e-energy fraction per rotated "
                     "coordinate (flat ~0.09 = fully mixed; "
                     "0/1 bands = unmixed)")
    print(f"[plot3d] {F.n3d} 3D, {F.nheat} heatmaps")
    json.dump(dict(n3d=F.n3d, nheat=F.nheat), open(os.path.join(
        rd, "tables", "figure_counts.json"), "w"))
    return 0


if __name__ == "__main__":
    sys.exit(main())

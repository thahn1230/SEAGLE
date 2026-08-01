#!/usr/bin/env python
"""27-item final terminal output for the EP3-P visualization study."""
import glob, json, os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    rd = os.path.join(ROOT, open(os.path.join(
        ROOT, "runs", "EP3PVIZ_RUN_DIR")).read().strip())
    man = json.load(open(os.path.join(
        rd, "metadata", "collection_manifest.json")))
    s = json.load(open(os.path.join(
        rd, "tables", "ep3p_viz_summary.json")))
    o = []

    def item(n, k, v):
        o.append(f"{n:2d}. {k}: {v}")

    br = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                        capture_output=True, text=True,
                        cwd=ROOT).stdout.strip()
    cm = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                        capture_output=True, text=True,
                        cwd=ROOT).stdout.strip()
    item(1, "Branch and commit", f"{br} @ {cm}")
    item(2, "Run directory", rd)
    item(3, "Checkpoint SHA256", man["anchor_sha256"])
    item(4, "beta_first", man["beta_first"])
    item(5, "beta_recurrent", man["beta_rec"])
    item(6, "m_first", round(man["m_first"], 4))
    item(7, "m_recurrent", round(man["m_rec"], 4))
    item(8, "First activation tensor shape", man["shapes"]["first"])
    item(9, "Recurrent activation shapes by depth",
         {k: man["shapes"][f"rec{k}"] for k in (1, 2, 3, 4)})
    item(10, "Projection weight original shape",
         man["shapes"]["W_before"])
    item(11, "Plotting orientation",
         man["weight_orientation"]["storage"]
         + f"; forward {man['weight_orientation']['forward']}; "
         "plot x=input channel (dim 1), y=output channel (dim 0)")
    item(12, "Embedding/hidden boundary index",
         f"input channel {man['D']} (of {2 * man['D']})")
    a = s["act_rms_ratio_first"]
    item(13, "First activation e/h RMS ratio before/after",
         f"{a['before']:.6f} -> {a['after']:.6f} "
         f"(x{a['after'] / a['before']:.2f} = m_first)")
    a = s["act_rms_ratio_recurrent"]
    item(14, "Recurrent activation e/h RMS ratio before/after",
         f"{a['before']:.6f} -> {a['after']:.6f} "
         f"(x{a['after'] / a['before']:.2f} = m_recurrent)")
    w = s["w_rms_ratio"]
    item(15, "Weight e/h RMS ratio before -> first migration",
         f"{w['before']:.4f} -> {w['first_after']:.4f}")
    item(16, "Weight e/h RMS ratio before -> recurrent migration",
         f"{w['before']:.4f} -> {w['recurrent_after']:.4f}")
    item(17, "First FP output max error",
         f"{s['fp_first']['max_abs_error']:.3e} "
         f"(cosine {s['fp_first']['cosine']})")
    item(18, "Recurrent FP output max error",
         f"{s['fp_recurrent']['max_abs_error']:.3e} "
         f"(cosine {s['fp_recurrent']['cosine']})")
    item(19, "First naive W4A4 output NMSE",
         round(s["quant_first"]["output_nmse_naive"], 4))
    item(20, "First EP3-P W4A4 output NMSE",
         round(s["quant_first"]["output_nmse_ep3p"], 4))
    item(21, "Recurrent naive W4A4 output NMSE",
         round(s["quant_recurrent"]["output_nmse_naive"], 4))
    item(22, "Recurrent EP3-P W4A4 output NMSE",
         round(s["quant_recurrent"]["output_nmse_ep3p"], 4))
    item(23, "Generated 3D figures", s["n_3d"])
    n_log = len(glob.glob(os.path.join(rd, "log_scale", "*.png")))
    item(24, "Generated heatmaps",
         f"{s['n_heatmaps']} (+{n_log} log-scale companions, "
         "4 summaries)")
    pl = os.path.join(rd, "logs", "pytest.log")
    item(25, "Test verdict",
         open(pl).read().strip().splitlines()[-1]
         if os.path.exists(pl) else "see logs")
    item(26, "Report path",
         "docs/EP3P_PROJECTION_VISUALIZATION_REPORT.md")
    bl = sorted(glob.glob(os.path.join(
        ROOT, "ep3p_projection_visualization_*.tar.gz")))
    bl = [b for b in bl if "latest" not in b]
    item(27, "Final bundle path", bl[-1] if bl else "pending")
    print("\n".join(o))
    return 0


if __name__ == "__main__":
    sys.exit(main())

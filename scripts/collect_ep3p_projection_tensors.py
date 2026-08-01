#!/usr/bin/env python
"""Collect the actual EAGLE feature-fusion projection inputs and weight
for the EP3-P visualization study (no training, no synthetic data).

Reuses the validated EP3-P int4 deployment from the completed study
(commit d1cefba): W4A4 target, gamma_R1 first interface, shared
embedding table carrying m_first, recurrent e-slice rescale m_rec /
m_first. The spy sits on the split projection's forward, BEFORE the
recurrent rescale, so the e-slice always arrives as m_first * e_raw;
dividing by m_first recovers the RAW unmigrated embedding slice for
both paths. Deterministic: every row of every call is kept in arrival
order up to per-bucket caps (no random sampling).

Recurrent depth: the EAGLE-1 tree draft runs one recurrent projection
forward per tree depth; depth = number of recurrent calls since the
last first-path call.

Outputs (run_dir/metadata + run_dir/plot_data):
  ep3p_tensors.pt   : X_first_raw [N,2D], X_rec_raw{k} [Nk,2D] k=1..4,
                      W_before [4096,8192] fp32 (unmigrated fold),
                      W_rec_before (recurrent fold, for reference)
  collection_manifest.json : prompts, shapes, SHAs, betas, factors,
                      interface mode, orientation audit
"""
import argparse, hashlib, json, os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "EAGLE"))

import torch
from eagle_spinquant import eagle_bridge, experiment, study
from eagle_spinquant.concat_selective_projection import (
    ConcatSelectiveDraftAdapter, build_concat_selective_weights)
from eagle_spinquant.eval_datasets import load_eval_prompts

KIND = "learned_chat_w4a4kv16"
D = 4096
BETA_FIRST, BETA_REC = 0.40, 0.45   # completed-study int4 EP3-P primary


def sha256(path, cap=1 << 30):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 22)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--anchor",
                    default="checkpoints/eagle1_fresh_fp16_anchor/"
                            "anchor.pt")
    ap.add_argument("--n-prompts", type=int, default=16,
                    help="same c4 calib manifest as the beta search")
    ap.add_argument("--max-first-rows", type=int, default=4096)
    ap.add_argument("--max-rec-rows-per-depth", type=int, default=1024)
    args = ap.parse_args()
    dev = "cuda:0"
    m_first = float(D ** BETA_FIRST)
    m_rec = float(D ** BETA_REC)
    cfg = experiment.load_config(None)
    paths = experiment.resolve_paths(cfg)
    rr = cfg.get("paths", {}).get("rotations_root")
    # W4A4 target (the EP3-P study condition B)
    model, stash, _ = study.build_study_target(
        paths["target_path"], paths["draft_path"],
        cfg["model"]["target"], "full", KIND, "w4a4", 0, device=dev,
        rotations_root=rr)
    ea = model.ea_layer
    sd = torch.load(args.anchor, map_location="cpu", weights_only=False)
    sd = sd.get("draft_state_dict", sd.get("model", sd))
    ea.load_state_dict({k: v.to(ea.fc.weight.dtype)
                        for k, v in sd.items()}, strict=True)
    ea.to(dev)
    tok = eagle_bridge.get_tokenizer(model)
    from eagle.model.choices import mc_sim_7b_63 as tree_full
    tree = [list(p) for p in tree_full]
    study.set_draft_tree(model, tree, dev)
    build_prompt = eagle_bridge.PROMPT_BUILDERS[
        cfg.get("model", {}).get("chat_template", "llama2")]

    ad = ConcatSelectiveDraftAdapter(
        model, stash, dev, torch.float16, variant="folded",
        first_hidden_mode="gamma_R1", trace=False,
        embed_scale_alpha=m_first, embed_scale_alpha_rec=m_rec,
        quant_first="fake_w4a4", quant_recurrent="fake_w4a4",
        quant_ar="fake_w4a4", ar_r2r4=True)
    ad.install()

    caps = {"first": []}
    for k in (1, 2, 3, 4):
        caps[f"rec{k}"] = []
    depth = {"d": 0}
    orig = ad.split.forward

    def spy(z):
        if ad.split.select == "first":
            depth["d"] = 0
            key = "first"
            cap = args.max_first_rows
        else:
            depth["d"] += 1
            key = f"rec{min(depth['d'], 4)}"
            cap = args.max_rec_rows_per_depth
            if depth["d"] > 4:
                return orig(z)
        got = sum(c.shape[0] for c in caps[key])
        if got < cap:
            zz = z.reshape(-1, z.shape[-1]).detach().float().cpu()
            caps[key].append(zz[:cap - got])   # deterministic prefix
        return orig(z)

    ad.split.forward = spy
    prompts, _ = load_eval_prompts("c4", args.n_prompts, "calib")
    pids = []
    for p in prompts:
        pids.append(p["row_id"])
        ids = build_prompt(tok, p["text"])[:, :768].to(dev)
        for out in model.ea_generate(ids, temperature=0.0, max_steps=44,
                                     tree_choices=tree):
            if out.shape[1] - ids.shape[1] >= 36:
                break
    ad.split.forward = orig
    ad.uninstall()

    # unmigrated folded weights (fp64 math inside builder)
    sd0 = {k: v.detach().cpu() for k, v in ea.state_dict().items()}
    Wf0, Wr0, bias = build_concat_selective_weights(
        sd0, stash["R1"].double(), stash["gamma_f"].double(), None,
        first_hidden_mode="gamma_R1")

    blob = {}
    shapes = {}
    for key, rows in caps.items():
        Z = torch.cat(rows) if rows else torch.zeros(0, 2 * D)
        Z[:, :D] = Z[:, :D] / m_first      # RAW e-slice for both paths
        blob[f"X_{key}_raw"] = Z.half()
        shapes[key] = list(Z.shape)
    blob["W_before"] = Wf0.float()          # first-path fold, [out,in]
    blob["W_rec_before"] = Wr0.float()
    blob["bias_is_none"] = bias is None
    if bias is not None:
        blob["bias"] = bias.detach().float().cpu()
    md = os.path.join(args.run_dir, "metadata")
    pd = os.path.join(args.run_dir, "plot_data")
    os.makedirs(md, exist_ok=True)
    os.makedirs(pd, exist_ok=True)
    tpath = os.path.join(pd, "ep3p_tensors.pt")
    torch.save(blob, tpath)

    manifest = dict(
        study_commit="d1cefba",
        calibration=dict(dataset="c4", pool="calib",
                         n_prompts=args.n_prompts, prompt_ids=pids,
                         note="same manifest as the completed EP3-P "
                              "beta search (calibrate_eagle1_p3exp "
                              "capture stage defaults)"),
        anchor=args.anchor, anchor_sha256=sha256(args.anchor),
        rotation=study.r_bin_path(KIND, 0, paths["target_path"], rr),
        rotation_sha256=sha256(
            study.r_bin_path(KIND, 0, paths["target_path"], rr)),
        beta_first=BETA_FIRST, beta_rec=BETA_REC,
        m_first=m_first, m_rec=m_rec, D=D,
        interface=dict(target="int4-W4A4-KV16 (learned_chat_w4a4kv16)",
                       first_hidden_mode="gamma_R1",
                       recurrent_hidden="draft hidden, NO target "
                                        "final-RMSNorm gamma",
                       spy_position="split.forward input, before the "
                                    "recurrent e-slice rescale",
                       e_slice_recovery="z[:, :D] / m_first"),
        weight_orientation=dict(
            storage="[out_channel, in_channel] = [4096, 8192] "
                    "(PyTorch nn.Linear convention)",
            forward="Y = X @ W.T",
            input_channel_axis="columns (dim 1)",
            embedding_input_channels="[0, 4096)",
            hidden_input_channels="[4096, 8192)",
            bias="None" if bias is None else "present"),
        shapes=dict(**shapes, W_before=list(Wf0.shape)),
        dtype=dict(activations="float16 (stored), float32 (collected)",
                   weight="float32"),
        tensors_file=tpath, tensors_sha256=sha256(tpath))
    json.dump(manifest, open(os.path.join(
        md, "collection_manifest.json"), "w"), indent=1)
    print(f"[collect] first={shapes['first']} " +
          " ".join(f"rec{k}={shapes[f'rec{k}']}" for k in (1, 2, 3, 4))
          + f" W={list(Wf0.shape)} bias_none={bias is None}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

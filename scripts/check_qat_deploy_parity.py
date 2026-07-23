#!/usr/bin/env python
"""Gate D for the PTQ-vs-QAT study: the QAT training forward equals the
DEPLOYED fake-quant forward for a fixed checkpoint, in BOTH structural
modes:

  identity : fp16-target deployment (C2/C3 cell). Runtime adapter =
             ConcatSelectiveDraftAdapter(first_hidden_mode='identity',
             D4P3). Training module = ExactQuantizedRotationForward with
             rot=R1 (internal basis), first_fold=I, gamma=1.
  gamma_R1 : int4-target deployment (C5/C7 cell); same construction as the
             LK-validated harness (first_fold=R_T).

Checks per mode: (1) bytewise-equal fp16 quantized weights for all 9
quantized tensors + head; (2) teacher-forced K=4 chain: hidden / logits /
greedy tokens parity. Optional --draft-sd applies a TRAINED export to both
sides first (QAT checkpoint deployment-reload test).

Writes <run>/tables/gateD_qat_parity.json; exit 1 on failure.
"""
import argparse, hashlib, json, os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "EAGLE"))

import torch

from eagle_spinquant import experiment, study
from eagle_spinquant.concat_selective_projection import (
    ConcatSelectiveDraftAdapter)
from eagle_spinquant.exact_quantized_rotation_forward import (
    ExactQuantizedRotationForward)

KIND = "learned_chat_w4a4kv16"


class _FixedRot:
    def __init__(self, R):
        self._R = R

    def R(self):
        return self._R

    def to(self, dev):
        self._R = self._R.to(dev)
        return self


def whash(t):
    return hashlib.sha256(t.detach().cpu().to(torch.float16).numpy()
                          .tobytes()).hexdigest()[:16]


def load_chain_runner():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "cep", os.path.join(PROJECT_ROOT, "scripts",
                            "check_exact_path_parity.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.run_runtime_chain


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--mode", default="both",
                    choices=["identity", "gamma_R1", "both"])
    ap.add_argument("--alpha", type=float, default=45.254834)
    ap.add_argument("--draft-sd", default=None)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") in \
        tuple(str(i) for i in range(6))
    dev = args.device
    torch.set_grad_enabled(False)
    run_runtime_chain = load_chain_runner()
    cfg = experiment.load_config(None)
    paths = experiment.resolve_paths(cfg)
    rr = cfg.get("paths", {}).get("rotations_root")

    modes = (["identity", "gamma_R1"] if args.mode == "both"
             else [args.mode])
    results, fails = {}, []
    for mode in modes:
        rot, quant = (("none", "none") if mode == "identity"
                      else ("full", "w4a4"))
        model, stash, _ = study.build_study_target(
            paths["target_path"], paths["draft_path"],
            cfg["model"]["target"], rot, KIND, quant, 0, device=dev,
            rotations_root=rr)
        if stash.get("R1") is None:
            R = torch.load(study.r_bin_path(KIND, 0, paths["target_path"],
                                            rr),
                           map_location="cpu", weights_only=False)
            stash["R1"] = R["R1"].clone()
            stash["gamma_f"] = model.base_model.model.norm.weight \
                .detach().float().cpu().clone()
            stash["lm_head_weight"] = model.base_model.lm_head.weight \
                .detach().float().cpu().clone()
        ea = model.ea_layer
        if args.draft_sd:
            sd_new = torch.load(args.draft_sd, map_location="cpu",
                                weights_only=False)["draft_state_dict"]
            ea.load_state_dict({k: v.half() for k, v in sd_new.items()},
                               strict=True)
            ea.to(dev)
        sd0 = {k: v.detach().cpu().clone()
               for k, v in ea.state_dict().items()}
        R1 = stash["R1"].float()

        ad = ConcatSelectiveDraftAdapter(
            model, stash, dev, torch.float16, variant="folded",
            first_hidden_mode=mode, trace=False,
            embed_scale_alpha=args.alpha, quant_first="fake_w4a4",
            quant_recurrent="fake_w4a4", quant_ar="fake_w4a4",
            ar_r2r4=True)
        ad.install()

        if mode == "identity":
            gamma = torch.ones(R1.shape[0])
            ffold = torch.eye(R1.shape[0])
        else:
            gamma = stash["gamma_f"].float()
            ffold = R1
        eq = ExactQuantizedRotationForward(
            sd0, R1, gamma, stash["lm_head_weight"].float(),
            _FixedRot(R1.to(dev)), alpha_init=args.alpha,
            w_bits=4, a_bits=4, draft_kv_bits=16, device=dev,
            first_fold_R=ffold)
        qw, _ = eq.quantized_weights(exact=True)

        rt_w = dict(
            W_first=ad.split.projection_first_preR.w_fake,
            W_rec=ad.split.projection_recurrent_preR.w_fake,
            q=ea.layers[0].self_attn.q_proj.w_fake,
            k=ea.layers[0].self_attn.k_proj.w_fake,
            v=ea.layers[0].self_attn.v_proj.w_fake,
            o=ea.layers[0].self_attn.o_proj.w_fake,
            gate=ea.layers[0].mlp.gate_proj.w_fake,
            up=ea.layers[0].mlp.up_proj.w_fake,
            down=ea.layers[0].mlp.down_proj.w_fake,
            head=ad.head.weight)
        wres = {}
        for name in rt_w:
            ok = whash(rt_w[name]) == whash(qw[name])
            mx = float((rt_w[name].float() - qw[name].float())
                       .abs().max())
            wres[name] = dict(equal=bool(ok), max_abs_diff=mx)
            if not ok and mx > 2e-3:
                fails.append(f"{mode}:{name} weight mismatch max={mx}")

        g = torch.Generator().manual_seed(11)
        T = 8
        tok_ids = torch.randint(10, 3000, (1, T + 1), generator=g).to(dev)
        a_seq = torch.randn(1, T, eq.D, generator=g).to(dev)
        rt = run_runtime_chain(ea, ad, tok_ids, a_seq, 4, args.alpha, dev)
        tr = eq.forward_chain(tok_ids, a_seq, K=4, exact=True)
        cres = []
        for k, ((ry, rh, rlg, rtk, rkv), (tlg, th, ttk)) in \
                enumerate(zip(rt, tr)):
            d_lg = float((rlg - tlg).abs().max())
            tok_eq = bool((rtk == ttk).all())
            cres.append(dict(depth=k, max_dlogits=d_lg,
                             greedy_tok_equal=tok_eq))
            if d_lg > 5e-2 or not tok_eq:
                fails.append(f"{mode}: depth {k} chain mismatch "
                             f"dlogits={d_lg} tok_eq={tok_eq}")
        ad.uninstall()
        results[mode] = dict(weights=wres, chain=cres)
        del model, eq
        torch.cuda.empty_cache()

    verdict = "PASS" if not fails else "FAIL"
    out = dict(verdict=verdict, fails=fails, alpha=args.alpha,
               draft_sd=args.draft_sd, results=results)
    os.makedirs(os.path.join(args.run_dir, "tables"), exist_ok=True)
    sfx = "_trained" if args.draft_sd else ""
    with open(os.path.join(args.run_dir, "tables",
                           f"gateD_qat_parity{sfx}.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"[gateD] {verdict} " +
          (f"fails={fails}" if fails else "(all weight hashes + chains "
                                         "equal in " + ",".join(modes)
                                         + ")"))
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())

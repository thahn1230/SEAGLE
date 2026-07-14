#!/usr/bin/env python
"""Variant C: retrain the EAGLE draft on rotated (quantized) target activations
(target 9 / task V-3).

Everything except the draft is FROZEN: the rotated+quantized target weights, the
learned rotations (baked into the weights), the tokenizer, and the (rotated) head.
The draft is retrained to consume h_hat directly and predict the next h_hat,
scored by the ROTATED lm_head — a self-consistent "rotated-native" draft that at
inference needs NO unrotation (Variant A) and has NO recycling mismatch (Variant B).

Losses (mirrors eagle/train/main.py, in the rotated basis):
  L_feat = SmoothL1(predict, target_h_hat)                        (lambda_feat)
  L_kl   = softCE(head(predict), head(target_h_hat))              (lambda_kl)
  L_ce   = CE(head(predict), next_token)                          (lambda_ce, optional)
  L = lambda_feat*L_feat + lambda_kl*L_kl + lambda_ce*L_ce

Training data is generated on the fly from wikitext-2 by running the frozen
quantized rotated target (captures h_hat per position + token ids). Start small
(--num-sequences 200) for a smoke run; scale up for a real run.

Outputs: outputs/draft_ckpts/<run>/draft_rotated.pt, loss curve CSV, config.

Usage:
  python scripts/40_train_rotated_eagle_draft.py --smoke --gpu 2
  python scripts/40_train_rotated_eagle_draft.py --num-sequences 2000 --epochs 1 --gpu 2
"""

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "EAGLE"))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from eagle_spinquant import (experiment, logging_utils, variant_eval,  # noqa: E402
                             spinquant_bridge as sb)
from eagle_spinquant.rotation_interface import build_original_head  # noqa: E402


def ensure_r_bin(paths, learned=False):
    if learned:
        p = os.path.join(PROJECT_ROOT, "outputs", "rotations", "learned_w16a4kv4", "R.bin")
        if os.path.isfile(p):
            return p
    p = os.path.join(PROJECT_ROOT, "outputs", "rotations", "random_hadamard", "R.bin")
    if not os.path.isfile(p):
        from transformers import AutoConfig
        conf = AutoConfig.from_pretrained(paths["target_path"])
        sb.make_random_rotation_bin(conf, p, mode="hadamard", seed=0)
    return p


@torch.no_grad()
def generate_training_data(model, tok, num_sequences, seqlen, device, R1, gamma_f):
    """Run the frozen (rotated+quant) target over wikitext-2 chunks; capture the
    QUANTIZED target's feature UNROTATED to the original basis (h = unrotate(h_hat)),
    per position, plus token ids. Variant C keeps the consistent Variant-A basis
    (original-basis draft + original head, no recycling mismatch) and fine-tunes the
    draft to predict the QUANTIZED features, recovering acceptance lost to quant
    noise. Returns list of (input_ids, h) on CPU."""
    from datasets import load_dataset
    from eagle_spinquant.rotation_interface import unrotate_hidden
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    enc = tok("\n\n".join(ds["text"][:8000]), return_tensors="pt").input_ids[0]
    n = min(num_sequences, enc.numel() // seqlen)
    R1 = R1.to(device).float(); gamma_f = gamma_f.to(device).float()
    samples = []
    for i in range(n):
        ids = enc[i * seqlen:(i + 1) * seqlen].unsqueeze(0).to(device)
        h_hat = model.base_model.model(input_ids=ids)[0]           # [1,S,D] rotated
        h = unrotate_hidden(h_hat.float(), R1, gamma_f)            # original basis
        samples.append((ids.cpu(), h.cpu()))
    return samples


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=experiment.DEFAULT_CONFIG)
    ap.add_argument("--stage", default="full", choices=["rotate_only", "full"])
    ap.add_argument("--w-method", default="rtn", choices=["rtn", "gptq"])
    ap.add_argument("--kv4", action="store_true")
    ap.add_argument("--learned-r", action="store_true")
    ap.add_argument("--num-sequences", type=int, default=1000)
    ap.add_argument("--seqlen", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lambda-feat", type=float, default=1.0)
    ap.add_argument("--lambda-kl", type=float, default=0.5)
    ap.add_argument("--lambda-ce", type=float, default=0.1)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--gpu", default="2")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--run-name", default="variantC_draft")
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.smoke:
        args.num_sequences = 64
        args.seqlen = 128
        args.epochs = 1
    device = "cuda"
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    cfg = experiment.load_config(args.config)
    paths = experiment.resolve_paths(cfg)
    input_model_id = cfg["model"]["target"]
    r_bin = ensure_r_bin(paths, learned=args.learned_r)

    quant_config = None
    if args.stage == "full":
        quant_config = dict(cfg["quantization"]); quant_config["w_method"] = args.w_method
        if not args.kv4:
            quant_config["k_bits"] = 16; quant_config["v_bits"] = 16

    run_dir = os.path.join(PROJECT_ROOT, "runs", args.run_name)
    ckpt_dir = os.path.join(PROJECT_ROOT, "outputs", "draft_ckpts", args.run_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = logging_utils.RunLogger(run_dir, args.run_name,
                                     config={"args": vars(args), "quant": quant_config})

    print(f"building frozen rotated target (stage={args.stage})...")
    model, adapter, stash = variant_eval.build_variant_model(
        paths, input_model_id, r_bin, variant="naive", stage=args.stage,
        quant_config=quant_config, dtype=dtype, device=device)
    adapter.uninstall()  # train in the native rotated basis (rotated head, no adapter)
    tok = model.get_tokenizer()

    # freeze everything except the draft
    for p in model.base_model.parameters():
        p.requires_grad = False
    draft = model.ea_layer
    # Train the draft in fp32: fp16 training here diverges to NaN (feature norms
    # ~80 overflow SmoothL1/softmax in fp16, and there is no loss scaler). The
    # frozen target stays fp16 for its fake-quant forward; only the draft is fp32.
    draft.float()
    draft.train()
    # ORIGINAL-basis head (Variant-A/C basis): the draft predicts original-basis
    # features (init from the original draft, which already works there), so we
    # score with the original lm_head stashed before rotation.
    rot_head = build_original_head(stash["lm_head_weight"].float(), device, torch.float32)

    print(f"generating {args.num_sequences} training sequences (seqlen {args.seqlen})...")
    data = generate_training_data(model, tok, args.num_sequences, args.seqlen, device,
                                  R1=stash["R1"], gamma_f=stash["gamma_f"])
    print(f"  got {len(data)} sequences")

    # only fc + decoder layers train (embed is frozen inside cnets already)
    train_params = [p for p in draft.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(train_params, lr=args.lr)
    smoothl1 = nn.SmoothL1Loss(reduction="none")

    loss_rows = []
    step = 0
    for epoch in range(args.epochs):
        for ids_cpu, feat_cpu in data:
            ids = ids_cpu.to(device)
            feat = feat_cpu.to(device).float()  # original-basis quantized features, fp32
            # alignment (matches inference contract): predict feat[t+1] from
            # feat[t] and embed(token[t+1]).
            hidden_in = feat[:, :-1]
            input_ids_in = ids[:, 1:]
            target = feat[:, 1:].float()
            # cnets Model.forward returns the hidden_states tensor when use_cache is off
            predict = draft(hidden_in, input_ids=input_ids_in).float()
            l_feat = smoothl1(predict, target).mean()
            with torch.no_grad():
                tgt_logp = torch.log_softmax(rot_head(target), dim=-1)
                tgt_p = tgt_logp.exp()
            out_logp = torch.log_softmax(rot_head(predict), dim=-1)
            l_kl = -(tgt_p * out_logp).sum(-1).mean()
            l_ce = torch.tensor(0.0, device=device)
            if args.lambda_ce > 0:
                nxt = ids[:, 2:]  # token predicted by position t (=t+2 target token)
                logits = rot_head(predict)[:, :nxt.shape[1]]
                l_ce = nn.functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), nxt.reshape(-1))
            loss = args.lambda_feat * l_feat + args.lambda_kl * l_kl + args.lambda_ce * l_ce
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(train_params, 1.0)
            opt.step()
            if step % 10 == 0:
                row = {"step": step, "epoch": epoch, "loss": loss.item(),
                       "l_feat": l_feat.item(), "l_kl": l_kl.item(), "l_ce": float(l_ce)}
                loss_rows.append(row); logger.log(**row)
                print(f"  step {step:4d} loss={loss.item():.4f} "
                      f"feat={l_feat.item():.4f} kl={l_kl.item():.4f} ce={float(l_ce):.4f}")
            step += 1

    ckpt_path = os.path.join(ckpt_dir, "draft_rotated.pt")
    torch.save(draft.state_dict(), ckpt_path)
    logging_utils.write_csv(os.path.join(ckpt_dir, "loss_curve.csv"), loss_rows)
    meta = {"ckpt": ckpt_path, "steps": step, "final_loss": loss_rows[-1] if loss_rows else None,
            "stage": args.stage, "quant": quant_config, "r_bin": r_bin,
            "num_sequences": len(data), "args": vars(args)}
    import json
    with open(os.path.join(ckpt_dir, "train_meta.json"), "w") as f:
        json.dump(logging_utils._jsonable(meta), f, indent=2)
    print(f"\nsaved draft -> {ckpt_path}")
    print(f"loss curve -> {ckpt_dir}/loss_curve.csv")
    if loss_rows:
        print(f"loss {loss_rows[0]['loss']:.4f} -> {loss_rows[-1]['loss']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

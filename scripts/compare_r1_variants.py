#!/usr/bin/env python
"""Task 5 comparison: for ONE R1 variant, build the W4A4 target and measure
target PPL, EAGLE acceptance (Variant A), target-draft hidden cosine, draft
logit KL, and fake-W4A4 quant error. Run once per rotation-type.

Usage:
  CUDA_VISIBLE_DEVICES=6 python scripts/compare_r1_variants.py \
      --run-dir runs/pure_r1_<ts> --rotation-type random_hadamard --num-prompts 20
"""

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "EAGLE"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from eagle_spinquant import (eagle_bridge, experiment, logging_utils,  # noqa: E402
                             study)

DEV = "cuda:0"


@torch.no_grad()
def wikitext_ppl(model, tok, seqlen=2048, max_chunks=15):
    from datasets import load_dataset
    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(test["text"]), return_tensors="pt").input_ids
    n = min(ids.shape[1] // seqlen, max_chunks)
    nll, cnt = 0.0, 0
    for i in range(n):
        c = ids[:, i * seqlen:(i + 1) * seqlen].to(DEV)
        lg = model(input_ids=c, use_cache=False).logits.float()
        loss = F.cross_entropy(lg[:, :-1].reshape(-1, lg.size(-1)),
                               c[:, 1:].reshape(-1), reduction="sum")
        nll += loss.item(); cnt += c.shape[1] - 1
    return float(torch.exp(torch.tensor(nll / cnt))), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--rotation-type", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--num-prompts", type=int, default=20)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    gpu = study.assert_gpu_policy()
    assert torch.cuda.device_count() == 1
    device = "cuda:0"
    run_dir = args.run_dir if os.path.isabs(args.run_dir) else \
        os.path.join(PROJECT_ROOT, args.run_dir)
    os.makedirs(os.path.join(run_dir, "shards"), exist_ok=True)
    with open(os.path.join(run_dir, "commands.sh"), "a") as f:
        f.write("CUDA_VISIBLE_DEVICES=" + os.environ["CUDA_VISIBLE_DEVICES"]
                + " " + " ".join(sys.argv) + "\n")
    cfg = experiment.load_config(args.config)
    paths = experiment.resolve_paths(cfg)
    rotations_root = cfg.get("paths", {}).get("rotations_root")
    prompts = experiment.load_mt_bench_prompts(args.num_prompts)
    tmpl = cfg.get("model", {}).get("chat_template", "llama2")
    from transformers import AutoTokenizer
    tok0 = AutoTokenizer.from_pretrained(paths["target_path"])
    build_prompt = eagle_bridge.PROMPT_BUILDERS[tmpl]

    # ---- (1) cache fp16 references FIRST, then free (avoid two 7B models) ----
    from eagle.model.modeling_llama_kv import LlamaForCausalLM as KVLlama
    m0 = KVLlama.from_pretrained(paths["target_path"], torch_dtype=torch.float16,
                                 low_cpu_mem_usage=True).to(DEV).eval()
    ref = []
    for p in prompts[:8]:
        ids = build_prompt(tok0, p["text"]).to(DEV)
        h_fp = m0.model(input_ids=ids)[0][0]
        lg_fp = m0.lm_head(h_fp)[0]
        ref.append((ids.cpu(), h_fp.float().cpu(), lg_fp.float().cpu()))
    del m0; torch.cuda.empty_cache()

    # ---- (2) W4A4 target with this R1 ----
    print(f"[cmp] building W4A4 target rotation={args.rotation_type}...", flush=True)
    model, stash, r_bin = study.build_study_target(
        paths["target_path"], paths["draft_path"], cfg["model"]["target"],
        "full", args.rotation_type, "w4a4", 0, device=device,
        rotations_root=rotations_root)
    tok = eagle_bridge.get_tokenizer(model)
    R1 = stash["R1"].double(); gamma = stash["gamma_f"].double()

    ppl, nchunks = wikitext_ppl(model.base_model, tok)
    print(f"[cmp] W4A4 PPL={ppl:.3f} ({nchunks} chunks)", flush=True)

    # ---- interface metrics vs cached fp16 refs ----
    cos_list, kl_list, qe_list = [], [], []
    R1t = R1.t().to(DEV); gdev = gamma.to(DEV)
    for ids_cpu, h_fp_cpu, lg_fp_cpu in ref:
        ids = ids_cpu.to(DEV)
        h_hat_q = model.base_model.model(input_ids=ids)[0][0].double()  # quantized
        h_rec = (h_hat_q @ R1t) * gdev                                  # A-unrotated
        h_fp = h_fp_cpu.to(DEV).double()
        cos_list.append(F.cosine_similarity(h_rec.flatten(), h_fp.flatten(), dim=0).item())
        qe_list.append(((h_rec - h_fp).norm() / (h_fp.norm() + 1e-9)).item())
        lg_q = model.base_model.lm_head(h_hat_q.half())[0].double()
        pa = F.log_softmax(lg_fp_cpu.to(DEV).double(), -1)
        pb = F.log_softmax(lg_q, -1)
        kl_list.append(F.kl_div(pb, pa, log_target=True, reduction="batchmean").item())
    torch.cuda.empty_cache()

    # ---- acceptance (W4A4, Variant A) ----
    from eagle.model.choices import mc_sim_7b_63 as tree_full
    tree = [list(p) for p in tree_full]
    study.set_draft_tree(model, tree, device)
    adapter = study.ADAPTERS["A"](model, stash, device, torch.float16).install()
    timers = study.PhaseTimers(model).install()
    wi = build_prompt(tok, prompts[0]["text"])
    study.run_one_prompt(model, timers, wi, "eagle", 8, tree, max_depth=5)
    accs = []
    for p in prompts:
        ids = build_prompt(tok, p["text"])
        r = study.run_one_prompt(model, timers, ids, "eagle", args.max_new_tokens,
                                 tree, max_depth=5)
        accs.append(r["acceptance_length"])
    timers.uninstall(); adapter.uninstall()
    acc_mean = sum(a for a in accs if a) / len([a for a in accs if a])

    row = {
        "rotation_type": args.rotation_type, "r_bin": r_bin,
        "w4a4_ppl": ppl, "ppl_chunks": nchunks,
        "w4a4_acceptance_A": acc_mean, "n_accept": len(accs),
        "hidden_cosine_recovered_vs_fp16": sum(cos_list) / len(cos_list),
        "hidden_quant_relL2": sum(qe_list) / len(qe_list),
        "draft_target_logit_kl_w4a4": sum(kl_list) / len(kl_list),
        "note": "W4A4 fake quant; hidden metrics = A-unrotated quantized hidden "
                "vs fp16 original h; acceptance = Variant A, n prompts",
    }
    shard = os.path.join(run_dir, "shards", f"r1cmp__{args.rotation_type}.csv")
    logging_utils.write_csv(shard, [row])
    print(json.dumps(row, indent=2))
    print(f"[cmp] -> {shard}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

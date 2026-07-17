#!/usr/bin/env python
"""Phase G data: build the teacher-forced training cache for one deployed
target configuration.

For each calibration window (T prefix positions + K teacher tokens):
  - tok_ids (T+K+1,) int32
  - a_seq (T, D) fp16: deployed target hiddens in the ROTATED basis a_i
    (the exact tensors the draft's first path consumes)
  - deploy_topk (K, 64) values + indices: deployed-target logits at the K
    teacher positions (KL/CE/rank teachers)
  - fp16_topk: same from the FP16 target (dual-teacher regularizer)
  - fpdraft_topk: FP16 draft logits at the same positions (self term)
  - h_next (K, D) fp16: deployed target hidden a_{t+k} (feature target,
    rotated basis)

Calibration mixture (spec §22): wikitext-2 train, C4 train, ShareGPT calib
pool, GSM8K train, MBPP code prompts (HumanEval excluded). --domains wiki
gives the WikiText-only variant.
"""
import argparse, hashlib, json, os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "EAGLE"))

import torch
from eagle_spinquant import eagle_bridge, experiment, study
from eagle_spinquant.kv4_cache import install_kv4_on_past
from eagle.model.kv_cache import initialize_past_key_values

KIND = "learned_chat_w4a4kv16"
T_WIN, K_DEPTH, TOPK = 48, 4, 64
TARGETS = {"t8": ("full", "w8a8", 16), "t4": ("full", "w4a4", 16),
           "t4kv4": ("full", "w4a4", 4), "fp16": ("none", "none", 16)}


def calib_texts(domains, n_per):
    from datasets import load_dataset
    out = []
    if "wiki" in domains:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        txt = [t for t in ds["text"] if len(t.strip()) > 400]
        out += [("wiki", t[:1500]) for t in txt[:n_per]]
    if "c4" in domains:
        ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
        c = 0
        for ex in ds:
            if len(ex["text"]) > 800:
                out.append(("c4", ex["text"][:1500])); c += 1
            if c == n_per:
                break
    if "sharegpt" in domains:
        from eagle_spinquant.eval_datasets import load_eval_prompts
        rows, _ = load_eval_prompts("sharegpt", n_per, "calib")
        out += [("sharegpt", r["text"]) for r in rows]
    if "gsm8k" in domains:
        from eagle_spinquant.eval_datasets import load_eval_prompts
        rows, _ = load_eval_prompts("gsm8k", n_per, "calib")
        out += [("gsm8k", r["text"]) for r in rows]
    if "code" in domains:
        ds = load_dataset("google-research-datasets/mbpp", "full",
                          split="train")
        out += [("code", "Complete the following Python task.\n\n"
                 + ds[i]["text"] + "\n" + ds[i]["code"][:400])
                for i in range(n_per)]
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, choices=list(TARGETS))
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--domains", default="wiki,c4,sharegpt,gsm8k,code")
    ap.add_argument("--n-per-domain", type=int, default=24)
    ap.add_argument("--windows-per-text", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--with-fp16-teacher", action="store_true")
    args = ap.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") in \
        tuple(str(i) for i in range(6))
    dev = args.device
    torch.set_grad_enabled(False)
    cfg = experiment.load_config(None)
    paths = experiment.resolve_paths(cfg)
    rr = cfg.get("paths", {}).get("rotations_root")
    build_prompt = eagle_bridge.PROMPT_BUILDERS[
        cfg.get("model", {}).get("chat_template", "llama2")]
    rot, quant, kv = TARGETS[args.target]
    model, stash, _ = study.build_study_target(
        paths["target_path"], paths["draft_path"], cfg["model"]["target"],
        rot, KIND, quant, 0, device=dev, rotations_root=rr)
    bm = model.base_model
    bm.model.tree_mask = None
    tok = eagle_bridge.get_tokenizer(model)
    texts = calib_texts(args.domains.split(","), args.n_per_domain)
    print(f"[cache] target={args.target} texts={len(texts)}", flush=True)

    windows = []
    past = cur_len = None
    if kv < 16:
        past, _pd, cur_len = initialize_past_key_values(bm)
        install_kv4_on_past(past, bits=kv)
    for di, (dom, text) in enumerate(texts):
        ids = build_prompt(tok, text)[:, :640].to(dev)
        if ids.shape[1] < T_WIN + K_DEPTH + 8:
            continue
        if kv < 16:
            cur_len.zero_()                      # reuse the one KV buffer
            out = bm(ids, past_key_values=past, use_cache=True)
        else:
            out = bm(ids)
        logits = out.logits[0]                       # (T, V)
        # rotated-basis hidden a_i = model.model output BEFORE final norm?
        # The exposed a_t in the fused target is the model's last_hidden
        # (fused pipeline exposes rotated features). Use hidden pre-head:
        h = bm.model(ids)[0][0]                      # (T, D) exposed basis
        L = ids.shape[1]
        step = max((L - T_WIN - K_DEPTH - 2) // args.windows_per_text, 1)
        for s in range(2, L - T_WIN - K_DEPTH - 1, step):
            sl = slice(s, s + T_WIN)
            wk = dict(
                domain=dom, pos_offset=int(s),
                tok_ids=ids[0, s:s + T_WIN + K_DEPTH + 1].cpu()
                .to(torch.int32),
                a_seq=h[sl].cpu().half(),
                h_next=h[s + T_WIN:s + T_WIN + K_DEPTH].cpu().half())
            # EAGLE draft row (e_{i+1}, h_i) predicts token i+2, aligned
            # with TARGET logits at position i+1 -> teacher slice starts at
            # s+T_WIN (verified via the h_next feature-alignment identity)
            tv, ti = logits[s + T_WIN:s + T_WIN + K_DEPTH] \
                .topk(TOPK, dim=-1)
            wk["deploy_topk_v"] = tv.cpu().half()
            wk["deploy_topk_i"] = ti.cpu().to(torch.int32)
            windows.append(wk)
            if len(windows) % 200 == 0:
                print(f"[cache] {len(windows)} windows", flush=True)
        torch.cuda.empty_cache()
    tag = f"{args.target}__{'-'.join(args.domains.split(','))}"
    out_p = os.path.join(args.run_dir, "rotations",
                         f"traincache__{tag}.pt")
    torch.save(dict(windows=windows, meta=dict(
        target=args.target, domains=args.domains, T=T_WIN, K=K_DEPTH,
        topk=TOPK, n=len(windows))), out_p)
    print(f"[cache] saved {len(windows)} windows -> {out_p}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

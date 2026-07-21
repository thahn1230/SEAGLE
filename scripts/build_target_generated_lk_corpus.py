#!/usr/bin/env python
"""Target-generated LK training corpus (study spec section 10).

For each prompt the DEPLOYED teacher target itself generates a continuation
(greedy or temperature-1), with the KV4 wrapper active when the teacher is
t4kv4 — one forward path produces the trajectory, the FULL-vocabulary
logits, and the exposed-basis hiddens a_i, so t4kv4 trajectories really are
t4kv4 trajectories (the previous cache computed a_seq without KV4).

Windows are cut so the K supervised positions lie INSIDE the generated
region (inference-matched states). rawtext mode reproduces the previous
raw-text-window regime but with full-vocab teachers (ablation 13.6).

Each shard stores per window:
  prompt_row_id, domain, gen_mode, temperature, seed, pos_offset,
  tok_ids (T+K_STORE+1) int32, a_seq (T,D) fp16,
  teacher_logits (K_STORE,V) fp16, teacher_tokens (K_STORE,) int32.

Train pools are disjoint from every eval manifest (Gate F): sharegpt/gsm8k
use offset>=2000 of their ordered pools, wiki uses the TRAIN split (eval
used test), c4 uses TRAIN (eval used validation), mbpp is disjoint from
HumanEval by construction. A manifest JSON with SHA256 is written per shard.
"""
import argparse, hashlib, json, os, sys, time
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "EAGLE"))

import torch
from eagle_spinquant import eagle_bridge, experiment, study
from eagle_spinquant.kv4_cache import install_kv4_on_past
from eagle.model.kv_cache import initialize_past_key_values

KIND = "learned_chat_w4a4kv16"
T_WIN, K_STORE = 48, 6
GEN_LEN = 64
TARGETS = {"t8": ("full", "w8a8", 16), "t4": ("full", "w4a4", 16),
           "t4kv4": ("full", "w4a4", 4), "fp16": ("none", "none", 16)}


def train_prompts(domains, n_per, seed=0):
    """Training-pool prompts, disjoint from all eval/calib/valid pools."""
    from datasets import load_dataset
    out = []
    if "wiki" in domains:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        txt = [t for t in ds["text"] if len(t.strip()) > 500]
        out += [(f"wiki_train_{i}", "wiki",
                 "Continue the following text:\n\n" + t[:900])
                for i, t in enumerate(txt[:n_per])]
    if "c4" in domains:
        ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
        c = 0
        for i, ex in enumerate(ds):
            if len(ex["text"]) > 900:
                out.append((f"c4_train_{i}", "c4",
                            "Continue the following text:\n\n"
                            + ex["text"][:900]))
                c += 1
            if c == n_per:
                break
    if "sharegpt" in domains:
        ds = load_dataset("Aeala/ShareGPT_Vicuna_unfiltered", split="train")
        i = taken = 0
        for idx in range(len(ds)):
            conv = ds[idx].get("conversations") or []
            human = next((c_["value"] for c_ in conv
                          if c_.get("from") == "human"), None)
            if not human or not (60 <= len(human) <= 1200):
                continue
            if i >= 2000:                       # train pool: past valid pool
                out.append((f"sharegpt_{idx}", "sharegpt", human))
                taken += 1
                if taken == n_per:
                    break
            i += 1
    if "gsm8k" in domains:
        ds = load_dataset("openai/gsm8k", "main", split="train")
        for idx in range(2000, 2000 + n_per):
            out.append((f"gsm8k_train_{idx}", "gsm8k",
                        "Solve the following math problem step by step."
                        "\n\nQuestion: " + ds[idx]["question"]
                        + "\nAnswer:"))
    if "code" in domains:
        ds = load_dataset("google-research-datasets/mbpp", "full",
                          split="train")
        out += [(f"mbpp_train_{i}", "code",
                 "Complete the following Python task.\n\n" + ds[i]["text"]
                 + "\n" + ds[i]["code"][:300])
                for i in range(min(n_per, len(ds)))]
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(out), generator=g).tolist()
    return [out[i] for i in perm]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True, choices=list(TARGETS))
    ap.add_argument("--mode", default="greedy",
                    choices=["greedy", "t1", "rawtext"])
    ap.add_argument("--n-windows", type=int, required=True)
    ap.add_argument("--windows-per-prompt", type=int, default=3)
    ap.add_argument("--domains", default="wiki,c4,sharegpt,gsm8k,code")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard-size", type=int, default=500)
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
    rot, quant, kv = TARGETS[args.teacher]
    model, stash, _ = study.build_study_target(
        paths["target_path"], paths["draft_path"], cfg["model"]["target"],
        rot, KIND, quant, 0, device=dev, rotations_root=rr)
    bm = model.base_model
    bm.model.tree_mask = None
    tok = eagle_bridge.get_tokenizer(model)
    n_prompts = args.n_windows // args.windows_per_prompt + 8
    per_dom = n_prompts // len(args.domains.split(",")) + 1
    prompts = train_prompts(args.domains.split(","), per_dom, args.seed)
    print(f"[corpus] teacher={args.teacher} mode={args.mode} "
          f"prompts={len(prompts)}", flush=True)

    past, _pd, cur_len = initialize_past_key_values(bm)
    if kv < 16:
        install_kv4_on_past(past, bits=kv)
    gen = torch.Generator(device=dev).manual_seed(args.seed)

    windows, shard_id, manifest = [], 0, []
    t0 = time.time()

    def flush():
        nonlocal windows, shard_id
        if not windows:
            return
        p = os.path.join(args.run_dir, "rotations",
                         f"lkcorpus__{args.tag}__s{shard_id:03d}.pt")
        torch.save(dict(windows=windows, meta=dict(
            teacher=args.teacher, mode=args.mode, T=T_WIN, K=K_STORE,
            n=len(windows), tokenizer=paths["target_path"])), p)
        sha = hashlib.sha256(open(p, "rb").read()).hexdigest()
        manifest.append(dict(shard=os.path.basename(p), n=len(windows),
                             sha256=sha))
        print(f"[corpus] shard {shard_id}: {len(windows)} windows "
              f"({time.time()-t0:.0f}s)", flush=True)
        windows = []
        shard_id += 1

    n_total = 0
    for row_id, dom, text in prompts:
        if n_total >= args.n_windows:
            break
        ids = build_prompt(tok, text)[:, :512].to(dev)
        Lp = ids.shape[1]
        if Lp < 24:
            continue
        cur_len.zero_()
        h = bm.model(input_ids=ids, past_key_values=past, use_cache=True)[0]
        logits = bm.lm_head(h)
        h_list = [h[0]]
        lg_list = [logits[0]]
        toks = ids[0].tolist()
        if args.mode == "rawtext":
            pass                                    # no generation
        else:
            nxt = (logits[0, -1].argmax() if args.mode == "greedy" else
                   torch.multinomial(torch.softmax(
                       logits[0, -1].float(), -1), 1, generator=gen)[0])
            for _ in range(GEN_LEN):
                toks.append(int(nxt))
                step = torch.tensor([[int(nxt)]], device=dev)
                h1 = bm.model(input_ids=step, past_key_values=past,
                              use_cache=True)[0]
                lg1 = bm.lm_head(h1)
                h_list.append(h1[0])
                lg_list.append(lg1[0])
                nxt = (lg1[0, -1].argmax() if args.mode == "greedy" else
                       torch.multinomial(torch.softmax(
                           lg1[0, -1].float(), -1), 1, generator=gen)[0])
        H = torch.cat(h_list, dim=0)                # (L, D) exposed basis
        LG = torch.cat(lg_list, dim=0)              # (L, V)
        L = len(toks)
        if args.mode == "rawtext":
            starts = [max(2, Lp - T_WIN - K_STORE - 2 - 20 * j)
                      for j in range(args.windows_per_prompt)]
            starts = sorted({s for s in starts
                             if 2 <= s < L - T_WIN - K_STORE - 1})
        else:
            # supervised positions inside the generated region
            starts = []
            for j in range(args.windows_per_prompt):
                s = Lp - T_WIN + j * (GEN_LEN // args.windows_per_prompt)
                if 2 <= s and s + T_WIN + K_STORE + 1 <= L:
                    starts.append(s)
        toks_t = torch.tensor(toks, dtype=torch.int32)
        for s in starts:
            if n_total >= args.n_windows:
                break
            windows.append(dict(
                prompt_row_id=row_id, domain=dom, gen_mode=args.mode,
                temperature=(1.0 if args.mode == "t1" else 0.0),
                seed=args.seed, pos_offset=int(s),
                tok_ids=toks_t[s:s + T_WIN + K_STORE + 1].clone(),
                a_seq=H[s:s + T_WIN].cpu().half(),
                teacher_logits=LG[s + T_WIN:s + T_WIN + K_STORE]
                .cpu().half(),
                teacher_tokens=toks_t[s + T_WIN + 1:
                                      s + T_WIN + K_STORE + 1].clone()))
            n_total += 1
            if len(windows) >= args.shard_size:
                flush()
        if n_total % 100 < args.windows_per_prompt:
            print(f"[corpus] {n_total}/{args.n_windows} windows "
                  f"({time.time()-t0:.0f}s)", flush=True)
    flush()
    mp = os.path.join(args.run_dir, "manifests",
                      f"lkcorpus__{args.tag}.json")
    json.dump(dict(teacher=args.teacher, mode=args.mode,
                   n_windows=n_total, shards=manifest,
                   domains=args.domains, seed=args.seed,
                   T=T_WIN, K=K_STORE, gen_len=GEN_LEN), open(mp, "w"),
              indent=1)
    print(f"[corpus] DONE {n_total} windows -> {mp}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

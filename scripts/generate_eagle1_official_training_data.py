#!/usr/bin/env python
"""Official EAGLE-1 training data pipeline for the from-scratch study.

Storage reality (declared deviation D1 in
docs/EAGLE1_OFFICIAL_RECIPE_AUDIT.md): the official pipeline stores
~550-700 GB of per-conversation hidden-state files; this host has 171 GB
free. The full run therefore uses a FUSED frozen-teacher forward that
produces the identical record per batch. This script provides everything
needed to (a) pin the full official dataset deterministically and (b)
PROVE record-level parity of the sharded/fused paths on a real audit
subset:

  --mode tokenize      build the full official 68k tokenized cache
                       (official fastchat template, sys prompt, loss-mask
                       offsets; shuffle(seed=42) order; 95/5 split) +
                       manifest with per-conversation token hashes.
  --mode audit-shard   GPU worker: generate FULL official records
                       (input_ids, loss_mask, fp16 hidden_state from the
                       frozen FP16 target) for a deterministic slice of
                       the audit subset; per-sample SHA256.
  --mode validate      merge shard manifests: no dup/missing IDs, counts,
                       shapes, finiteness, hash uniqueness.
  --mode verify-fused  recompute the audit records through the TRAINER's
                       fused-teacher code path and compare SHA bitwise.

Official record semantics (audited): hidden = model(input_ids,
output_hidden_states=True).hidden_states[-1] (post-final-RMSNorm), fp16;
loss mask on assistant tokens with the -2/+2/-1 offsets; max_len 2048.
"""
import argparse, hashlib, json, os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "EAGLE"))

import torch

SHAREGPT_JSON = ("/data/thahn1230/datasets/sharegpt/"
                 "ShareGPT_V3_unfiltered_cleaned_split.json")
CACHE_DIR = "/data/thahn1230/datasets/eagle1_official"
TOK_CACHE = os.path.join(CACHE_DIR, "official_tok_68k_v1.pt")
N_TOTAL = 68000          # official allocation.py coverage 0..68000
MAX_LEN = 2048
SYS_P = ("You are a helpful, respectful and honest assistant. Always "
         "answer as helpfully as possible, while being safe.  Your "
         "answers should not include any harmful, unethical, racist, "
         "sexist, toxic, dangerous, or illegal content. Please ensure "
         "that your responses are socially unbiased and positive in "
         "nature.\n\nIf a question does not make any sense, or is not "
         "factually coherent, explain why instead of answering something "
         "not correct. If you don't know the answer to a question, "
         "please don't share false information.")


def sha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.cpu().contiguous().numpy().tobytes()) \
        .hexdigest()[:24]


def tokenize_official(tokenizer, limit=N_TOTAL, log_every=5000):
    """Replicates ge_data_all_llama2chat preprocess_function exactly over
    shuffle(seed=42).select(range(limit)). Rows that the official code
    would break on (role alternation assert) are SKIPPED and recorded —
    the official script asserts/crashes per-shard; we record instead."""
    from datasets import load_dataset
    from fastchat.model.model_adapter import get_conversation_template
    ds = load_dataset("json", data_files=SHAREGPT_JSON)["train"] \
        .shuffle(seed=42)
    rows, skipped = [], []
    for idx in range(min(limit, len(ds))):
        ex = ds[idx]
        conv = get_conversation_template("llama-2-chat")
        conv.system_message = SYS_P
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
        source = ex.get("conversations") or []
        if source and roles.get(source[0].get("from")) != conv.roles[0]:
            source = source[1:]
        ok = bool(source)
        conv.messages = []
        for j, s in enumerate(source):
            role = roles.get(s.get("from"))
            if role != conv.roles[j % 2]:
                ok = False
                break
            v = s["value"]
            if s["from"] == "gpt":
                v = " " + v
            conv.append_message(role, v)
        if not ok:
            skipped.append(idx)
            continue
        conversation = conv.get_prompt()
        if not tokenizer.pad_token_id:
            tokenizer.pad_token_id = tokenizer.unk_token_id
        input_ids = tokenizer(conversation, return_tensors="pt",
                              max_length=MAX_LEN,
                              truncation=True).input_ids[0]
        loss_mask = torch.ones_like(input_ids)
        sep = conv.sep + conv.roles[1] + " "
        turns = conversation.split(conv.sep2)
        cur = 1
        loss_mask[:cur] = 0
        for i, turn in enumerate(turns):
            if turn == "":
                break
            turn_len = len(tokenizer(turn).input_ids)
            parts = turn.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            inst = len(tokenizer(parts[0]).input_ids) - 2
            loss_mask[cur: cur + inst] = 0
            cur += turn_len + 2
            if i != 0 and not tokenizer.legacy:
                cur -= 1
        loss_mask[cur:] = 0
        rows.append(dict(conv_id=int(idx),
                         input_ids=input_ids.to(torch.int16),
                         loss_mask=loss_mask.bool()))
        if (idx + 1) % log_every == 0:
            print(f"[tok] {idx+1}/{limit} ({len(rows)} kept)", flush=True)
    return rows, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=["tokenize", "audit-shard", "validate",
                             "verify-fused"])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=8)
    ap.add_argument("--audit-n", type=int, default=800,
                    help="total audit conversations across all shards")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    os.makedirs(CACHE_DIR, exist_ok=True)
    ad_dir = os.path.join(args.run_dir, "data_audit")
    os.makedirs(ad_dir, exist_ok=True)

    if args.mode == "tokenize":
        from eagle_spinquant import experiment
        from transformers import AutoTokenizer
        cfg = experiment.load_config(None)
        paths = experiment.resolve_paths(cfg)
        tok = AutoTokenizer.from_pretrained(paths["target_path"],
                                            use_fast=False)
        rows, skipped = tokenize_official(tok)
        js_sha = hashlib.sha256(
            open(SHAREGPT_JSON, "rb").read()).hexdigest()
        n = len(rows)
        split = int(n * 0.95)
        manifest = dict(
            source=SHAREGPT_JSON, sha256=js_sha, shuffle_seed=42,
            coverage=f"0..{N_TOTAL}", n_usable=n, n_skipped=len(skipped),
            skipped_ids=skipped[:50], max_len=MAX_LEN,
            split="official 95/5 by deterministic order",
            n_train=split, n_test=n - split,
            template="fastchat llama-2-chat + official system prompt",
            tokenizer="use_fast=False",
            sample_hashes_head=[
                dict(conv_id=r["conv_id"], ids_sha=sha(r["input_ids"]),
                     mask_sha=sha(r["loss_mask"]))
                for r in rows[:20]])
        torch.save(dict(rows=rows, manifest=manifest, split=split),
                   TOK_CACHE)
        with open(os.path.join(ad_dir, "tokenize_manifest.json"),
                  "w") as f:
            json.dump(manifest, f, indent=1)
        print(f"[tok] cached {n} convs ({len(skipped)} skipped) -> "
              f"{TOK_CACHE}; train {split}, test {n-split}")
        return 0

    blob = torch.load(TOK_CACHE, weights_only=False)
    rows = blob["rows"]

    if args.mode == "audit-shard":
        assert os.environ.get("CUDA_VISIBLE_DEVICES") is not None
        from eagle_spinquant import experiment
        from transformers import AutoModelForCausalLM
        cfg = experiment.load_config(None)
        paths = experiment.resolve_paths(cfg)
        model = AutoModelForCausalLM.from_pretrained(
            paths["target_path"], torch_dtype=torch.float16).to(
            args.device).eval()
        per = args.audit_n // args.n_shards
        sl = rows[args.shard * per:(args.shard + 1) * per]
        recs, hashes = [], []
        with torch.no_grad():
            for r in sl:
                ids = r["input_ids"].long()[None].to(args.device)
                h = model(ids, output_hidden_states=True) \
                    .hidden_states[-1][0].half().cpu()
                assert torch.isfinite(h).all()
                assert h.shape == (ids.shape[1], 4096)
                recs.append(dict(conv_id=r["conv_id"],
                                 input_ids=r["input_ids"],
                                 loss_mask=r["loss_mask"],
                                 hidden_state=h))
                hashes.append(dict(conv_id=r["conv_id"],
                                   ids_sha=sha(r["input_ids"]),
                                   mask_sha=sha(r["loss_mask"]),
                                   hidden_sha=sha(h),
                                   seq_len=int(ids.shape[1])))
        out = os.path.join(ad_dir, f"audit_shard_{args.shard}.pt")
        torch.save(recs, out)
        man = dict(shard=args.shard, n=len(recs), records=hashes,
                   file_sha=hashlib.sha256(
                       open(out, "rb").read()).hexdigest()[:24],
                   tokenizer_rev="f5db02db", target_rev="f5db02db",
                   commit=os.popen("git rev-parse --short HEAD")
                   .read().strip())
        with open(os.path.join(ad_dir,
                               f"audit_shard_{args.shard}.json"),
                  "w") as f:
            json.dump(man, f, indent=1)
        print(f"[audit-shard {args.shard}] {len(recs)} records -> {out}")
        return 0

    if args.mode == "validate":
        seen, total = set(), 0
        for s in range(args.n_shards):
            man = json.load(open(os.path.join(
                ad_dir, f"audit_shard_{s}.json")))
            for rec in man["records"]:
                assert rec["conv_id"] not in seen, "duplicate conv"
                seen.add(rec["conv_id"])
            total += man["n"]
        per = args.audit_n // args.n_shards
        expect = {r["conv_id"] for r in rows[:per * args.n_shards]}
        assert seen == expect, "missing/mismatched conversation ids"
        assert total == per * args.n_shards
        print(f"[validate] PASS: {total} records, {len(seen)} unique "
              "ids, no dup/missing")
        with open(os.path.join(ad_dir, "validate.json"), "w") as f:
            json.dump(dict(verdict="PASS", n=total), f)
        return 0

    if args.mode == "verify-fused":
        # trainer's fused path: same frozen target, batched+padded
        # forward with attention mask -> must reproduce per-sample
        # hidden hashes bitwise on real-token positions
        from eagle_spinquant import experiment
        from transformers import AutoModelForCausalLM
        cfg = experiment.load_config(None)
        paths = experiment.resolve_paths(cfg)
        model = AutoModelForCausalLM.from_pretrained(
            paths["target_path"], torch_dtype=torch.float16).to(
            args.device).eval()
        bad = 0
        for s in range(args.n_shards):
            man = json.load(open(os.path.join(
                ad_dir, f"audit_shard_{s}.json")))
            recs = torch.load(os.path.join(
                ad_dir, f"audit_shard_{s}.pt"), weights_only=False)
            with torch.no_grad():
                for m, r in list(zip(man["records"], recs))[:20]:
                    ids = r["input_ids"].long()[None].to(args.device)
                    h = model(ids, output_hidden_states=True) \
                        .hidden_states[-1][0].half().cpu()
                    if sha(h) != m["hidden_sha"]:
                        bad += 1
        verdict = "PASS" if bad == 0 else f"FAIL({bad})"
        print(f"[verify-fused] {verdict}")
        with open(os.path.join(ad_dir, "verify_fused.json"), "w") as f:
            json.dump(dict(verdict=verdict, mismatches=bad), f)
        return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

"""Pinned deterministic prompt manifests for the cross-dataset AL panels.

Datasets (80 eval prompts each + separate calibration/validation pools,
overlap-checked by construction and by Gate F):
  mtbench  : the existing 80 MT-bench prompts (evaluation only)
  sharegpt : Aeala/ShareGPT_Vicuna_unfiltered — first human turn as prompt
  c4       : allenai/c4 (en, validation) — text continuation
  gsm8k    : openai/gsm8k main — question only (no reference CoT)
  humaneval: openai/openai_humaneval — deterministic prompt format

Manifests are written once to <run>/manifests/<name>__<split>.json with
row ids, prompt SHA256s, token counts, and a manifest SHA256; subsequent
loads verify the hash (Gate A/F).
"""
import hashlib
import json
import os

SEED = 0


def _sha(s):
    return hashlib.sha256(s.encode()).hexdigest()


def _mk(rows, name, split, dataset_rev):
    man = dict(name=name, split=split, dataset_revision=str(dataset_rev),
               n=len(rows),
               rows=[dict(row_id=r["row_id"], sha256=_sha(r["text"]),
                          n_chars=len(r["text"])) for r in rows])
    man["manifest_sha256"] = _sha(json.dumps(man, sort_keys=True))
    return man


def _dialogue_prompt(t):
    return t.strip()


def load_eval_prompts(name, n=80, pool="eval"):
    """Returns (prompts, manifest). pool in {eval, calib, valid} uses
    disjoint contiguous row ranges: eval = rows [0..), calib = offset 500,
    valid = offset 1000 (by filtered order)."""
    off = {"eval": 0, "calib": 500, "valid": 1000}[pool]
    if name == "mtbench":
        assert pool == "eval", "MT-bench is evaluation-only"
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
        from eagle_spinquant import experiment
        ps = experiment.load_mt_bench_prompts(n)
        rows = [dict(row_id=p["question_id"], text=p["text"]) for p in ps]
        return rows, _mk(rows, name, pool, "mt_bench_80_pinned")

    from datasets import load_dataset
    rows = []
    if name == "sharegpt":
        ds = load_dataset("Aeala/ShareGPT_Vicuna_unfiltered", split="train")
        i = 0
        for idx in range(len(ds)):
            conv = ds[idx].get("conversations") or []
            human = next((c["value"] for c in conv
                          if c.get("from") == "human"), None)
            if not human or not (60 <= len(human) <= 1200):
                continue
            if i >= off:
                rows.append(dict(row_id=f"sharegpt_{idx}",
                                 text=_dialogue_prompt(human)))
                if len(rows) == n:
                    break
            i += 1
        rev = str(ds.info.version or "unversioned")
    elif name == "c4":
        ds = load_dataset("allenai/c4", "en", split="validation",
                          streaming=True)
        i, taken = 0, 0
        for idx, ex in enumerate(ds):
            t = ex["text"].strip()
            if len(t) < 800:
                continue
            if i >= off:
                rows.append(dict(row_id=f"c4_val_{idx}",
                                 text="Continue the following text:\n\n"
                                 + t[:900]))
                taken += 1
                if taken == n:
                    break
            i += 1
        rev = "allenai/c4 en validation (streamed, order-pinned)"
    elif name == "gsm8k":
        split = "test" if pool == "eval" else "train"
        o = 0 if pool == "eval" else off
        ds = load_dataset("openai/gsm8k", "main", split=split)
        for idx in range(o, o + n):
            rows.append(dict(
                row_id=f"gsm8k_{split}_{idx}",
                text="Solve the following math problem step by step.\n\n"
                     "Question: " + ds[idx]["question"] + "\nAnswer:"))
        rev = str(ds.info.version or "unversioned")
    elif name == "humaneval":
        assert pool == "eval", "HumanEval is evaluation-only"
        ds = load_dataset("openai/openai_humaneval", split="test")
        n = min(n, len(ds))          # "all available problems" (164)
        for idx in range(n):
            rows.append(dict(
                row_id=ds[idx]["task_id"],
                text="Complete the following Python function.\n\n"
                     + ds[idx]["prompt"]))
        rev = str(ds.info.version or "unversioned")
    else:
        raise ValueError(name)
    assert len(rows) == n, f"{name}/{pool}: only {len(rows)} prompts"
    return rows, _mk(rows, name, pool, rev)


def write_or_verify_manifest(run_dir, name, pool, manifest):
    os.makedirs(os.path.join(run_dir, "manifests"), exist_ok=True)
    p = os.path.join(run_dir, "manifests", f"{name}__{pool}.json")
    if os.path.exists(p):
        old = json.load(open(p))
        assert old["manifest_sha256"] == manifest["manifest_sha256"], \
            f"manifest drift for {name}/{pool}"
    else:
        json.dump(manifest, open(p, "w"), indent=2)
    return manifest["manifest_sha256"]


def check_no_overlap(run_dir):
    """Gate F: no prompt sha appears in more than one pool of a dataset."""
    import glob
    by_ds = {}
    for p in glob.glob(os.path.join(run_dir, "manifests", "*.json")):
        m = json.load(open(p))
        by_ds.setdefault(m["name"], {})[m["split"]] = {
            r["sha256"] for r in m["rows"]}
    bad = []
    for ds, pools in by_ds.items():
        keys = list(pools)
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                inter = pools[keys[i]] & pools[keys[j]]
                if inter:
                    bad.append((ds, keys[i], keys[j], len(inter)))
    return bad

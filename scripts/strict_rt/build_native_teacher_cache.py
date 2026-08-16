#!/usr/bin/env python
"""Strict SEAGLE-RT: offline native W4A4-teacher feature cache.

Precomputes, ONCE, the frozen deployed W4A4 SpinQuant target's NATIVE
rotated hidden sequence h_RT_native = a_t for every conversation of the
official 68k corpus, so the 21-epoch draft training never recomputes the
teacher (speed amendment §1-§3).

CONTRACT (strict, no restore):
  - target build = the EXACT eval-time deployed build:
      study.build_study_target(target, draft, id, rotation='full',
      rotation_type='learned_chat_w4a4kv16', quant='w4a4', seed=0)
    (R.bin sha256 4b7e91d2..., RTN W4 per-out-channel sym + MSE clip,
     A4 per-token asym dynamic, KV16, R4 online Hadamard on down_proj)
  - tap = base_model.model(input_ids).last_hidden_state
        = modeling_llama_kv.py:1074 `self.norm(hidden_states)` with
          norm.weight == ones (gamma_f folded into lm_head), basis = R1
    -> stored as RAW fp16 bytes, shape (T, 4096), the exact tensor the
       online fused teacher would produce (bit-parity gated separately).
  - NO h @ R1.T anywhere; NO gamma multiply anywhere.

SHARDING (amendment §2): worker k of N handles conv indices with
idx % N == k; each sample processed exactly once; workers write disjoint
part files (~part_gib GiB each) + a JSON index; merge is manifest-only.

MODES:
  --analyze              exact storage analytics from the token cache
                         (no GPU); prints §17 numbers and exits.
  --worker K --n-workers N   generate shard K (one GPU).
  --budget-gib G         hard cap on TOTAL cache size; conv indices are
                         admitted in ascending order by a global
                         deterministic rule until the analytic total
                         hits the cap (hybrid cache+online training).

Artifacts (rank 0 / worker 0 additionally):
  <cache-dir>/native_head.pt   deployed fused lm_head weight (fp16) +
                               provenance + sha256 — the ploss head.
  <cache-dir>/target_manifest.json  full provenance record.
"""
import argparse, hashlib, json, os, sys, time

PROJECT_ROOT = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "EAGLE"))

import numpy as np
import torch

TOK_CACHE = ("/data/thahn1230/datasets/eagle1_official/"
             "official_tok_68k_v1.pt")
KIND = "learned_chat_w4a4kv16"
D = 4096
BYTES_PER_TOK = 2 * D          # fp16


def sha16(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


def load_rows():
    blob = torch.load(TOK_CACHE, weights_only=False)
    return blob["rows"], blob["split"], blob.get("manifest", {})


def admitted_set(rows, budget_gib, split=None):
    """Deterministic global admission until the analytic total exceeds
    the budget. Order: the 200 validation conversations FIRST (the
    trainer's validate() cap — keeps rank-0 validation fully cached),
    then ascending conv index. Returns (set, total_bytes)."""
    cap = budget_gib * 2**30 if budget_gib else float("inf")
    order = []
    if split is not None:
        order += list(range(split, min(split + 200, len(rows))))
    seen = set(order)
    order += [i for i in range(len(rows)) if i not in seen]
    tot, adm = 0, set()
    for i in order:
        b = len(rows[i]["input_ids"]) * BYTES_PER_TOK
        if tot + b > cap:
            break
        tot += b
        adm.add(i)
    return adm, tot


def analyze(rows, split, budget_gib):
    ntok = [len(r["input_ids"]) for r in rows]
    total_tok = sum(ntok)
    raw = total_tok * BYTES_PER_TOK
    adm, adm_bytes = admitted_set(rows, budget_gib, split)
    st = os.statvfs("/data")
    free = st.f_bavail * st.f_frsize
    rep = dict(
        n_convs=len(rows), n_train=split, n_test=len(rows) - split,
        total_tokens=total_tok, mean_tokens=round(total_tok / len(rows), 1),
        dtype="fp16", bytes_per_value=2, hidden_dim=D,
        cached_hidden_values=total_tok * D,
        raw_gib_full=round(raw / 2**30, 2),
        budget_gib=budget_gib,
        admitted_convs=len(adm),
        admitted_gib=round(adm_bytes / 2**30, 2),
        admitted_token_frac=round(
            sum(ntok[i] for i in adm) / total_tok, 4) if budget_gib else 1.0,
        expected_shard_count_4gib=int(np.ceil(
            (adm_bytes if budget_gib else raw) / (4 * 2**30))),
        extra_labels="none (labels = shift of the same array; "
                     "token labels live in the token cache; teacher "
                     "logits derived from the frozen head at train time)",
        free_disk_gib_data=round(free / 2**30, 2),
        frac_of_free=round((adm_bytes if budget_gib else raw) / free, 3),
    )
    print(json.dumps(rep, indent=1))
    return rep


def build_target(dev):
    from eagle_spinquant import experiment, study
    cfg = experiment.load_config(None)
    paths = experiment.resolve_paths(cfg)
    model, stash, _ = study.build_study_target(
        paths["target_path"], paths["draft_path"], cfg["model"]["target"],
        "full", KIND, "w4a4", 0, device=dev, rotations_root=None)
    bm = model.base_model
    # contract assertions: gamma folded away, R1 present in stash
    nw = bm.model.norm.weight.detach().float()
    assert torch.allclose(nw, torch.ones_like(nw)), \
        "norm.weight != ones — gamma not folded; wrong build"
    return model, bm, stash, paths


def head_weight(bm):
    lm = bm.lm_head
    for attr in ("module", "linear"):
        if hasattr(lm, attr):
            lm = getattr(lm, attr)
    return lm.weight.detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir",
                    default="/data/thahn1230/strict_rt_cache")
    ap.add_argument("--worker", type=int, default=-1)
    ap.add_argument("--n-workers", type=int, default=8)
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--budget-gib", type=float, default=0)
    ap.add_argument("--part-gib", type=float, default=4.0)
    args = ap.parse_args()

    rows, split, tok_manifest = load_rows()
    if args.analyze:
        analyze(rows, split, args.budget_gib)
        return 0

    assert 0 <= args.worker < args.n_workers
    dev = "cuda:0"
    torch.set_grad_enabled(False)
    adm, _ = admitted_set(rows, args.budget_gib, split)
    my = [i for i in range(len(rows))
          if i % args.n_workers == args.worker and i in adm]
    os.makedirs(args.cache_dir, exist_ok=True)

    model, bm, stash, paths = build_target(dev)

    if args.worker == 0:
        W = head_weight(bm).half().cpu()
        # algebraic contract check: fused head == orig · diag(gamma) · R1
        Wref = (stash["lm_head_weight"].double()
                * stash["gamma_f"].double().unsqueeze(0)) \
            @ stash["R1"].double().cpu()
        rel = float((W.double() - Wref).norm() / Wref.norm())
        assert rel < 5e-3, f"fused-head algebra mismatch rel={rel}"
        r_bin = os.path.join(PROJECT_ROOT, "outputs", "rotations",
                             KIND, "R.bin")
        torch.save(dict(
            weight=W, sha16=sha16(W.numpy().tobytes()),
            algebra_rel_err=rel,
            note="deployed fused lm_head = W_lm·diag(gamma_f)·R1 "
                 "(fp16); ploss head for strict SEAGLE-RT"),
            os.path.join(args.cache_dir, "native_head.pt"))
        json.dump(dict(
            target_path=paths["target_path"],
            draft_path=paths["draft_path"],
            rotation="full", rotation_type=KIND, quant="w4a4",
            r_bin_sha256=hashlib.sha256(
                open(r_bin, "rb").read()).hexdigest(),
            quantizer=("W4 per-out-channel symmetric RTN + MSE clip "
                       "(lm_head/embed excluded); A4 per-token asym "
                       "dynamic (lm_head input 16-bit); KV16; R1+R2 "
                       "from R.bin; R4 online Hadamard on down_proj"),
            tap=("third_party/EAGLE/eagle/model/modeling_llama_kv.py"
                 ":1074 self.norm(hidden_states) -> last_hidden_state; "
                 "norm.weight==ones; basis=R1-rotated, gamma-excluded"),
            dtype="fp16", hidden_dim=D,
            tok_cache=TOK_CACHE,
            tok_manifest_keys={k: tok_manifest.get(k) for k in
                               ("source_sha256", "n_usable", "n_skipped")
                               if isinstance(tok_manifest, dict)},
            n_workers=args.n_workers,
            budget_gib=args.budget_gib,
            head_sha16=sha16(W.numpy().tobytes()),
            created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
            open(os.path.join(args.cache_dir, "target_manifest.json"),
                 "w"), indent=1)

    part_bytes_cap = int(args.part_gib * 2**30)
    part_id, off, fh = 0, 0, None
    index = {}
    t0, ntok_done = time.time(), 0

    def open_part():
        nonlocal fh, off
        p = os.path.join(args.cache_dir,
                         f"h_w{args.worker:02d}_p{part_id:04d}.bin")
        fh = open(p, "wb")
        off = 0
        return p

    pname = open_part()
    for k, i in enumerate(my):
        ids = rows[i]["input_ids"].long()[None].to(dev)
        h = bm.model(input_ids=ids).last_hidden_state[0].half()
        b = h.cpu().numpy().tobytes()
        if off + len(b) > part_bytes_cap and off > 0:
            fh.close()
            part_id += 1
            pname = open_part()
        index[i] = dict(part=os.path.basename(pname), off=off,
                        ntok=h.shape[0], sha16=sha16(b))
        fh.write(b)
        off += len(b)
        ntok_done += h.shape[0]
        if k % 500 == 0:
            el = time.time() - t0
            print(f"[cache w{args.worker}] {k}/{len(my)} convs "
                  f"{ntok_done/1e6:.1f}M tok {el/60:.1f}min "
                  f"{ntok_done/max(el,1e-9)/1e3:.1f}k tok/s", flush=True)
    fh.close()
    json.dump(dict(worker=args.worker, n_convs=len(my),
                   n_tokens=ntok_done,
                   wall_sec=round(time.time() - t0, 1),
                   index=index),
              open(os.path.join(args.cache_dir,
                                f"index_w{args.worker:02d}.json"), "w"))
    print(f"[cache w{args.worker}] DONE {len(my)} convs "
          f"{ntok_done} tok {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

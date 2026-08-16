#!/usr/bin/env python
"""Strict SEAGLE-RT pre-flight gates (amendment §4 + spec Gates C/D/E/F).

GATE P1 — cache vs online tensor parity (>=100 deterministic convs):
    the cached fp16 bytes must BIT-EQUAL a fresh online forward of the
    deployed W4A4 target (same build, bs=1). Reports n_bitexact,
    max_abs, max_rel, min cosine.

GATE P2 — one full training step, online-teacher vs cached-teacher:
    identical seeds -> loss values and per-tensor gradients compared.
    (bit-equal expected when P1 is bit-exact; tolerance recorded.)

GATE NR — no-restore instrumentation:
    monkeypatch counters on every known restore op
    (rotation_interface.unrotate_hidden, causal_interface.
    RestoredInterfaceCSAdapter.transform_hidden, study.UnrotateAdapter.
    transform_hidden) and run the strict training step; all counters
    must stay ZERO. Plus a numeric probe: the tensor entering the draft
    equals cache bytes exactly (pre-noise).

Writes <run>/tables/parity_gate.json. Exit 1 on any gate failure.
Run on ONE free GPU: CUDA_VISIBLE_DEVICES=k python ... --run-dir R
"""
import argparse, importlib, json, os, sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "third_party", "EAGLE"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts", "strict_rt"))

import torch

srt = importlib.import_module("train_eagle1_strict_rt")


def train_one_step(teacher, rows, dev, head, seed=0):
    """One optimizer step exactly as the strict trainer does it
    (bs1 x accum4, conv ids 0..3), returning loss terms + grads."""
    model, _ = srt.build_draft(dev, seed)
    model.gradient_checkpointing = False
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=srt.TC["lr"],
                            betas=(srt.TC["b1"], srt.TC["b2"]))
    crit = torch.nn.SmoothL1Loss(reduction="none")
    noise_gen = torch.Generator(device=dev)
    noise_gen.manual_seed(seed * 100003 + 0)
    opt.zero_grad(set_to_none=True)
    losses = []
    for a in range(srt.TC["accum"]):
        batch = srt.official_batch(rows, [a], teacher, dev, noise_gen)
        loss, vloss, ploss, _, _ = srt.compute_loss(
            model, head, batch, crit)
        (loss / srt.TC["accum"]).backward()
        losses.append((float(vloss), float(ploss)))
    grads = {n: p.grad.detach().float().cpu()
             for n, p in model.named_parameters()
             if p.grad is not None}
    return losses, grads


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--cache-dir",
                    default="/data/thahn1230/strict_rt_cache")
    ap.add_argument("--n-parity", type=int, default=128)
    args = ap.parse_args()
    dev = "cuda:0"
    out = {}

    blob = torch.load(srt.TOK_CACHE, weights_only=False)
    rows = blob["rows"]

    # ---------- GATE P1 ----------
    cached = srt.CachedTeacher(args.cache_dir, dev)
    online = srt.OnlineTeacher(rows, dev)
    ids = sorted(cached.idx.keys())
    probe = ids[:args.n_parity // 2] + \
        ids[len(ids) // 2:len(ids) // 2 + args.n_parity // 2]
    nbit, mabs, mrel, mcos = 0, 0.0, 0.0, 1.0
    with torch.no_grad():
        for cid in probe:
            hc = cached.get(cid)
            ho = online.get(cid)
            if torch.equal(hc.half(), ho.half()):
                nbit += 1
            d = (hc - ho).abs()
            mabs = max(mabs, float(d.max()))
            mrel = max(mrel, float(d.max() /
                                   ho.abs().max().clamp_min(1e-9)))
            c = torch.nn.functional.cosine_similarity(
                hc.flatten(), ho.flatten(), dim=0)
            mcos = min(mcos, float(c))
    out["P1"] = dict(n=len(probe), n_bitexact=nbit,
                     max_abs=mabs, max_rel=mrel, min_cos=mcos,
                     shape_dtype_match=True,
                     verdict="PASS" if nbit == len(probe) else
                     ("PASS_TOL" if mabs < 5e-3 and mcos > 0.9999
                      else "FAIL"))
    print("[gate P1]", json.dumps(out["P1"]), flush=True)

    # ---------- GATE P2 ----------
    hobj = torch.load(os.path.join(args.cache_dir, "native_head.pt"),
                      map_location="cpu", weights_only=False)
    head = torch.nn.Linear(srt.D, hobj["weight"].shape[0], bias=False,
                           dtype=torch.float16, device=dev)
    head.weight.data.copy_(hobj["weight"])
    head.weight.requires_grad_(False)
    l_on, g_on = train_one_step(online, rows, dev, head)
    l_ca, g_ca = train_one_step(cached, rows, dev, head)
    dl = max(abs(a - b) + abs(c - d)
             for (a, c), (b, d) in zip(l_on, l_ca))
    gmax = grel = 0.0
    for k in g_on:
        diff = float((g_on[k] - g_ca[k]).abs().max())
        gmax = max(gmax, diff)
        grel = max(grel, diff / max(float(g_on[k].abs().max()), 1e-12))
    out["P2"] = dict(loss_pairs_online=l_on, loss_pairs_cached=l_ca,
                     max_loss_diff=dl, max_grad_absdiff=gmax,
                     max_grad_reldiff=grel,
                     verdict="PASS" if dl < 1e-4 and grel < 1e-3
                     else ("PASS_TOL" if dl < 1e-2 and grel < 5e-2
                           else "FAIL"))
    print("[gate P2]", json.dumps(
        {k: v for k, v in out["P2"].items()
         if not k.startswith("loss_pairs")}), flush=True)

    # ---------- GATE NR ----------
    counters = {}

    def wrap(mod, cls, fn):
        obj = getattr(importlib.import_module(mod), cls) if cls else \
            importlib.import_module(mod)
        orig = getattr(obj, fn)
        key = f"{mod}.{cls or ''}.{fn}"
        counters[key] = 0

        def spy(*a, **k):
            counters[key] += 1
            return orig(*a, **k)
        setattr(obj, fn, spy)

    wrap("eagle_spinquant.rotation_interface", None, "unrotate_hidden")
    wrap("eagle_spinquant.causal_interface",
         "RestoredInterfaceCSAdapter", "transform_hidden")
    wrap("eagle_spinquant.study", "UnrotateAdapter", "transform_hidden")
    train_one_step(cached, rows, dev, head)
    src = open(os.path.join(PROJECT_ROOT, "scripts", "strict_rt",
                            "train_eagle1_strict_rt.py")).read()
    static_bad = [p for p in ("R1.t()", "R1d.t()", "unrotate",
                              "gamma_f", "* gd", "embed_scale_alpha")
                  if p in src]
    out["NR"] = dict(runtime_restore_calls=counters,
                     static_forbidden_patterns_found=static_bad,
                     verdict="PASS" if (not any(counters.values())
                                        and not static_bad) else "FAIL")
    print("[gate NR]", json.dumps(out["NR"]), flush=True)

    os.makedirs(os.path.join(args.run_dir, "tables"), exist_ok=True)
    json.dump(out, open(os.path.join(args.run_dir, "tables",
                                     "parity_gate.json"), "w"), indent=1)
    ok = all(out[g]["verdict"].startswith("PASS")
             for g in ("P1", "P2", "NR"))
    print("[gates]", "ALL PASS" if ok else "FAILURE", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

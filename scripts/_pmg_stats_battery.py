#!/usr/bin/env python
"""PMG statistics battery: paired bootstrap for every pre-registered
comparison (spec §16) on every dataset, then Holm adjustment.

Comparison families (per dataset):
  A-grid: T16D16 vs each of the other 8 cells; draft effect within each
          target row (D16:D8, D8:D4); target effect within each draft
          column (T16:T8, T8:T4 at matched draft).
  B-methods (per target): B1:QFgen, B1:B3, B3:B5, QFgen:QFep3g,
          QFep3g:QFep3p, B5:B7, QFep3p:QFrd (B6:B8), bestPTQ:bestQAT,
          B7:XFER (transfer, T16/T8 only).
Tags resolved from median_seeds.json (QAT) with rdv2 override for T16.
"""
import json, os, subprocess, sys

rd = sys.argv[1]
ms = json.load(open(f"{rd}/tables/median_seeds.json"))
TN = {"fp16": "T16", "w8a8": "T8", "int4": "T4"}


def qf(meth, tgt):
    key = f"{'rdv2' if (meth, tgt) == ('rd', 'fp16') else meth}_{TN[tgt]}"
    s = ms[key]["seed"]
    m = "rdv2" if key.startswith("rdv2") else meth
    return f"QF_{m}_{TN[tgt]}_s{s}"


def run(ds, pairs):
    cmd = ["python", "scripts/bootstrap_eagle_tau.py", "--run-dir", rd,
           "--dataset", ds]
    for name, a, b in pairs:
        cmd += ["--pair", f"{name}={a}:{b}"]
    subprocess.run(cmd, check=True)


for ds in ("mtbench", "gsm8k", "sharegpt", "humaneval"):
    pairs = []
    # A-grid
    cells = {f"{t}{d}": f"A_{t}{d}@{tgt}"
             for t, tgt in (("T16", "fp16"), ("T8", "w8a8"),
                            ("T4", "int4")) for d in ("D16", "D8", "D4")}
    for c, spec in cells.items():
        if c != "T16D16":
            pairs.append((f"grid_T16D16_vs_{c}",
                          cells["T16D16"], spec))
    for t in ("T16", "T8", "T4"):
        pairs.append((f"draft_{t}_D16vD8", cells[f"{t}D16"],
                      cells[f"{t}D8"]))
        pairs.append((f"draft_{t}_D8vD4", cells[f"{t}D8"],
                      cells[f"{t}D4"]))
    for d in ("D16", "D8", "D4"):
        pairs.append((f"target_{d}_T16vT8", cells[f"T16{d}"],
                      cells[f"T8{d}"]))
        pairs.append((f"target_{d}_T8vT4", cells[f"T8{d}"],
                      cells[f"T4{d}"]))
    # B-methods
    for tgt in ("fp16", "w8a8", "int4"):
        T = TN[tgt]
        B = lambda n: f"{n}_{T}@{tgt}"
        Q = lambda m: f"{qf(m, tgt)}@{tgt}"
        pairs += [
            (f"{T}_naive_ptq_v_gen_qat", B("B1"), Q("gen")),
            (f"{T}_naive_v_ep3g_ptq", B("B1"), B("B3")),
            (f"{T}_ep3g_v_ep3p_ptq", B("B3"), B("B5")),
            (f"{T}_gen_v_ep3g_qat", Q("gen"), Q("ep3g")),
            (f"{T}_ep3g_v_ep3p_qat", Q("ep3g"), Q("ep3p")),
            (f"{T}_ep3p_ptq_v_rd_ptq", B("B5"), B("B7")),
            (f"{T}_ep3p_qat_v_rd_qat", Q("ep3p"), Q("rd")),
            (f"{T}_bestptq_v_bestqat", B("B7"), Q("rd")),
        ]
        if tgt != "int4":
            pairs.append((f"{T}_rdxfer_v_rdmatched",
                          f"XFER_RD4_{T}@{tgt}", B("B7")))
    # drop pairs with missing shards (bootstrap reports MISSING anyway,
    # but keep the json clean)
    ok = []
    for name, a, b in pairs:
        fa = a.split("@")
        fb = b.split("@")
        pa = f"{rd}/shards/al__{fa[0]}__{fa[1]}__{ds}.csv"
        pb = f"{rd}/shards/al__{fb[0]}__{fb[1]}__{ds}.csv"
        if os.path.exists(pa) and os.path.exists(pb):
            ok.append((name, a, b))
        else:
            print(f"[stats] SKIP {ds}:{name} (missing shard)")
    run(ds, ok)
subprocess.run(["python", "scripts/holm_adjust_bootstrap.py",
                "--run-dir", rd], check=True)
subprocess.run(["python", "scripts/compute_macro_al.py",
                "--run-dir", rd], check=True)
print("[stats] battery complete")

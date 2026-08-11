"""FIDI §29 aggregate tables from capture__*.json shards.

Flattens every (config, dataset, tensor) stats/qparams entry into the master
four_stage_activation_stats.csv, then writes filtered views:
target_hidden_stats.csv (A_src*/B_concat), ht_stats.csv (C_*/S3_*),
ht_qparam_stats.csv, kv_projection_stats.csv (S4_*_lin*),
kv_cache_stats.csv (S4_*_stored*), hypothetical_kv_quant.csv (KV4_/KV8_),
cross_source_imbalance.csv (§5). Idempotent — regenerates from whatever
shards exist; run again as more captures finish.
"""
import argparse
import csv
import glob
import json
import os


def flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flat(v, key + "."))
        elif isinstance(v, (int, float, bool)):
            out[key] = v
        elif isinstance(v, list) and len(v) <= 16 and all(
                isinstance(x, (int, float)) for x in v):
            out[key] = ";".join(str(round(float(x), 6)) for x in v)
    return out


def write(path, rows):
    if not rows:
        return 0
    keys = ["config", "dataset", "tensor"]
    rest = sorted({k for r in rows for k in r} - set(keys))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys + rest, restval="")
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    td = os.path.join(args.run_dir, "tables")
    srows, qrows = [], []
    shards = sorted(glob.glob(os.path.join(td, "capture__*.json")))
    for p in shards:
        d = json.load(open(p))
        base = {"config": d["config"], "dataset": d["dataset"]}
        for name, s in d.get("stats", {}).items():
            srows.append({**base, "tensor": name, **flat(s)})
        for name, q in d.get("qparams", {}).items():
            qrows.append({**base, "tensor": name, **flat(q)})

    n = write(os.path.join(td, "four_stage_activation_stats.csv"), srows)
    write(os.path.join(td, "target_hidden_stats.csv"),
          [r for r in srows if r["tensor"].startswith(("A_src", "B_concat"))])
    write(os.path.join(td, "ht_stats.csv"),
          [r for r in srows if r["tensor"].startswith(("C_Zt", "C_Ht",
                                                       "S3_"))])
    write(os.path.join(td, "ht_qparam_stats.csv"),
          [r for r in qrows if r["tensor"].startswith(("C_Ht", "S3_",
                                                       "B_concat"))])
    write(os.path.join(td, "kv_projection_stats.csv"),
          [r for r in srows if "_lin" in r["tensor"]])
    write(os.path.join(td, "kv_cache_stats.csv"),
          [r for r in srows if "_stored" in r["tensor"]])
    write(os.path.join(td, "hypothetical_kv_quant.csv"),
          [r for r in qrows if r["tensor"].startswith(("KV4_", "KV8_"))])

    # §5 cross-source imbalance per (config, dataset)
    imb = []
    bykey = {}
    for r in srows:
        if r["tensor"].startswith("A_src") and "origbasis" not in r["tensor"]:
            bykey.setdefault((r["config"], r["dataset"]), []).append(r)
    for (cfg, ds), rs in sorted(bykey.items()):
        rms = [r.get("rms") for r in rs if r.get("rms")]
        am = [r.get("absmax") for r in rs if r.get("absmax")]
        ku = [r.get("kurtosis") for r in rs if r.get("kurtosis")]
        if len(rms) == 5:
            imb.append({"config": cfg, "dataset": ds,
                        "tensor": "A_src_1..29",
                        "rms_max_over_min": max(rms) / min(rms),
                        "absmax_max_over_min": max(am) / min(am),
                        "kurtosis_min": min(ku), "kurtosis_max": max(ku)})
    write(os.path.join(td, "cross_source_imbalance.csv"), imb)
    print(f"[fidi_tables] {len(shards)} shards -> {n} stat rows, "
          f"{len(qrows)} qparam rows, {len(imb)} imbalance rows")


if __name__ == "__main__":
    main()

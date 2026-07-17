#!/usr/bin/env python
"""S27 mechanism instrumentation from raw acceptance records (CPU-only).

From every kv4mat / panel / drot shard:
  - tau histogram (tau = accepted+1 per verification cycle, 1..7)
  - micro-AL by cycle-index bucket (early 0-4, mid 5-14, late 15+):
    does KV4 error accumulation degrade acceptance as generation
    proceeds?
Writes tables/tau_histograms.csv and tables/tau_by_cycle_bucket.csv.
"""
import csv, glob, json, os, sys

RD = sys.argv[1]
BUCKETS = [(0, 5, "cyc00_04"), (5, 15, "cyc05_14"), (15, 10 ** 9,
                                                     "cyc15plus")]
hist_rows, bucket_rows = [], []
for p in sorted(glob.glob(os.path.join(RD, "shards", "*.csv"))):
    base = os.path.basename(p)[:-4]
    if base.startswith("ctx__"):
        continue
    hist = {}
    buckets = {name: [] for _, _, name in BUCKETS}
    with open(p) as f:
        for r in csv.DictReader(f):
            taus = json.loads(r["acceptance_list"])
            for ci, t in enumerate(taus):
                hist[t] = hist.get(t, 0) + 1
                for lo, hi, name in BUCKETS:
                    if lo <= ci < hi:
                        buckets[name].append(t)
    if not hist:
        continue
    n = sum(hist.values())
    row = dict(shard=base, n_cycles=n)
    for t in range(1, 8):
        row[f"tau{t}"] = round(hist.get(t, 0) / n, 4)
    hist_rows.append(row)
    brow = dict(shard=base)
    for _, _, name in BUCKETS:
        b = buckets[name]
        brow[name] = round(sum(b) / len(b), 4) if b else ""
        brow[name + "_n"] = len(b)
    bucket_rows.append(brow)
os.makedirs(os.path.join(RD, "tables"), exist_ok=True)
for fname, rows in [("tau_histograms.csv", hist_rows),
                    ("tau_by_cycle_bucket.csv", bucket_rows)]:
    with open(os.path.join(RD, "tables", fname), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
print(f"[mech] {len(hist_rows)} shards -> tau_histograms.csv, "
      "tau_by_cycle_bucket.csv")

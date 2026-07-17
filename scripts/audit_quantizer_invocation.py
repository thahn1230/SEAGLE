#!/usr/bin/env python
"""B3 audit: prove KV4 fake-quant was actually invoked wherever claimed.

Reads every kv4mat__*.json shard and checks, per cell:
  - draft_kv_bits==4  => draft_kv_tokens>0 and 1e-4 < draft NMSE < 0.2
  - target_kv_bits==4 => target_kv_tokens>0 and 1e-4 < target NMSE < 0.2
  - *_kv_bits==16     => the corresponding counters are absent/zero
NMSE bounds: >1e-4 rules out a silent no-op (identity would give ~0);
<0.2 rules out a broken quantizer. Also verifies no fp16-fallback counter
fired. Writes tables/kv4_invocation_audit.csv; exits 1 on any FAIL.
"""
import csv, glob, json, os, sys

RD = sys.argv[1] if len(sys.argv) > 1 else None
assert RD, "usage: audit_quantizer_invocation.py <run_dir>"
rows, bad = [], 0
for p in sorted(glob.glob(os.path.join(RD, "shards", "kv4mat__*.json"))):
    d = json.load(open(p))
    cell = d["cell"]
    checks = []
    if d.get("draft_kv_bits", 16) == 4:
        ok = (d.get("draft_kv_tokens", 0) > 0
              and 1e-4 < d.get("draft_k_nmse", 0) < 0.2
              and 1e-4 < d.get("draft_v_nmse", 0) < 0.2)
        checks.append(("draft_kv4_invoked", ok))
    else:
        checks.append(("draft_kv16_clean", "draft_k_nmse" not in d))
    if d.get("target_kv_bits", 16) == 4:
        ok = (d.get("target_kv_tokens", 0) > 0
              and 1e-4 < d.get("target_k_nmse", 0) < 0.2
              and 1e-4 < d.get("target_v_nmse", 0) < 0.2)
        checks.append(("target_kv4_invoked", ok))
    else:
        checks.append(("target_kv16_clean", "target_k_nmse" not in d))
    checks.append(("no_fp16_fallback", not d.get("fp16_fallback_count")))
    checks.append(("fake_quant_mode", d.get("quant_mode") == "fake"))
    for name, ok in checks:
        rows.append(dict(cell=cell, check=name,
                         result="PASS" if ok else "FAIL"))
        bad += 0 if ok else 1
out = os.path.join(RD, "tables", "kv4_invocation_audit.csv")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["cell", "check", "result"])
    w.writeheader(); w.writerows(rows)
n_pass = sum(r["result"] == "PASS" for r in rows)
print(f"[audit] {n_pass}/{len(rows)} checks PASS -> {out}")
sys.exit(1 if bad else 0)

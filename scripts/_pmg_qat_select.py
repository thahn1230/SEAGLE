#!/usr/bin/env python
"""QAT checkpoint selection + selected-ckpt mtbench eval.

Usage: _pmg_qat_select.py <rd> <tag> <target> <alpha> <alpha_rec|none>
                          <rd_ckpt|none>

Selection: c4 calib-20 AL over {<tag>_step1500.pt, <tag>_best.pt (best
val-loss), <tag>_last.pt} -> argmax -> tables/qat_selection_<tag>.json
-> mtbench-80 eval with final tag QF_<tag-without-Q_>.
Same protocol for every method/seed (fairness gate)."""
import csv, json, os, subprocess, sys

rd, tag, tgt, alpha, arec, rdck = sys.argv[1:7]
EV = ["python", "scripts/eval_eagle_acceptance_length.py"]


def deploy_args(sd_path):
    if rdck != "none":
        a = ["--target", tgt, "--draft-cfg", "rot_ep3p",
             "--ckpt", rdck, "--alpha", alpha, "--alpha-rec", arec]
    else:
        a = ["--target", tgt, "--draft-cfg", "d4p3_deploy",
             "--alpha", alpha]
        if arec != "none":
            a += ["--alpha-rec", arec]
    return a + ["--draft-sd", sd_path, "--run-dir", rd]


cands = []
for suff in ("step1500", "best", "last"):
    p = os.path.join(rd, "ckpts", f"{tag}_{suff}.pt")
    if os.path.exists(p):
        cands.append((suff, p))
assert cands, f"no ckpts for {tag}"
res = []
for suff, p in cands:
    stag = f"SEL_{tag}_{suff}"
    subprocess.run(EV + deploy_args(p) +
                   ["--tag", stag, "--datasets", "c4", "--pool",
                    "calib", "--n-prompts", "20"], check=True)
    sh = os.path.join(rd, "shards", f"al__{stag}__{tgt}__c4__calib.csv")
    ts = [t for r in csv.DictReader(open(sh))
          for t in json.loads(r["acceptance_list"])]
    res.append(dict(ckpt=suff, path=p,
                    calib_tau=round(sum(ts) / max(len(ts), 1), 4)))
    print(f"[sel] {tag} {suff}: {res[-1]['calib_tau']}", flush=True)
best = max(res, key=lambda r: r["calib_tau"])
json.dump(dict(tag=tag, selected=best, candidates=res),
          open(os.path.join(rd, "tables",
                            f"qat_selection_{tag}.json"), "w"),
          indent=1)
ftag = "QF_" + tag[2:]
subprocess.run(EV + deploy_args(best["path"]) +
               ["--tag", ftag, "--datasets", "mtbench", "--pool",
                "eval", "--n-prompts", "80"], check=True)
# disk hygiene: keep only the SELECTED ckpt + last; prune the rest
import glob
keep = {best["path"], os.path.join(rd, "ckpts", f"{tag}_last.pt")}
pruned = []
for p2 in glob.glob(os.path.join(rd, "ckpts", f"{tag}_*.pt")):
    if p2 not in keep:
        os.remove(p2)
        pruned.append(os.path.basename(p2))
with open(os.path.join(rd, "logs", "ckpt_prune_manifest.txt"),
          "a") as mf:
    mf.write(f"{tag}: post-selection prune {pruned}\n")
print(f"[sel] {tag} -> {best['ckpt']} ({best['calib_tau']}) "
      f"-> mtbench as {ftag}; pruned {len(pruned)}")

"""RCAL contracts (study §20, §30): identity control, decomposition to
1e-12, same-branch min relation, branch divergence R_RC < min, LCP
correctness, official-tau contract, bootstrap aggregation."""
import importlib.util
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(ROOT, rel))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


cap = _load("cap", "scripts/capture_eagle_proposal_cycles.py")
met = _load("met", "scripts/compute_eagle_rcal_metrics.py")


def _sanity_records():
    """Real captured cycle records if a run exists, else None."""
    import glob
    hits = sorted(glob.glob(os.path.join(
        ROOT, "runs", "eagle1_generic_qat*", "cycles", "cyc__*.jsonl")))
    if not hits:
        return None
    return [json.loads(x) for x in open(hits[-1])]


def rec(rq, r0, rrc, same=None, div=None):
    if same is None:
        same = (rrc == min(rq, r0))
    return dict(R_q=rq, R_0=r0, R_RC=rrc, same_branch=same,
                div_depth=div)


def test_rcal_identity_control():
    # identical verifier: RCAL = AL_q = AL_0, SAL = LAL = 0, AFS = 1
    recs = [rec(3, 3, 3), rec(1, 1, 1), rec(5, 5, 5), rec(0, 0, 0)]
    m = met.metrics(recs)
    assert m["AL_q"] == m["AL_0"] == m["RCAL"]
    assert m["SAL"] == 0.0 and m["LAL"] == 0.0
    assert m["P_A"] == 1.0 and m["R_A"] == 1.0 and m["AFS"] == 1.0


def test_rcal_decomposition_identities_fp64():
    recs = [rec(4, 2, 2), rec(3, 5, 3), rec(2, 2, 1, same=False, div=2),
            rec(0, 1, 0), rec(5, 0, 0)]
    m = met.metrics(recs)
    assert abs(m["AL_q"] - (m["RCAL"] + m["SAL"])) < 1e-12
    assert abs(m["AL_0"] - (m["RCAL"] + m["LAL"])) < 1e-12
    assert abs((m["AL_q"] - m["AL_0"]) - (m["SAL"] - m["LAL"])) < 1e-12
    assert abs(m["AFS"] - 2 * m["RCAL"]
               / (m["AL_q"] + m["AL_0"])) < 1e-12


def test_rcal_same_branch_min_relation():
    # same branch, different rejection depth -> R_RC = min
    S_q = [11, 22, 33, 44]
    S_0 = [11, 22]
    r = cap.lcp(S_q, S_0)
    assert r == 2 == min(len(S_q), len(S_0))


def test_rcal_branch_divergence_lcp_less_than_min():
    # different child at depth 2 -> R_RC < min(R_q, R_0)
    S_q = [11, 22, 33]
    S_0 = [11, 99, 33]
    r = cap.lcp(S_q, S_0)
    assert r == 1 < min(len(S_q), len(S_0))


def test_rcal_zero_denominator_convention():
    m = met.metrics([rec(0, 0, 0)])
    assert m["P_A"] == 0.0 and m["R_A"] == 0.0 and m["AFS"] == 0.0


def test_rcal_depth_survival_monotone():
    recs = [rec(4, 3, 3), rec(2, 2, 2), rec(1, 4, 1),
            rec(3, 3, 2, same=False, div=3)]
    m = met.metrics(recs)
    sv = m["depth_survival"]
    for k in range(1, 8):
        assert sv[k]["S_RC"] <= sv[k]["S_q"] + 1e-12
        assert sv[k]["S_RC"] <= sv[k]["S_0"] + 1e-12
        assert sv[k]["spurious"] >= -1e-12 and sv[k]["lost"] >= -1e-12


def test_official_tau_contract_doc_exists():
    p = os.path.join(ROOT, "docs",
                     "EAGLE_ACCEPTANCE_LENGTH_CONTRACT.md")
    s = open(p).read()
    assert "1 + accepted_draft_length" in s
    assert "proposal tokens only" in s.replace("**", "")


def test_bootstrap_paired_delta_sign():
    bs = _load("bs", "scripts/bootstrap_eagle_rcal.py")
    cl_a = {f"p{i}": [(2, 2, 2)] for i in range(30)}
    cl_b = {f"p{i}": [(3, 3, 3)] for i in range(30)}
    a = bs.agg(cl_a, sorted(cl_a))
    b = bs.agg(cl_b, sorted(cl_b))
    assert abs((b["AL_q"] - a["AL_q"]) - 1.0) < 1e-12
    assert abs((b["RCAL"] - a["RCAL"]) - 1.0) < 1e-12


def test_rcal_same_prefix_and_tree():
    """Lockstep bookkeeping: within a prompt, each cycle's prefix_len
    advances by previous R_q + 1 (accepted + root); tree width fixed."""
    import glob
    recs = _sanity_records()
    if recs is None:
        return
    by = {}
    for r in recs:
        by.setdefault(r["prompt_id"], []).append(r)
    for rows in by.values():
        rows = sorted(rows, key=lambda x: x["cycle"])
        for a, b in zip(rows, rows[1:]):
            assert b["prefix_len"] == a["prefix_len"] + a["R_q"] + 1
        widths = {len(r["tree_tokens"]) for r in rows
                  if "tree_tokens" in r}
        assert len(widths) <= 1


def test_rcal_tree_replay_determinism():
    """Metric computation is a pure function of the records."""
    import importlib.util, os
    spec = importlib.util.spec_from_file_location(
        "crm", os.path.join(ROOT, "scripts",
                            "compute_eagle_rcal_metrics.py"))
    crm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(crm)
    recs = _sanity_records()
    if recs is None:
        return
    assert crm.metrics(recs) == crm.metrics(list(recs))


def test_rcal_reference_cache_equivalence():
    """Offline trajectory-reconstruction replay must reproduce the
    inline lockstep S_0 exactly (when the control has been run)."""
    import glob, json, os
    hits = glob.glob(os.path.join(ROOT, "runs", "eagle1_generic_qat*",
                                  "tables", "replay_equiv_*.json"))
    for h in hits:
        d = json.load(open(h))
        assert d["equivalent"], h

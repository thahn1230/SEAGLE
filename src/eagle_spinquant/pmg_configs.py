"""Precision × Method Grid (PMG) arm registry — single source of truth.

Experiment A: 9 precision cells (naive drafts, precision effect only).
Experiment B: 8 W4A4-draft methods × 3 targets = 24 primary arms.
Every arm resolves to evaluator CLI fragments; uniqueness is gated by
tests/test_precision_method_grid_gates.py::test_precision_grid_config_unique.
"""
D = 4096
EP3P = {                     # (m_f, m_r) per target — calib-AL validated
    "fp16": (D ** 0.46, D ** 0.46),
    "w8a8": None,            # filled by T8 calibration (this study)
    "int4": (D ** 0.40, D ** 0.45),
}
EP3G = {                     # single global m per target
    "fp16": D ** 0.46,
    "w8a8": None,            # filled by T8 calibration
    "int4": D ** 0.42,
}
TARGETS = ("fp16", "w8a8", "int4")
_DQ = {"D16": None, "D8": "fake_w8a8", "D4": "fake_w4a4"}


def grid_a_arms():
    """9 precision cells. Naive quantized drafts (no P3/R_D/QAT)."""
    arms = {}
    for tname, tgt in (("T16", "fp16"), ("T8", "w8a8"), ("T4", "int4")):
        for dname, dq in _DQ.items():
            aid = f"A_{tname}{dname}"
            if dq is None:
                cfg = dict(target=tgt, draft_cfg="stock")
            else:
                # naive: alpha=1.0 (bitwise-identity fold), quant overrides
                cfg = dict(target=tgt, draft_cfg="d4p3_deploy", alpha=1.0,
                           quant_first=dq, quant_recurrent=dq,
                           quant_ar=dq)
            arms[aid] = cfg
    return arms


def grid_b_arms():
    """24 primary method arms (draft W4A4)."""
    arms = {}
    for tname, tgt in (("T16", "fp16"), ("T8", "w8a8"), ("T4", "int4")):
        ep3p = EP3P[tgt]
        ep3g = EP3G[tgt]
        base = dict(target=tgt)
        arms[f"B1_{tname}"] = dict(base, draft_cfg="d4p3_deploy",
                                   alpha=1.0)                    # naive PTQ
        arms[f"B2_{tname}"] = dict(base, draft_cfg="naive_w4a4_qat",
                                   alpha=1.0, qat=True)
        arms[f"B3_{tname}"] = dict(base, draft_cfg="d4p3_deploy",
                                   alpha=ep3g)                   # EP3-G PTQ
        arms[f"B4_{tname}"] = dict(base, draft_cfg="d4p3_deploy",
                                   alpha=ep3g, qat=True)
        arms[f"B5_{tname}"] = dict(base, draft_cfg="d4p3_deploy",
                                   alpha=(ep3p[0] if ep3p else None),
                                   alpha_rec=(ep3p[1] if ep3p else None))
        arms[f"B6_{tname}"] = dict(arms[f"B5_{tname}"], qat=True)
        arms[f"B7_{tname}"] = dict(base, draft_cfg="rot_ep3p",
                                   alpha=(ep3p[0] if ep3p else None),
                                   alpha_rec=(ep3p[1] if ep3p else None),
                                   rd="target_matched")
        arms[f"B8_{tname}"] = dict(arms[f"B7_{tname}"], qat=True)
    return arms


def all_arm_ids():
    return list(grid_a_arms()) + list(grid_b_arms())

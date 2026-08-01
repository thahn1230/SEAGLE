"""Recurrent path uses m_recurrent = D**0.45, distinct from m_first."""
from _ep3p_viz_common import D, load_manifest, run_dir


def test_ep3p_recurrent_migration_values():
    rd = run_dir()
    if rd is None:
        return
    man = load_manifest(rd)
    assert man["beta_rec"] == 0.45
    m = man["m_rec"]
    assert abs(m - D ** 0.45) < 1e-9
    assert abs(m - 42.2243) < 1e-3
    assert abs(m - man["m_first"]) > 10     # genuinely independent

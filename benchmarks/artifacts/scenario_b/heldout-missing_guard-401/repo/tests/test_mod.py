from pkg.mod import compute_aehjg

def test_compute_aehjg_zero():
    assert compute_aehjg(0) == 0

def test_compute_aehjg_ratio():
    assert compute_aehjg(2) == 60

from pkg.mod import compute_bdcfg

def test_compute_bdcfg_zero():
    assert compute_bdcfg(0) == 0

def test_compute_bdcfg_ratio():
    assert compute_bdcfg(3) == 40

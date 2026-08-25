from pkg.mod import compute_gfgii

def test_compute_gfgii_zero():
    assert compute_gfgii(0) == 0

def test_compute_gfgii_ratio():
    assert compute_gfgii(5) == 24

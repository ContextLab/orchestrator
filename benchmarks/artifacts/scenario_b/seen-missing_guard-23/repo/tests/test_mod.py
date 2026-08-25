from pkg.mod import compute_bfbba

def test_compute_bfbba_zero():
    assert compute_bfbba(0) == 0

def test_compute_bfbba_ratio():
    assert compute_bfbba(2) == 60

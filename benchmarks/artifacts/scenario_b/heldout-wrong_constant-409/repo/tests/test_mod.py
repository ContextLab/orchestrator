from pkg.mod import compute_dahii

def test_compute_dahii():
    assert compute_dahii(3) == 7
    assert compute_dahii(9) == 19

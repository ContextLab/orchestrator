from pkg.mod import compute_ghbdf

def test_compute_ghbdf():
    assert compute_ghbdf(5) == 11
    assert compute_ghbdf(11) == 23

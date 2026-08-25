from pkg.mod import compute_hachi

def test_compute_hachi():
    assert compute_hachi(2) == 5
    assert compute_hachi(8) == 17

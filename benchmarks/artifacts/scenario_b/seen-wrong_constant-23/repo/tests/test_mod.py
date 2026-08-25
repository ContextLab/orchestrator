from pkg.mod import compute_jdjjh

def test_compute_jdjjh():
    assert compute_jdjjh(3) == 7
    assert compute_jdjjh(9) == 19

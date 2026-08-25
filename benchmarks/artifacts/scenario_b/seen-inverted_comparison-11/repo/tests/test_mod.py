from pkg.mod import compute_ifhec

def test_compute_ifhec():
    assert compute_ifhec(3, 10) == 10

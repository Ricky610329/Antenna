import numpy as np
import torch

from script import r82_delta_sm as dm


def _patterns(n=60, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.random(625) < 0.5
    out = [base]
    for _ in range(n - 1):
        q = out[int(rng.integers(len(out)))].copy()
        idx = rng.choice(625, size=int(rng.integers(1, 20)), replace=False)
        q[idx] = ~q[idx]
        out.append(q)
    return np.asarray(out)


def test_make_pairs_distance_within_dmax_and_matches_hamming():
    P = _patterns()
    Pk = np.packbits(P, 1)
    q, c = dm.make_pairs(Pk, np.random.default_rng(1), draws=6, dmax=10)
    assert len(q) > 0
    d = (P[q] != P[c]).sum(1)
    assert d.min() >= 1 and d.max() <= 10
    assert (dm.hamming(Pk[:5], Pk)[np.arange(5), np.arange(5)] == 0).all()


def test_features_encode_flip_mask_and_distance():
    g = 4
    Pc = torch.zeros(2, 625, dtype=torch.uint8)
    Pq = Pc.clone()
    Pq[1, :5] = 1
    Y = torch.ones(2, 3, g)
    M = torch.ones(2, 3, g, dtype=torch.bool)
    img, cur = dm.features(Pc, Pq, Y, M)
    assert img.shape == (2, 1250) and cur.shape == (2, 3 * g * 2 + 1)
    assert img[0, 625:].sum() == 0 and img[1, 625:].sum() == 5
    assert torch.isclose(cur[1, -1], torch.tensor(5 / dm.DMAX))
    _, cur_h = dm.features(Pc, Pq, Y, M, dS=torch.zeros(2, 3, g))
    assert cur_h.shape == (2, 3 * g * 3 + 1)


def test_deltanet_cnn_and_mlp_shapes():
    g = 4
    img = torch.zeros(3, 1250)
    for arch in ("mlp", "cnn"):
        for hyb in (False, True):
            net = dm.DeltaNet(g, arch, hyb)
            cur = torch.zeros(3, 3 * g * (3 if hyb else 2) + 1)
            assert net(img, cur).shape == (3, 3, g)


def test_inband_score_nan_when_band_not_fully_measured():
    slices = [("s21_pass", 1, 0, 2, "ge", -3.0), ("s11_match", 0, 1, 2, "le", -10.0)]
    Y = np.zeros((2, 3, 3), np.float32)
    Y[:, 1] = -2.0
    Y[:, 0] = -12.0
    M = np.ones((2, 3, 3), bool)
    M[1, 0, 2] = False
    s = dm.inband_score(Y, M, slices)
    assert np.isclose(s[0], 1.0) and np.isnan(s[1])

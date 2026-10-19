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


def test_neighbor_bank_pairs_within_dmax_and_allow():
    P = _patterns()
    Pk = np.packbits(P, 1)
    rng = np.random.default_rng(1)
    allow = np.ones(len(P), bool)
    allow[:10] = False
    bank = dm.neighbor_bank(Pk, rng, allow=allow, dmax=10, per=4)
    q, c = dm.sample_pairs(bank, rng, 6)
    assert len(q) > 0
    d = (P[q] != P[c]).sum(1)
    assert d.min() >= 1 and d.max() <= 10
    assert allow[q].all() and allow[c].all()
    assert (dm.hamming(Pk[:5], Pk)[np.arange(5), np.arange(5)] == 0).all()


def test_drop_future_removes_disallowed_neighbors():
    P = _patterns()
    Pk = np.packbits(P, 1)
    rng = np.random.default_rng(2)
    bank = dm.neighbor_bank(Pk, rng, dmax=10, per=4)
    ok = np.arange(len(P)) < 40
    bank["nb"][~ok] = -1
    dm._drop_future(bank, ok)
    q, c = dm.sample_pairs(bank, rng, 4)
    assert len(q) > 0 and ok[q].all() and ok[c].all()


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


def test_deltanet_shapes_and_train_smoke():
    g = 4
    for arch in ("mlp", "cnn"):
        assert dm.DeltaNet(g, arch)(torch.zeros(3, 1250), torch.zeros(3, 3 * g * 2 + 1)).shape == (3, 3, g)
    P = _patterns(30)
    Y = np.random.default_rng(3).normal(size=(30, 3, g)).astype(np.float32)
    M = np.ones((30, 3, g), bool)
    rng = np.random.default_rng(4)
    bank = dm.neighbor_bank(np.packbits(P, 1), rng, dmax=20, per=4)
    ms = dm.train(P, Y, M, np.zeros(30, bool), bank, 2, arch="cnn", seeds=(0,), epochs=2, ft_bank=bank,
                  ft_draws=1, ft_epochs=1, rng=rng, log=lambda *_: None)
    assert dm.predict(ms, P[:3], P[3:6], Y[:3], M[:3]).shape == (1, 3, 3, g)


def test_score_candidates_uses_nearest_full_context_and_flags_far():
    g = 4
    P = _patterns(30)
    rng = np.random.default_rng(5)
    Y = rng.normal(size=(30, 3, g)).astype(np.float32)
    M = np.ones((30, 3, g), bool)
    M[:5, 1, :] = False                                    # 前 5 筆缺 S21：不能當上下文
    data = {"P": P, "Y": Y, "M": M, "wide": np.ones(30, bool)}
    slices = [("s21_pass", 1, 0, 3, "ge", -3.0)]
    far = ~P[:1]                                           # 與所有設計都很遠
    out = dm.score_candidates(np.concatenate([P[5:8], far]), slices, data, seeds=(0,), epochs=1,
                              log=lambda *_: None)
    assert out["ctx_d"][:3].tolist() == [0, 0, 0]
    assert np.isfinite(out["wm_mean"][:3]).all() and np.isnan(out["wm_mean"][3])


def test_inband_score_nan_when_band_not_fully_measured():
    slices = [("s21_pass", 1, 0, 2, "ge", -3.0), ("s11_match", 0, 1, 2, "le", -10.0)]
    Y = np.zeros((2, 3, 3), np.float32)
    Y[:, 1] = -2.0
    Y[:, 0] = -12.0
    M = np.ones((2, 3, 3), bool)
    M[1, 0, 2] = False
    s = dm.inband_score(Y, M, slices)
    assert np.isclose(s[0], 1.0) and np.isnan(s[1])

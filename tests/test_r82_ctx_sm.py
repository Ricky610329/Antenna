import numpy as np

from script import r82_ctx_sm as cx
from script import r82_delta_sm as dm


def _patterns(n=40, seed=0):
    rng = np.random.default_rng(seed)
    out = [rng.random(625) < 0.5]
    for _ in range(n - 1):
        q = out[int(rng.integers(len(out)))].copy()
        idx = rng.choice(625, size=int(rng.integers(1, 15)), replace=False)
        q[idx] = ~q[idx]
        out.append(q)
    return np.asarray(out)


def test_near_list_and_episodes_respect_bank():
    P = _patterns()
    Pk = np.packbits(P, 1)
    rng = np.random.default_rng(1)
    allow = np.arange(len(P)) >= 5
    bank = dm.neighbor_bank(Pk, rng, allow=allow, dmax=20, per=4)
    near = cx.near_list(bank, 3)
    q, C = cx.sample_episodes(bank, near, rng, 3, 3)
    assert len(q) > 0 and C.shape == (len(q), 3)
    assert allow[q].all() and allow[C].all()
    d = (P[q][:, None, :] != P[C]).sum(2)
    assert d.min() >= 1 and d.max() <= 20


def test_ctx_model_trains_and_masks_missing_context_bands():
    g = 4
    P = _patterns(30)
    rng = np.random.default_rng(2)
    Y = rng.normal(size=(30, 3, g)).astype(np.float32)
    M = np.ones((30, 3, g), bool)
    bank = dm.neighbor_bank(np.packbits(P, 1), rng, dmax=20, per=4)
    ms = cx.train_ctx(P, Y, M, np.zeros(30, bool), bank, 2, 3, seeds=(0,), epochs=2, rng=rng, log=lambda *_: None)
    C = np.array([[0, 1, 2], [3, 4, 5]])
    M2 = M.copy()
    M2[[1, 2], 1, :] = False                                # 第 0 個查詢的 S21 只剩上下文 0 有
    out = cx.predict_ctx(ms, P, Y, M2, C, P[10:12])
    assert out.shape == (2, 3, g) and np.isfinite(out).all()


def test_variable_k_model_predicts_with_fewer_contexts():
    g = 4
    P = _patterns(30)
    rng = np.random.default_rng(3)
    Y = rng.normal(size=(30, 3, g)).astype(np.float32)
    M = np.ones((30, 3, g), bool)
    bank = dm.neighbor_bank(np.packbits(P, 1), rng, dmax=20, per=4)
    ms = cx.train_ctx(P, Y, M, np.zeros(30, bool), bank, 2, 3, kmin=1, seeds=(0,), epochs=2, rng=rng,
                      log=lambda *_: None)
    C = np.array([[0, 1, 2], [3, 4, 5]])
    full = cx.predict_ctx(ms, P, Y, M, C, P[10:12])
    one = cx.predict_ctx(ms, P, Y, M, C, P[10:12], use_k=1)
    one_only = cx.predict_ctx(ms, P, Y, M, C[:, :1], P[10:12])
    assert np.isfinite(one).all() and not np.allclose(full, one)
    assert np.allclose(one, one_only, atol=1e-4)            # 遮掉的位置不影響結果


def test_nearest_k_excludes_self():
    P = _patterns(20)
    Pk = np.packbits(P, 1)
    C, D = cx.nearest_k(Pk, Pk[:3], np.arange(20), 2)
    assert C.shape == (3, 2) and (D >= 1).all()
    assert not any(i in row for i, row in enumerate(C))

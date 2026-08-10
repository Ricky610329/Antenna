"""
tests/test_sm_dual.py — dual-port SM 排序器（`script/sm_dual.py`）最小契約。

只測「不靠 NAS、不靠訓練好的權重」也該成立的三件事：
  1. **split 決定性**：同一鍋 + 同 seed → 同一份 held-out（換 seed 才會變），且十分位每箱都抽到
     （分層抽樣的定義；沒有這條，「held-out 上的品質閘數字」不可重現）。
  2. **輸出形狀**：模型吃 (B,625)/(B,25,25)/(B,1,25,25) 都回 (B,3,17) 響應 + (B,4) margin。
  3. **rank 排序正確**：以 ensemble 平均 margin 的 min（＝wm_dual 口徑）由高到低排，--top 截斷。
另含去重鍵（pattern bits）與 --pool 資料夾讀取的契約。全部用合成小資料。
"""
import json

import numpy as np
import pytest
import torch
from torch import nn

from script.sm_dual import (DualNet, HELDOUT_FRAC, N_POINTS, SPLIT_BINS, _load_pool_arg,
                            make_split, pattern_key, rank_pool)


# ── 1. split 決定性 / 分層 ────────────────────────────────────────────────
def _wm(n=500, seed=7):
    return np.random.default_rng(seed).normal(size=n) * 8 - 17     # 量級比照真鍋 (dB)


def test_split_is_deterministic():
    wm = _wm()
    tr1, ho1 = make_split(wm)
    tr2, ho2 = make_split(wm)
    assert np.array_equal(ho1, ho2)
    assert np.array_equal(tr1, tr2)


def test_split_partitions_and_sizes():
    wm = _wm()
    tr, ho = make_split(wm)
    assert set(tr.tolist()) | set(ho.tolist()) == set(range(len(wm)))
    assert not (set(tr.tolist()) & set(ho.tolist()))
    assert len(ho) == pytest.approx(len(wm) * HELDOUT_FRAC, abs=SPLIT_BINS)


def test_split_is_stratified_every_decile_sampled():
    """每個 wm 十分位都要有 held-out 成員——這就是「分位分層」的可檢驗定義。"""
    wm = _wm()
    _, ho = make_split(wm)
    hoset = set(ho.tolist())
    for b in np.array_split(np.argsort(wm, kind="stable"), SPLIT_BINS):
        assert hoset & set(b.tolist()), "有分位箱完全沒被抽到 → 不是分層抽樣"


def test_split_changes_with_seed():
    wm = _wm()
    _, a = make_split(wm, seed=58)
    _, b = make_split(wm, seed=59)
    assert not np.array_equal(a, b)


# ── 2. 去重鍵 ────────────────────────────────────────────────────────────
def test_pattern_key_normalizes_dtype():
    """float32 / bool / 0-1 uint8 的同一張 pattern 必須是同一個去重鍵（跨店格式不一致的坑）。"""
    bits = (np.random.default_rng(0).random(625) > 0.5)
    assert pattern_key(bits.astype(np.float32)) == pattern_key(bits)
    assert pattern_key(bits.astype(np.uint8)) == pattern_key(bits.astype(np.float32).reshape(25, 25))
    flipped = bits.copy()
    flipped[3] = not flipped[3]
    assert pattern_key(flipped) != pattern_key(bits)


# ── 3. 模型輸出形狀 ──────────────────────────────────────────────────────
@pytest.mark.parametrize("shape", [(4, 625), (4, 25, 25), (4, 1, 25, 25)])
def test_model_output_shape(shape):
    m = DualNet().eval()
    with torch.no_grad():
        resp, marg = m(torch.zeros(*shape))
    assert resp.shape == (4, 3, N_POINTS)
    assert marg.shape == (4, 4)


# ── 4. rank 排序 ─────────────────────────────────────────────────────────
class _StubNet(nn.Module):
    """margin = 第 0 個像素決定（值越大排越前），供排序契約測試用。"""

    def __init__(self, bias=0.0):
        super().__init__()
        self.bias = bias

    def forward(self, x):
        v = x.reshape(x.shape[0], -1)[:, 0]
        marg = torch.stack([v + self.bias, v + 10, v + 20, v + 30], dim=1)
        return torch.zeros(len(v), 3, N_POINTS), marg


def test_rank_sorts_desc_by_pred_wm_and_truncates():
    X = np.zeros((5, 625), dtype=np.float32)
    X[:, 0] = [1.0, 5.0, 3.0, 2.0, 4.0]
    ids = ["a", "b", "c", "d", "e"]
    rows = rank_pool([_StubNet()], X, ids, top=3)
    assert [r[0] for r in rows] == ["b", "e", "c"]
    assert [round(r[1], 3) for r in rows] == [5.0, 4.0, 3.0]      # wm = min(m1..m4) = m1
    assert rows[0][2]["m3"] == pytest.approx(25.0)                 # per-margin 也回傳


def test_rank_uses_ensemble_mean():
    """兩個成員的 margin 取平均後才排序（不是各自排完再投票）。"""
    X = np.zeros((2, 625), dtype=np.float32)
    X[:, 0] = [0.0, 1.0]
    rows = rank_pool([_StubNet(bias=0.0), _StubNet(bias=4.0)], X, ["lo", "hi"])
    assert [r[0] for r in rows] == ["hi", "lo"]
    assert rows[0][1] == pytest.approx(3.0)                        # (1+0 + 1+4)/2
    assert rows[1][1] == pytest.approx(2.0)                        # (0+0 + 0+4)/2


# ── 5. --pool 讀取契約 ───────────────────────────────────────────────────
def test_load_pool_from_manifest_dir(tmp_path):
    ids = ["x_01", "x_02"]
    for i, pid in enumerate(ids):
        torch.save(torch.full((25, 25), float(i)), tmp_path / f"{pid}.pt")
    (tmp_path / "manifest.json").write_text(
        json.dumps([{"id": i} for i in ids] + [{"id": "missing_99"}]), encoding="utf-8")
    X, got = _load_pool_arg(str(tmp_path))
    assert got == ids                       # manifest 有、檔案沒有的 id 要被略過
    assert X.shape == (2, 625)
    assert X[1].max() == 1.0


def test_load_pool_from_store_dir(tmp_path):
    """SampleStore 夾：<hash>.pt 存 (x, y) tuple → 只取 x。"""
    torch.save((torch.ones(25, 25), torch.zeros(3, N_POINTS)), tmp_path / "aa11.pt")
    X, ids = _load_pool_arg(str(tmp_path))
    assert ids == ["aa11"] and X.shape == (1, 625) and X.min() == 1.0


def test_load_pool_from_npz(tmp_path):
    p = tmp_path / "pool.npz"
    np.savez(p, X=np.zeros((3, 625), dtype=np.float32), ids=np.array(["p0", "p1", "p2"]))
    X, ids = _load_pool_arg(str(p))
    assert X.shape == (3, 625) and ids == ["p0", "p1", "p2"]

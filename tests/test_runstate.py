"""
RunState (結果夾即資料庫) 的單元測試。

驗證：純量 append/讀取語義 (對齊 Record，golden 保真)、early_stop、
metrics.csv 持久化與斷點續跑載回、patterns/ 去重快取、epoch→pattern 反查。
"""
import torch

from antenna.utils.runstate import RunState


def _fill_epoch(state, epoch, loss, pattern=None):
    pattern = pattern if pattern is not None else torch.rand(25, 25)
    h = state.add_pattern(pattern, torch.rand(2, 17), loss)
    state.append("sim_loss", loss)
    state.append("sim_loss_avg", state.average("sim_loss"))
    state.append("best_loss", min(state.last("best_loss", float("inf")), loss))
    state.append("pattern_hash", h)
    state.append("r_feed", 0.5)
    state.append("gen_loss", loss * 2)
    state.append("epoch", epoch)
    state.append("time", 1.0)
    state.save_row()
    return pattern


def test_scalar_semantics(tmp_path):
    """append/last/series/average 語義與 Record 對齊 (average 含當前筆)。"""
    s = RunState(tmp_path, verbose=False)
    s.append("sim_loss", 4.0)
    s.append("sim_loss", 2.0)
    assert s.last("sim_loss") == 2.0
    assert s.series("sim_loss") == [4.0, 2.0]
    assert s.average("sim_loss") == 3.0
    assert s.last("nothing", "df") == "df"


def test_early_stop_semantics(tmp_path):
    """照 Record：資料不足 False；最近 patience 筆皆未優於先前最佳 → True。"""
    s = RunState(tmp_path, verbose=False)
    for v in (5.0, 3.0):                    # 資料不足 (需 patience+1)
        s.append("sim_loss", v)
    assert s.early_stop("sim_loss", 2) is False
    s.append("sim_loss", 3.5); s.append("sim_loss", 4.0)   # 最近 2 筆都 >= 3.0
    assert s.early_stop("sim_loss", 2) is True
    s.append("sim_loss", 1.0)              # 出現新低 → 最近窗口含改善
    assert s.early_stop("sim_loss", 2) is False


def test_csv_persistence_and_resume(tmp_path):
    """save_row → metrics.csv；新實例載回 → last_epoch / 序列 / 去重快取都在。"""
    s1 = RunState(tmp_path, verbose=False)
    p = _fill_epoch(s1, 1, 3.0)
    _fill_epoch(s1, 2, 2.0)

    s2 = RunState(tmp_path, verbose=False)              # 模擬斷點續跑
    assert s2.last_epoch == 2
    assert s2.series("sim_loss") == [3.0, 2.0]
    assert s2.lookup(p) is not None                     # 去重快取跨實例存活 (檔案制)
    assert (tmp_path / "metrics.csv").exists()


def test_pattern_dedup_and_lookup(tmp_path):
    s = RunState(tmp_path, verbose=False)
    pat = torch.rand(25, 25); resp = torch.rand(2, 17)
    assert s.lookup(pat) is None                        # 沒模擬過
    h = s.add_pattern(pat, resp, 1.5)
    got_resp, got_loss, got_h = s.lookup(pat.clone())   # 同內容不同物件也命中
    assert got_h == h and got_loss == 1.5
    assert torch.equal(got_resp, resp)
    assert s.add_pattern(pat, resp, 1.5) == h           # 重複寫不增檔
    assert len(list((tmp_path / "patterns").glob("*.pt"))) == 1


def test_best_epoch_and_pattern_at(tmp_path):
    """best_epoch = 最小 loss「首次」出現的 epoch；pattern_at 反查該 epoch 的圖。"""
    s = RunState(tmp_path, verbose=False)
    p1 = _fill_epoch(s, 1, 3.0)
    p2 = _fill_epoch(s, 2, 1.0)
    _fill_epoch(s, 3, 1.0, pattern=p1)                  # 同 loss 再現 → 仍取首次 (epoch 2)
    assert s.best_epoch("sim_loss") == 2
    pattern, response = s.pattern_at(2)
    assert torch.equal(pattern, p2)
    assert response.shape == (2, 17)

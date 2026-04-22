"""針對 antenna.utils.{record, path, figure} 的純單元測試。

僅使用 pytest + tmp_path，不觸及真實專案路徑或網路磁碟。
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # 避免 GUI 依賴

import numpy as np
import pytest
import torch

from antenna.utils.figure import FIG_CONFIG, TQDM_CONFIG, Figure, plot
from antenna.utils.hashing import TID, get_shake_128
from antenna.utils.json_utils import json as JsonFile
from antenna.utils.path import Path
from antenna.utils.record import Record

# --------------------------------------------------------------------------- #
# Record                                                                      #
# --------------------------------------------------------------------------- #


class TestRecord:
    def test_append_and_access(self, tmp_path):
        r = Record("temp", rootdir=str(tmp_path))
        r["loss"] = 0.5
        r["loss"] = 0.4
        r["loss"] = 0.3

        # __getitem__ 回傳完整 list
        assert r["loss"] == [0.5, 0.4, 0.3]
        # __contains__
        assert "loss" in r
        assert "missing" not in r

    def test_end_returns_last_value(self, tmp_path):
        r = Record("end", rootdir=str(tmp_path))
        r["a"] = 1
        r["a"] = 2
        assert r.end("a") == 2
        # __call__ 等於 end
        assert r("a") == 2

    def test_end_default_and_append(self, tmp_path):
        r = Record("end2", rootdir=str(tmp_path))
        # key 不存在且 append=False -> 回傳 default
        assert r.end("missing", default=10) == 10
        assert "missing" not in r
        # append=True -> 寫入 default 並回傳
        assert r.end("new", default=5, append=True) == 5
        assert r["new"] == [5]

    def test_add(self, tmp_path):
        r = Record("add", rootdir=str(tmp_path))
        r.add("cnt", 1, default=0)
        r.add("cnt", 2)
        r.add("cnt", 3)
        assert r["cnt"] == [1, 3, 6]

    def test_delete_and_reset(self, tmp_path):
        r = Record("reset", rootdir=str(tmp_path))
        r["x"] = 1
        r["y"] = 2
        del r["x"]
        assert "x" not in r
        r.reset("y")
        assert r["y"] == []
        r["y"] = 99
        r.reset()
        assert len(r._data) == 0

    def test_reset_delete(self, tmp_path):
        r = Record("resetdel", rootdir=str(tmp_path))
        r["k"] = 1
        r.reset("k", delete=True)
        assert "k" not in r

    def test_average(self, tmp_path):
        r = Record("avg", rootdir=str(tmp_path))
        for v in [1.0, 2.0, 3.0, 4.0]:
            r["v"] = v
        assert r.average("v") == pytest.approx(2.5)
        assert r.average("missing") is None

    def test_index_and_find(self, tmp_path):
        r = Record("idx", rootdir=str(tmp_path))
        for epoch, (a, b) in enumerate(zip(["a1", "a2", "a3"], ["b1", "b2", "b3"]), start=1):
            r["epoch"] = epoch
            r["a"] = a
            r["b"] = b

        assert r.index("a", "a2") == 1
        assert r.index("a", "missing") is None
        assert r.index("missing_key", "anything") is None
        assert r.find("a", "a1", "epoch") == 1
        assert r.find("epoch", 3, ("a", "b")) == ["a3", "b3"]
        assert r.find("a", "missing", "epoch") is None

    def test_index_with_ndarray(self, tmp_path):
        r = Record("np", rootdir=str(tmp_path))
        r["arr"] = np.array([1, 2])
        r["arr"] = np.array([3, 4])
        assert r.index("arr", np.array([3, 4])) == 1
        assert r.index("arr", np.array([9, 9])) is None

    def test_index_with_tensor(self, tmp_path):
        r = Record("tensor", rootdir=str(tmp_path))
        r["t"] = torch.tensor([1.0, 2.0])
        r["t"] = torch.tensor([3.0, 4.0])
        assert r.index("t", torch.tensor([3.0, 4.0])) == 1
        assert r.index("t", torch.tensor([9.0, 9.0])) is None

    def test_save_load_roundtrip(self, tmp_path):
        r = Record("roundtrip", rootdir=str(tmp_path))
        r["loss"] = 0.1
        r["loss"] = 0.2
        r.save(description="unit-test")

        r2 = Record("roundtrip", rootdir=str(tmp_path), load=True)
        assert r2["loss"] == [0.1, 0.2]
        # history 是 DataFrame
        assert len(r2.history) >= 1

    def test_load_creates_when_missing(self, tmp_path):
        # load=True 時若檔案不存在應該自動建立（呼叫 save）
        r = Record("missing_file", rootdir=str(tmp_path), load=True)
        assert r.path.exists()

    def test_state_dict_roundtrip(self, tmp_path):
        r = Record("sd", rootdir=str(tmp_path))
        r["a"] = 1
        r["a"] = 2
        sd = r.state_dict()

        r2 = Record("sd2", rootdir=str(tmp_path))
        r2.load_state_dict(sd)
        assert r2["a"] == [1, 2]

    def test_early_stop_not_enough_data(self, tmp_path):
        r = Record("es", rootdir=str(tmp_path))
        for v in [1.0, 2.0]:
            r["loss"] = v
        assert r.early_stop("loss", patience=5) is False

    def test_early_stop_minimize(self, tmp_path):
        r = Record("es_min", rootdir=str(tmp_path))
        # 先有一個最佳值 0.1，之後都沒有改善
        for v in [0.5, 0.4, 0.1, 0.2, 0.3, 0.25]:
            r["loss"] = v
        # 目前最佳（window 之前）是 0.1，之後 [0.2,0.3,0.25] 全都 >= 0.1 -> True
        assert r.early_stop("loss", patience=3, is_maximize=False) is True

    def test_early_stop_maximize(self, tmp_path):
        r = Record("es_max", rootdir=str(tmp_path))
        # 最大化：window 之前最佳為 0.9，之後 [0.5,0.6,0.7] 全 <= 0.9 -> True
        for v in [0.1, 0.5, 0.9, 0.5, 0.6, 0.7]:
            r["acc"] = v
        assert r.early_stop("acc", patience=3, is_maximize=True) is True

    def test_early_stop_improving(self, tmp_path):
        r = Record("es_impr", rootdir=str(tmp_path))
        # 持續改善 -> 不應該停
        for v in [0.5, 0.4, 0.3, 0.2, 0.1]:
            r["loss"] = v
        assert r.early_stop("loss", patience=2, is_maximize=False) is False

    def test_custom(self, tmp_path):
        r = Record("custom", rootdir=str(tmp_path))
        for v in [1, 2, 3, 4]:
            r["x"] = v
        assert r.custom("x", max) == 4
        assert r.custom("x", sum) == 10
        # 空 key 回傳 default
        assert r.custom("empty", sum, default=-1) == -1

    def test_dataframe_property(self, tmp_path):
        r = Record("df", rootdir=str(tmp_path))
        r["a"] = 1
        r["a"] = 2
        r["b"] = 3
        r["b"] = 4
        df = r.dataframe
        assert list(df.columns) == ["a", "b"]
        assert len(df) == 2
        assert len(r) == 2

    def test_dataframe_with_tensor(self, tmp_path):
        r = Record("df_t", rootdir=str(tmp_path))
        r["t"] = torch.tensor(1.0)
        r["t"] = torch.tensor(2.0)
        df = r.dataframe
        assert list(df["t"]) == [1.0, 2.0]

    def test_repr_and_str(self, tmp_path):
        r = Record("repr", rootdir=str(tmp_path))
        r["a"] = 1
        assert "Record(repr" in repr(r)
        assert isinstance(str(r), str)

    def test_getitem_missing_raises(self, tmp_path):
        r = Record("missing", rootdir=str(tmp_path))
        with pytest.raises(KeyError):
            r["nope"]


# --------------------------------------------------------------------------- #
# Path                                                                        #
# --------------------------------------------------------------------------- #


class TestPath:
    def test_joinpath_and_stem(self, tmp_path):
        p = Path(str(tmp_path)).joinpath("sub", "file.txt")
        assert p.stem == "file"
        assert p.suffix == ".txt"
        assert p.parent == Path(str(tmp_path)).joinpath("sub")

    def test_not_exist_create_directory(self, tmp_path):
        target = Path(str(tmp_path / "newdir"))
        target.not_exist_create()
        assert target.exists()
        assert target.is_dir()

    def test_not_exist_create_file(self, tmp_path):
        target = Path(str(tmp_path / "sub" / "file.txt"))
        target.not_exist_create(create_file=True)
        assert target.exists()
        assert target.is_file()

    def test_not_exist_create_file_no_touch(self, tmp_path):
        target = Path(str(tmp_path / "sub2" / "file.txt"))
        target.not_exist_create(create_file=False)
        # 只會建立 parent，不建立檔案
        assert target.parent.exists()
        assert not target.exists()

    def test_rmtree_on_directory(self, tmp_path):
        d = Path(str(tmp_path / "to_remove"))
        d.mkdir()
        (d / "inner.txt").write_text("x", encoding="utf-8")
        assert d.rmtree() is True
        assert not d.exists()

    def test_rmtree_on_file_returns_false(self, tmp_path):
        f = Path(str(tmp_path / "a_file.txt"))
        f.write_text("x", encoding="utf-8")
        assert f.rmtree() is False
        assert f.exists()

    def test_del_from_glob_pattern(self, tmp_path):
        base = Path(str(tmp_path))
        for i in range(3):
            (base / f"x{i}.pth").write_text("x", encoding="utf-8")
        (base / "keep.txt").write_text("keep", encoding="utf-8")
        base.del_from_glob("*.pth")
        remaining = sorted(p.name for p in base.iterdir())
        assert remaining == ["keep.txt"]

    def test_del_from_glob_single_file(self, tmp_path):
        f = Path(str(tmp_path / "one.pth"))
        f.write_text("x", encoding="utf-8")
        f.del_from_glob("*.pth")  # suffix -> 直接 unlink
        assert not f.exists()

    def test_manage_file_count(self, tmp_path):
        base = Path(str(tmp_path))
        files = []
        for i in range(5):
            f = base / f"ck_{i}.pth"
            f.write_text("x", encoding="utf-8")
            files.append(f)
        # 保留最新 2 個
        changed = base.manage_file_count("*.pth", keep_latest=2)
        assert changed is True
        remaining = sorted(p.name for p in base.glob("*.pth"))
        assert len(remaining) == 2

    def test_manage_file_count_noop_when_under_limit(self, tmp_path):
        base = Path(str(tmp_path))
        (base / "a.pth").write_text("x", encoding="utf-8")
        changed = base.manage_file_count("*.pth", keep_latest=5)
        assert changed is False

    def test_manage_file_count_none_skips(self, tmp_path):
        base = Path(str(tmp_path))
        assert base.manage_file_count("*.pth", keep_latest=None) is False

    def test_manage_file_count_missing_dir_raises(self, tmp_path):
        missing = Path(str(tmp_path / "does_not_exist"))
        with pytest.raises(FileNotFoundError):
            missing.manage_file_count("*.pth", keep_latest=1)

    def test_pickling_roundtrip(self, tmp_path):
        """Path 的 __reduce__ 應允許 pickle。"""
        import pickle

        p = Path(str(tmp_path / "a" / "b.txt"))
        data = pickle.dumps(p)
        p2 = pickle.loads(data)
        assert isinstance(p2, Path)
        assert str(p2) == str(p)


# --------------------------------------------------------------------------- #
# Figure                                                                      #
# --------------------------------------------------------------------------- #


class TestFigure:
    def test_module_level_configs(self):
        # 純 static 驗證 module 級設定的健全性
        assert FIG_CONFIG["format"] == "png"
        assert "dpi" in FIG_CONFIG
        assert "unit" in TQDM_CONFIG

    def test_figure_context_manager(self, tmp_path):
        with Figure("smoke1", nrowcol=(1, 1), rootdir=str(tmp_path)) as fig:
            ax = fig.index(1)
            ax.plot([1, 2, 3])
            ax.set_title("smoke")
        # 不 save 的情況下，檔案不應該存在
        assert not (tmp_path / "smoke1.png").exists()

    def test_figure_save(self, tmp_path):
        with Figure("smoke2", nrowcol=(1, 1), save=True, rootdir=str(tmp_path)) as fig:
            ax = fig.index(1)
            ax.plot([1, 2, 3])
        assert (tmp_path / "smoke2.png").exists()

    def test_figure_addAll_and_len(self, tmp_path):
        with Figure("smoke3", nrowcol=(2, 2), rootdir=str(tmp_path)) as fig:
            assert len(fig) == 4
            fig.addAll()
            # addAll 後應該有 4 個 axes
            assert len(fig.fig.get_axes()) == 4
            # __getitem__ 透過 axes list
            ax0 = fig[0]
            ax0.set_title("t0")

    def test_figure_ncols_tuple(self, tmp_path):
        # (total=5, cols=2) -> nrowcol=(3,2)
        with Figure("smoke4", ncols=(5, 2), rootdir=str(tmp_path)) as fig:
            assert fig.nrowcol == (3, 2)

    def test_figure_convert_to(self, tmp_path):
        with Figure("conv", rootdir=str(tmp_path)) as fig:
            # 自訂 callback
            result = fig.convert_to(lambda f: ("ok", type(f).__name__))
        assert result[0] == "ok"

    def test_figure_repr(self, tmp_path):
        with Figure("rep", rootdir=str(tmp_path)) as fig:
            assert "rep" in repr(fig)
            assert "Figure" in repr(fig)

    def test_figure_requires_grad_toggle(self, tmp_path):
        # 預設 requires_grad=False -> 進入時關閉 grad
        before = torch.is_grad_enabled()
        with Figure("grad", rootdir=str(tmp_path), requires_grad=False):
            assert torch.is_grad_enabled() is False
        # 離開後恢復原狀
        assert torch.is_grad_enabled() is before

    def test_plot_smoke(self, tmp_path):
        # plot 是 module-level helper，純 smoke（無 label 會有 legend 警告，但不 crash）
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plot([1, 2, 3])  # 不指定 file_name


# --------------------------------------------------------------------------- #
# hashing                                                                     #
# --------------------------------------------------------------------------- #


class TestHashing:
    def test_tid_roundtrip_string(self):
        # 固定 timestamp 便於回推還原
        ts = 1_700_000_000
        tid = TID.generate(ts)
        assert isinstance(tid, str)
        assert TID.decode(tid) == ts

    def test_tid_roundtrip_int(self):
        ts = 1_700_000_000
        offset = TID.generate(ts, as_int=True)
        assert isinstance(offset, int)
        assert TID.decode(offset) == ts

    def test_tid_zero_delta(self):
        # delta = 0 -> 回傳 ALPHABET[0] ('0')
        tid = TID.generate(TID.CUSTOM_EPOCH)
        assert tid == TID.ALPHABET[0]

    def test_tid_before_epoch_raises(self):
        with pytest.raises(ValueError):
            TID.generate(TID.CUSTOM_EPOCH - 1)

    def test_tid_decode_invalid_char(self):
        with pytest.raises(ValueError):
            TID.decode("!invalid")

    def test_get_shake_128_deterministic(self):
        # 相同輸入永遠產生相同摘要
        a = get_shake_128("hello", length=6)
        b = get_shake_128("hello", length=6)
        assert a == b
        assert len(a) == 6

    def test_get_shake_128_differs_per_input(self):
        assert get_shake_128("a") != get_shake_128("b")

    def test_get_shake_128_various_lengths(self):
        for length in [1, 2, 4, 8, 16]:
            assert len(get_shake_128("x", length=length)) == length


# --------------------------------------------------------------------------- #
# json_utils                                                                  #
# --------------------------------------------------------------------------- #


class TestJsonFile:
    def test_create_and_load_empty(self, tmp_path):
        p = tmp_path / "cfg.json"
        j = JsonFile(str(p))
        assert p.exists()
        assert j.load() == {}

    def test_missing_no_create_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            JsonFile(str(tmp_path / "nope.json"), create=False)

    def test_set_and_get_simple(self, tmp_path):
        j = JsonFile(str(tmp_path / "s.json"))
        j("foo", "bar")
        assert j("foo") == "bar"
        assert j["foo"] == "bar"

    def test_set_and_get_nested(self, tmp_path):
        j = JsonFile(str(tmp_path / "n.json"))
        j("base/path", "/data")
        j("base/name", "demo")
        assert j("base/path") == "/data"
        assert j("base/name") == "demo"
        # 未 nested 層可讀回完整 dict
        assert j("base") == {"path": "/data", "name": "demo"}

    def test_value_coercion(self, tmp_path):
        j = JsonFile(str(tmp_path / "c.json"))
        j("a", "null")
        assert j("a") is None
        j("b", "true")
        assert j("b") is True
        j("c", "False")
        assert j("c") is False

    def test_get_with_default_persists(self, tmp_path):
        j = JsonFile(str(tmp_path / "g.json"))
        # 讀不到時以 default 寫回並回傳
        assert j.get("missing", default=123) == 123
        # 第二次直接讀得到
        assert j("missing") == 123

    def test_setitem_and_getitem(self, tmp_path):
        j = JsonFile(str(tmp_path / "i.json"))
        j["k1"] = "v1"
        assert j["k1"] == "v1"

    def test_delete(self, tmp_path):
        j = JsonFile(str(tmp_path / "d.json"))
        j("a/b", "x")
        assert j.delete("a/b") is True
        # 中間節點仍存在，但葉節點被移除
        assert j("a") == {}
        # 刪不存在的 key 回傳 False
        assert j.delete("nope/nope") is False

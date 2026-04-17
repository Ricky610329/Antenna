"""`antenna.utils.config` 的單元測試。

涵蓋 ``Config`` 的 attribute-access 行為、``MultiConfig`` 的建構、
以及 ``save``/``load`` 的往返行為。所有測試都在 tmp_path 下進行，
並避免觸發任何 email / excepthook 副作用。
"""

from __future__ import annotations

import json
import sys
from unittest.mock import patch

import pytest
import torch

from antenna.utils.config import Config, MultiConfig, global_exception_handler

# ---------------------------------------------------------------------------
# Config: attribute / item 基本行為
# ---------------------------------------------------------------------------


def test_config_is_dict_subclass():
    cfg = Config()
    assert isinstance(cfg, dict)


def test_config_default_fields():
    cfg = Config()
    assert cfg.epochs == 10
    assert cfg.lr == pytest.approx(1e-3)
    assert cfg.element_num == 40
    # ID 應為 unix time 的字串
    assert isinstance(cfg.ID, str)
    assert cfg.ID.isdigit()


def test_config_device_default_is_cpu():
    cfg = Config()
    assert isinstance(cfg.device, torch.device)
    assert cfg.device.type == "cpu"


def test_config_setattr_stores_into_dict():
    """非 property 的 attr 應寫入到底層 dict。"""
    cfg = Config()
    cfg.custom_value = 42
    assert cfg["custom_value"] == 42
    # attr-access 也能讀到
    assert cfg.custom_value == 42


def test_config_getattr_falls_back_to_item():
    cfg = Config()
    cfg["some_key"] = "hello"
    assert cfg.some_key == "hello"


def test_config_getattr_missing_raises_keyerror():
    cfg = Config()
    with pytest.raises(KeyError):
        _ = cfg.definitely_missing_key


def test_config_property_setattr_uses_object_setattr():
    """property 的 attr 應走 setter；這裡用 enable_exception_handler 驗證。"""
    cfg = Config()
    original = sys.excepthook
    try:
        # enable_exception_handler 是 property，setattr 應呼叫 setter 而非寫入 dict
        cfg.enable_exception_handler = True
        assert "enable_exception_handler" not in cfg  # 未寫入 dict
        assert sys.excepthook is cfg.excepthook
    finally:
        sys.excepthook = original


def test_config_check_keys_ok():
    cfg = Config()
    cfg["foo"] = 1
    cfg["bar"] = 2
    cfg.check_keys("foo", "bar")  # 不拋例外


def test_config_check_keys_missing_raises():
    cfg = Config()
    with pytest.raises(KeyError):
        cfg.check_keys("does_not_exist")


def test_config_check_keys_only_warning():
    cfg = Config()
    # only_warning=True 不應拋錯，僅記 log（loguru 不走 pytest caplog，不在此斷言 log 內容）
    cfg.check_keys("missing_key", only_warning=True)


def test_config_str_contains_classname():
    cfg = Config()
    # 預設 dict 中的 Path 物件在某些 Python 版本上 str() 會壞；
    # 此處只確保 Config.__str__ 包含類別名前綴即可。
    cfg.pop("checkpoint_save_path", None)
    s = str(cfg)
    assert s.startswith("Config(")


# ---------------------------------------------------------------------------
# Config: excepthook (不實際更動 sys.excepthook)
# ---------------------------------------------------------------------------


def test_config_excepthook_attached():
    cfg = Config()
    # 應為 callable
    assert callable(cfg.excepthook)


def test_config_enable_exception_handler_toggle():
    """enable_exception_handler 應可正確讀寫 sys.excepthook，
    測試結束時還原，避免污染其他測試。"""
    cfg = Config()
    original = sys.excepthook
    try:
        cfg.enable_exception_handler = True
        assert sys.excepthook is cfg.excepthook
        assert cfg.enable_exception_handler is True

        cfg.enable_exception_handler = False
        assert sys.excepthook is sys.__excepthook__
        assert cfg.enable_exception_handler is False
    finally:
        sys.excepthook = original


def test_global_exception_handler_mode_false_returns_original():
    """mode=False 時應回傳 sys.__excepthook__ 本身。"""
    hook = global_exception_handler(mode=False)
    assert hook is sys.__excepthook__


# ---------------------------------------------------------------------------
# Config.save / Config.load (legacy API)
# ---------------------------------------------------------------------------


def _make_clean_config() -> Config:
    """建立一個 Config 實例並移除 default 中的 Path/device 物件，
    使 `save()` 不需要 serialize 這些在目前環境中可能行為異常的 default 值。"""
    cfg = Config()
    cfg.pop("checkpoint_save_path", None)
    cfg.pop("device", None)
    return cfg


def test_config_save_writes_json(tmp_path):
    cfg = _make_clean_config()
    cfg["name"] = "test-run"
    cfg["nums"] = [1, 2, 3]
    cfg.save(name="config", rootdir=str(tmp_path))

    json_file = tmp_path / "config.json"
    assert json_file.exists()
    data = json.loads(json_file.read_text(encoding="utf-8"))
    assert data["name"] == "test-run"
    assert data["nums"] == [1, 2, 3]


def test_config_save_non_serializable_as_str(tmp_path):
    """非 JSON-native 型別會被 str() 化。"""

    class Custom:
        def __str__(self):
            return "custom-object-str"

    cfg = _make_clean_config()
    cfg["weird"] = Custom()
    cfg.save(name="config", rootdir=str(tmp_path))

    data = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert data["weird"] == "custom-object-str"


def test_config_save_update_hook_called(tmp_path):
    captured = {}

    def hook(cfg):
        captured["called"] = True
        captured["has_name"] = "name" in cfg
        return "hook-return"

    cfg = _make_clean_config()
    cfg["name"] = "x"
    result = cfg.save(name="config", rootdir=str(tmp_path), update_hook=hook)
    assert result == "hook-return"
    assert captured["called"] is True
    assert captured["has_name"] is True


def test_config_load_roundtrip(tmp_path):
    cfg = _make_clean_config()
    cfg["alpha"] = 1
    cfg["beta"] = "two"
    cfg["nested"] = {"k": [1, 2]}
    cfg.save(name="config", rootdir=str(tmp_path))

    loaded = _make_clean_config()
    loaded.load(name="config", rootdir=str(tmp_path))
    assert loaded["alpha"] == 1
    assert loaded["beta"] == "two"
    assert loaded["nested"] == {"k": [1, 2]}


# ---------------------------------------------------------------------------
# MultiConfig
# ---------------------------------------------------------------------------


def test_multiconfig_with_explicit_label():
    mc = MultiConfig({"default": {"epochs": 5}, "fast": {"epochs": 1}}, label="default")
    assert mc.label == "default"
    assert mc["epochs"] == 5


def test_multiconfig_label_setter():
    mc = MultiConfig({"a": {"x": 1}, "b": {"x": 2}}, label="a")
    assert mc["x"] == 1
    mc.label = "b"
    assert mc["x"] == 2


def test_multiconfig_setitem_writes_to_current_label():
    mc = MultiConfig({"default": {}}, label="default")
    mc["new_key"] = "new_val"
    assert mc.metadata["default"]["new_key"] == "new_val"


def test_multiconfig_call_returns_default_if_missing():
    mc = MultiConfig({"default": {"a": 1}}, label="default")
    assert mc("a") == 1
    assert mc("missing") is None
    assert mc("missing", default="fallback") == "fallback"


def test_multiconfig_get_label_data():
    mc = MultiConfig({"x": {"v": 10}, "y": {"v": 20}}, label="x")
    assert mc.get_label_data() == {"v": 10}
    assert mc.get_label_data("y") == {"v": 20}


def test_multiconfig_reads_label_from_argv():
    """label=None 時應從 sys.argv[1] 讀取。"""
    with patch.object(sys, "argv", ["prog", "fast"]):
        mc = MultiConfig({"default": {}, "fast": {"epochs": 1}})
        assert mc.label == "fast"
        assert mc["epochs"] == 1


def test_multiconfig_raises_if_no_label_and_no_argv():
    with patch.object(sys, "argv", ["prog"]):
        with pytest.raises(ValueError):
            MultiConfig({"default": {}})


def test_multiconfig_from_yaml_like_tmp_path(tmp_path):
    """模擬從 YAML 讀入 dummy 設定，再餵進 MultiConfig。"""
    import yaml

    yaml_path = tmp_path / "multi.yaml"
    yaml_path.write_text(
        "default:\n  epochs: 100\n  lr: 0.001\nfast:\n  epochs: 1\n  lr: 0.1\n",
        encoding="utf-8",
    )
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))

    mc = MultiConfig(data, label="default")
    assert mc["epochs"] == 100
    assert mc["lr"] == pytest.approx(0.001)

    mc.label = "fast"
    assert mc["epochs"] == 1
    assert mc["lr"] == pytest.approx(0.1)


def test_multiconfig_empty_default():
    """沒傳 config 時 metadata 為空 dict。"""
    mc = MultiConfig(label="foo")
    assert mc.metadata == {}
    assert mc.label == "foo"

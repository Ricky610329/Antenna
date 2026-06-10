"""
YAML config 載入與 port 解析的測試。
"""
import os

import pytest

from antenna.training import load_config, TrainConfig, PORT_SPECS, build_feeds

FIX = os.path.join(os.path.dirname(__file__), "fixtures")
CONFIGS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs")


def test_load_single():
    cfg = load_config(os.path.join(FIX, "single_test.yaml"))
    assert cfg.port == "single"
    assert cfg.epochs == 6 and cfg.lr == 0.005 and cfg.patience == 50
    assert cfg.loss["total_variation"] == 0.01
    assert cfg.loss["spectral_connectivity"] == 0.0005
    assert cfg.targets["S11"]["method"] == "low"
    assert cfg.targets["Gain"]["width"] == [5, 0, 7, 0, 5]


def test_load_dual():
    cfg = load_config(os.path.join(FIX, "dual_test.yaml"))
    assert cfg.port == "dual"
    assert set(cfg.targets) == {"S11", "S21", "S22"}
    assert cfg.targets["S21"]["interval"] == [-1, 1]


def test_production_configs_parse():
    """configs/ 下的 production 範例都要能解析。"""
    for name in ("single_base.yaml", "dual_base.yaml"):
        cfg = load_config(os.path.join(CONFIGS, name))
        assert cfg.port in PORT_SPECS
        # targets 須涵蓋該 port 的所有標籤 (否則 __post_init__ 已擋下)
        assert set(PORT_SPECS[cfg.port]["labels"]).issubset(cfg.targets)


def test_surrogate_section_parsed():
    """模型載入 (surrogate) 區段要能從 YAML 載入。"""
    s = load_config(os.path.join(CONFIGS, "single_base.yaml"))
    assert s.surrogate["pretrained"] == "old_sm.pth"
    assert s.surrogate["offline_dataset"] == "patch_single_mirror"
    d = load_config(os.path.join(CONFIGS, "dual_base.yaml"))
    assert d.surrogate["pretrained"] == "patch_dual.pth"
    # 測試 fixture 無 surrogate 區段 → 預設空 dict (prepare_models 變 no-op)
    t = load_config(os.path.join(FIX, "single_test.yaml"))
    assert t.surrogate == {}


def test_generator_section_parsed():
    """generator: <名字> 簡寫要正規化成 {name: ...}；無區段 → 空 dict (zoo 預設)。"""
    s = load_config(os.path.join(CONFIGS, "single_base.yaml"))
    assert s.generator == {"name": "sigmoid"}
    t = load_config(os.path.join(FIX, "single_test.yaml"))
    assert t.generator == {}


def test_warmup_section_parsed():
    """KuoHung 暖身編號要能從 YAML 載入 (補齊舊 single 3/4)。"""
    cfg = load_config(os.path.join(CONFIGS, "single_tv.yaml"))
    assert cfg.surrogate["warmup"] == "1"


def test_port_resolves_to_components():
    single = load_config(os.path.join(FIX, "single_test.yaml"))
    dual = load_config(os.path.join(FIX, "dual_test.yaml"))
    assert len(build_feeds(single)) == 1      # 只有 lower
    assert len(build_feeds(dual)) == 2        # lower + upper
    assert PORT_SPECS["single"]["labels"] == ["S11", "Gain"]
    assert PORT_SPECS["dual"]["labels"] == ["S11", "S21", "S22"]


def test_bad_port_rejected():
    with pytest.raises(ValueError):
        TrainConfig(name="x", port="triple", targets={})


def test_missing_target_rejected():
    with pytest.raises(ValueError):
        TrainConfig(name="x", port="single", targets={"S11": {}})  # 缺 Gain

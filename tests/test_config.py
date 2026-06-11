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
    # offline_dataset 已從學長的 patch_single_mirror 改指向自己工作區收割的資料集
    assert s.surrogate["offline_dataset"] == "harvest_single"
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


def _ok_targets():
    return {
        "S11": {"side": 0, "center": -10, "width": [5, 0, 7, 0, 5], "method": "low"},
        "Gain": {"side": -19, "center": 4, "width": [5, 0, 7, 0, 5], "method": "high"},
    }


def test_unknown_section_key_rejected():
    """區段內鍵打錯不可默默變預設值 (歷史教訓：dual island_suppression 鍵名 bug)。"""
    with pytest.raises(ValueError, match="loss"):
        TrainConfig(name="x", port="single", targets=_ok_targets(),
                    loss={"total_variaton": 1.0})            # 少個 t
    with pytest.raises(ValueError, match="sm_train"):
        TrainConfig(name="x", port="single", targets=_ok_targets(), sm_train={"lrr": 1})
    with pytest.raises(ValueError, match="scheduler"):
        TrainConfig(name="x", port="single", targets=_ok_targets(), scheduler={"on_plato": "linear"})
    with pytest.raises(ValueError, match="generator"):
        TrainConfig(name="x", port="single", targets=_ok_targets(), generator={"hiden": [64]})
    with pytest.raises(ValueError, match="surrogate"):
        TrainConfig(name="x", port="single", targets=_ok_targets(), surrogate={"pretrain": "x.pth"})


def test_unknown_target_key_rejected():
    t = _ok_targets(); t["S11"]["widht"] = [1, 2, 3]          # 拼錯
    with pytest.raises(ValueError, match="S11"):
        TrainConfig(name="x", port="single", targets=t)


def test_seed_parsed():
    """seed 進 config (可重現性)；未設 → None (維持現行為)。"""
    cfg = TrainConfig(name="x", port="single", targets=_ok_targets(), seed=7)
    assert cfg.seed == 7
    assert TrainConfig(name="x", port="single", targets=_ok_targets()).seed is None


def test_scheduler_params_flow_to_acp():
    """YAML scheduler 區段可調 ACP 超參數 (預設值=原寫死值，golden 不變)。"""
    import torch
    from antenna.training import build_scheduler
    cfg = TrainConfig(name="x", port="single", targets=_ok_targets(),
                      lr=0.005, scheduler={"T_0": 50, "temp_max": 2.0, "patience": 7})
    opt = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))], lr=0.005)
    sch = build_scheduler(cfg, opt)
    assert sch.T_i == 50 and sch.temp_max == 2.0 and sch.patience == 7
    assert sch.lr_max == 0.005                                 # 與 cfg.lr 綁定
    # 預設值 = 原本寫死在 run_training 的值
    sch2 = build_scheduler(TrainConfig(name="y", port="single", targets=_ok_targets(), lr=0.005), opt)
    assert sch2.T_i == 100 and sch2.temp_max == 4.0 and sch2.patience == 25
    assert sch2.warmup_ratio == 0.2 and sch2.factor == 0.7


def test_bad_port_rejected():
    with pytest.raises(ValueError):
        TrainConfig(name="x", port="triple", targets={})


def test_missing_target_rejected():
    with pytest.raises(ValueError):
        TrainConfig(name="x", port="single", targets={"S11": {}})  # 缺 Gain

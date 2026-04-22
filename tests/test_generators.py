"""Generator 純單元測試。

僅測試 forward shape / 二值化輸出範圍 / device 一致性，
不會呼叫 HFSS、不會讀寫網路磁碟，完全 CPU-only。
"""

import pytest
import torch

from antenna.core.pattern import AntennaPattern
from antenna.core.response import AntennaResponse
from antenna.utils.config import config

# 測試用小尺寸，避免跑太久且仍可驗證結構。
PATTERN_COORD = (0, 8, 0, 8)  # 8 * 8 = 64 pixels
RESPONSE_LABELS = ("s11",)
RESPONSE_X = (0, 7, 8)  # 8 frequency points per label


@pytest.fixture(scope="module", autouse=True)
def _register_antenna_shapes():
    """在整個模組測試前設定 AntennaPattern / AntennaResponse 的全域狀態。

    注意：`config` 預設就是 CPU（見 Config.__init__），不要再去 set，
    否則 `Config.device` 的 setter 會嘗試 `cuda.set_device` 而對 CPU 報錯。
    """
    AntennaPattern.setDefaultCoordinate(PATTERN_COORD)
    AntennaResponse.registerLabels(*RESPONSE_LABELS, x=RESPONSE_X)
    yield


@pytest.fixture
def response_input() -> torch.Tensor:
    """構造一個 1D 響應輸入（SigmoidGEN / OldGEN / GumbelSigmoidGEN 接受）。"""
    return torch.randn(AntennaResponse.size(flatten=True), device=config.device)


# ─────────────── 共用工具 ───────────────


def _is_binary(tensor: torch.Tensor) -> bool:
    """檢查張量值是否僅包含 {0, 1}。"""
    unique = torch.unique(tensor.detach())
    return bool(torch.all((unique == 0) | (unique == 1)).item())


# ─────────────── SigmoidGEN ───────────────


def test_sigmoid_gen_forward_shape(response_input):
    from antenna.models.generators import SigmoidGEN

    model = SigmoidGEN()
    out = model(response_input)

    # 輸出應展平回 pattern 的 2D shape
    expected_shape = AntennaPattern.size(flatten=False)
    assert out.shape == expected_shape


def test_sigmoid_gen_binarization_outputs_zero_one(response_input):
    """tau 極小時 STE 應輸出純 {0, 1}。"""
    from antenna.models.generators import SigmoidGEN

    model = SigmoidGEN()
    out = model(response_input, tau=1e-4)
    assert _is_binary(out)


# ─────────────── OldGEN ───────────────


def test_old_gen_forward_shape(response_input):
    from antenna.models.generators import OldGEN

    model = OldGEN()
    out = model(response_input)

    # OldGEN 將 (8*8=64,) flatten 後送入 fc_patch，輸出 (64,)
    assert out.shape == (AntennaPattern.size(flatten=True),)


def test_old_gen_outputs_zero_or_one(response_input):
    """sign_f → /2 + 0.5 應只會得到 {0, 1}。"""
    from antenna.models.generators import OldGEN

    model = OldGEN()
    out = model(response_input)
    assert _is_binary(out)


# ─────────────── GumbelSigmoidGEN ───────────────


def test_gumbel_sigmoid_gen_forward_shape(response_input):
    from antenna.models.generators import GumbelSigmoidGEN

    model = GumbelSigmoidGEN()
    out = model(response_input)
    # GumbelSigmoid 為機率近似輸出，shape 會跟 logits 一樣 (pattern_size,)
    assert out.shape == (AntennaPattern.size(flatten=True),)


def test_gumbel_sigmoid_gen_output_in_unit_interval(response_input):
    from antenna.models.generators import GumbelSigmoidGEN

    model = GumbelSigmoidGEN()
    out = model(response_input)
    # Sigmoid 輸出 [0, 1]
    assert torch.all((out >= 0.0) & (out <= 1.0)).item()


def test_gumbel_sigmoid_gen_binarize_method(response_input):
    """`binarize()` 必須回傳 {0, 1} 的 AntennaPattern。"""
    from antenna.models.generators import GumbelSigmoidGEN

    model = GumbelSigmoidGEN()
    _ = model(response_input)  # 先 forward 填入 self.logits
    pattern = model.binarize()
    assert isinstance(pattern, AntennaPattern)
    assert _is_binary(pattern.merge())


def test_gumbel_sigmoid_gen_tau_history_grows(response_input):
    from antenna.models.generators import GumbelSigmoidGEN

    model = GumbelSigmoidGEN()
    assert len(model.tau_history) == 0
    model(response_input)
    model(response_input)
    assert len(model.tau_history) == 2


# ─────────────── Device 一致性 ───────────────


@pytest.mark.parametrize(
    "generator_name",
    ["SigmoidGEN", "OldGEN", "GumbelSigmoidGEN"],
)
def test_generator_device_matches_config(generator_name, response_input):
    """模型參數應放在 config.device 上。"""
    import antenna.models.generators as gens

    model_cls = getattr(gens, generator_name)
    model = model_cls()

    # 所有參數應在同一 device
    devices = {p.device for p in model.parameters()}
    assert len(devices) == 1
    assert next(iter(devices)).type == config.device.type


@pytest.mark.parametrize(
    "generator_name",
    ["SigmoidGEN", "OldGEN", "GumbelSigmoidGEN"],
)
def test_generator_forward_device_matches_input(generator_name, response_input):
    """forward 輸出應與輸入 (CPU) 同 device。"""
    import antenna.models.generators as gens

    model_cls = getattr(gens, generator_name)
    model = model_cls()
    out = model(response_input)
    assert out.device.type == response_input.device.type


# ─────────────── SPGEN ───────────────


def test_sp_gen_forward_shape():
    """SPGEN 以子 pattern table 拼合出 grid_size * patern_size 的大圖。"""
    from antenna.models.generators import SPGEN

    # 4 種 2x2 小圖案
    pattern_table = (
        [[0, 0], [0, 0]],
        [[1, 0], [0, 1]],
        [[0, 1], [1, 0]],
        [[1, 1], [1, 1]],
    )
    size = 8  # 8 // 2 = 4 grid
    model = SPGEN(pattern_table, size=size)

    out = model(tau=1.0, hard=True)
    # [batch=1, size, size]
    assert out.shape == (1, size, size)


def test_sp_gen_hard_output_is_binary():
    """hard=True 時輸出應為 {0, 1}（小 pattern table 都是 0/1）。"""
    from antenna.models.generators import SPGEN

    pattern_table = (
        [[0, 0], [0, 0]],
        [[1, 1], [1, 1]],
    )
    model = SPGEN(pattern_table, size=4)
    out = model(tau=0.5, hard=True)
    assert _is_binary(out)


# ─────────────── CVAE ───────────────


def test_cvae_forward_shape():
    """CVAE forward 回傳 (recon_logits, mu, logvar)，三者皆應帶批次維度。"""
    from antenna.models.generators import CVAE

    latent_dim = 8
    batch = 4
    model = CVAE(latent_dim=latent_dim)

    pattern = torch.randn(batch, AntennaPattern.size(flatten=True), device=config.device)
    response = torch.randn(batch, AntennaResponse.size(flatten=True), device=config.device)

    recon, mu, logvar = model(pattern, response)
    assert recon.shape == (batch, AntennaPattern.size(flatten=True))
    assert mu.shape == (batch, latent_dim)
    assert logvar.shape == (batch, latent_dim)


def test_cvae_generate_shape():
    """generate() 以單一條件生成 n_samples 個 logits。"""
    from antenna.models.generators import CVAE

    latent_dim = 8
    n_samples = 3
    model = CVAE(latent_dim=latent_dim)

    response = torch.randn(1, AntennaResponse.size(flatten=True), device=config.device)
    logits = model.generate(response, n_samples=n_samples)
    assert logits.shape == (n_samples, AntennaPattern.size(flatten=True))


# ─────────────── re-export 完整性 ───────────────


def test_generators_reexport_all_names():
    import antenna.models.generators as gens

    expected = {
        "SigmoidGEN",
        "GumbelSigmoidGEN",
        "OldGEN",
        "SPGEN",
        "CVAE",
        "MirrorCVAE",
        "GradientEstimator",
    }
    assert expected.issubset(set(gens.__all__))
    for name in expected:
        assert hasattr(gens, name), f"antenna.models.generators 缺少 {name}"


def test_top_level_models_reexport():
    """確認 `antenna.models` top-level 仍可取得各 generator。"""
    from antenna.models import CVAE, SPGEN, GumbelSigmoidGEN, OldGEN, SigmoidGEN

    for obj in (SigmoidGEN, GumbelSigmoidGEN, OldGEN, SPGEN, CVAE):
        assert obj is not None

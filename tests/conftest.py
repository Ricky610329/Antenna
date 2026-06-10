"""
特徵化測試的共用設定（重構安全網）。

目的：在重構前先「pin 住」核心純函式的現有行為。設定方式刻意對齊
train_single.py 的全域註冊（座標、響應標籤、目標、loss hook），讓測試
能直接驗證 criterion / 正則化 / 二值化等行為。

不觸碰 HFSS、不掛 NAS、不寫結果夾——純函式層級、確定性。
"""
import os
import json
import pytest
import torch

from antenna.utils import config
config.device = "cpu"  # 強制 CPU，確保確定性

from antenna import AntennaPattern, AntennaResponse, TargetResponse
from antenna.patch import custom_loss_minmax


@pytest.fixture(scope="session", autouse=True)
def _setup_globals():
    """安裝單埠 (S11+Gain) 的響應規格：建 spec → AntennaResponse.use(spec) 原子安裝。"""
    config.device = "cpu"
    AntennaPattern.setDefaultCoordinate((0, 25, 0, 25))
    spec = TargetResponse(labels=("S11", "Gain"), x="n257")
    s11 = spec(0, -10, (5, 0, 7, 0, 5), label="S11", add=True)
    spec.register_loss_fn("S11", custom_loss_minmax, target=s11, method="low")
    gain = spec(-19, 4, (5, 0, 7, 0, 5), label="Gain", add=True)
    spec.register_loss_fn("Gain", custom_loss_minmax, target=gain, method="high")
    AntennaResponse.use(spec)
    yield


# ---- golden 快照機制（approval testing）----------------------------------
# 第一次執行：golden.json 不存在 → 自動把目前數值寫入（捕捉現況）。
# 之後執行（重構後）：與 golden 比對，數值漂移即 fail。
#
# 容差雙軌制（見 docs/development.md 第 3 節）：
#   本機 = 精檢 (絕對 1e-4)，是數值真相的來源。
#   CI   = 粗檢 (相對 1%)：GitHub runner 的 CPU 與開發機不同，SIMD/MKL kernel
#   路徑差異造成跨硬體浮點漂移 (特徵值分解最敏感、迴圈逐 epoch 放大，實測 ~0.4%)。
_CI = bool(os.environ.get("CI"))  # GitHub Actions 自帶 CI=true


def golden_tol(base, tol=1e-4):
    """本機回傳絕對容差；CI 放寬為 max(絕對, 相對 1%)。"""
    return max(tol, 0.01 * abs(base)) if _CI else tol


_GOLDEN = os.path.join(os.path.dirname(__file__), "golden.json")


@pytest.fixture(scope="session")
def golden():
    data = json.load(open(_GOLDEN, encoding="utf-8")) if os.path.exists(_GOLDEN) else {}
    rec = dict(data)

    class _Golden:
        def check(self, key, value, tol=1e-4):
            v = float(value)
            if key in data:
                eff = golden_tol(data[key], tol)
                assert abs(v - data[key]) <= eff, (
                    f"[golden drift] {key}: 現在={v:.8g} vs 基準={data[key]:.8g} "
                    f"(Δ={abs(v - data[key]):.2e} > tol={eff:.2e})"
                )
            rec[key] = v  # 通過比對（或新鍵）才記錄

    g = _Golden()
    yield g
    # session 結束時寫回（新鍵會被補進 golden.json）
    json.dump(rec, open(_GOLDEN, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

"""
tests/test_dual_simulator.py — DualPortSimulator 的三層護欄（不啟動 HFSS COM）。

`PatchSimulator.__init__` 只建目錄、不連 COM，故求解/掃頻參數化可在開發機直接驗。
真正的建模/求解（`__call__`）需要 HFSS，不在單元測試範圍。

護欄三件（2026-08-10 dual 開線施工包 A）：
  a. 三條 S 參數回傳前一律 `align_curve` 對齊 17 點 + `assert len == 17`（原本 iloc[0:17] 截取、
     卻回傳 iloc[:, 1] 全長 → 點數 ≠ 17 時靜默錯位）。
  b. 求解/掃頻參數化（sweep_type / max_delta_s / max_passes / min_passes / min_converged）。
  c. 匯出前刪同名舊 CSV（跨批共用工作目錄時，殘留檔會被當成本次結果讀回＝無聲污染）。
"""
import inspect

from antenna.patch.patch_simulator import dual_port, single_port
from antenna.patch.patch_simulator.dual_port import DualPortSimulator


def test_dual_defaults_match_harvest_dual_settings(tmp_path):
    """預設值必須＝本檔歷來寫死的值：**換掉等於換分佈**（harvest_dual 一萬筆全是這組設定）。"""
    sim = DualPortSimulator(str(tmp_path))
    assert sim.sweep_type == "Fast"          #! 不是 single 的 Interpolating —— 兩者重建演算法不同
    assert sim.max_delta_s == 0.02
    assert sim.max_passes == 6
    assert sim.min_passes == 5
    assert sim.min_converged == 5


def test_dual_solver_params_overridable(tmp_path):
    """批次線（hfss_setup.json）要能覆蓋求解/掃頻設定，且型別被正規化。"""
    sim = DualPortSimulator(str(tmp_path), sweep_type="Discrete", max_delta_s=0.005,
                            max_passes=20, min_passes=3, min_converged=2)
    assert sim.sweep_type == "Discrete"
    assert sim.max_delta_s == 0.005 and isinstance(sim.max_delta_s, float)
    assert (sim.max_passes, sim.min_passes, sim.min_converged) == (20, 3, 2)


def test_dual_solver_params_are_used_not_hardcoded():
    """COM 呼叫必須讀 self.*（防「加了參數但 InsertSetup 仍寫死」的半套修法）。"""
    src = inspect.getsource(DualPortSimulator.__call__)
    for attr in ("self.max_delta_s", "self.max_passes", "self.min_passes",
                 "self.min_converged", "self.sweep_type"):
        assert attr in src, f"{attr} 沒被 __call__ 用到 —— 參數化只做了一半"
    assert '"MaxDeltaS:=", 0.02' not in src and '"Type:=", "Fast"' not in src


def test_dual_shares_single_align_curve():
    """頻率對位共用 single 的實作（兩份實作＝兩把尺，遲早漂）。"""
    assert dual_port.align_curve is single_port.align_curve


def test_dual_call_aligns_and_asserts_17_points():
    """回傳前必須 align_curve 三條 + 斷言長度 17，且**回傳的就是對齊後的那份**。

    舊 bug 的形狀正是「上面截了 17 點、下面回傳全長」——所以光有 align_curve 不夠，
    要釘死 `_result` 的三個值都來自 `*_vals`。
    """
    src = inspect.getsource(DualPortSimulator.__call__)
    assert src.count("align_curve(") == 3
    for label in ("S11", "S21", "S22"):
        assert f"assert len({label}_vals) == 17" in src
        assert f"'{label}': tensor({label}_vals" in src


def test_dual_deletes_stale_csv_before_export():
    """匯出前刪三個同名舊檔（無聲污染防線，照 single_port 的理由）。"""
    src = inspect.getsource(DualPortSimulator.__call__)
    assert "_stale.unlink()" in src
    export_at = src.index("ExportToFile")
    assert src.index("_stale.unlink()") < export_at, "刪舊檔必須在匯出之前"
    for label in ("S11", "S21", "S22"):
        assert f'Sparameter_{{self.num}}_{label}.csv")' in src[:export_at]

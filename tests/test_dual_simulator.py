"""
tests/test_dual_simulator.py — DualPortSimulator 的三層護欄（不啟動 HFSS COM）。

`PatchSimulator.__init__` 只建目錄、不連 COM，故求解/掃頻參數化可在開發機直接驗。
真正的建模/求解（`__call__`）需要 HFSS，不在單元測試範圍。

護欄三件（2026-08-10 dual 開線施工包 A）：
  a. 三條 S 參數回傳前一律 `align_curve` 對齊 17 點 + `assert len == 17`（原本 iloc[0:17] 截取、
     卻回傳 iloc[:, 1] 全長 → 點數 ≠ 17 時靜默錯位）。
  b. 求解/掃頻參數化（sweep_type / max_delta_s / max_passes / min_passes / min_converged）。
  c. 匯出前刪同名舊 CSV（跨批共用工作目錄時，殘留檔會被當成本次結果讀回＝無聲污染）。

R60 亞像素耦合縫（`slot_spec`）另加一段：純函式 `slot_boxes` 的座標/驗證契約，
外加一組**假 COM** 端到端幾何測（見 `_FakeEditor`）——挖縫要對「Unite 後還活著的物件」下刀，
存活名單的推算是本次唯一有 off-by-one 風險的邏輯，字串比對測不出來。
"""
import inspect

import numpy as np
import pytest
import torch

from antenna.patch.patch_simulator import dual_port, single_port
from antenna.patch.patch_simulator.dual_port import DualPortSimulator, slot_boxes


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


# ---------------------------------------------------------------- R60 亞像素耦合縫（slot_spec）
def test_slot_boxes_axis_mapping_row_is_x_col_is_y():
    """★座標對應（本施工包最容易搞反的一件事，以 code 為準）：`pixel_matrix[r][c]` 在
    `__call__` 裡是 XPosition 用第一維、YPosition 用第二維 ⇒ **列 r 走 X、欄 c 走 Y**。
    故「第 r 列的縫」＝厚度落在 X、沿 Y 延伸；搞反的話縫會開成縱向（與兩埠直通路平行
    ＝根本不切耦合路徑），而且 S 參數看起來仍然「合理」——所以要用測試釘死。"""
    b, = slot_boxes([{"rows": [11], "cols": [10, 16], "width_mm": 0.05}], 25)
    assert b["row"] == 11 and (b["c0"], b["c1"]) == (10, 16)
    assert b["dx"] == pytest.approx(0.05)                      # X＝厚度＝縫寬
    assert b["x0"] == pytest.approx(11.5 * 0.2 - 0.025)        # 對準第 11 列的中心線
    assert b["y0"] == pytest.approx(10 * 0.2)                  # Y＝延伸方向
    assert b["dy"] == pytest.approx(7 * 0.2)                   # 跨 c0..c1 **含**（10-16＝7 欄）


def test_ports_are_separated_along_x_axis():
    """縫「厚度落在 X」的物理依據：兩個 port 的 IntLine 只差在 X（+12.5 / −7.5mm）、Y 都是
    2.5mm ⇒ 兩埠沿 X 分居兩端 ⇒ 直通路沿 X ⇒ 切耦合的縫必須垂直於 X。
    旁證：`dedust.dual_pads` 的饋墊 q[0:5,10:15] / q[20:25,10:15]——列→X 給出 0-1mm 與
    4-5mm 兩端、欄→Y 給出 2-3mm 正中，恰好對上 IntLine 的 y=2.5mm。"""
    src = inspect.getsource(DualPortSimulator.__call__)
    assert '"12.5mm", "2.5mm"' in src and '"-7.5mm", "2.5mm"' in src
    b, = slot_boxes([{"rows": [12], "cols": [10, 14], "width_mm": 0.05}], 25)
    assert b["y0"] == pytest.approx(2.0) and b["y0"] + b["dy"] == pytest.approx(3.0)
    assert b["dx"] < b["dy"]


def test_slot_width_converges_to_full_row_clear():
    """括號自證的幾何前提（round-60 §1-1）：w → 像素邊長 時縫盒收斂到「整列清空」的那條帶
    ⇒ 縫寬掃描的上端能對上已知的全切值。"""
    b, = slot_boxes([{"rows": [12], "cols": [0, 24], "width_mm": 0.2 - 1e-9}], 25)
    assert b["x0"] == pytest.approx(12 * 0.2, abs=1e-6)               # 帶下緣＝第 12 列下緣
    assert b["x0"] + b["dx"] == pytest.approx(13 * 0.2, abs=1e-6)     # 帶上緣＝第 13 列下緣
    assert b["y0"] == 0.0 and b["dy"] == pytest.approx(5.0)           # 整列＝跨滿 25 欄＝5mm


def test_slot_boxes_scales_with_pixel_count():
    """格距是算出來的（5mm / pixel_count），不是寫死 0.2——換域時縫位不會靜默錯位。"""
    b, = slot_boxes([{"rows": [4], "cols": [1, 2], "width_mm": 0.05}], 50)
    assert b["x0"] == pytest.approx(4.5 * 0.1 - 0.025) and b["dy"] == pytest.approx(0.2)


def test_slot_boxes_expands_rows_and_does_not_mirror():
    """一項可帶多列（鏡像列由呼叫端自己寫）；模擬器**不自動鏡像**——偷偷幫忙鏡像會讓
    「不對稱縫」這種構型永遠做不出來，也會讓 manifest 記的東西與實際幾何對不上。"""
    out = slot_boxes([{"rows": [7, 17], "cols": [10, 12], "width_mm": 0.05},
                      {"rows": [12], "cols": [0, 24], "width_mm": 0.1}], 25)
    assert [b["row"] for b in out] == [7, 17, 12]
    assert [b["w"] for b in out] == [0.05, 0.05, 0.1]


@pytest.mark.parametrize("spec", [
    {"rows": [11], "cols": [10, 16], "width_mm": 0.0},      # 無縫請不要給 slot_spec
    {"rows": [11], "cols": [10, 16], "width_mm": -0.05},
    {"rows": [11], "cols": [10, 16], "width_mm": 0.2},      # ≥ 像素邊長：單像素體會被挖成空物件
    {"rows": [25], "cols": [10, 16], "width_mm": 0.05},     # 列越界
    {"rows": [11], "cols": [16, 10], "width_mm": 0.05},     # c1 < c0
    {"rows": [11], "cols": [10, 25], "width_mm": 0.05},     # 欄越界
    {"rows": [11], "cols": [10], "width_mm": 0.05},         # cols 不是閉區間對
    {"rows": [], "cols": [10, 16], "width_mm": 0.05},       # 空 rows＝什麼都沒挖
    {"row": 11, "cols": [10, 16], "width_mm": 0.05},        # 打錯鍵（rows 少 s）＝靜默不挖，必須擋
])
def test_slot_boxes_rejects_bad_spec(spec):
    """壞規格一律 ValueError：這些錯法的共同特徵是**幾何上會靜默變成沒挖或挖歪**，
    而 HFSS 照樣跑得完、S 參數照樣「合理」——只能在進 COM 前擋。"""
    with pytest.raises(ValueError):
        slot_boxes([spec], 25)


def test_slot_boxes_rejects_bare_dict():
    """`hfss_setup.json` 手寫成單一 dict（漏了外層 list）是最容易犯的格式錯。"""
    with pytest.raises(ValueError):
        slot_boxes({"rows": [11], "cols": [10, 16], "width_mm": 0.05}, 25)


def test_dual_slot_spec_defaults_to_none(tmp_path):
    """向後相容：不給就是 None＝現行幾何（既有 dual 真值全是這條路徑跑出來的）。"""
    assert DualPortSimulator(str(tmp_path)).slot_spec is None


def test_dual_slot_spec_stored_and_validated_at_construct(tmp_path):
    """批次線把 hfss_setup.json 的鍵直接 pass-through 給建構子 → 規格錯要在**開 HFSS 之前**
    就炸（不然是跑到一半才發現，整夾白燒）。"""
    spec = [{"rows": [11, 13], "cols": [10, 16], "width_mm": 0.05}]
    assert DualPortSimulator(str(tmp_path), slot_spec=spec).slot_spec == spec
    with pytest.raises(ValueError):
        DualPortSimulator(str(tmp_path), slot_spec=[{"rows": [11], "cols": [10, 16], "width_mm": 0.2}])


# ---- 假 COM：驗「對哪些物件下刀」（存活名單推算是唯一有 off-by-one 風險的邏輯） ----
class _FakeModule:
    def __getattr__(self, name):
        return lambda *a, **k: None


class _FakeEditor:
    """假 3D Modeler：只實作要驗的三個動詞，其餘吞掉。CreateBox 回傳「HFSS 會給的名字」
    ——同 base 重複時加序號後綴（第 0 個 Patch、之後 Patch_1…），與 dual_port 推算
    Patch_<n> 的規則同一套。"""

    def __init__(self):
        self.boxes, self.unites, self.subtracts, self.verbs = [], [], [], []

    def CreateBox(self, params, attrs):
        base = attrs[attrs.index("Name:=") + 1]
        n = sum(1 for b in self.boxes if b["base"] == base)
        name = base if n == 0 else f"{base}_{n}"
        self.boxes.append(dict(base=base, name=name, **dict(zip(params[1::2], params[2::2]))))
        self.verbs.append("CreateBox")
        return name

    def Unite(self, sel, params):
        self.unites.append(sel[sel.index("Selections:=") + 1])
        self.verbs.append("Unite")

    def Subtract(self, sel, params):
        self.subtracts.append((sel[sel.index("Blank Parts:=") + 1],
                               sel[sel.index("Tool Parts:=") + 1]))
        self.verbs.append("Subtract")

    def __getattr__(self, name):
        return lambda *a, **k: None


class _FakeDesign:
    def __init__(self, editor):
        self._editor = editor

    def SetActiveEditor(self, _name):
        return self._editor

    def GetModule(self, _name):
        return _FakeModule()

    def __getattr__(self, name):
        return lambda *a, **k: None


def _run_geometry(tmp_path, monkeypatch, mat, slot_spec=None):
    """跑一次 `__call__`（HFSS 全假、S 參數走假 CSV），回傳假 editor 的動詞紀錄。"""
    import pandas as pd
    monkeypatch.setattr(dual_port, "read_csv",
                        lambda *_a, **_k: pd.DataFrame({0: np.linspace(24, 32, 17), 1: np.zeros(17)}))
    sim = DualPortSimulator(str(tmp_path), slot_spec=slot_spec)
    ed = _FakeEditor()
    sim.oDesign = _FakeDesign(ed)
    sim.num = 7
    out = sim(torch.tensor(mat.astype("float32")).reshape(-1))
    assert set(out) == {"S11", "S21", "S22"}
    return ed


def _mat_3cols():
    """手算得出存活名單的最小圖形（三種欄型各一）：
    欄3 兩格＝撞 "Patch_1" 邊界分支（聯不了）／欄7 三格＝真的 Unite／欄9 一格＝單顆不聯。"""
    m = np.zeros((25, 25), bool)
    m[2, 3] = m[4, 3] = True
    m[0, 7] = m[1, 7] = m[5, 7] = True
    m[8, 9] = True
    return m


def test_slot_subtract_targets_surviving_bodies(tmp_path, monkeypatch):
    """★挖縫要對「逐欄 Unite 之後還活著的物件」下刀。被吃掉的 Patch_3/Patch_4 若混進
    Blank Parts，HFSS 會在選不到名稱時整筆炸掉（或更糟：選到別的東西）。"""
    ed = _run_geometry(tmp_path, monkeypatch, _mat_3cols(),
                       slot_spec=[{"rows": [4], "cols": [3, 7], "width_mm": 0.05}])
    # 建立順序→HFSS 名：欄3(列2,4)=Patch,Patch_1 → 欄7(列0,1,5)=Patch_2..4 → 欄9(列8)=Patch_5
    assert ed.unites == ["Patch_2,Patch_3,Patch_4"]
    (blank, tool), = ed.subtracts
    assert blank == "Patch,Patch_1,Patch_2,Patch_5"     # 存活體全上；Patch_3/4 已被吃掉
    assert tool == "Slot_1"
    assert "feedline1" not in blank and "feedline2" not in blank   # dual 的貼片與饋線從不合體
    order = ed.verbs
    assert order.index("Subtract") > max(i for i, v in enumerate(order) if v == "Unite")


def test_slot_box_geometry_and_z_clearance(tmp_path, monkeypatch):
    """縫盒座標＝slot_boxes 的結果原封不動進 CreateBox；Z 上下各留 0.01mm 餘裕貫穿銅層
    （共面布林是 HFSS 的經典失敗點）。"""
    ed = _run_geometry(tmp_path, monkeypatch, _mat_3cols(),
                       slot_spec=[{"rows": [4], "cols": [3, 7], "width_mm": 0.05}])
    slot, = [b for b in ed.boxes if b["base"] == "Slot_1"]

    def _mm(v):
        return float(v[:-2])

    assert _mm(slot["XPosition:="]) == pytest.approx(4.5 * 0.2 - 0.025)   # 第 4 列中心線 ± w/2
    assert _mm(slot["XSize:="]) == pytest.approx(0.05)
    assert _mm(slot["YPosition:="]) == pytest.approx(0.6)                 # 欄 3 起
    assert _mm(slot["YSize:="]) == pytest.approx(1.0)                     # 欄 3-7 含＝5 欄
    assert slot["ZPosition:="] == "0.498mm" and slot["ZSize:="] == "CooperH+0.02mm"


def test_slot_spec_none_leaves_geometry_untouched(tmp_path, monkeypatch):
    """向後相容鐵則：slot_spec=None 的路徑一個 COM 動詞都不能多——Subtract 零次、
    CreateBox 只有像素盒、Unite 選取字串與有縫時**逐字相同**。"""
    m = _mat_3cols()
    base = _run_geometry(tmp_path, monkeypatch, m)
    withslot = _run_geometry(tmp_path, monkeypatch, m,
                             slot_spec=[{"rows": [4], "cols": [3, 7], "width_mm": 0.05}])
    assert base.subtracts == []
    assert len(base.boxes) == int(m.sum()) == len(withslot.boxes) - 1
    assert base.unites == withslot.unites
    assert [b["name"] for b in base.boxes] == [b["name"] for b in withslot.boxes[:-1]]


def test_slot_spec_skips_when_pattern_has_no_metal(tmp_path, monkeypatch):
    """全空 pattern：沒有金屬可挖 → 連縫盒都不建（免在模型裡留一塊懸空銅片＝假天線）。"""
    ed = _run_geometry(tmp_path, monkeypatch, np.zeros((25, 25), bool),
                       slot_spec=[{"rows": [4], "cols": [3, 7], "width_mm": 0.05}])
    assert ed.boxes == [] and ed.subtracts == []

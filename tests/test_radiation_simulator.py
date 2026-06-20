"""
SinglePortRadSimulator 的「無 HFSS」結構測試。

方向圖的「實際萃取」需要真實 HFSS COM (正式機)，本機無法測 —— 由 script/verify_radiation.py
在正式機驗證。這裡只 pin 住「不搞壞」的不變式 (純結構、不開 COM、不掛 NAS)：
  - 是 SinglePortSimulator 的子類 (管線/build 視同單埠模擬器)。
  - 建構只建目錄、不開 COM (沒有 oDesktop)。
  - 萃取參數 (phi 切面 / 物理量) 符合操作截圖配方。
  - last_radiation 預設 None (尚未跑過)。
"""


def test_rad_simulator_is_single_port_subclass():
    from antenna.patch import SinglePortRadSimulator, SinglePortSimulator
    assert issubclass(SinglePortRadSimulator, SinglePortSimulator)


def test_rad_simulator_constructs_without_com(tmp_path):
    """建構只建目錄、不連 HFSS (無 oDesktop)；回傳通道與父類別一致。"""
    from antenna.patch import SinglePortRadSimulator, SinglePortSimulator
    sim = SinglePortRadSimulator(record_path=str(tmp_path))

    assert isinstance(sim, SinglePortSimulator)          # 管線視同單埠模擬器
    assert not hasattr(sim, "oDesktop")                  # 建構不開 COM (open() 才連)
    assert sim.last_radiation is None                    # 尚未萃取
    assert sim.path_result.exists()                      # HFSS/result 目錄已建
    assert sim.path_project.exists()


def test_rad_simulator_recipe_matches_screenshots():
    """方向圖萃取配方 (phi 切面 / 物理量 / 頻率) 鎖定，避免日後誤改。"""
    from antenna.patch import SinglePortRadSimulator
    assert SinglePortRadSimulator.RAD_PHIS == (0, 90)    # E-plane / H-plane 兩切面
    assert SinglePortRadSimulator.RAD_FREQ == "28GHz"    # 設計中心頻
    assert SinglePortRadSimulator.RAD_SPHERE == "3D"     # 沿用父類別 3D 無限球面
    # __call__ 確實被覆寫 (在子類別自己身上，而非繼承父類別的)
    assert "SinglePortRadSimulator" in SinglePortRadSimulator.__call__.__qualname__


def test_parse_rad_csv_reads_theta_column_not_freq():
    """回歸：HFSS CSV 欄位 = [Freq, Phi, Theta, dB(GainTotal)]，theta 在第 2 欄、不是第 0 欄。
    (踩過的雷：寫死 iloc[:,0] 把 Freq(全 28) 當成 theta → 方向圖 x 軸塌成一點、整個畫錯。)"""
    import pandas as pd
    from antenna.patch.patch_simulator.single_port_rad import _parse_rad_csv
    df = pd.DataFrame({
        "Freq [GHz]": [28, 28, 28],
        "Phi [deg]": [0, 0, 0],
        "Theta [deg]": [-180, 0, 180],
        "dB(GainTotal) []": [-12.0, 3.5, -11.0],
    })
    theta, gain = _parse_rad_csv(df)
    assert list(theta) == [-180.0, 0.0, 180.0]           # 抓到 Theta 欄 (非 Freq 的全 28)
    assert list(gain) == [-12.0, 3.5, -11.0]

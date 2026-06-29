"""
tests/test_hook_remind.py — config-sync 提醒 hook 的純函式測試。

只測 reminder_for(file_path)：命中 configs/*.yaml → 給提醒;其餘 → None(零打擾)。
(I/O / stdin 解析在 main(),不在此測;hook 失敗一律靜默不阻擋。)
"""
from script.hooks.remind_config_sync import reminder_for


def test_config_yaml_triggers_reminder():
    """改 configs/*.yaml → 回提醒(含 README/log/ONGOING 同步檢查)。"""
    for p in ("configs/single_x.yaml",
              "C:/Users/x/Antenna/configs/single_r2_ens_harvest.yaml",
              r"C:\Users\x\Antenna\configs\dual_base.yaml"):       # 反斜線路徑也要命中
        msg = reminder_for(p)
        assert msg and "configs/README.md" in msg, p


def test_non_config_is_silent():
    """非 configs/*.yaml(程式碼/文件/其他)→ None,不打擾。"""
    for p in ("antenna/training.py", "script/round_report.py",
              "docs/log/round-01.md", "configs/README.md", ""):
        assert reminder_for(p) is None, p

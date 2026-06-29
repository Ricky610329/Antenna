# -*- coding: utf-8 -*-
"""
PostToolUse hook：Edit/Write 到 `configs/*.yaml` 時,回注一段提醒,確保「改了實驗 config
卻忘了同步追蹤」不會發生(README 全集 / docs/log 研究日誌 / ONGOING live 板)。

機制:讀 stdin 的 hook JSON → 取 tool_input.file_path → 命中 configs/*.yaml 才輸出
additionalContext;否則靜默 exit 0(零打擾、非阻擋)。純 stdlib,任何 python3 可跑。
"""
import json
import re
import sys

_REMINDER = (
    "你改了 configs/*.yaml(實驗定義)。追蹤同步檢查:"
    "① configs/README.md 對照表同步了嗎(硬規則)? "
    "② 這是新實驗 round → 在 docs/log/ 開 round-NN(用 _TEMPLATE)+ ONGOING.md 加一行 🔵 指向它? "
    "③ 改的是現有 round 的臂 → 更新對應 round 檔與 ONGOING?"
)


def reminder_for(file_path):
    """命中 configs/*.yaml → 回傳提醒字串;否則 None。(可單元測試的純函式)"""
    fp = (file_path or "").replace("\\", "/")
    return _REMINDER if re.search(r"configs/.*\.yaml$", fp) else None


def main():
    try:
        data = json.load(sys.stdin)
    except Exception:
        return                                  # 壞 JSON / 無輸入 → 靜默,絕不阻擋工具
    msg = reminder_for((data.get("tool_input") or {}).get("file_path", ""))
    if msg:
        print(json.dumps({"hookSpecificOutput": {
            "hookEventName": "PostToolUse", "additionalContext": msg}}))


if __name__ == "__main__":
    main()

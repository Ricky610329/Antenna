"""備份對帳：逐類比對 NAS 與本機備份的筆數，任何一類短少就 exit 1。

用法（repo 根目錄）：
    python .claude/skills/nas-backup/verify.py
    python .claude/skills/nas-backup/verify.py --nas <NAS路徑> --local <本機路徑>

只讀、不寫、不修任何東西。
"""
import argparse
import os
import sys

NAS_DEFAULT = r"T:\碩二_鄒穎麒's\antenna"
LOCAL_DEFAULT = r"C:\Users\Ricky\antenna_nas_backup"


def count_ext(d, ext):
    """數 d 底下（不遞迴）副檔名為 ext 的檔案數；d 不存在回 0。"""
    try:
        return sum(1 for e in os.scandir(d) if e.is_file() and e.name.endswith(ext))
    except OSError:
        return 0


def count_ext_deep(d, ext):
    """遞迴版，給 result/ 這種深樹用。"""
    n = 0
    for _, _, files in os.walk(d):
        n += sum(1 for f in files if f.endswith(ext))
    return n


def tally(root):
    """把一棵備份樹數成各類筆數。NAS 與本機用同一把尺。"""
    t = {"我們產的批次線": 0, "學長收割 harvest_*": 0, "原始方向圖 rad/": 0,
         "輸入 pattern (_input)": 0, "results.json": 0,
         "result/ 的 online 量測": 0, "result/ 的 patterns": 0, "SM 權重 .pth": 0}

    dataset = os.path.join(root, "dataset")
    try:
        entries = sorted(os.scandir(dataset), key=lambda e: e.name)
    except OSError:
        entries = []
    for e in entries:
        if e.is_file():
            if e.name.endswith(".pth"):
                t["SM 權重 .pth"] += 1
            continue
        n = count_ext(e.path, ".pt")
        if e.name.endswith("_input"):
            t["輸入 pattern (_input)"] += n
        elif e.name.startswith("harvest_"):
            t["學長收割 harvest_*"] += n
        else:
            t["我們產的批次線"] += n
        t["原始方向圖 rad/"] += count_ext(os.path.join(e.path, "rad"), ".pt")
        t["results.json"] += 1 if os.path.exists(os.path.join(e.path, "results.json")) else 0

    result = os.path.join(root, "result")
    try:
        runs = [e for e in os.scandir(result) if e.is_dir()]
    except OSError:
        runs = []
    for r in runs:
        t["result/ 的 online 量測"] += count_ext(os.path.join(r.path, "online"), ".pt")
        t["result/ 的 patterns"] += count_ext(os.path.join(r.path, "patterns"), ".pt")
        t["SM 權重 .pth"] += count_ext_deep(os.path.join(r.path, "checkpoint"), ".pth")
    return t


def main():
    #! Windows 主控台預設 cp950，直接 print 非中日韓字元（✓/✗）會 UnicodeEncodeError 炸掉整個對帳。
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--nas", default=NAS_DEFAULT)
    ap.add_argument("--local", default=LOCAL_DEFAULT)
    args = ap.parse_args()

    print(f"NAS  : {args.nas}\n本機 : {args.local}\n")
    nas, loc = tally(args.nas), tally(args.local)

    print(f"{'類別':<24}{'NAS':>10}{'本機':>10}{'差':>9}")
    print("-" * 55)
    short = []
    for k in nas:
        d = loc[k] - nas[k]
        flag = "" if d >= 0 else "  ← 短少"
        if d < 0:
            short.append(k)
        print(f"{k:<24}{nas[k]:>10,}{loc[k]:>10,}{d:>+9,}{flag}")

    print()
    if short:
        print(f"✗ 對帳未過：{len(short)} 類短少（{'、'.join(short)}）")
        print("  → 重跑 SKILL.md §2 的 robocopy（/XO 只補缺的），再對帳一次。")
        return 1
    print("✓ 對帳通過：本機備份不少於 NAS。")
    return 0


if __name__ == "__main__":
    sys.exit(main())

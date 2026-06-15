"""
script/verify_radiation.py — 驗證 SinglePortRadSimulator 能否在真實 HFSS 上「把方向圖資料抓出來」。

用途 (正式機，需有 Ansys HFSS)：
    拿同一份 single config，用「方向圖版」單埠模擬器跑「一個」pattern，
    確認求解後 self.path_result 下出現 NN_patch_RadGain_{num}_phi0.csv / _phi90.csv，
    且讀回的 theta / gain 形狀、數值合理。

    本機 (開發機, 無 HFSS) 不可跑；這支是給你在正式機驗證「資料抓得出來」用的。
    它「不碰」訓練核心、不改既有模擬器，只是直接實例化新模擬器跑一發。

跑法 (repo 根目錄)：
    python -m script.verify_radiation                                   # 用 configs/single_base.yaml + 預設置中方塊圖樣
    python -m script.verify_radiation --config configs/single_peak.yaml
    python -m script.verify_radiation --pattern some_pattern.pt         # 用自備的 25x25 二元圖樣 (.pt)
    python -m script.verify_radiation --out D:/rad_verify --num 0
"""
import argparse
from pathlib import Path

import torch

from antenna.patch import SinglePortRadSimulator
from antenna.training import load_config


def _build_pattern(pattern_path: str | None, pixel: int) -> torch.Tensor:
    """取得一張 (pixel, pixel) 的二元圖樣。

    有給 --pattern 就載入 (.pt)；否則用「置中實心方塊」當測試圖樣 ——
    這只是為了驗證資料抓得出來，不在意它是不是好天線。
    """
    if pattern_path:
        p = torch.load(pattern_path)
        p = torch.as_tensor(p).float().reshape(pixel, pixel)
        # 保險二值化 (模擬器要求純 0/1)
        return (p > 0.5).float()

    #? 預設：置中 ~1/3 邊長的實心方塊 (純 0/1)
    p = torch.zeros(pixel, pixel)
    lo, hi = pixel // 3, pixel - pixel // 3
    p[lo:hi, lo:hi] = 1.0
    return p


def main():
    ap = argparse.ArgumentParser(description="驗證方向圖萃取 (需正式機 HFSS)")
    ap.add_argument("--config", default="configs/single_base.yaml", help="同一份 single config")
    ap.add_argument("--out", default="_radiation_verify", help="模擬輸出根目錄 (CSV 會在 <out>/HFSS/result)")
    ap.add_argument("--pattern", default=None, help="選填：自備 .pt 二元圖樣路徑")
    ap.add_argument("--pixel", type=int, default=25, help="像素邊長 (預設 25)")
    ap.add_argument("--num", type=int, default=0, help="本回合編號 (組檔名用)")
    args = ap.parse_args()

    #* 用「同一份 config」確認這是 single 實驗 (方向圖萃取目前只做 single)。
    cfg = load_config(args.config)
    print(f"[config] name={cfg.name!r} port={cfg.port!r}")
    if cfg.port != "single":
        raise SystemExit(f"本驗證只支援 port=single，但 config 是 {cfg.port!r}")

    pattern = _build_pattern(args.pattern, args.pixel)
    print(f"[pattern] shape={tuple(pattern.shape)} 金屬像素數={int(pattern.sum())}")

    out = Path(args.out)
    sim = SinglePortRadSimulator(record_path=str(out))
    print(f"[sim] {sim}")
    print(f"[sim] CSV 將輸出到: {sim.path_result}")

    #* HFSS 生命週期：open 一次 → start/呼叫/end → quit。包 try/finally 確保關 HFSS。
    sim.open()
    try:
        sim.start(args.num)
        result = sim(pattern)
        elapsed = sim.end()
    finally:
        sim.quit()

    print(f"\n[result] S11/Gain keys = {list(result.keys())} (回傳與既有模擬器相同) 耗時≈{elapsed}s")

    #* 驗證方向圖資料：CSV 檔 + self.last_radiation。
    rad = sim.last_radiation
    csv0 = sim.path_result.joinpath(f"NN_patch_RadGain_{args.num}_phi0.csv")
    csv90 = sim.path_result.joinpath(f"NN_patch_RadGain_{args.num}_phi90.csv")

    ok = True
    print("\n=== 方向圖資料驗證 ===")
    for csv in (csv0, csv90):
        exists = csv.exists()
        ok &= exists
        print(f"  {'✓' if exists else '✗'} {csv}")

    if isinstance(rad, dict) and "error" in rad:
        ok = False
        print(f"  ✗ 萃取時發生錯誤: {rad['error']}")
    elif isinstance(rad, dict) and rad.get("theta") is not None:
        theta = rad["theta"]
        print(f"  ✓ theta 點數={theta.numel()} 範圍=[{theta.min():.0f}, {theta.max():.0f}] deg")
        for phi in SinglePortRadSimulator.RAD_PHIS:
            g = rad.get(f"phi{phi}")
            if g is not None:
                print(f"  ✓ phi={phi}° gain(dB): 點數={g.numel()} min={g.min():.2f} max={g.max():.2f}")
            else:
                ok = False
                print(f"  ✗ phi={phi}° 沒有資料")
    else:
        ok = False
        print("  ✗ self.last_radiation 沒有方向圖資料")

    print(f"\n{'✅ 通過：方向圖資料已抓出' if ok else '❌ 失敗：方向圖資料缺失'}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()

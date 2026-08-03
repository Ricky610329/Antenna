# -*- coding: utf-8 -*-
"""script/diffsim/run.py — diffsim 驅動：預測、擬合純量、仿射校準、gate 報數。

紀律（`docs/log/analysis-08-diffsim.md` §1）：
  - 調參/診斷 **只看 `dev`**；`val` 只在 gate 報數時看一次。
  - 仿射校準的係數只能在 `fit` 上擬（與 val/dev 不相交）；凍結尺永不進擬合。

用法（L1 腔模型）：
    python -m script.diffsim.run predict --split dev --n 200            # 跑 L1 存快取
    python -m script.diffsim.run scan    --split dev --n 200            # (er, Q) 網格掃 ρ
    python -m script.diffsim.run fitscan --n 200                        # 在 fit 分割選純量
    python -m script.diffsim.run gate1                                  # L1 gate：val 報數
用法（L2 MoM）：
    python -m script.diffsim.run l2cal                                  # 核的解析校準（不用 HFSS）
    python -m script.diffsim.run l2eval  --split dev --solver l3fl      # 跑 L2 報 ρ
    python -m script.diffsim.run l2fit   --steps 120                    # 擬核（只能 dcim）
    python -m script.diffsim.run gate2   --solver l3fl                  # L2 gate：val 報數
    python -m script.diffsim.run head    --model l2                     # 殘差頭
**`--solver` 選的是一整組物理設定**（核 + 埠 + 遠場），登記表在 `l2.SOLVERS`：
`dcim`（可擬核，特徵化快照守的那條）／`l3`（精確分層核 + 分層遠場 + 半屋頂埠）／
`l3fl`（真饋線 + 駐波 wave port，§37）／`l3fld`（再加對角連通，§45/§47 —— ⚠ 物理更對但選批崩掉，歸因中，暫非出貨值，見 §49.5）。
"""
import argparse
import hashlib
import os
import sys
import time

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from . import data as D                        # noqa: E402
from .eval import margins, rank_rho, report_rho  # noqa: E402
from .l2 import SOLVERS                        # noqa: E402

CACHE = os.path.join(D.CACHE_DIR, "pred")


def pick(idx, split_arr, which: str, n_per: int = None, strata=None, seed: int = 0):
    """挑樣本：回 index 陣列（每 stratum 取 n_per 筆，決定性）。"""
    strata = strata or ["clean", "neg", "senior", "frozen"]
    out = []
    for s in strata:
        m = np.where((split_arr == which) & (idx["stratum"] == s))[0]
        if n_per is not None and len(m) > n_per:
            #! 種子不可用 Python 的 hash()——它每個 process 都不同（PYTHONHASHSEED 隨機化），
            #  同一條指令跑兩次會抽到不同樣本。改用內容穩定的雜湊。
            tag = int(hashlib.sha1(s.encode()).hexdigest()[:8], 16)
            m = np.sort(np.random.default_rng(seed + tag).choice(m, n_per, replace=False))
        out.append(m)
    return np.concatenate(out) if out else np.array([], dtype=int)


def run_l1(idx, sel, *, er, q, n_modes=None, device="cpu", batch=16, dtype=None, **kw):
    from .l1 import CavityL1
    import torch
    m = CavityL1(n_modes=n_modes, er_eff=er, q=q, device=device,
                 dtype=dtype or torch.float64, **kw)
    t = time.time()
    pred = m.predict(idx["x"][sel].astype(np.float64), batch=batch)
    return pred, time.time() - t


def affine_fit(pred_fit, y_fit):
    """每頻點仿射 a·x+b（最小平方）。回 (a (34,), b (34,))。"""
    a = np.ones(34)
    b = np.zeros(34)
    for j in range(34):
        x, y = pred_fit[:, j], y_fit[:, j]
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 5 or np.std(x[ok]) < 1e-9:
            b[j] = np.mean(y[ok]) - np.mean(x[ok]) if ok.any() else 0.0
            continue
        A = np.stack([x[ok], np.ones(ok.sum())], 1)
        sol, *_ = np.linalg.lstsq(A, y[ok], rcond=None)
        a[j], b[j] = sol
    return a, b


def affine_apply(pred, a, b):
    return pred * a[None, :] + b[None, :]


def _cache_path(tag):
    os.makedirs(CACHE, exist_ok=True)
    return os.path.join(CACHE, f"{tag}.npz")



def _save_report(tag, pred, sel, idx, **meta):
    """把「只跑一次」的報數預測落地。

    #! 沒有這個，要換口徑重算（例如稽核建議的「層內 ρ 平均」）就得**再看一次 val**
    #  ——而那正是 Goodhart 護欄禁止的事。gate1/gate2/head 先前一筆都沒存。
    """
    import json
    path = _cache_path(f"report_{tag}")
    np.savez_compressed(path, pred=pred, sel=sel, y=idx["y"][sel],
                        stratum=idx["stratum"][sel], meta=json.dumps(meta, ensure_ascii=False))
    print(f"  報數預測已落地：{path}")


def cmd_predict(args):
    idx = D.load()
    split, _ = D.assign_split(idx)
    sel = pick(idx, split, args.split, args.n)
    print(f"{args.split}: {len(sel)} 筆 | er={args.er} Q={args.q} modes={args.modes} dev={args.device}")
    pred, dt = run_l1(idx, sel, er=args.er, q=args.q, n_modes=args.modes,
                      device=args.device, batch=args.batch)
    print(f"  {dt:.1f}s ({dt / max(len(sel), 1) * 1000:.0f} ms/筆)")
    np.savez_compressed(_cache_path(args.tag or f"l1_{args.split}"), pred=pred, sel=sel,
                        er=args.er, q=args.q, modes=args.modes)
    wm_p, _, _ = margins(pred)
    wm_t, _, _ = margins(idx["y"][sel])
    report_rho(wm_p, wm_t, idx["stratum"][sel], tag=f"L1 裸 ({args.split})")


def cmd_scan(args):
    """(er, Q) 網格：**只在 dev 上掃**，找可用起點。"""
    idx = D.load()
    split, _ = D.assign_split(idx)
    sel = pick(idx, split, args.split, args.n)
    y = idx["y"][sel]
    wm_t, _, _ = margins(y)
    ers = [float(v) for v in args.ers.split(",")]
    qs = [float(v) for v in args.qs.split(",")]
    print(f"掃描 {len(sel)} 筆（{args.split}）：er {ers} × Q {qs}")
    print("| er | Q | ρ(ALL) | ρ(clean) | ρ(neg) | ρ(senior) |")
    print("|---|---|---|---|---|---|")
    best = None
    for er in ers:
        for q in qs:
            pred, dt = run_l1(idx, sel, er=er, q=q, n_modes=args.modes,
                              device=args.device, batch=args.batch)
            wm_p, _, _ = margins(pred)
            row = [er, q]
            for s in ["ALL", "clean", "neg", "senior"]:
                m = np.ones(len(sel), bool) if s == "ALL" else (idx["stratum"][sel] == s)
                row.append(rank_rho(wm_p[m], wm_t[m])[0] if m.sum() > 3 else float("nan"))
            print("| " + " | ".join(f"{v:+.3f}" if i > 1 else f"{v:g}" for i, v in enumerate(row)) + " |",
                  flush=True)
            if best is None or row[2] > best[2]:
                best = row
    print(f"\n最佳（dev）: er={best[0]} Q={best[1]} ρ={best[2]:+.3f}")


#? 網格刻意小（27 組 × 600 筆）——參數多、樣本少就變成對 fit 分割過擬合。
#  rad_eff（輻射效率）是**先驗的物理選擇**不是超參數：D₀ 只管方向性，抓不到
#  「會共振但不輻射」；dev 上量到 pooled ρ +0.363→+0.413，主要修的正是 Gain 那一路。
#  精度：一律 CPU float64——float32 下 ρ 掉 0.07（Cholesky+eigh 對 B 的條件數敏感），
#  GPU float64 的 cusolverDnXsyevd 在本機直接報 INTERNAL_ERROR。
L1_GRID = [dict(er=er, q=q, gap=g, diag=g, rad_eff=True)
           for er in (3.0, 3.3, 3.55)
           for q in (8.0, 15.0, 30.0)
           for g in (1, 2, 3)]


def cmd_fitscan(args):
    """L1 純量擬合：**只在 fit 分割上選**（與 dev/val 不相交），選 pooled ρ(wm) 最大者。"""
    idx = D.load()
    split, _ = D.assign_split(idx)
    sel = pick(idx, split, "fit", args.n)
    y = idx["y"][sel]
    wm_t, _, _ = margins(y)
    st = idx["stratum"][sel]
    print(f"fit 選參：{len(sel)} 筆（{args.n}/stratum）× {len(L1_GRID)} 組")
    print("| er | Q | gap | diag | ρ(pooled) | ρ(clean) | ρ(neg) | ρ(senior) |")
    print("|---|---|---|---|---|---|---|---|")
    rows = []
    for cfg in L1_GRID:
        pred, _ = run_l1(idx, sel, batch=args.batch, device=args.device, **cfg)
        wm_p, _, _ = margins(pred)
        pooled = rank_rho(wm_p, wm_t)[0]
        per = [rank_rho(wm_p[st == s], wm_t[st == s])[0] for s in ("clean", "neg", "senior")]
        rows.append((pooled, cfg, per))
        print(f"| {cfg['er']} | {cfg['q']:g} | {cfg['gap']} | {cfg['diag']} | {pooled:+.3f} | "
              + " | ".join(f"{v:+.3f}" for v in per) + " |", flush=True)
    rows.sort(key=lambda t: -t[0])
    print(f"\n**fit 最佳**：{rows[0][1]} → ρ(pooled)={rows[0][0]:+.3f}")
    with open(os.path.join(D.CACHE_DIR, "l1_params.json"), "w", encoding="utf-8") as fh:
        import json
        json.dump(dict(best=rows[0][1], rho_fit=rows[0][0], n=len(sel)), fh, ensure_ascii=False)


def cmd_gate1(args):
    """L1 gate：**val 只跑這一次**。判準寫死在 `docs/log/analysis-08-diffsim.md` §1。

    ① 主 KPI = 裸 diffsim-wm 對 HFSS-wm 的 pooled Spearman ρ（判準 ≥0.40）
    ② 仿射校準（係數只在 fit 上擬）後的 S11 每頻點 MAE
    ③ 負片域 ρ 單獨報；凍結尺 30 單獨報
    """
    import json
    idx = D.load()
    split, _ = D.assign_split(idx)
    pf = os.path.join(D.CACHE_DIR, "l1_params.json")
    cfg = json.load(open(pf, encoding="utf-8"))["best"] if os.path.exists(pf) else \
        dict(er=3.55, q=15.0, gap=2, diag=2)
    print(f"L1 參數（fit 分割選出，未看過 val）：{cfg}")

    sel = pick(idx, split, "val")
    pred, dt = run_l1(idx, sel, batch=args.batch, device=args.device, **cfg)
    st, y = idx["stratum"][sel], idx["y"][sel]
    wm_p, _, _ = margins(pred)
    wm_t, _, _ = margins(y)
    print(f"val {len(sel)} 筆，{dt:.0f}s")
    _save_report("gate1", pred, sel, idx, model="l1", cfg=cfg, stamp=_model_stamp("l1"))
    rhos = report_rho(wm_p, wm_t, st, tag="L1 裸（val，主 KPI ①）")

    selc = pick(idx, split, "fit", args.calib)
    pc, _ = run_l1(idx, selc, batch=args.batch, device=args.device, **cfg)
    a, b = affine_fit(pc, idx["y"][selc])
    pa = affine_apply(pred, a, b)
    mae = np.abs(pa[:, :17] - y[:, :17]).mean(0)
    print(f"\n② 仿射校準（fit {len(selc)} 筆擬）後 S11 每頻點 MAE(dB)：中位 {np.median(mae):.2f}，"
          f"帶內 5:12 {mae[5:12].mean():.2f}，全帶 {mae.mean():.2f}")
    wm_a, _, _ = margins(pa)
    print(f"   （校準後 wm 的 pooled ρ = {rank_rho(wm_a, wm_t)[0]:+.3f}，僅供對照，非判準）")

    print(f"\n③ 負片域 ρ = {rhos.get('neg', float('nan')):+.3f}；"
          f"凍結尺 ρ = {rhos.get('frozen', float('nan')):+.3f}")
    ok = rhos.get("ALL", 0) >= 0.40
    print(f"\n===== GATE 1：pooled ρ = {rhos.get('ALL', float('nan')):+.3f} vs 判準 0.40 → "
          f"{'通過' if ok else '**不通過**'} =====")
    return ok


L2_PARAMS = os.path.join(D.CACHE_DIR, "l2_params.json")


def build_l2(scale=None, raw=False, solver="l3fl"):
    """依名字＋存檔建 L2 求解器。**`solver` 決定物理設定，存檔只決定 DCIM 核的參數。**

    #! 2026-08-02 教訓：本函式原本**漏掉載入 `kernel` 的那段**（用字串替換改 code、
    #  替換靜默失敗而我沒驗證）。後果是 `l2eval`/`gate2` 全部跑在解析初值上，
    #  「擬核後 ρ 一模一樣」看起來像「模型不敏感」，其實是根本沒載入。
    #  現在載入完會印出實際生效的核，讓這種錯不可能再靜默發生。
    #! 2026-08-03：同一個病的**第四次**——本函式當時只生得出 `dcim`，而 analysis-10
    #  §31 之後所有報出去的數字都跑在 L3 核 + 分層遠場 + 半屋頂埠上，
    #  也就是**出貨路徑生不出研究路徑用的那個模型**。設定改住 `l2.SOLVERS`。
    """
    import json
    from . import l2 as L2
    import torch
    saved = json.load(open(L2_PARAMS, encoding="utf-8")) if os.path.exists(L2_PARAMS) else {}
    ker = None
    if L2.SOLVERS[solver]["kernel"] == "dcim":
        if scale is None:
            #! `a_scale` 是舊 key（那時縮放乘在 G_A 上）。2026-08-03 修正後縮放改除在 G_V，
            #  語義不同但**數值相同**（同一個相速），所以舊檔可直接沿用。
            scale = saved.get("v_scale", saved.get("a_scale", 2.356))
        sd = None if raw else saved.get("kernel")
        learned = bool(sd) and any(k.startswith("net.") for k in sd)     # 黑盒核的 state_dict
        n_img = len(sd["base.b" if learned else "b"][0]) if sd else 3
        #! **一定要走 `v_scale=`，不可以在外面 `a.data[0,:2,0] *= scale`。**
        #  後者就是 §10 修掉的病灶（把相速縮放乘在 A 項 → 偽輻射跟著放大 ~n³）。
        #  2026-08-03 稽核抓到：`DCIMKernel` 改好了但這裡沒跟著改 → 出貨核 η = 0.372
        #  而不是宣稱的 0.872，且常駐能量守恆測試守的是**沒人走的那條路**。
        base = L2.DCIMKernel(n_img=n_img, v_scale=scale)
        ker = L2.LearnedKernel(base=base) if learned else base
        if sd:
            #! 一定要用 load_state_dict 的嚴格模式：舊版這裡是逐 key copy_，
            #  key 對不上就靜默跳過 → 整個擬合結果沒生效卻看不出來（2026-08-02 踩過）。
            ker.load_state_dict({k: torch.as_tensor(v, dtype=torch.float64)
                                 for k, v in sd.items()}, strict=True)
        src = "黑盒核（DCIM 骨架 + MLP 修正）" if learned else ("擬合核" if sd else "解析校準初值")
        src += f"（n_img={n_img}, {saved.get('steps', '?')} 步, v_scale={scale:.3f}）"
    else:
        #? L3 表是**物理常數**、不可擬 ⇒ 存檔的 v_scale/kernel 與它無關，刻意不套用。
        src = "L3 精確 Sommerfeld 表（物理常數，不吃存檔參數）"
    m = L2.build(solver, kernel=ker)
    print(f"  L2 solver＝`{solver}`：{L2.SOLVERS[solver]}")
    print(f"  L2 核來源：{src}")
    return m, (scale if scale is not None else float("nan"))


def cmd_l2cal(args):
    """L2 核的解析校準：**不用 HFSS 資料**，只對閉式微帶物理。"""
    import json
    from .l2 import calibrate_analytic
    scale, err = calibrate_analytic()
    os.makedirs(D.CACHE_DIR, exist_ok=True)
    with open(L2_PARAMS, "w", encoding="utf-8") as fh:
        json.dump(dict(v_scale=scale, rms_log_err=err), fh)


def cmd_l2eval(args):
    """L2 在指定分割上的 ρ（迭代看 dev；gate 走 `gate2`）。"""
    idx = D.load()
    split, _ = D.assign_split(idx)
    sel = pick(idx, split, args.split, args.n)
    m, scale = build_l2(args.scale, solver=args.solver)
    print(f"L2 在 {args.split}：{len(sel)} 筆")
    t = time.time()
    pred = m.predict(idx["x"][sel].astype(np.float64), batch=args.batch)
    print(f"  {time.time() - t:.0f}s（{(time.time() - t) / len(sel) * 1000:.0f} ms/筆）")
    np.savez_compressed(_cache_path(f"l2_{args.split}_{args.solver}"), pred=pred, sel=sel,
                        scale=scale, solver=args.solver)
    wm_p, ms, mg = margins(pred)
    wm_t, mst, mgt = margins(idx["y"][sel])
    report_rho(wm_p, wm_t, idx["stratum"][sel], tag=f"L2 裸（{args.split}）")
    print(f"  通道：pooled ρ(mS11)={rank_rho(ms, mst)[0]:+.3f}  ρ(mGain)={rank_rho(mg, mgt)[0]:+.3f}")



def _wm_torch(s11, gain):
    """可微的 worst_margin（與 eval.margins 同定義；擬合時要走梯度所以另寫一份）。"""
    import torch
    n = s11.shape[-1]
    lo, hi = (5, 12) if n == 17 else (int(round(5 / 17 * n)), int(round(12 / 17 * n)))
    return torch.minimum(-10.0 - s11[:, lo:hi].max(1).values,
                         gain[:, lo:hi].min(1).values - 4.0)


def _pair_rank_loss(pred, true, tau: float = 1.0, dead: float = 0.2):
    """batch 內兩兩順序的 logistic loss；真值差距 < `dead` dB 的配對不計（雜訊不當監督）。"""
    import torch
    dp = pred[:, None] - pred[None, :]
    dt = true[:, None] - true[None, :]
    m = dt.abs() > dead
    if not m.any():
        return pred.sum() * 0.0
    return torch.nn.functional.softplus(-dp * torch.sign(dt) / tau)[m].mean()


def cmd_l2fit(args):
    """擬核（`docs/diffsim.md` §3 L2 方案 b）：**對解算器反傳**，不另寫擬合器。

    只用 `fit` 分割。解析校準已經把共振頻率調對（<6%），這裡要修的是**輻射電阻**——
    初值的兩項鏡像都用 n=√εr，但輻射本身是以 k₀ 在空氣中傳的，所以 Re(Zin) 系統性偏低
    ~10×、S11 谷深不夠（dev 中位 −2.7dB vs 真值 −12.4dB）。
    """
    import json
    import torch
    idx = D.load()
    split, _ = D.assign_split(idx)
    sel = pick(idx, split, "fit", args.n)
    from .l2 import MoML2, DCIMKernel, LearnedKernel
    scale = args.scale if args.scale is not None else 2.356
    base = DCIMKernel(n_img=args.n_img, v_scale=scale)   # ← 必須走 v_scale，見 build_l2 的註記
    m = MoML2(kernel=LearnedKernel(base=base) if args.learned else base)
    init = {k: v.detach().clone() for k, v in m.kernel.state_dict().items()}
    x = torch.as_tensor(idx["x"][sel].astype(np.float64)).reshape(-1, 25, 25)
    y = torch.as_tensor(idx["y"][sel].astype(np.float64))
    f = np.linspace(24e9, 32e9, args.nfreq)
    jf = np.round(np.linspace(0, 16, args.nfreq)).astype(int)          # 對齊資料集頻點
    if args.learned:      # 學習節用大 lr、物理節用小 lr（骨架已經對，不該被大步打壞）
        opt = torch.optim.Adam([{"params": m.kernel.net.parameters(), "lr": args.lr},
                                {"params": m.kernel.base.parameters(), "lr": args.lr * 0.1}])
    else:
        opt = torch.optim.Adam(m.kernel.parameters(), lr=args.lr)
    g = torch.Generator().manual_seed(0)
    hub = torch.nn.HuberLoss(delta=3.0)
    bad, hist, best = 0, [], (float("inf"), None)
    print(f"擬核：fit {len(sel)} 筆、{args.nfreq} 頻點、{args.steps} 步、lr={args.lr}")
    for step in range(args.steps):
        ix = torch.randperm(len(sel), generator=g)[:args.batch]
        out = m.solve(x[ix], freqs=f)
        yb = y[ix]
        loss = hub(out["S11"], yb[:, :17][:, jf]) + hub(out["Gain"], yb[:, 17:][:, jf])
        if args.rank_w > 0:
            #? KPI 是 rank ρ，曲線 MSE 只是代理——實測 loss 減半（24.5→12.7）但 ρ 幾乎不動。
            #  這一項直接優化「batch 內兩兩順序」：真值差距 >0.2dB 的配對才算（避免拿雜訊當監督）。
            loss = loss + args.rank_w * _pair_rank_loss(
                _wm_torch(out["S11"], out["Gain"]), _wm_torch(yb[:, :17], yb[:, 17:]))
        opt.zero_grad()
        if not torch.isfinite(loss):                 # 奇異矩陣/離群 dB → 跳過，別讓 NaN 汙染參數
            bad += 1
            print(f"  step {step:3d}  loss 非有限，跳過（累計 {bad}）", flush=True)
            continue
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(m.kernel.parameters(), 0.5)
        if not torch.isfinite(gn):
            bad += 1
            continue
        opt.step()
        hist.append(float(loss))
        if float(loss) < best[0]:
            best = (float(loss), {k: v.detach().clone() for k, v in m.kernel.state_dict().items()})
        if step % max(1, args.steps // 20) == 0 or step == args.steps - 1:
            print(f"  step {step:3d}  loss {float(loss):.3f}（最佳 {best[0]:.3f}）", flush=True)
    if best[1] is None:
        raise SystemExit("擬核全程沒有有效步——降 lr 或檢查核參數範圍")
    m.kernel.load_state_dict(best[1])                # 取最佳而非最後一步
    print(f"最佳 loss {best[0]:.3f}（{len(hist)} 有效步 / {bad} 跳過）")
    #! 常駐輸出「參數到底動了多少」——2026-08-02 教訓：第一次擬核 loss 看起來有降，
    #  其實參數只動了 0.4%（Adam 每步位移上界＝lr，lr 被 NaN 嚇到設太小），
    #  ρ 與初值一模一樣。少了這三行，「沒收斂」與「根本沒在擬」分不出來。
    for k, v in m.kernel.state_dict().items():
        d = (v - init[k]).abs()
        rel = (d / init[k].abs().clamp_min(1e-9))[init[k].abs() > 1e-9]
        print(f"  參數 {k}: |Δ|max={float(d.max()):.3e}"
              + (f"  相對Δmax={float(rel.max()):.2%}" if rel.numel() else ""))
    st = {k: v.detach().tolist() for k, v in m.kernel.state_dict().items()}
    with open(L2_PARAMS, "w", encoding="utf-8") as fh:
        json.dump(dict(v_scale=scale, kernel=st, fitted=True, n=len(sel),
                       steps=args.steps, nfreq=args.nfreq), fh)
    print(f"核參數落地 {L2_PARAMS}")


def _model_stamp(model: str, solver: str = None) -> str:
    """模型的完整內容指紋（參數檔 + 原始碼 + solver 設定 + L3 表）。

    #! 快取 key 一定要含這個：核/參數換了但檔名沒變 → 靜默回傳舊模型的預測，
    #  而且看起來完全正常。同一類「靜默用到舊東西」在本輪踩過**三次**：
    #  ① `build_l2` 沒載入擬合核 ② 字串替換靜默失敗
    #  ③ **本函式自己**——它漏了 `solver`、`l3.py`、L3 表（2026-08-03 bug 獵捕 agent 抓到）。
    #  第三次特別值得記：這個函式的存在理由就是防這件事，它自己還是犯了，
    #  因為「模型」的定義在本輪擴大了（solver ＝ 核 + 埠 + 遠場一整組），而指紋沒跟著擴大。
    #  ⇒ **指紋的涵蓋範圍必須跟著「什麼算模型」一起長。**
    #  實測：漏 solver 時 `l3` 與 `l3fl` 的 `_cache_path` 完全相同，而同一貼片
    #  `S11@res` 是 −5.19 vs −2.00 dB ⇒ `head` 的 A/B 直接失效。
    """
    import hashlib as _h
    h = _h.sha1()
    h.update((solver or "").encode())
    f = os.path.join(D.CACHE_DIR, "l2_params.json" if model == "l2" else "l1_params.json")
    if os.path.exists(f):
        h.update(open(f, "rb").read())
    #! 也要指紋**原始碼**：本輪的修正全在 code 裡（`_mode_q` 歸一化、半格相位、
    #  v_scale 換邊），這些**不會改動 json** → 只指紋 json 的話換了物理還是重用舊預測。
    here = os.path.dirname(os.path.abspath(__file__))
    for fn in ("l1.py", "l2.py", "l3.py", "geom.py"):
        h.update(open(os.path.join(here, fn), "rb").read())
    #? L3 表是**離線算的物理常數**，不在原始碼裡 ⇒ 重建表（換 b_reg、換積分參數）
    #  必須讓快取失效。用 size+mtime 而不是內容（749 KiB，每次呼叫都讀太浪費）。
    tab = os.path.join(D.CACHE_DIR, "l3_table.npz")
    if os.path.exists(tab):
        st = os.stat(tab)
        h.update(f"{st.st_size}:{int(st.st_mtime)}".encode())
    return h.hexdigest()[:8]


def _phys_pred(idx, sel, model, args, tag):
    """diffsim 對一組樣本的預測（帶檔案快取——L2 每筆 ~310ms，重算很貴）。"""
    sv = getattr(args, "solver", "l3fl")
    path = _cache_path(f"phys_{model}_{tag}_{len(sel)}_{_model_stamp(model, sv)}")
    if os.path.exists(path):
        z = np.load(path)
        if len(z["sel"]) == len(sel) and (z["sel"] == sel).all():
            return z["pred"]
    if model == "l1":
        import json
        pf = os.path.join(D.CACHE_DIR, "l1_params.json")
        cfg = json.load(open(pf, encoding="utf-8"))["best"]
        pred, _ = run_l1(idx, sel, batch=24, **cfg)
    else:
        m, _ = build_l2(solver=sv)
        pred = m.predict(idx["x"][sel].astype(np.float64), batch=8)
    np.savez_compressed(path, pred=pred, sel=sel)
    return pred


def cmd_head(args):
    """殘差頭三臂 A/B/C + **paired bootstrap over samples**。

    三臂（同資料、同架構、同超參，只差輸入）：
      (P+Φ) pattern + 物理錨   (Φ) **只有物理錨**   (P) 只有 pattern（純資料對照）
    第二臂是可辨識性對照：少了它，「(P+Φ) ≈ (P)」分不出「物理沒資訊」還是
    「架構讓 MLP 有能力把物理錨抵銷掉」（Kennedy & O'Hagan 的 model discrepancy）。

    ⚠ 統計：多個 seed 只反映**訓練隨機性**；「換一批驗證樣本結論還在不在」要靠
    **重抽樣本**的 paired bootstrap（n=30 時這才是主要不確定性來源）。
    """
    from .head import train_head, predict_head
    from .eval import boot_delta_rho
    idx = D.load()
    split, _ = D.assign_split(idx)
    tr = pick(idx, split, "fit", args.n)
    va = pick(idx, split, "val")
    mods = ["l1", "l2"] if args.model == "both" else [args.model]
    print(f"殘差頭三臂（錨 = {args.model}）：訓練 {len(tr)}（fit）、驗證 {len(va)}（val）、"
          f"seeds {args.seeds}、device {args.device}")
    t = time.time()
    p_tr = np.concatenate([_phys_pred(idx, tr, k, args, "tr") for k in mods], 1)
    p_va = np.concatenate([_phys_pred(idx, va, k, args, "va") for k in mods], 1)
    print(f"  diffsim 預測 {time.time() - t:.0f}s")
    x_tr, y_tr = idx["x"][tr].astype(np.float32), idx["y"][tr].astype(np.float32)
    x_va, y_va = idx["x"][va].astype(np.float32), idx["y"][va].astype(np.float32)
    wm_t, _, _ = margins(y_va)
    st = idx["stratum"][va]
    ARMS = [("P+Φ", True, True), ("Φ(只有物理錨)", True, False), ("P(純資料對照)", False, True)]

    preds = {k: [] for k, _, _ in ARMS}
    for sd in range(args.seeds):
        for name, up, upat in ARMS:
            m, _ = train_head(x_tr, p_tr, y_tr, use_phys=up, use_pattern=upat,
                              epochs=args.epochs, seed=sd, device=args.device, verbose=False)
            preds[name].append(margins(predict_head(m, x_va, p_va))[0])
        print(f"  seed {sd} 完成（{time.time() - t:.0f}s）", flush=True)

    strata = ["ALL"] + sorted(set(st.tolist()))
    print("\n== 三臂 ρ（seed 平均）==")
    print("| 臂 | " + " | ".join(strata) + " |")
    print("|" + "---|" * (len(strata) + 1))
    mean_rho = {}
    for name, _, _ in ARMS:
        row = []
        for g in strata:
            msk = np.ones(len(va), bool) if g == "ALL" else (st == g)
            row.append(float(np.mean([rank_rho(w[msk], wm_t[msk])[0] for w in preds[name]])))
        mean_rho[name] = row
        print(f"| {name} | " + " | ".join(f"{v:+.3f}" for v in row) + " |")

    #? 用 seed 平均後的預測做 paired bootstrap：先平均掉訓練隨機性，再量樣本隨機性
    avg = {k: np.mean(v, 0) for k, v in preds.items()}
    print("\n== Δρ = (P+Φ) − (P)：paired bootstrap over **樣本**（4000 次重抽）==")
    print("| 分層 | n | Δρ | 95% CI | P(Δ>0) | 判讀 |")
    print("|---|---|---|---|---|---|")
    for g in strata:
        msk = np.ones(len(va), bool) if g == "ALL" else (st == g)
        d, lo, hi, pg = boot_delta_rho(avg["P+Φ"][msk], avg["P(純資料對照)"][msk], wm_t[msk])
        verdict = "**物理錨勝**" if lo > 0 else ("**對照組勝**" if hi < 0 else "CI 跨 0 → 分不出")
        print(f"| {g} | {int(msk.sum())} | {d:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {pg:.2f} | {verdict} |")
    print("\n== 可辨識性檢查：Φ(只有物理錨) 是否 ≫ 隨機？==")
    for g in strata:
        msk = np.ones(len(va), bool) if g == "ALL" else (st == g)
        r = mean_rho["Φ(只有物理錨)"][strata.index(g)]
        print(f"  {g:8s} ρ(Φ only) = {r:+.3f}"
              + ("   → 物理有資訊，但在 P+Φ 裡沒轉成加值 ⇒ 疑似被 discrepancy 吸收"
                 if r > 0.25 and mean_rho["P+Φ"][strata.index(g)] <=
                 mean_rho["P(純資料對照)"][strata.index(g)] + 0.02 else ""))


def cmd_gate2(args):
    """L2 gate：判準＝**裸** pooled ρ ≥ 0.60（`docs/diffsim.md` §5，發車前寫死）。"""
    idx = D.load()
    split, _ = D.assign_split(idx)
    sel = pick(idx, split, "val")
    m, scale = build_l2(args.scale, solver=args.solver)
    pred = m.predict(idx["x"][sel].astype(np.float64), batch=args.batch)
    y, st = idx["y"][sel], idx["stratum"][sel]
    wm_p, _, _ = margins(pred)
    wm_t, _, _ = margins(y)
    _save_report("gate2", pred, sel, idx, model="l2", solver=args.solver, v_scale=scale,
                 stamp=_model_stamp("l2", args.solver))
    rhos = report_rho(wm_p, wm_t, st, tag="L2 裸（val，主 KPI ①）")
    selc = pick(idx, split, "fit", args.calib)
    pc = m.predict(idx["x"][selc].astype(np.float64), batch=args.batch)
    a, b = affine_fit(pc, idx["y"][selc])
    mae = np.abs(affine_apply(pred, a, b)[:, :17] - y[:, :17]).mean(0)
    print(f"\n② 仿射後 S11 每頻點 MAE：中位 {np.median(mae):.2f}，帶內 {mae[5:12].mean():.2f} dB")
    print(f"③ 負片域 ρ = {rhos.get('neg', float('nan')):+.3f}；"
          f"凍結尺 ρ = {rhos.get('frozen', float('nan')):+.3f}")
    ok = rhos.get("ALL", 0) >= 0.60
    print(f"\n===== GATE 2：pooled ρ = {rhos.get('ALL', float('nan')):+.3f} vs 判準 0.60 → "
          f"{'通過' if ok else '**不通過**'} =====")


def main():
    ap = argparse.ArgumentParser(description="diffsim 驅動")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("l2cal")
    le = sub.add_parser("l2eval")
    le.add_argument("--split", default="dev")
    le.add_argument("--n", type=int, default=None)
    le.add_argument("--batch", type=int, default=8)
    le.add_argument("--scale", type=float, default=None)
    le.add_argument("--solver", default="l3fl", choices=tuple(SOLVERS))
    g2 = sub.add_parser("gate2")
    g2.add_argument("--batch", type=int, default=8)
    g2.add_argument("--calib", type=int, default=150)
    g2.add_argument("--scale", type=float, default=None)
    g2.add_argument("--solver", default="l3fl", choices=tuple(SOLVERS))
    g1 = sub.add_parser("gate1")
    g1.add_argument("--batch", type=int, default=24)
    g1.add_argument("--device", default="cpu")
    g1.add_argument("--calib", type=int, default=300, help="仿射校準用的 fit 筆數/stratum")
    hd = sub.add_parser("head")
    hd.add_argument("--model", default="l1", choices=("l1", "l2", "both"))
    hd.add_argument("--n", type=int, default=1200, help="每 stratum 取幾筆（fit 分割）")
    hd.add_argument("--epochs", type=int, default=60)
    hd.add_argument("--seeds", type=int, default=3, help="訓練 seed 數（平均掉訓練隨機性）")
    #? 預設 cpu：GPU 讓給批次線（SESSION_COORDINATION §2）；要用先查 nvidia-smi
    hd.add_argument("--device", default="cpu")
    hd.add_argument("--solver", default="l3fl", choices=tuple(SOLVERS))
    lf = sub.add_parser("l2fit")
    lf.add_argument("--n", type=int, default=400, help="每 stratum 取幾筆（fit 分割）")
    lf.add_argument("--batch", type=int, default=24)
    lf.add_argument("--nfreq", type=int, default=9)
    lf.add_argument("--steps", type=int, default=120)
    lf.add_argument("--lr", type=float, default=0.02)
    lf.add_argument("--scale", type=float, default=None)
    lf.add_argument("--n-img", type=int, default=3, dest="n_img")
    lf.add_argument("--learned", action="store_true", help="核 K 黑盒化（殘差式 MLP 修正）")
    lf.add_argument("--rank-w", type=float, default=0.0, dest="rank_w")
    fs = sub.add_parser("fitscan")
    fs.add_argument("--n", type=int, default=200, help="每 stratum 取幾筆（fit 分割）")
    fs.add_argument("--batch", type=int, default=24)
    fs.add_argument("--device", default="cpu")
    for name in ("predict", "scan"):
        p = sub.add_parser(name)
        p.add_argument("--split", default="dev")
        p.add_argument("--n", type=int, default=None, help="每 stratum 取幾筆")
        #! 預設必須是 None(全模)——與 fitscan/gate1 同一個模型。
        #  截斷 30 模對全模的 |ΔS11| 中位 4.1dB/最大 23.6dB（模態和以 1/kₙ² 收斂，要 ~300 模），
        #  舊的 30 讓「調參看的模型」與「gate 報數的模型」不是同一個。
        p.add_argument("--modes", type=int, default=None)
        p.add_argument("--device", default="cpu")
        p.add_argument("--batch", type=int, default=16)
        p.add_argument("--tag", default=None)
        if name == "predict":
            p.add_argument("--er", type=float, default=3.55)
            p.add_argument("--q", type=float, default=20.0)
        else:
            p.add_argument("--ers", default="2.8,3.1,3.55")
            p.add_argument("--qs", default="8,20,50")
    a = ap.parse_args()
    {"predict": cmd_predict, "scan": cmd_scan, "fitscan": cmd_fitscan, "gate1": cmd_gate1,
     "l2cal": cmd_l2cal, "l2eval": cmd_l2eval, "l2fit": cmd_l2fit, "head": cmd_head,
     "gate2": cmd_gate2}[a.cmd](a)


if __name__ == "__main__":
    main()

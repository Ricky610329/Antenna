# Pattern Browser — 規格契約(v1,2026-08-06)

Ricky 需求:快速瀏覽/比對全史 pattern;Hamming 距離鄰域查找;按對角/組塊/有無對角消融(菱形·挖空變體)篩選;總攬;自訂群組比對。

## 架構
- `build_index.py`:掃 NAS 一次 → `data/`(patterns.npz + meta.json)。重跑=增量刷新。
- `server.py`:**stdlib http.server**(零新依賴)。靜態檔+JSON API;numpy 做 Hamming。
- `static/`:單頁 SPA(vanilla JS+canvas,無 build step)。
- `data/` **gitignore**(可重生);程式碼進 git。

## 資料契約
- `data/patterns.npz`:`packed` uint8 [N,79](np.packbits(625 bits));`ids` 陣列(同序)。
- `data/meta.json`:list[N],每筆:
  `{id, folder, store|null, wm|null, rad|null, lo|null, sel|null, total, n4, n8, largest8_frac,
    ndiag, has_db100(bool), has_sl100(bool), db100_wm|null, sl100_wm|null, kind|null}`
  - 變體 id(含 `~`)**不進主列表**,收進親本的 has_*/\*_wm 欄。
  - n8=scipy.ndimage.label(structure=ones(3,3));n4=預設;ndiag=repo `diag_bridge_sites(mat,0.10,0.2)` 的 len(sites)。
  - 指標源=res_index 快取(scratchpad)+新 store 增量掃;lo=oob_gain_max_lo。
- 渲染方向鐵則:**i(第一索引)朝下、j 朝右 → 饋線邊=圖下緣**(memory reference-pattern-render-convention)。

## API(全部 GET,JSON)
- `/api/list?offset&limit&sort=wm|lo|rad|ndiag|n8|total&dir&f_qual=1&f_diag_min&f_diag_max&f_n8_min&f_n8_max&f_has_db100=1&f_has_sl100=1&f_wm_min&f_lo_max&q=<id子串>` → {total, rows:[meta...]}
- `/api/pattern/<id>` → meta+`bits`(625 個 0/1)。變體 id 也可查(bits 同親本,meta 標變體)。
- `/api/hamming?id=<id>&maxd=<int>&limit` → 距離排序 [{id, d, wm, lo,...}]
- `/api/compare?ids=a,b,c` → 各 bits+meta(前端畫 XOR)。
- `/api/stats` → 總攬聚合(筆數/合格數/ndiag 分布/n8 分布/消融覆蓋數)。

## 前端頁面
1. **總攬**:統計卡+分布長條(n8/ndiag/wm);篩選器+表格(縮圖 canvas 48px+指標欄+排序)。
2. **詳情**:大圖(可切換疊菱形站點)、指標、消融變體對照(db100/sl100 wm 並排)、「找鄰居」按鈕(Hamming)。
3. **比對**:2-4 格並排+XOR 差異圖(共同=灰、A獨有=藍、B獨有=紅)。
4. **群組**:localStorage 建群組(新增/移除/命名/匯出入 JSON);群組頁=成員縮圖牆+指標表+一鍵送比對。

## 慣例
- 啟動:`python -m application.pattern_browser.server --port 8321`(repo 根;ant env)。
- 中文 UI;深色可免;效能目標:list/hamming <300ms(N≈36k,numpy 向量化)。
- 測試:`tests/` 不動;瀏覽器工具自帶 `application/pattern_browser/selftest.py`(API 冒煙)。

---

# v2 增補契約(2026-08-06;Ricky:response/rad 疊圖比對·雙視角·多渲染·tooltip+說明頁)

## v2 資料(build_index 增產;全部與 patterns.npz 的 ids 同序對齊)
- `data/resp.npz`:`resp` float16 [N,2,17](S11,Gain;24-32GHz 17 點;缺=NaN)+`has_resp` bool[N]。
  來源=store 樣本 .pt(hash 檔名):每店載入樣本以 pattern bytes 對映 id;增量刷新同主索引。
- `data/rad.npz`:`theta` float16[181]+`phi0`/`phi90` float16[N,181](缺=NaN)+`has_rad` bool[N]。
  來源=store/rad/{id}.pt(id 檔名,直取)。
- `data/variant_resp.json`:{變體id: {"s11":[17],"gain":[17],"phi0":[181]|null,"phi90":[181]|null}}
  (~db100/~sl100 變體曲線;消融疊圖用)。
- meta.json 每筆加 `has_resp`/`has_rad`(bool)。

## v2 API
- `/api/resp?ids=a,b,c` → {id:{s11:[17],gain:[17]}}(親本查 npz,變體查 variant_resp;缺=null)。
- `/api/radc?ids=a,b,c` → {id:{theta:[181],phi0:[..],phi90:[..]}}(缺=null)。
- `/api/targets` → {band:[26.5,29.5], s11_max:-10, gain_min:4, wm_buffer:0.15, rad_window:45, rad_floor:3}
  (規格常數單一來源,前端畫目標線/合格門檻全由此取)。
- `/api/list` 每列加 has_resp/has_rad。

## v2 前端
- **比對頁**分頁籤:Pattern(XOR,現行)/S11 疊圖/Gain 疊圖/rad 極座標疊圖(φ0/φ90 兩圖;主波束朝上、
  金=±45° 窗、紅虛圈=G0−3dB——同 repo 報告圖語言);目標線取 /api/targets。
- **詳情頁**加:S11+Gain 曲線(含目標線與內帶底色)、rad 極座標兩切面;消融對照卡升級=
  原始/菱形/挖空**曲線疊圖**(不只 wm 數字)。
- **總攬多渲染模式**(切換鈕):表格(現行)/縮圖牆(大縮圖 grid+hover 指標)/散點(X-Y 軸可選
  wm/lo/rad/ndiag/n8,點=pattern,框選→送比對/建群組)。
- **雙視角頁籤**:
  - **製造視角(日月光)**:排行榜(軸可選 wm/rad/lo/sel;硬閘門開關=合格 gates)、
    「可製造欄」=db100_wm(菱形化後餘裕=實際出貨值)並可按它排序、規格達成卡(距各目標多少 dB)、
    一鍵匯出 top-N CSV+pattern PNG。
  - **研究視角**:帕累托前緣圖(lo-wm 平面,合格點高亮)、家族分布(id 前綴聚合)、
    消融覆蓋統計、與散點模式互通。
- **全面 tooltip**:每個按鈕/篩選器/欄位標題 hover 出說明(自製 tooltip,非僅 title);
  **/help 說明頁**:系統導覽(四視圖+雙視角)、名詞表(wm/rad/lo/sel/ndiag/n8/消融)、
  資料來源與刷新方式、常見工作流(找最好/找鄰居/建群組比對)。

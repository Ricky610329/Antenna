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

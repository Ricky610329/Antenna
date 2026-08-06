# Pattern Browser — 全史 pattern 瀏覽/比對工具

規格契約=`SPEC.md`(v1+v2 增補)。定位:研究輔助瀏覽器,非 production system;資料唯讀,不碰訓練管線。

## 啟動(repo 根、ant env)

```bash
python -m application.pattern_browser.server --port 8321      # 讀 data/(先跑 build_index 產生)
python -m application.pattern_browser.server --fixture        # 無 data/ 的開發模式:200 筆決定性假資料+假曲線
```

開 `http://127.0.0.1:8321/`。`data/` 由 `build_index.py` 掃 NAS 產生,gitignore 不進版控;
`patterns.npz`+`meta.json` 缺→server 印訊息退出;**v2 曲線檔(`resp.npz`/`rad.npz`/`variant_resp.json`)缺→容錯**:
照常啟動,`has_resp`/`has_rad` 全 False、曲線 API 回 null,前端顯示「無曲線資料」。

## 視圖

- **總攬**:統計卡(含曲線覆蓋)+n8/ndiag/wm 分布;篩選;三種呈現切換——**表格**(排序欄)/**縮圖牆**(96px,hover 浮層指標)/**散點**(X-Y 軸自選 wm·lo·rad·ndiag·n8·total;點=進詳情、拖框=送比對/建群組、Shift+拖=累加)。
- **詳情**:大圖(疊菱形站點)、全指標、S11/Gain 曲線(紅虛線=門檻、金底=26.5–29.5GHz 頻帶)、rad 極座標 φ0/φ90(金扇=±45° 窗、橘虛線=窗界、紅虛圈=G0−3dB、每環 5dB)、消融=wm 數字卡+**原始/菱形/挖空曲線疊圖**(有才畫)、Hamming 找鄰居。
- **比對**:2–4 筆,分頁籤=Pattern(XOR)/S11 疊圖/Gain 疊圖/rad 極座標疊圖;曲線 hover 有 crosshair+數值。
- **群組**:localStorage 群組(建/改名/刪/匯出入 JSON)、勾選送比對。
- **製造視角**:排行榜(軸=wm/rad/lo/sel/**db100_wm 可製造欄**,欄頭可點排序;合格閘門開關)、規格達成卡(點列切換)、匯出 top-N CSV/PNG。
- **研究視角**:lo–wm 帕累托前緣(合格高亮、前緣橘線、點擊進詳情)、家族統計(id 底線前綴,top20,點列→總攬篩選)、消融/曲線覆蓋卡。
- **說明(/#help)**:文件站等級——左側 sticky 章節目錄(平滑捲動+scroll-spy 高亮+`#help/<錨點>` 深連結)+快速開始/資料與刷新/四視圖逐元件詳解/雙視角/五個 step-by-step 使用範例/完整名詞表(定義·單位·門檻·怎麼讀)/FAQ;內容在 `static/help.js`(零依賴 template)。全站按鈕/篩選器/欄頭皆有自製 hover tooltip。

渲染方向鐵則:第一索引 i 朝下、j 朝右=**饋線邊在圖下緣**。

## API(全 GET/JSON;※=對 SPEC 契約的加欄擴充)

| 路徑 | 回傳 |
|---|---|
| `/api/list?offset&limit&sort&dir&f_*&q` | `{total, rows}`;每列含 has_resp/has_rad;※`bits_b64`(79B packed base64);※`lite=1`=無 bits_b64 且 limit 上限 50000;※sort 多收 `sel`/`db100_wm` |
| `/api/pattern/<id>` | meta+`bits`(625);變體 id(含 `~`)可查;※`diag_sites`、`bits_b64` |
| `/api/hamming?id&maxd&limit` | 距離排序陣列;※每列多 `bits_b64` |
| `/api/compare?ids=a,b,c` | 各 bits+meta 的陣列(2–4 筆) |
| `/api/resp?ids=a,b,c` | v2:`{id:{s11:[17],gain:[17]}\|null}`;親本查 resp.npz、變體查 variant_resp;※未知/無曲線回 null 不 404(批次語意) |
| `/api/radc?ids=a,b,c` | v2:`{id:{theta:[181],phi0:[181],phi90:[181]}\|null}`;變體切面可個別 null |
| `/api/targets` | v2:規格常數單一來源 `{band,s11_max,gain_min,wm_buffer,rad_window,rad_floor}`;※多 `freqs`(24–32GHz 17 點) |
| `/api/stats` | 筆數/合格數(wm≥0.15∧rad≥0)/ndiag·n8 分布/消融覆蓋;※`wm_hist`、`wm_nonnull`、`resp_count`、`rad_count` |

Hamming=packed uint8 XOR+popcount 查表向量化(36k 列遠低於 100ms)。
曲線資料與 patterns.npz 的 ids 同序對齊;server 載入時套同一列重排,形狀對不上→警告+當作缺(容錯)。

## 冒煙測試

```bash
python -X utf8 application/pattern_browser/selftest.py   # 自啟 --fixture server 打全部 API+缺曲線檔容錯線,退出碼 0/1
```

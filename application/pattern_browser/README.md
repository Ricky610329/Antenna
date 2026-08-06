# Pattern Browser — 全史 pattern 瀏覽/比對工具

規格契約=`SPEC.md`。定位:研究輔助瀏覽器,非 production system;資料唯讀,不碰訓練管線。

## 啟動(repo 根、ant env)

```bash
python -m application.pattern_browser.server --port 8321      # 讀 data/(先跑 build_index 產生)
python -m application.pattern_browser.server --fixture        # 無 data/ 的開發模式:200 筆決定性假資料
```

開 `http://127.0.0.1:8321/`。`data/`(patterns.npz+meta.json)由 `build_index.py` 掃 NAS 產生,gitignore 不進版控;缺檔時 server 會印清楚訊息並退出。

## 四個視圖

- **總攬**:統計卡+n8/ndiag/wm 分布長條;篩選(合格/ndiag/n8/wm/lo/消融覆蓋/id 子串)+排序表格(點欄頭排序)+縮圖;點列進詳情。
- **詳情**:大圖(可疊 45° 菱形站點,server 算)、全指標、db100/sl100 消融 wm 對照、Hamming 找鄰居。
- **比對**:2–4 格並排+XOR 差異(共同灰/A獨藍/B獨紅)+兩兩距離矩陣;基準可切。
- **群組**:localStorage 自訂群組(建/改名/刪/匯出入 JSON)、成員縮圖牆+指標表、勾選一鍵送比對。

渲染方向鐵則:第一索引 i 朝下、j 朝右=**饋線邊在圖下緣**。

## API(全 GET/JSON;※=對 SPEC 契約的加欄擴充)

| 路徑 | 回傳 |
|---|---|
| `/api/list?offset&limit&sort&dir&f_*&q` | `{total, rows}`;※每列多 `bits_b64`(79B packed base64,縮圖用) |
| `/api/pattern/<id>` | meta+`bits`(625);變體 id(含 `~`)可查;※多 `diag_sites`([[cx,cy,w]] HFSS mm)與 `bits_b64` |
| `/api/hamming?id&maxd&limit` | 距離排序陣列;※每列多 `bits_b64` |
| `/api/compare?ids=a,b,c` | 各 bits+meta 的陣列(2–4 筆) |
| `/api/stats` | 筆數/合格數(wm≥0.15∧rad≥0=專案口徑)/ndiag·n8 分布/消融覆蓋;※多 `wm_hist`、`wm_nonnull` |

Hamming=packed uint8 XOR+popcount 查表向量化(36k 列遠低於 100ms)。

## 冒煙測試

```bash
python -X utf8 application/pattern_browser/selftest.py   # 自啟 --fixture server 打全部 API,退出碼 0/1
```

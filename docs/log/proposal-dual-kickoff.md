# proposal-dual-kickoff — Dual-port 濾波器線開線提案

- **狀態**: proposed(2026-08-10;D0 論文精讀+D1 資料體檢+D2 配管盤點三前置完成;等 Ricky 裁三個決策點)
- **背景**: 學長點頭、single-port 收尾中;dual=論文的「濾波器」線(實為**會輻射的二埠濾波天線**,
  通帶僅 ~56% 能量留在埠上)。工作模式=任務管理+Opus agent 施工。
- **指向**: [senior-thesis-dual-port](../reference/notes/senior-thesis-dual-port.md)(D0 全文)·
  D1/D2 任務報告(本檔為其收斂;關鍵結論已錄)·configs/dual_base.yaml·decisions「Single-port 收尾期程」

## 1. 資產盤點(三前置的結論)

- **規格=論文 p.44,與 configs/dual_base.yaml 逐點吻合**:S11/S22 帶內 26.5-29.5GHz ≤−12dB;
  S21 通帶 25.5-30.5 ≥−3dB、阻帶 ≤−20dB;17 點 @24-32GHz。**合成判準論文沒有,自定**(§2)。
- **harvest_dual 10,023 筆體檢全綠**:格式 (25,25)+(3,17) 零壞檔;**通道序=(S11,S21,S22) 已用能量守恆
  物理鑑定結案**;能量自證 |S11|²+|S21|²≤1 全過(max 0.91=輻射 ~46% 實錘)。
- **起跑線=零合格**:學長萬筆離規格很遠(六項全過 0/400,best worst −8.2dB)——首個像樣解即全場基準,
  必須自公證。帶外反射 m5/m6=零通過的最難兩項(能量帳使然),與論文筆記預測相反。
- **結構槓桿**:dual=上下對向雙埠;**上下鏡像 pattern ⇒ S11≡S22**(判準 6→4 項、自由度砍半)——
  學長資料零對稱樣本=全未開發。
- **模型層 dual-ready**:MLPSurrogate 預設輸出就是 (3,17);patch_dual.pth 可直接暖啟。
- **single/dual pattern 空間結構性不相交**(上緣饋墊),工具鏈相容但資產不搬。

## 2. 判準草案(發車前寫死於 round 檔;三決策點見 §4)

- 六項 margin(**mask 口徑**,與 loss 同尺;⚠ 不可沿用 single 的 width 切片算術——有斜邊會靜默切錯帶):
  m1/m2=帶內 S11/S22 反射、m3=通帶 S21、m4=阻帶 S21、m5/m6=帶外 S11/S22。
- **wm_dual = min(m1..m4)**;m5/m6 記帳不進 min(待決策③)。
- 每筆自證:`energy_max=max(|Sii|²+|S21|²) ≤ 1`(>1=壞檔);`assert len==17` ×3 通道。
- 紀錄制:`docs/records_dual.json` 另立(不混 single);首紀錄照公證鐵則 ×2。
- 埠位渲染核對=開線前必做 1 筆 smoke(上下對稱幾何會讓轉 90° bug 完全隱形)。

## 3. 施工清單(D2 全文=任務報告;此處=執行序)

**🔴 最小可跑鏈 11 項(4M+7S,無 L)**——Opus agent 分工,估 1-2 天:
1. dual_port.py 三層護欄(align_curve 17 點+assert/求解掃頻參數化/匯出前刪舊 CSV)[M]
2. losses.py 新 `worst_margin_dual`(mask 口徑;**不動 single 路徑一個 byte**)+回歸測試 [M]
3. dedust 新 `dual_metrics()`(六項分項+energy_max)[S]
4. dedust run() port 分派(模擬器選型/entry 泛化 n+1 欄/rad·oob·sel 走 single 分支)[M]
5. 派工鏈帶 config(jobs-add --config/job dict/worker 讀取)[S]
6. check-dup 分域(dual 不與 single 全史交叉——同 pattern 兩域各測是合法對照)[S]
7. configs/dual_r1_eval.yaml+README 一行 [S]
8. select-dual(harvest 錨+隨機+鏡像試點;可製造閘=雙饋墊 [(24,12),(0,12)],勿用 FEED/R_feed)[M]
9. report() port-aware [S]
10. 埠位渲染核對 smoke [S,不可省]
11. records_dual.json 開檔+round 檔判準寫死 [S]

**🟡 後補**(觸發制):雙側 loss(待決策①)/可製造代理量(禁 R_feed)/sm_reanchor port 參數化 [L]/
skill dual 分支/瀏覽器 3 通道 [L]/**上下鏡像 generator(結構槓桿,訓練線開跑即做)**/DualPortRadSimulator/
幾何對齊(待決策②)/analyze·chain 全家 [L]。

## 4. 三個 Ricky 決策點(施工前拍板)

1. **loss 換不換**:現行 dual 路徑=interval_loss(論文自評較差);換=+~20 行 two_sided_minmax
   (mask 口徑與 margin 同尺,golden 零漂移)。⚠ 改 loss 需 Ricky 同意(既定規矩)。
   不換→首批數字不可與論文圖 4-10 比(round 檔註明)。
2. **幾何對齊 single 與否**:dual 建模無 0.01mm 重疊(與 single 不同)。改=數值更穩但與 harvest_dual
   分佈脫鉤(SM 冷啟動價值打折)。**建議先不改**,第一批記已知風險。
3. **m5/m6 進不進 min**:實測=最難兩項(零通過)且可達性未證。**建議記帳不進 min**,
   第一批順帶驗「m5/m6 在此底板可達性」再定案。

## 5. 首輪(暫名 R57 dual 元年)設計草案

施工 11 項 → smoke 1 筆(埠位目檢+能量自證)→ 首批 ~100 筆
(harvest 最佳 20 錨+鄰域變異 40+純隨機 20+**上下鏡像試點 20**〔驗 S11≡S22 假說=判準減半的門票〕)
→ 判讀(六項分佈/energy/鏡像假說)→ SM 冷啟動(patch_dual 暖啟+harvest_dual 鍋)另批。
機隊:三台已閒置,隨時可發。

## 6. 排程關係

收尾線(R54/R56 收輪+方法論文件+交付包)與 dual 施工**並行**(施工=開發機 agent 工作,不搶機時);
dual 首批發車=收尾 round 收完後(Ricky 裁示的順序)。

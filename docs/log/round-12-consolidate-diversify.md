# Round 12 — 收斂（穩健冠軍）× 破單一化（第二山頭）

- **狀態**: queued（2026-07-08 備妥,兩批已生於 NAS＋查重過;R11 收檔後接跑）
- **提出 / 開跑 / 結論**: 2026-07-08 / — / —
- **一句話問題**: 50 個三標過全同 w17 家族——(A) 哪個是可送製造的穩健冠軍？(B) w17/F2 是唯一能過三標的家族,還是我們沒深挖別家？
- **一句話結論 (TL;DR)**: 待跑
- **指向**: 討論起點 = `docs/report/status-2026-07-08.md` §「很多過標」討論;工具 select-crown/select-family2;
  前作 [round-11](round-11-robustness.md)

## 1. 假設 (Propose)
- 背景（R11 產出）：去重後 50 個互異三標過 pattern,但 **Hamming>60 聚類=1 家族**（全 w17 高原）、
  中位 wm 僅 +0.07、只 8 個公證過。→ 瓶頸已從「找到任何三標解」轉為「收斂＋破單一化」。
- 假設 A（收斂）：top 候選間的差異不在 margin（已逼近雜訊）而在**局部缺陷存活率**（tol:c21 10/18 vs 薄冠軍 1-2/18）;
  公證＋缺陷掃描能選出唯一該送製造的穩健冠軍。
- 假設 B（破單一化）：pool 非 w17 家族（F0 wm+0.38/34 員、F1/F4/F6,對 w17 Hamming>230）R9 每族只試 leader
  一個就放棄;用「除塵＋對稱化{10,12}＋鄰域擾動＋SM 篩」深掘,測第二座山頭存不存在。

## 2. 實驗設計 (Design)
| 批 | 機器 | 內容 | 筆數 | 狀態 |
|---|---|---|---|---|
| crown | 37 | 8 top 候選 × (公證×2 + 邊緣缺陷 k1×4);erode/dilate 不跑(tol 已證全滅) | 48 ≈2.4hr | queued |
| family2 | 218 | 非 w17 家族 F0/F1/F4/F6 × (除塵/對稱10/對稱12/鄰域k4,k8) SM 篩 | 45 ≈2.2hr | queued |

判準（發車前寫死）：
- crown：公證 2 次一致（防假象）→ 存活率=(缺陷後三標數/4);**穩健冠軍=存活率最高者**（平手取 wm 高）。
- family2：任一非 w17 家族出現 wm≥−0.5 的乾淨三標解 → 第二山頭成立、升 R13 深耕;全部 <−1 → w17 家族特殊性確立（也是結論）。

## 3. 執行紀錄 (Run)
```
# 37 (R11 probes 收完後):  python -m script.dedust run --input dedust_crown_input --store dedust_crown
# 218 (R11 wide 收完後):   python -m script.dedust run --input dedust_family2_input --store dedust_family2
```

## 4. 分析 (Analyze)
（待收檔）

## 5. 結論 (Conclude)
- 待。

## 6. 後續決策 (Next)
- 穩健冠軍出爐 → 送實作量測（關掉「模擬≠量測」最大 limitation）。
- 第二山頭成立 → R13 深耕該家族;不成立 → w17 特殊性寫進碩論（分布≫策略的最強版本）。

## 7. 歸檔指向 (Archive)
- 結果夾: `dedust_crown/`、`dedust_family2/`;memory [[project_w17_champion]] [[project_benchmark_vs_random]]

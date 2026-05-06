# DEEPFOLD-SOLVER 使用說明

> v1.1.0 新功能完整指南 — Runout Report、Combo Drill、Memory Profile

**[English](USER_GUIDE.md) · [中文](USER_GUIDE.zh.md)**

---

## 目錄

1. [第一次使用](#1-第一次使用)
2. [Runout Report — 一眼看完所有 turn 走勢](#2-runout-report--一眼看完所有-turn-走勢)
   - [2.1 By Card 模式（13×4 grid）](#21-by-card-模式134-grid)
   - [2.2 By Class 模式（5 個 texture bucket）](#22-by-class-模式5-個-texture-bucket)
   - [2.3 排序模式](#23-排序模式)
   - [2.4 CSV 匯出](#24-csv-匯出)
3. [Combo Drill — 用 blocker 拆 1326 specific combos](#3-combo-drill--用-blocker-拆-1326-specific-combos)
   - [3.1 開啟方式](#31-開啟方式)
   - [3.2 Class 選擇器](#32-class-選擇器)
   - [3.3 Blocker 解讀](#33-blocker-解讀)
4. [Memory Profile — solver 資源策略](#4-memory-profile--solver-資源策略)
5. [常見問題](#5-常見問題)
6. [附錄: 首次安裝 Windows SmartScreen 警告](#附錄-首次安裝-windows-smartscreen-警告)

---

## 1. 第一次使用

| 步驟 | 動作 |
|---|---|
| 1 | 安裝 → 啟動 |
| 2 | 點 **以 Google 登入**（系統瀏覽器會自動開） |
| 3 | 設置 board (例 `AsKsQs`) + ranges + bet sizes |
| 4 | 點 **Solve** |
| 5 | 等 progress bar 跑完（小 spot 數秒、turn spot 數十秒、river full tree 視 memory profile 而定） |

解完之後右側 **StrategyPanel** 會看到兩個新按鈕：

```
┌───────────────────────────────────┐
│  [📊 Runout Report   23 turns]    │  ← 第 2 章
│  [🎴 Combo Drill     AKs    ]    │  ← 第 3 章
└───────────────────────────────────┘
```

> 💡 **Runout Report** 只在 board 有 enumerate 出多張 turn 時才顯示
> （monotone / two-tone / paired flop 會 enumerate；rainbow flop 為了
> 控 memory 只會 collapse 成單一 chance child）。

> 💡 **Combo Drill** 永遠都在（只要解完）。如果你之前點過 grid 上某張
> hand class，按鈕會帶該 class 名稱（例 `AKs`），打開 modal 就直接顯示
> 那 4 個 specific combos。

---

## 2. Runout Report — 一眼看完所有 turn 走勢

**這個解什麼問題**：傳統 solver UI 一次只給你看「當前 turn 的策略」。要知道
「整個 turn 街跨 23 張可能 turn cards 的策略分布」要點 23 次切。Runout
Report 把所有 turn 一次攤開，**1 秒內看完整個 turn 街**。

### 2.1 By Card 模式（13×4 grid）

預設模式。攤成 13 欄（rank） × 4 列（suit）。每個 cell 顏色 = 該 turn 上
**最高頻率的動作**。

```
       2  3  4  5  6  7  8  9  T  J  Q  K  A
  ♣  [.][.][.][.][.][.][.][.][.][.][.][.][.]
  ♦  [.][.][.][.][.][.][.][.][.][.][.][.][.]
  ♥  [.][.][.][.][.][.][.][.][.][.][.][.][.]
  ♠  [.][.][.][.][.][.][.][.][.][.][.][.][.]
```

**顏色含義**：

| 顏色 | 動作 |
|---|---|
| 灰 | Check / Call |
| 深灰 | Fold |
| 綠 | Bet ≤ 33% pot |
| 橘 | Bet 34–75% pot |
| 紅 | Bet > 75% pot |
| 紫 | Raise |
| 暗紅 | All-in |
| 藍 | 其他 |

**Cell 內容**：

- 大字 = dominant action 的 frequency（例：`67%`）
- 小字 `×N` = 這張 turn 是 iso class，代表 N 張真實 turn cards（例
  monotone flop 的 hearts/diamonds/clubs 對 spade flop 來說是對稱
  的，會 collapse 成 1 個 canonical rep with weight 3）

**flop card 處理**：原本是 flop 上的 3 張 cards 用斜線方框 + `flop` 字標出來
（不可能再當 turn 出現）。

**Hover 看細節**：hover 任一 cell → 下方 DetailPane 出現：

```
┌─────────────────────────────────────┐
│  9♠   IP acts · iso class ×1        │
│  [Check 40%] [Bet_33 35%] [All-in 25%] │
│  Mean EV: 8.4 chips                 │
└─────────────────────────────────────┘
```

### 2.2 By Class 模式（5 個 texture bucket）

點 modal 上方的 **By Class** tab → 把 23 張 turns 按 board texture 分成
5 個 bucket，weighted 算出每個 bucket 的策略 mix 跟 EV：

| Bucket | 定義 | 例（AsKsQs flop） |
|---|---|---|
| **Pair** | turn rank 與 flop 任一 rank 相同 | A♣ K♣ Q♣ |
| **Flush completion** | turn 花色在 flop 上有 ≥2 張 | 2♠ 3♠ 4♠ … J♠ T♠ |
| **Straight completion** | turn rank 在 [min flop rank − 1, max flop rank + 1] 區間內 | J♣ |
| **Overcard** | turn rank 嚴格大於所有 flop ranks | （此 board 沒有，A 已是頂） |
| **Brick** | 以上皆否 | 2♣–9♣, T♣ |

每個 bucket 一張卡片，左邊有色條：

```
┃ ⬤ Flush completion       10 cards · weight 10
┃ Mean EV: 7.3 chips
┃ ▓▓▓▓▓▓░░░░░░░░░░░░░░░░░░░  ← strategy bar (sized by freq)
┃ [Check 32%] [Bet_33 33%] [All-in 35%]
┃ 2S · 3S · 4S · 5S · 6S · 7S · 8S · 9S · TS · JS
```

**Weighted 算法**：每張 turn 用 `iso_weight × strategy_freq` 加總後除以總
weight。一張 weight=3 的 canonical rep 對 bucket 平均的影響等於 3 張
weight=1 的 turns。EV 同樣 weighted。

> ⚠️ **straight bucket 是粗略近似**：簡化版只看 rank 落在
> `[min−1, max+1]`，不檢查能否實際接成 5 張連續。對 AKQ flop，T turn
> 其實能跟 J 配成 TJQKA，但簡化規則會把它分到 Brick。準確的 straight
> 完成偵測需要更精細的牌型分析，列在 Day 3 之後的 polish round。

### 2.3 排序模式

**By Card** 模式下右上有 4 個排序按鈕：

| 按鈕 | 用途 | 排序鍵 |
|---|---|---|
| **Card** | 預設順序 | suit (♣♦♥♠) → rank (2→A) |
| **Best EV** | 找最甜的 turn | meanEV 降冪 |
| **Worst EV** | 找最爛的 turn（防守規劃） | meanEV 升冪 |
| **Aggressive** | 找最常下注的 turn | sum(Bet/Raise/All-in freq) 降冪 |

切換到非 Card 排序 → grid 從 13×4 變成 1D wrap（避免 13×4 的 suit/rank
位置在排序後誤導）。每個 cell 改顯示 `card icon` + freq + EV。

### 2.4 CSV 匯出

modal header 右上角的 **CSV** 按鈕 → 下載一個 `runout_<board>_<history>.csv`
檔。欄位：

```csv
card,acting,All-in_pct,Bet_33_pct,Check_pct,mean_ev,iso_weight
2c,OOP,30.9,34.4,34.6,9.40,3
2s,OOP,72.3,15.8,11.9,1.20,1
3c,OOP,28.1,35.2,36.8,9.21,3
...
```

- `<action>_pct` 欄位是純數字（沒有 `%`），方便試算表 / pandas 算
- `iso_weight` 給你還原真實 frequency（multiply by weight）
- 檔名含 board + history → 多 spot 對照不會混

---

## 3. Combo Drill — 用 blocker 拆 1326 specific combos

**這個解什麼問題**：169 hand class（例如 `AKs`）裡有 4 個 specific combos
（A♠K♠ / A♥K♥ / A♦K♦ / A♣K♣）。當 class 顯示 mixed strategy 時（例
55% bet / 45% check），你要決定**手上這個具體 combo** 走哪條線。

撲克的標準 tie-breaker 是 **blocker**：你拿走的 cards 是不是擋到對手的 value
range。Combo Drill 用 board-aware 的 blocker math 把 4/6/12 個 specific combos
按 blocker 強度排好給你看。

### 3.1 開啟方式

兩條路徑：

| 路徑 A | 路徑 B |
|---|---|
| 先在主畫面 **169 grid** 點某個 class（例 `AKs`） | 直接點 StrategyPanel 的 **Combo Drill** 按鈕 |
| → StrategyPanel 顯示 target analysis | → modal 開，預設選 hero 最常打的 class |
| → **Combo Drill** 按鈕標籤變成 `AKs` | |
| → 點按鈕 → modal 開、初始 class = AKs | |

兩條路徑進到的 modal 內容相同。

### 3.2 Class 選擇器

modal 上方有 search box + 169 個 class 按鈕：

```
🔍 [Filter class (e.g. AK, 99, AKs)…]

[22] [33] [44] [55] [66] [77] [88] [99] [TT] [JJ] [QQ] [KK] [AA]
[A2s][A3s][A4s][A5s][A6s][A7s][A8s][A9s][ATs][AJs][AQs][AKs]
[A2o][A3o][A4o][A5o][A6o][A7o][A8o][A9o][ATo][AJo][AQo][AKo]
...
```

- **粗的高亮按鈕**：當前選中的 class
- **實心背景**：opp 的 range 裡有的 class
- **空心**：opp 的 range 裡 weight 為 0 的 class（仍可以選）

打字到 search box 過濾（例打 `AK` → 顯示 `AKs` `AKo`）。

### 3.3 Blocker 解讀

選中 class 後，下方 grid 會顯示該 class 的 4/6/12 個 specific combos，按
blocker% 降冪排（dead combos 沉到最下）。每張 combo card：

```
┌──────────────────────────────┐
│  A♥ K♥                  DEAD │  ← 如果 combo 用到 board card
├──────────────────────────────┤   會標 DEAD 並 fade
│  Blocks opp range    8.1%    │
│  ▓▓▓▓░░░░░░░░░░░░░░░░░░     │  ← blocker bar
│                              │
│  Top blocked:                │
│  [AKo ×4] [AJo ×3] [KJo ×3] │
│                              │
│  ▓▓▓▓▓▓░░░░░░░░░░░░░░       │  ← class 共用 strategy bar
└──────────────────────────────┘
```

**Blocker% 怎麼算**：

```
total_live_opp_combos = sum(opp_range[label] × live_combos_in_label)
                       對所有 169 labels 加總（live 是扣掉 board cards）

blocked_opp_combos = sum(opp_range[label] × combos_using_hero_cards)
                    對所有 labels 加總

blocker% = blocked_opp_combos / total_live_opp_combos × 100
```

**Top blocked**：把所有 opp class 按「擋到的 combo 數 × range weight」排序
取前 5 名。例如 `AKo ×4` 表示對手 AKo 那 12 個 specific combos 裡，有 4
個含有你拿的 card。

**色階**：

| 顏色 | blocker% | 含義 |
|---|---|---|
| 🟢 綠 | < 4% | 弱 blocker — 可以放給對手 fold |
| 🟡 黃 | 4–8% | 中 blocker — 中性 |
| 🔴 紅 | ≥ 8% | 強 blocker — 適合 bluff / aggressive line |

### 重要 caveat

modal 中央有提示條：

> ⓘ Strategy & EV are class-shared. Blocker analysis is per-combo —
> use it to break mixed-strategy ties.

意思是：4 個 AKs combos 顯示的 strategy bar / EV 數字 **都是 class 平均**，
不是 per-specific-combo 的真值。要拿到 per-combo 的真實 strategy / EV 需要
engine 動工（已列在 roadmap）。**Blocker 是純前端可算的、唯一 per-combo
真實的差異化資訊**。

---

## 4. Memory Profile — solver 資源策略

solve 之前可以在 advanced settings 選 memory profile（CLI flag
`--memory-profile`）：

| Profile | host RAM 預算 | JSON 預算 | strategy tree node 上限 | 適用 |
|---|---|---|---|---|
| `safe` | 2 GB | 50 MB | 500 | 老電腦 / 測試 / 同時跑其他重 app |
| `balanced` | 6 GB | 100 MB | 2,000 | **預設** — 一般使用 |
| `performance` | 12 GB | 150 MB | 5,000 | 16+ GB RAM 工作站 / 想要最深 strategy_tree |

**衝突保護**：如果預估 footprint 超過 budget，solver 不會 OOM，會：

1. 嘗試 GPU AUTO fallback（GPU OOM 但 CPU 預算夠 → 自動切 CPU）
2. 用 byte-based runout cap 削減 chance enumeration
3. 截斷 strategy_tree（只 emit 上限節點，flag `truncated: true`）
4. 走完所有路徑都不行 → 結構化錯誤，UI badge 標 reason

**怎麼看是否被截斷**：StrategyPanel 上方有 resource badge（只在 truncate /
fallback / 非 ok decision 時出現）：

```
┃ ⚠ Strategy tree truncated to 2000/4321 nodes
┃   (50 MB JSON budget)
┃ ⚠ GPU OOM → fell back to CPU backend
```

---

## 5. 常見問題

**Q: 為什麼 Runout Report 按鈕有時候不出現？**

A: 該 board 沒 enumerate 出多張 turn。Rainbow flop（3 張不同花色 + 沒有
pair）為了控 memory 會 collapse 成單一 chance child，沒有 per-turn 資料
可聚合。試試 monotone / two-tone / paired flop。

---

**Q: Combo Drill 裡的 strategy 跟 EV 為什麼跟 169 grid 一樣？**

A: 對。current build 還沒做 per-specific-combo 的 strategy / EV emission
（要動 engine）。**只有 blocker 是 per-combo 的真實差異化資訊**，請以
blocker% 為決策依據，strategy / EV 當參考即可。Per-combo strategy 在
roadmap 上。

---

**Q: 為什麼某些 combo 顯示 DEAD？**

A: 該 combo 的兩張 card 至少有一張在 board 上。例如 board 有 A♠，那
A♠K♠ 就 DEAD（你不可能拿到一張已經在公共牌上的 card）。

---

**Q: Blocker% 多少算「強 blocker」？**

A: 經驗值參考：

| 範圍 | 解讀 |
|---|---|
| < 3% | 幾乎沒擋 — 跟對手 range 沒交集 |
| 3–6% | 中性 |
| 6–10% | 顯著 — 通常該走 aggressive line |
| > 10% | 罕見、極強 blocker — 通常是雙頂對之類 |

對 monotone flop 的 flush 完成牌，blocker% 會更密集分布在 6–12%（因為
hero 有花色 card 時擋掉很多對手的 flush）。

---

**Q: CSV 開出來中文/特殊字元亂碼？**

A: CSV 是 UTF-8，Excel 預設用 ANSI 開會亂。Excel 開啟方式：
**資料 → 從文字／CSV → 選 UTF-8 → 載入**。或用 LibreOffice / Numbers /
Pandas 直接讀就沒問題。

---

**Q: Memory profile 切到 `performance` 還是 OOM？**

A: 通常代表你的 board 真的太大（極端的 turn full tree + 寬 ranges）。
看 resource badge 的 reason，用 fallback / 鎖 range / 改 bet sizes 縮 tree。

---

**Q: 怎麼知道 solver 跑到收斂了？**

A: solve 結束後右上會顯示 `exploitability_pct`。一般：
- < 0.5% — 高品質
- 0.5–2% — 可用
- > 2% — iter 太少或 ranges 太寬，加 iteration 或縮 range

---

**Q: 解一次 solve 需要多久？**

A: 視 board 跟 backend：

| Board 類型 | CPU | GPU |
|---|---|---|
| flop, narrow ranges | 1–3s | < 1s |
| flop, full ranges | 5–15s | 1–5s |
| turn, full ranges | 30–120s | 5–30s |
| river full tree | 不建議 | 視 VRAM |

---

## 附錄: 首次安裝 Windows SmartScreen 警告

### 你會看到什麼

雙擊安裝檔後，Windows 會跳出**藍色全螢幕**警告視窗：

```
┌──────────────────────────────────────────────┐
│  Windows 已保護你的電腦                       │
│                                              │
│  Microsoft Defender SmartScreen 已防止無法    │
│  辨識的應用程式啟動。執行此應用程式可能會    │
│  讓您的電腦面臨風險。                        │
│                                              │
│  [更多資訊]                                   │  ← 點這裡（不是「不執行」）
│                                              │
│              [不要執行]                       │
└──────────────────────────────────────────────┘
```

點 **「更多資訊」** 之後，視窗會展開多兩行文字 + 一個新按鈕：

```
┌──────────────────────────────────────────────┐
│  Windows 已保護你的電腦                       │
│  ...（同上）                                  │
│                                              │
│  應用程式: DEEPFOLD-SOLVER_1.x.x_x64-setup.exe │
│  發行者: 不明的發行者                          │
│                                              │
│  [仍要執行]                  [不要執行]        │  ← 點「仍要執行」
└──────────────────────────────────────────────┘
```

點 **「仍要執行」** → 進入正常 NSIS 安裝流程。

### 為什麼會跳這個警告

Windows SmartScreen 對所有**沒有 code signing 憑證**的 .exe 都會顯示
這個警告。原因不是「我們的 app 不安全」，而是 **Microsoft 不認識
publisher**（任何人都可以叫自己「DEEPFOLD」，沒有第三方驗證過）。

要消除這個警告，需要購買 **EV (Extended Validation) Code Signing
Certificate**（一年約 $300–600 USD）。**目前我們在等使用者數累積到一定
規模再投資這筆費用**，先請使用者忍耐這一個多餘步驟。

### 安全性

- DEEPFOLD-SOLVER 的所有 source code **公開在 GitHub** 可審視：
  https://github.com/a9876543245/DEEPFOLD-SOLVER
- 安裝檔 hash 在每次 release 的 GitHub release page 可比對
- 全程**本機執行**，不上傳任何 solver 資料 — 唯一的網路請求是登入時跟
  deepfold.co 確認會員資格
- 不會跳警告 = 沒有惡意，也不代表跳警告 = 有惡意。判斷依據應該是
  **source code 公開度** + **下載來源**

### 自動更新會不會再跳警告

**會**。每個版本的 installer hash 不同，SmartScreen 對 unsigned executable
的 reputation 是 per-file 的，不會延續。意思是 v1.1.0 點過「仍要執行」，
v1.1.1 還是要再點一次。

這也是我們把 EV code signing 列在 roadmap 的原因 — 簽完之後**所有**新
版本都會自動信任。

### 還是不放心？

可以從原始碼編譯（README 有完整步驟）。編出來的 .exe 沒有 publisher
資訊但也不會跳警告（因為是你自己執行你自己編的東西，不是從網路下載
的 unknown publisher）。

---

更多技術細節 → [README](README.md) / [English Guide](USER_GUIDE.md)

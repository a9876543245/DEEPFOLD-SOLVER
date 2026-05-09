# DEEPFOLD-SOLVER

> GPU 加速 GTO 撲克 solver · CPU 自動 fallback · 三語介面 · 一鍵安裝

**[English](README.md) · [中文](README.zh.md) · [日本語](README.ja.md)**

📘 **[使用說明 (中文)](USER_GUIDE.zh.md)** · **[User Guide (English)](USER_GUIDE.md)**

![Platform](https://img.shields.io/badge/platform-Windows%2010%2F11-blue)
![Backend](https://img.shields.io/badge/backend-CUDA%20%2B%20CPU%20%28AVX2%2BMulti--core%29-orange)

DEEPFOLD-SOLVER 是 [DEEPFOLD](https://deepfold.co) 的桌面端 GTO solver。GPU 加速 DCFR 引擎搭配**真正會吃滿所有 CPU 核心**的 multi-core 後端,加上 **runout 聚合報告、per-combo blocker 分析、EV / 激進度熱力圖、2,500+ preflop 圖庫**,單一 Windows 安裝檔搞定。

## DEEPFOLD-SOLVER 的核心特點

### 引擎會吃滿你機器上的每一顆核心

- **雙後端 DCFR** — 有 GPU 用 GPU、沒 GPU 用 CPU,輸出策略**逐位相同**。GPU 路徑內建 Turing / Ampere / Ada / Hopper 的原生 CUDA SASS,Blackwell 走 PTX-JIT 前向相容。CPU 路徑用 BFS-flat「levelized」CFR 後端,能線性 scale 到所有實體核心,不再卡 2 執行緒上限。
- **Runtime CPUID 動態分派** — Haswell 之後的 CPU 跑 AVX2 kernel,更老的硬體自動 fallback 到 scalar。單一安裝檔、不需要另外編譯、2013 之前的 CPU 也不會啟動就 crash。
- **每次 commit 都過 parity gate** — `reference vs levelized`、`scalar vs AVX2`、`單執行緒 vs 多執行緒` 三組路徑必須輸出 bit-identical 策略才能 build pass。快速路徑永遠不會悄悄偏離正確性 oracle。

### Solve mode 預設真的會準時停

- **Quick / Standard / Deep** 三個 pill 按鈕把 iter 上限 + 時間預算 + exploitability 目標一鍵打包,任一條件先到就停。CFR 是 anytime 演算法,iter N 的 running average **就是**當前策略,所以預算停下來拿到的是「夠用」的結果,不是半成品。
- **解算前 ETA banner** — 按 Solve 後先用 sub-second 的 `--estimate-only` 引擎呼叫拿到 wall-clock 預估,跟 standard benchmark 對齊校準。AUTO 因 GPU 被排除而 fallback 的原因(例如「Pascal 需要 CUDA-12.x build」)會直接顯示在預估旁邊。
- **Stop 按鈕 + Quality badge** — 純 abort、不保留部分結果(「停下來給我現在這版」走時間預算)。結果面板按最終 exploitability 顯示 🟢 高品質 / 🟡 可用 / 🟠 粗略 / 🔴 低信心。
- **`--time-budget-seconds` 每個 iter 都會檢查**,慢 per-iter 的 spot 也能精準到 budget 停,不會晚個 5 分鐘。

### 別家 solver 沒有的解後分析工具

- **Runout Report** — 解完一鍵把所有 canonical turn card 攤成 13×4 grid,按 dominant action 上色。切到 **By Class** view,23+ 張 turn 自動分到 **Pair / Flush / Straight / Overcard / Brick** 五個 texture bucket,每 bucket 有加權 strategy + EV。可按 Best EV / Worst EV / Most aggressive 排序、一鍵匯出 CSV。
- **1326 Combo Drill** — 點任一 169-class label,展開成 4 / 6 / 12 specific combos,每張附 **per-combo blocker analysis**:你的 hand 擋掉對手 range 的百分比 + Top-5 最常被擋到的對手 class。撲克 mixed-strategy 的標準 tie-breaker,終於是一級 UI。
- **策略 grid 視圖模式** — 169 grid 上方工具列切換 Strategy Mix(預設多動作漸層)、**EV**(per-class 熱力圖,按 in-range cells 範圍 normalize 紅→灰→綠)、**Aggression**(Bet/Raise/All-in 頻率合計,冷→熱)、單動作 Heatmap。EV 模式一眼看出「哪些 combo 是利潤中心、哪些在虧」。

### 記憶體控制可信任

- **Memory Profile 預設** — `safe / balanced / performance` 三組預設一鍵把 host RAM、JSON、strategy-tree-node 預算定死,選的時候即時預覽。Solver 全程遵守預算 — Pre-backend gate 在 allocation **前**就評估完 CPU host / GPU VRAM / AUTO fallback,OOM 變成結構化錯誤 + UI badge,不是 crash。
- **共同 host budget gate 在 GPU 後端也會檢查** — matchup tables、strategy-tree EV cache、JSON response 不論用哪個 backend 都常駐 host RAM,通通都會檢查。錯誤訊息會明確說「換 GPU 也救不了 common-host 超量」。
- **Chunked GPU matchup 上傳** 拿掉 host 端 `flat_ev` / `flat_valid` 重複,改成 per-runout 的 `cudaMemset + cudaMemcpy`,GPU prep 時 peak host RAM 降低。

### 內建內容

- **2,550+ preflop 場景** app 內可瀏覽,一鍵套用為 IP / OOP range。
- **120+ 預解 flop spot** 一鍵載入。
- **下注尺度預設** — Standard / Polar / Small Ball — 同時影響 solver tree 跟 UI 按鈕。
- **Range 編輯器 + 節點鎖** — 強制任一手牌的 frequency 然後重 solve。
- **訓練模式** — 10 題 drill 對你的答案打分(vs 均衡解)。

### 細節打磨

- **三語介面** — English / 中文 / 日本語,隨時切換。
- **自動更新** — banner 一鍵安裝、release 有簽章、install mode `passive`。
- **花色同構** 在 monotone / 三花色板自動 3–7× 加速;GPU per-runout matchup table 在 iso 啟用時比 CPU 快 6–10×。
- **Route A 導航 cache** — O(1) 動作切換、不用重 solve。**Path B runout selector** 提供 PioSolver 風格的 chance-aware 導航。
- **可重現 benchmark** — `deepsolver_core --benchmark standard` 跑 AsKd7c rainbow / 100 iter 場景,輸出緊湊的 perf JSON(`iterations_per_sec` / `nodes_per_sec` / `memory_estimate_mb` + 完整 timing 內訳)。CI regression 追蹤可直接 grep。

## 下載

**Windows 10 / 11 (x64)** — [最新安裝檔](https://github.com/a9876543245/DEEPFOLD-SOLVER/releases/latest)

裝完後 app 會自動更新:有新版時左上角會出現 banner,點一下即可安裝重啟。

> ⚠️ **首次安裝會看到 Windows 警告**:執行安裝檔時,Windows 會跳出
> 「**Windows 已保護你的電腦**」(SmartScreen) 警告 — 這是正常的,因為
> DEEPFOLD-SOLVER 還沒申請 EV code signing 憑證,Windows 不認識這個 publisher。
> 請點 **更多資訊 (More info)** → **仍要執行 (Run anyway)** 繼續安裝。
> 完整步驟見 [使用說明 — 首次安裝警告](USER_GUIDE.zh.md#附錄-首次安裝-windows-smartscreen-警告)。

## 系統需求

| | 最低 | 建議 |
|---|---|---|
| 作業系統 | Windows 10 64-bit | Windows 11 64-bit |
| CPU | x86-64 雙核(任何年份) | 4 顆以上實體核心、AVX2(Haswell 2013 / Excavator 2015 之後) |
| RAM | 4 GB | 8 GB+ |
| GPU | — *(沒 GPU 也完整可用 — CPU 後端功能對齊)* | NVIDIA RTX 2000 系列以上、4 GB+ VRAM |
| 硬碟 | 200 MB | 200 MB |

GPU 跟 SIMD 都自動偵測:右上角狀態指示燈一眼看出 **CUDA** / **CPU**,CPU 後端會根據硬體支援度標 `AVX2` 或 `scalar` — Pre-Haswell CPU 跑 scalar kernels,永遠不會碰到 AVX2 opcode。

## 開始使用

1. 安裝並啟動 app
2. 點 **以 Google 登入** — 系統瀏覽器會自動開啟做 OAuth
3. DEEPFOLD PRO 會員直接進入 solver

還不是會員?到 [deepfold.co](https://deepfold.co) 升級。

## 功能總覽

| 功能 | 說明 |
|---|---|
| **GTO Solver** | Discounted CFR 配上向量化 GPU kernel 跟 multi-core CPU 後端。一般 turn spot 幾秒內收斂到 sub-percent exploitability |
| **每手策略 grid** | 13×13 grid 在當前決策點按動作 mix 上色。hover 看花色變體拆解 |
| **行動方/對手視角切換** | 同節點切換你的策略 vs 對手的 reach-weighted range |
| **Grid 視圖模式** | 169 grid 上方工具列切換 Strategy Mix / **EV** / **Aggression** / 單動作 heatmap。EV 模式按 in-range cells 範圍 normalize 紅→灰→綠 |
| **Memory Profile selector** | Advanced 區的 `safe / balanced / performance` pill,即時顯示預算。透過 `--memory-profile` 通到引擎 |
| **Benchmark CLI** | `deepsolver_core --benchmark standard` 跑可重現的 AsKd7c+100iter 場景,輸出緊湊 perf 追蹤 JSON |
| **Runout Report** | 解完任何 spot 後一鍵把所有 turn card 攤成 13×4 grid + texture bucket view + 4 種排序 + CSV 匯出。看[使用說明](USER_GUIDE.zh.md#2-runout-report--一眼看完所有-turn-走勢) |
| **1326 Combo Drill** | 任一 169-class 展開成 4/6/12 specific combos,附 per-combo blocker analysis。看[使用說明](USER_GUIDE.zh.md#3-combo-drill--用-blocker-拆-1326-specific-combos) |
| **Memory Profile** | `safe / balanced / performance` 預設管 host RAM / JSON / strategy-tree-node 預算。不再沉默 OOM |
| **Runout 選擇器** | iso 列舉啟用時,點任一 canonical river card 切換子樹 |
| **GTO 圖庫** | 內建 2,550+ preflop 場景,app 內瀏覽。一鍵套用為 IP / OOP range |
| **下注尺度預設** | Standard / Polar / Small Ball — 同時影響 solver tree 跟 UI 按鈕 |
| **訓練模式** | 10 題 drill 對你的答案打分(vs 均衡) |
| **預解 spot 圖庫** | 120+ 常見 flop spot,一鍵載入 |
| **Range 編輯器 + 節點鎖** | 強制任一手牌的 frequency 然後重 solve |

## 架構

```
┌─────────────────────────────────────────────┐
│  React + TypeScript UI (Tauri webview)      │
│  ├── 策略 grid · Runout 選擇器              │
│  └── GTO 圖庫瀏覽器                         │
├─────────────────────────────────────────────┤
│  Rust (Tauri) — IPC + 圖庫載入器            │
├─────────────────────────────────────────────┤
│  C++ 引擎 (deepsolver_core)                 │
│  ├── DCFR (CPU)                             │
│  ├── CUDA kernels (GPU)                     │
│  └── 花色同構 + per-runout matchup          │
└─────────────────────────────────────────────┘
```

引擎是獨立的 CLI(`deepsolver_core.exe`),以 Tauri sidecar 形式打包。Tauri 每次 solve 啟一個 process 然後 parse JSON 結果,內含完整策略樹供 client 端導航。

## 從原始碼編譯

需要:
- **Node.js 20+** + **npm**
- **Rust 1.78+**(`rustup`)
- **CMake 3.20+** + **MSVC 2022**(Windows)
- **CUDA Toolkit 12.x**(選用 — CPU build 不需要)

```sh
git clone https://github.com/a9876543245/DEEPFOLD-SOLVER.git
cd DEEPFOLD-SOLVER
npm install

# 編譯 C++ 引擎
cd core && mkdir -p build && cd build
cmake .. -DBUILD_TESTS=ON
cmake --build . --config Release
ctest

# 開發模式(自動找 ../core/build/Release/ 的引擎 binary)
cd ../..
npm run tauri dev
```

> **注意**:內建的 GTO preflop 圖庫資料(`gto_output/`,~31MB)跟預編譯的
> 引擎 sidecar(`src-tauri/binaries/`)**沒有放在這個 repo**,只在官方
> 安裝檔裡有。從 source build 出來的版本不會有圖庫資料,除非你自己提供
> `gto_output/` 目錄並符合相同 JSON schema。登入功能還需要在 build 環境
> 設 `DEEPFOLD_GOOGLE_CLIENT_SECRET` env var(沒設的話 OAuth 會在 runtime fail)。

## 支援

- **Bug 回報 / 功能建議**:[開 issue](https://github.com/a9876543245/DEEPFOLD-SOLVER/issues)
- **會員相關**:[contact@deepfold.co](mailto:contact@deepfold.co)

回報 bug 時請附上:
- App 版本(視窗右上 / **關於**)
- 當下的 backend 指示燈:**CUDA** / **CPU**
- Windows 版本(設定 → 關於)
- UI 問題請附截圖或螢幕錄影

## 常見問題

**沒有 GPU 能用嗎?**
可以。會自動偵測 fallback 到 CPU。比較慢但策略完全相同。

**支援 macOS / Linux 嗎?**
目前沒有。在 roadmap 上。

**Solver 策略會被上傳嗎?**
不會。全部本機計算。網路只用一次:登入時跟 deepfold.co 確認會員資格。

**更新怎麼運作?**
啟動時 app 會檢查 GitHub 上最新 release。簽章對且更新就會出 banner,一鍵安裝。

## 授權

DEEPFOLD-SOLVER 原始碼公開以利透明。安裝檔僅供 DEEPFOLD PRO 會員使用。© DEEPFOLD — All rights reserved.

[deepfold.co](https://deepfold.co)

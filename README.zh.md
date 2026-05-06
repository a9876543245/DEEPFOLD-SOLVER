# DEEPFOLD-SOLVER

> GPU 加速 GTO 撲克 solver · CPU 自動 fallback · 三語介面 · 一鍵安裝

**[English](README.md) · [中文](README.zh.md) · [日本語](README.ja.md)**

📘 **[使用說明 (中文)](USER_GUIDE.zh.md)** · **[User Guide (English)](USER_GUIDE.md)**

![Platform](https://img.shields.io/badge/platform-Windows%2010%2F11-blue)
![Version](https://img.shields.io/badge/version-1.1.0-green)
![Backend](https://img.shields.io/badge/backend-CUDA%20%2B%20CPU-orange)

DEEPFOLD-SOLVER 是 [DEEPFOLD](https://deepfold.co) 的桌面端 GTO solver。GPU 加速的 DCFR 引擎（含 CPU fallback）+ **runout 聚合報告、per-combo blocker 分析、2,500+ preflop 圖庫**，單一 Windows 安裝檔搞定。

## v1.1.0 新功能

> v1.1.0 是 feature drop，主打**市面 solver 沒有的解後分析工具**。
> 引擎本體也做了一輪嚴肅的資源安全升級。

### 三個旗艦功能

- **Runout Report**（runout 聚合報告） — 解完一次按一鍵，所有 canonical
  turn card 攤成 13×4 grid，按 dominant action 上色。切到 **By Class**
  view，23+ 張 turns 自動分到 **Pair / Flush / Straight / Overcard /
  Brick** 五個 texture bucket，每 bucket 有 weighted strategy + EV。可按
  Best EV / Worst EV / Most aggressive 排序。一鍵匯出 CSV。
- **1326 Combo Drill**（specific combo 拆解） — 點任一 169-class label，
  展開成 4/6/12 specific combos，每張附 **per-combo blocker analysis**：
  你的 hand 擋掉對手 range 的百分比 + Top-5 最常被擋到的對手 class。
  撲克 mixed-strategy 的標準 tie-breaker，終於是一級 UI。
- **Memory Profile** 預設 — `safe / balanced / performance` 三種 profile
  在 solve 前就把 host RAM / JSON / strategy-tree-node 預算定死。Solver
  全程遵守預算 — 不會再有沉默 OOM。

### 引擎資源安全

- **Pre-backend budget gate** — CPU host / GPU VRAM / AUTO fallback 全
  在 allocation **前**評估完。OOM 變成結構化錯誤 + UI badge，不是 crash。
- **CUDA exception-based 錯誤處理** — `CUDA_CHECK` 改成 throw `CudaError`
  而非 `exit()`。失敗時 partial allocation 乾淨 rollback。
- **Chunked GPU matchup 上傳** — 拿掉 host-side `flat_ev` / `flat_valid`
  duplication。改成 per-runout 的 `cudaMemset + cudaMemcpy`。GPU prep 時
  peak host RAM 降低。
- **Strategy tree EV emission 模式** — `none | visible | full` 讓你裁
  JSON 輸出（例 headless benchmark 場景）。
- **測試分層** — ctest 現在有 labeled suite：`smoke`（~13s）、
  `correctness`（~106s）、`stress`（nightly），加 `gpu` / `memory`。

### v1.0.4–1.0.11 延續

- Phase 2 花色同構（monotone / 三花色板 3–7× 加速）
- GPU per-runout matchup table（iso 啟用時 6–10× 比 CPU 快）
- Route A 導航 cache（O(1) 動作切換，不用重 solve）
- Path B runout 選擇器（PioSolver 風格 chance-aware 導航）
- GameContextSelector — Cash 6max/8max + MTT + stack picker

## 下載

**Windows 10 / 11 (x64)** — [最新安裝檔](https://github.com/a9876543245/DEEPFOLD-SOLVER/releases/latest)

裝完後 app 會自動更新：有新版時左上角會出現 banner，點一下即可安裝重啟。

> ⚠️ **首次安裝會看到 Windows 警告**：執行安裝檔時，Windows 會跳出
> 「**Windows 已保護你的電腦**」(SmartScreen) 警告 — 這是正常的，因為
> DEEPFOLD-SOLVER 還沒申請 EV code signing 憑證，Windows 不認識這個 publisher。
> 請點 **更多資訊 (More info)** → **仍要執行 (Run anyway)** 繼續安裝。
> 完整步驟見 [使用說明 — 首次安裝警告](USER_GUIDE.zh.md#附錄-首次安裝-windows-smartscreen-警告)。

## 系統需求

| | 最低 | 建議 |
|---|---|---|
| 作業系統 | Windows 10 64-bit | Windows 11 64-bit |
| CPU | 雙核 | 四核以上 |
| RAM | 4 GB | 8 GB+ |
| GPU | — (CPU fallback 可用) | NVIDIA RTX 2000 系列以上、4 GB+ VRAM |
| 硬碟 | 200 MB | 200 MB |

GPU 自動偵測。右上角的狀態指示燈一眼就能看出 **CUDA** / **CPU**。

## 開始使用

1. 安裝並啟動 app
2. 點 **以 Google 登入** — 系統瀏覽器會自動開啟做 OAuth
3. DEEPFOLD PRO 會員直接進入 solver

還不是會員？到 [deepfold.co](https://deepfold.co) 升級。

## 功能總覽

| 功能 | 說明 |
|---|---|
| **GTO Solver** | Discounted CFR 配上向量化 GPU kernel。一般 turn spot 幾秒內收斂到 sub-percent exploitability |
| **每手策略 grid** | 13×13 grid 在當前決策點按動作 mix 上色。hover 看花色變體拆解 |
| **行動方／對手視角切換** | 同節點切換你的策略 vs 對手的 reach-weighted range |
| **🆕 Runout Report (v1.1.0)** | 解完任何 spot 後一鍵把所有 turn card 攤成 13×4 grid + texture bucket view + 4 種排序 + CSV 匯出。看[使用說明](USER_GUIDE.zh.md#2-runout-report--一眼看完所有-turn-走勢) |
| **🆕 1326 Combo Drill (v1.1.0)** | 任一 169-class 展開成 4/6/12 specific combos，附 per-combo blocker analysis。看[使用說明](USER_GUIDE.zh.md#3-combo-drill--用-blocker-拆-1326-specific-combos) |
| **🆕 Memory Profile (v1.1.0)** | `safe / balanced / performance` 預設管 host RAM / JSON / strategy-tree-node 預算。不再沉默 OOM |
| **Runout 選擇器** | iso 列舉啟用時，點任一 canonical river card 切換子樹 |
| **GTO 圖庫** | 內建 2,550+ preflop 場景，app 內瀏覽。一鍵套用為 IP / OOP range |
| **下注尺度預設** | Standard / Polar / Small Ball — 同時影響 solver tree 跟 UI 按鈕 |
| **訓練模式** | 10 題 drill 對你的答案打分（vs 均衡） |
| **預解 spot 圖庫** | 120+ 常見 flop spot，一鍵載入 |
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

引擎是獨立的 CLI（`deepsolver_core.exe`），以 Tauri sidecar 形式打包。Tauri 每次 solve 啟一個 process 然後 parse JSON 結果，內含完整策略樹供 client 端導航。

## 從原始碼編譯

需要：
- **Node.js 20+** + **npm**
- **Rust 1.78+**（`rustup`）
- **CMake 3.20+** + **MSVC 2022**（Windows）
- **CUDA Toolkit 12.x**（選用 — CPU build 不需要）

```sh
git clone https://github.com/a9876543245/DEEPFOLD-SOLVER.git
cd DEEPFOLD-SOLVER
npm install

# 編譯 C++ 引擎
cd core && mkdir -p build && cd build
cmake .. -DBUILD_TESTS=ON
cmake --build . --config Release
ctest

# 開發模式（自動找 ../core/build/Release/ 的引擎 binary）
cd ../..
npm run tauri dev
```

> **注意**：內建的 GTO preflop 圖庫資料（`gto_output/`，~31MB）跟預編譯的
> 引擎 sidecar（`src-tauri/binaries/`）**沒有放在這個 repo**，只在官方
> 安裝檔裡有。從 source build 出來的版本不會有圖庫資料，除非你自己提供
> `gto_output/` 目錄並符合相同 JSON schema。登入功能還需要在 build 環境
> 設 `DEEPFOLD_GOOGLE_CLIENT_SECRET` env var（沒設的話 OAuth 會在 runtime fail）。

## 支援

- **Bug 回報 / 功能建議**：[開 issue](https://github.com/a9876543245/DEEPFOLD-SOLVER/issues)
- **會員相關**：[contact@deepfold.co](mailto:contact@deepfold.co)

回報 bug 時請附上：
- App 版本（視窗右上 / **關於**）
- 當下的 backend 指示燈：**CUDA** / **CPU**
- Windows 版本（設定 → 關於）
- UI 問題請附截圖或螢幕錄影

## 常見問題

**沒有 GPU 能用嗎？**
可以。會自動偵測 fallback 到 CPU。比較慢但策略完全相同。

**支援 macOS / Linux 嗎？**
v1.0.x 沒有。在 roadmap 上。

**Solver 策略會被上傳嗎？**
不會。全部本機計算。網路只用一次：登入時跟 deepfold.co 確認會員資格。

**更新怎麼運作？**
啟動時 app 會檢查 GitHub 上最新 release。簽章對且更新就會出 banner，一鍵安裝。

## 授權

DEEPFOLD-SOLVER 原始碼公開以利透明。安裝檔僅供 DEEPFOLD PRO 會員使用。© DEEPFOLD — All rights reserved.

[deepfold.co](https://deepfold.co)

# DEEPFOLD-SOLVER

> GPU 加速 GTO 撲克 solver · CPU 自動 fallback · 三語介面 · 一鍵安裝

**[English](README.md) · [中文](README.zh.md) · [日本語](README.ja.md)**

![Platform](https://img.shields.io/badge/platform-Windows%2010%2F11-blue)
![Version](https://img.shields.io/badge/version-1.0.4-green)
![Backend](https://img.shields.io/badge/backend-CUDA%20%2B%20CPU-orange)

DEEPFOLD-SOLVER 是 [DEEPFOLD](https://deepfold.co) 的桌面端 GTO solver。把 GPU 加速的 DCFR 引擎（含 CPU fallback）跟 **2,500+ preflop 場景圖庫**綁在一起，單一 Windows 安裝檔搞定。

## v1.0.4 新功能

- **Phase 2 花色同構** — runout 列舉採用 PioSolver / GTO+ 風格的對稱壓縮。Monotone 跟三花色板可加速 **3-7 倍**，策略品質完全沒退步
- **GPU per-runout matchup table** — chance node 列舉走 CUDA。iso 啟用時 **6-10 倍** 比 CPU 快
- **Route A 導航 cache** — 解一次，到處點。UI 動作切換現在是 **O(1) cache 查找**，不用每次重 solve。再也不用每點一下等 8 秒
- **Path B runout 選擇器** — iso 把多張 canonical river card 列出來時，UI 會顯示 runout 選擇器。點任一張 canonical card 切換到該子樹策略。PioSolver 風格的 chance-aware 導航
- **GTO 圖庫** — 內建 2,550+ preflop 場景（Cash 6max/8max + MTT），可在 app 內瀏覽，一鍵套用為 IP / OOP range

## 下載

**Windows 10 / 11 (x64)** — [最新安裝檔](https://github.com/a9876543245/DEEPFOLD-SOLVER/releases/latest)

裝完後 app 會自動更新：有新版時左上角會出現 banner，點一下即可安裝重啟。

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
| **Runout 選擇器 (v1.0.4)** | iso 列舉啟用時，點任一 canonical river card 切換子樹 |
| **GTO 圖庫 (v1.0.4)** | 內建 2,550+ preflop 場景，app 內瀏覽。一鍵套用為 IP / OOP range |
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

# DEEPFOLD-SOLVER

> GPU 加速 GTO 撲克 solver · CPU 自動 fallback · 三語介面 · 一鍵安裝

**[English](README.md) · [中文](README.zh.md) · [日本語](README.ja.md)**

📘 **[使用說明 (中文)](USER_GUIDE.zh.md)** · **[User Guide (English)](USER_GUIDE.md)**

![Platform](https://img.shields.io/badge/platform-Windows%2010%2F11-blue)
![Version](https://img.shields.io/badge/version-1.3.1-green)
![Backend](https://img.shields.io/badge/backend-CUDA%20%2B%20CPU-orange)

DEEPFOLD-SOLVER 是 [DEEPFOLD](https://deepfold.co) 的桌面端 GTO solver。GPU 加速的 DCFR 引擎（含 CPU fallback）+ **runout 聚合報告、per-combo blocker 分析、EV / 激進度熱力圖、2,500+ preflop 圖庫**，單一 Windows 安裝檔搞定。

## v1.3.1 新功能（慢 CPU 上時間預算正確生效）

v1.3.0 ship 後使用者用 GTX 1070 Max-Q laptop 撞到 bug：300 iter 的 spot
在 engine 的 time_budget(300s) 來得及觸發前，Tauri 外層 subprocess timeout
(720s) 先把 engine 砍了。Root cause：那台 laptop CPU 太慢，9k 節點的
turn solve **單一 iter 就跑了超過 720s**，永遠走不到下一個 budget check 點。

Tauri 外層 timeout 現在會跟 time_budget 連動：
`min(budget × 3 + 90s, 1800s)`。給正在跑的 iter 足夠時間完成，engine 自己
的 budget 檢查才能觸發。大 spot 在慢機器上現在會正確在 budget 停下，
而不是死在「Engine timed out」。

錯誤訊息也變聰明了 — timeout 觸發 + budget 有設時，會說「**你的 spot
太大、超過這台機器在 X 秒預算內能解的範圍**」而不是泛用的「請減少 iter」。

## v1.3.0 新功能（時間預算 + Stop 按鈕）

解決等待懸崖。源於 v1.2.2 使用者回報：「既然 5 分鐘要停了，秀 3 小時 ETA 沒意義」。

### Solve mode 預設

Solve 按鈕上方多了三個 pill，把 iter cap + 時間預算 + exploitability 目標一鍵打包：

| 模式 | iter 上限 | 時間預算 | exploit 目標 | 用途 |
|---|---|---|---|---|
| **Quick**    | 100  | 60 秒   | 1.5%   | 快速看大方向 |
| **Standard** | 300  | 5 分鐘  | 0.5%   | **預設** — Pro 等級玩家可用 |
| **Deep**     | 1000 | 15 分鐘 | 0.2%   | 細部研究 |

任一條件先到就停。CFR 是 anytime — iter N 的 running average 就是當前策略，
budget 到就停拿到的是「夠用」的結果，不是半成品。

### Stop 按鈕 + Quality badge

- **Stop 按鈕**（loading 時取代 Solve）— 純 abort，不保留部分結果。「停下來給我現在這版」
  走時間預算（自動觸發）。
- **Quality badge** 在結果面板：🟢 高品質 / 🟡 可用 / 🟠 粗略 / 🔴 低信心，
  按最終 exploitability 分級。如果是因為 budget 停的，會提示「換 Deep 模式」。

### ETA banner 重做

標題現在顯示 `min(estimate, time_budget)`，不再無上限預估。不會再對使用者說
「Estimated 5 hours on CPU」結果只等 5 分鐘。下方副標仍給原始 estimate
讓使用者了解「budget 是真的會停下來，不是說我們會在那時間內收斂」。

### Throughput 重 calibrate

Pre-solve estimator 之前 GPU 端低估 50×（用 hand-wave 估的；現在用 RTX 5090
跑 `--benchmark standard` 實測 568 Gops/s 重訂）。CPU rate 也提 3×。
之前估「11 分鐘」的 spot 現在估「12 秒」，跟實際對得起來。

### 引擎

- 新 `--time-budget-seconds <s>` CLI flag，**每個 iter** 檢查，慢 per-iter
  的 spot 也能精準到 budget 停（不會晚個 5 分鐘）。
- 新 `early_stop_reason` field 在 result JSON：`iter_cap` / `time_budget`。
- 新 `cancel_solve` Tauri command（taskkill 殺 process）。

## v1.2.2 新功能（解算前 ETA + 多 arch CUDA）

使用者看得到的：

- **解算前 ETA banner** — 按 *Solve* 之後在 iteration 開始前，先用 sub-second
  的 `--estimate-only` 引擎呼叫拿到「Estimated 12 minutes on CPU」這種預估。
  以前等了 5 分鐘還沒結果不知道在幹嘛的 spot，現在 click solve 馬上就知道
  wall-clock 大概多久。超過 60s 是琥珀色警告、超過 30 分鐘是紅色警告。
  AUTO 因 GPU 被排除（例如 Pascal 卡需要 CUDA 12.x build）而 fallback 到 CPU
  時，原因會直接顯示在 banner 上。
- **多 arch CUDA build** — installer 現在內建 Turing (RTX 20 系列)、Ampere
  (RTX 30)、Ada (RTX 40)、Hopper (H100) 的原生 SASS，加上給 Blackwell
  (RTX 5090) 的 PTX-JIT 前向相容。之前只有 Ada 是原生的，其他卡每次第一次
  跑都要付 JIT 編譯成本。
- **AUTO fallback 診斷** — AUTO 因 GPU 被排除而走 CPU 時，resources block
  會載明實際原因。Pascal (GTX 10 系列) 跟 Volta (Titan V) 的卡會看到
  「current build uses CUDA 13.x which dropped Pascal/Volta」說明，並指出
  Pascal-friendly CUDA-12.x build 預計 v1.3.0 推出。

引擎內部：

- `SolveResources` 加了 `ops_per_iteration` / `backend_for_estimate` /
  `estimated_solve_seconds`，post-solve calibration 跟 pre-solve estimate 對
  得起來（benchmark CLI / regression tracking 也用得到）。
- 新 `Solver::estimate_only()` method — 只跑 iso + tree build，不做
  precompute_matchups 也不跑 iterations。一般 spot sub-second。
- 新 `--estimate-only` CLI flag 跟 Tauri `estimate_solve` command。

## v1.2.1 新功能（記憶體控制強化）

外部 review 抓出兩個 memory budget 漏洞，這個 patch release 修掉：

- **VISIBLE EV cache 現在會吃 `strategy_tree_max_nodes` 上限**：之前 EV cache
  的 pre-walk 沒有 cap，大 tree + 小 node cap 時 EV cache RAM（兩個
  `std::map<uint32_t, std::vector<float>>`）可能比實際 JSON payload 大幾個
  數量級。現在會跟 JSON walk 用同一個 cap。`resources.estimated_strategy_tree_bytes`
  也會隨 cap 正確下降。
- **共同 host budget gate 現在 GPU backend 也會檢查**：之前 host gate 只在
  `selected_backend == CPU` 時跑，GPU solve 對 matchup tables、strategy_tree
  EV cache、JSON response 都靜默 bypass host RAM 上限（這三者不論 backend
  都在 host RAM）。拆成 common-host gate（永遠檢查）+ CPU-specific add
  (cpu_state)。錯誤訊息會明確說「換 GPU 也救不了 common-host 超量」。
- 新的 ctest regression guard：`CliEvCacheRespectsCap`、`CliGpuCommonHostBudgetReject`。

## v1.2.0 新功能

> v1.2.0 加入兩個新的策略 grid 視圖、一個記憶體配置選擇器，以及多項
> solver 端可靠性改進。

### Grid 視圖模式

169 策略 grid 上方多了切換工具列，4 個視圖：

- **Strategy Mix**（預設）— 每格的多動作漸層
- **EV**（新）— per-class EV 熱力圖，按 in-range cells 的 EV 範圍 normalize 紅→灰→綠。一眼看出「哪些 combo 是利潤中心、哪些在虧」
- **Aggression**（新）— Bet/Raise/All-in 頻率合計，冷→熱漸層。回答「這個 class 多常打 aggressive line vs passive？」
- **Heatmap** — 單一動作強度（例如「只看 Bet 75% 頻率」）

### Solve 控制

- **Memory Profile selector** — Advanced 區加 `safe / balanced / performance` pill 按鈕，即時顯示每個 profile 的 host RAM / JSON / strategy-tree-node 預算。預設 `balanced` 跟引擎與 Rust resolver 一致。

### 引擎

- **JSON cap as action** — 當預估 JSON 回應超過設定預算時，solver 現在會在 budget gate 之前**自動降低** `strategy_tree_max_nodes` 來 fit。之前的行為是直接讓 gate 失敗；現在使用者拿到較小但可用的導航 cache，加上 `resources.diagnostic` 說明降了多少、為什麼。
- **`--benchmark standard` CLI flag** — 可重現的 perf 追蹤預設（AsKd7c rainbow、full ranges、100 iter）。輸出緊湊的 JSON 含 `iterations_per_sec` / `nodes_per_sec` / `memory_estimate_mb` + 完整 timing breakdown。CI regression 追蹤可直接 grep。

### 可靠性

- **Tauri timeout integration test**（`src-tauri/tests/timeout_kill.rs`）— Phase 5 引擎清理修復的 regression guard。Spawn engine、讓 timeout 觸發、確認 child process 被乾淨砍掉（不會留 zombie 佔住 GPU 記憶體）。
- **Release script 可靠性** — `scripts/prepare-release.ps1` 現在會繞過 Tauri 內建 minisign 簽章（在 PowerShell 5.1 + npm.cmd 環境會卡 stdin 密碼提示），改在 build 後用 `--password` CLI flag 明確簽章。對使用者端驗證流程沒變，但不會再遇到「.exe 出來但 .sig 沒生」的 release 卡關。

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
| **🆕 Grid 視圖模式 (v1.2.0)** | 169 grid 上方工具列切換 Strategy Mix / **EV** / **Aggression** / 單動作 heatmap。EV 模式按 in-range cells 範圍 normalize 紅→灰→綠 |
| **🆕 Memory Profile selector (v1.2.0)** | Advanced 區加 `safe / balanced / performance` pill，即時顯示預算。透過 `--memory-profile` 通到引擎 |
| **🆕 Benchmark CLI (v1.2.0)** | `deepsolver_core --benchmark standard` 跑可重現的 AsKd7c+100iter 場景，輸出緊湊 perf 追蹤 JSON |
| **Runout Report (v1.1.0)** | 解完任何 spot 後一鍵把所有 turn card 攤成 13×4 grid + texture bucket view + 4 種排序 + CSV 匯出。看[使用說明](USER_GUIDE.zh.md#2-runout-report--一眼看完所有-turn-走勢) |
| **1326 Combo Drill (v1.1.0)** | 任一 169-class 展開成 4/6/12 specific combos，附 per-combo blocker analysis。看[使用說明](USER_GUIDE.zh.md#3-combo-drill--用-blocker-拆-1326-specific-combos) |
| **Memory Profile (v1.1.0)** | `safe / balanced / performance` 預設管 host RAM / JSON / strategy-tree-node 預算。不再沉默 OOM |
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

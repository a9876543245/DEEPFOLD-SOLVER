# DEEPFOLD-SOLVER

> GPU アクセラレーション GTO ポーカーソルバー · CPU フォールバック対応 · 三言語 UI · ワンクリックインストーラー

**[English](README.md) · [中文](README.zh.md) · [日本語](README.ja.md)**

📘 **[User Guide (English)](USER_GUIDE.md)** · **[使用說明 (中文)](USER_GUIDE.zh.md)**

![Platform](https://img.shields.io/badge/platform-Windows%2010%2F11-blue)
![Backend](https://img.shields.io/badge/backend-CUDA%20%2B%20CPU%20%28AVX2%2BMulti--core%29-orange)

DEEPFOLD-SOLVER は [DEEPFOLD](https://deepfold.co) のデスクトップ GTO ソルバーです。GPU アクセラレーション DCFR エンジンに、**全ての CPU コアを線形にスケールさせるマルチコア後端**を組み合わせ、**runout 集計、コンボ別ブロッカー解析、EV/アグレ度ヒートマップ、2,500+ プリフロップ チャート** とともに Windows ワンクリックインストーラーに同梱しています。

## DEEPFOLD-SOLVER の核心となる特徴

### 全ての CPU コアを使い切るエンジン

- **デュアル後端 DCFR** — GPU があれば GPU、なければ CPU、どちらでも数値戦略は完全一致。GPU パスは Turing / Ampere / Ada / Hopper のネイティブ CUDA SASS を同梱、Blackwell は PTX-JIT 前方互換。CPU パスは BFS-flat「levelized」CFR 後端で、2 スレッドで頭打ちするのではなく物理コア数まで線形にスケール。
- **ランタイム CPUID ディスパッチ** — Haswell 以降は AVX2 カーネル、それより古い CPU は scalar に自動フォールバック。単一バイナリ、別ビルド不要、2013 年以前の CPU でも起動時クラッシュなし。
- **コミット毎の parity gate** — `reference vs levelized`、`scalar vs AVX2`、`1 スレッド vs N スレッド` の全パスがビット単位で同一の戦略を出力することを毎ビルド検証。高速パスが正確性 oracle から無音で乖離することはありません。

### 時間予算を本当に守る Solve mode プリセット

- **Quick / Standard / Deep** ピルが iter 上限 + 時間予算 + exploitability 目標を一括設定。最初に達した条件で停止 — CFR は anytime アルゴリズムで、iter N の running average が**そのまま**戦略なので、予算で停止しても使える戦略であって半端な結果ではありません。
- **解算前 ETA バナー** — Solve クリックで sub-second の `--estimate-only` エンジン呼び出しが走り、イテレーション開始前に標準ベンチマーク校正済みの wall-clock 予測を表示。AUTO が GPU 除外でフォールバックした場合の理由(例:「Pascal は CUDA-12.x build が必要」)も予測の隣に表示。
- **Stop ボタン + Quality バッジ** — 完全 abort、部分結果は保存しない(「今あるものを返せ」は時間予算経由)。結果パネルは最終 exploitability に応じて 🟢 高 / 🟡 良 / 🟠 粗 / 🔴 低信頼度 のバッジ表示。
- **`--time-budget-seconds` は iter 毎に確認**、遅い per-iter スポットでも予算を 5 分超過することなく正確に停止。

### 他のソルバーにない解析後インサイトツール

- **Runout Report** — ソルブ後ワンクリックで全ての正準 turn カードを 13×4 グリッドに展開、dominant action で色分け。**By Class** ビューに切り替えると 23+ 枚の turn が **Pair / Flush / Straight / Overcard / Brick** の 5 つのテクスチャバケットに自動分類され、加重ストラテジー + EV を表示。Best EV / Worst EV / Most aggressive でソート可能、CSV エクスポート対応。
- **1326 Combo Drill** — 任意の 169-class ラベルをクリックすると、4 / 6 / 12 specific combo を **per-combo ブロッカー解析** 付きで展開。各ハンドが相手レンジをどれだけブロックしているか + 最もブロックされる相手クラスの Top-5 を表示。ポーカーで mixed strategy を選ぶ際の標準的なタイブレーカーが、ついに一級 UI として実装。
- **戦略グリッド表示モード** — 169 グリッド上部のツールバーで Strategy Mix(既定のマルチアクション勾配)、**EV**(per-class ヒートマップ、in-range セルの EV 範囲で normalize、赤→グレー→緑)、**Aggression**(Bet/Raise/All-in 頻度の合計、クール→ホット)、単一アクション Heatmap を切替。EV モードは「どのコンボが利益源か、どれが負けているか」が一目瞭然。

### 信頼できるメモリ制御

- **Memory Profile プリセット** — `safe / balanced / performance` で host RAM、JSON、strategy-tree-node の予算を事前にライブプレビュー付きで決定。ソルバーは予算を全工程で遵守 — pre-backend ゲートが CPU host / GPU VRAM / AUTO fallback をアロケーション**前**に評価、OOM は構造化エラー + UI バッジになり、クラッシュしません。
- **共通ホストバジェットゲートは GPU バックエンドにも適用** — matchup tables、strategy-tree EV cache、JSON response はバックエンドに関わらず host RAM 常駐、全て検査されます。診断メッセージは「GPU に切り替えても common-host のオーバーフローは解決しない」と明示。
- **Chunked GPU マッチアップ アップロード** が host 側の `flat_ev` / `flat_valid` 重複を排除、runout 単位の `cudaMemset + cudaMemcpy` に変更し、GPU prep 時のピーク host RAM を低減。

### 同梱コンテンツ

- **2,550+ プリフロップ シナリオ** をアプリ内ブラウズ。ワンクリックで IP / OOP レンジに適用。
- **120+ 解析済み flop スポット** をワンクリックで読込。
- **ベットサイジング プリセット** — Standard / Polar / Small Ball — ソルバーツリーと UI ボタンの両方に反映。
- **レンジエディタ + ノードロック** — 任意のコンボの頻度を上書きして再ソルブ。
- **トレーニングモード** — 10 問のドリルが均衡解と比較して回答を採点。

### 運用上の磨き

- **三言語 UI** — English / 中文 / 日本語、いつでも切替可能。
- **自動更新** — バナーからのワンクリックインストール、署名付きリリース、install mode `passive`。
- **スート同型** がモノトーン / 3-of-suit ボードで自動的に 3〜7 倍高速化、GPU per-runout マッチアップテーブルが iso 有効時に CPU 比 6〜10 倍。
- **Route A ナビゲーションキャッシュ** — O(1) アクション切替、再ソルブ不要。**Path B runout セレクタ** で PioSolver 風の chance-aware ナビゲーション。
- **再現可能なベンチマーク** — `deepsolver_core --benchmark standard` が AsKd7c rainbow / 100 iter シナリオを実行、コンパクトな perf JSON(`iterations_per_sec` / `nodes_per_sec` / `memory_estimate_mb` + 完全な timing 内訳)を出力。CI regression 追跡で grep 可能。

## ダウンロード

**Windows 10 / 11 (x64)** — [最新インストーラー](https://github.com/a9876543245/DEEPFOLD-SOLVER/releases/latest)

インストール後、アプリは自動更新されます。新バージョンが利用可能になると左上にバナーが表示され、ワンクリックでインストール&再起動。

> ⚠️ **初回インストール時に Windows の警告が表示されます**:インストーラーを
> 起動すると Windows が「**WindowsによってPCが保護されました**」(SmartScreen)
> 警告を表示します。これは想定内です — DEEPFOLD-SOLVER はまだ EV コード署名
> 証明書を取得していないため、Windows が発行元を認識しません。
> **詳細情報** → **実行** をクリックしてインストールを続行してください。
> 完全な手順は [User Guide — SmartScreen warning](USER_GUIDE.md#appendix-first-install-windows-smartscreen-warning) を参照。

## システム要件

| | 最小 | 推奨 |
|---|---|---|
| OS | Windows 10 64-bit | Windows 11 64-bit |
| CPU | x86-64 デュアルコア(年代不問) | 4 物理コア以上、AVX2 対応(Haswell 2013 / Excavator 2015 以降) |
| RAM | 4 GB | 8 GB+ |
| GPU | — *(CPU 後端のみで完全機能)* | NVIDIA RTX 2000 シリーズ以上、4 GB+ VRAM |
| ディスク | 200 MB | 200 MB |

GPU と SIMD はどちらも自動検出。右上のステータスピルで **CUDA** / **CPU** が一目瞭然、CPU 後端はハードウェアの対応状況に応じて `AVX2` または `scalar` を表示します — Pre-Haswell CPU は scalar kernels で動作し、AVX2 opcode に触れることはありません。

## 始め方

1. インストールしてアプリを起動
2. **Google でサインイン** をクリック — システムブラウザが OAuth 用に開きます
3. DEEPFOLD PRO メンバーはそのままソルバーへ

まだメンバーでない方は [deepfold.co](https://deepfold.co) でアップグレード。

## 機能一覧

| 機能 | 説明 |
|---|---|
| **GTO ソルバー** | ベクトル化 GPU カーネルとマルチコア CPU 後端を備えた Discounted CFR。一般的な turn スポットを数秒で sub-percent exploitability まで収束 |
| **コンボ別戦略グリッド** | 13×13 グリッドが現在の決定ノードでアクションミックスごとに色分け。ホバーでスーテッド変種の内訳 |
| **アクション側 ↔ 相手視点切替** | 同じノードで自分の戦略 vs 相手の reach 加重レンジを切替 |
| **グリッド表示モード** | 169 グリッド上部のツールバーで Strategy Mix / **EV** / **Aggression** / 単一アクションヒートマップを切替 |
| **Memory Profile セレクタ** | Advanced 設定の `safe / balanced / performance` ピル + 予算ライブプレビュー |
| **Benchmark CLI** | `deepsolver_core --benchmark standard` で再現可能な AsKd7c+100iter シナリオを実行、コンパクトなパフォーマンス JSON を出力 |
| **Runout Report** | ソルブ後ワンクリックで全 turn を 13×4 グリッド + テクスチャバケットビュー + 4 ソートモード + CSV エクスポート。詳細は [User Guide](USER_GUIDE.md#2-runout-report--see-every-turn-at-once) |
| **1326 Combo Drill** | 任意の 169-class を 4/6/12 specific combo に展開、相手レンジ vs per-combo ブロッカー解析付き。詳細は [User Guide](USER_GUIDE.md#3-combo-drill--break-169-classes-into-specific-combos) |
| **Memory Profile** | `safe / balanced / performance` プリセットで host RAM / JSON / strategy-tree-node 予算を境界決定。サイレント OOM なし |
| **Runout ピッカー** | iso 列挙有効時、任意の正準 river カードをクリックして枝切替 |
| **GTO チャートライブラリ** | 2,550 件以上のプリフロップシナリオをアプリ内ブラウズ。ワンクリックで IP / OOP レンジに適用 |
| **ベットサイジング プリセット** | Standard / Polar / Small Ball — ソルバーツリーと UI ボタンの両方に反映 |
| **トレーニングモード** | 10 問のドリルが均衡解と比較して回答を採点 |
| **解析済みスポットライブラリ** | 120 件以上の一般的な flop スポット、ワンクリックで読込 |
| **レンジエディタ + ノードロック** | 任意のコンボの頻度を上書きして再ソルブ |

## アーキテクチャ

```
┌─────────────────────────────────────────────┐
│  React + TypeScript UI (Tauri webview)      │
│  ├── 戦略グリッド · Runout ピッカー         │
│  └── GTO チャートブラウザ                   │
├─────────────────────────────────────────────┤
│  Rust (Tauri) — IPC + チャートローダー      │
├─────────────────────────────────────────────┤
│  C++ エンジン (deepsolver_core)             │
│  ├── DCFR (CPU)                             │
│  ├── CUDA カーネル (GPU)                    │
│  └── スート同型 + per-runout マッチアップ    │
└─────────────────────────────────────────────┘
```

エンジンは独立した CLI(`deepsolver_core.exe`)で、Tauri サイドカーとして同梱されます。Tauri はソルブごとにプロセスを起動し JSON 結果をパース。クライアント側ナビゲーション用の完全な戦略ツリーを含みます。

## ソースからビルド

必要なもの:
- **Node.js 20+** + **npm**
- **Rust 1.78+**(`rustup`)
- **CMake 3.20+** + **MSVC 2022**(Windows)
- **CUDA Toolkit 12.x**(オプション — CPU ビルドは不要)

```sh
git clone https://github.com/a9876543245/DEEPFOLD-SOLVER.git
cd DEEPFOLD-SOLVER
npm install

# C++ エンジンをビルド
cd core && mkdir -p build && cd build
cmake .. -DBUILD_TESTS=ON
cmake --build . --config Release
ctest

# 開発モード(../core/build/Release/ のエンジンバイナリを自動検出)
cd ../..
npm run tauri dev
```

> **注意**:同梱の GTO プリフロップチャートデータ(`gto_output/`、約31MB)と
> プリビルド済みエンジンサイドカー(`src-tauri/binaries/`)は **このリポジトリ
> には含まれていません**。公式インストーラーにのみ同梱されています。ソースから
> ビルドした版にはチャートデータが入らないため、必要なら同じ JSON スキーマで
> `gto_output/` ディレクトリをリポジトリルートに自分で用意してください。サインインは
> ビルド環境に `DEEPFOLD_GOOGLE_CLIENT_SECRET` の env var を設定する必要があります
> (未設定の場合、OAuth は実行時に失敗します)。

## サポート

- **バグ報告 / 機能リクエスト**:[Issue を開く](https://github.com/a9876543245/DEEPFOLD-SOLVER/issues)
- **会員関連**:[contact@deepfold.co](mailto:contact@deepfold.co)

バグ報告には以下を添えてください:
- アプリのバージョン(ウィンドウ右上 / **About**)
- そのときの backend ピル:**CUDA** / **CPU**
- Windows のバージョン(設定 → 詳細情報)
- UI の問題の場合はスクリーンショットまたは画面録画

## よくある質問

**GPU なしでも動きますか?**
はい。自動検出して CPU にフォールバックします。遅くなりますが戦略は完全に同一。

**macOS / Linux 対応は?**
現在はなし。ロードマップにあります。

**ソルバーの戦略はどこかにアップロードされますか?**
いいえ。すべてローカルで実行されます。ネットワーク通信は deepfold.co への会員資格チェック(サインイン時)のみ。

**アップデートはどう動きますか?**
起動時にアプリが GitHub の最新リリースをチェック。署名が有効で新しいものがあればバナーを表示、ワンクリックでインストール。

## ライセンス

DEEPFOLD-SOLVER のソースコードは透明性のために公開されています。インストーラーは DEEPFOLD PRO メンバー向けです。© DEEPFOLD — All rights reserved.

[deepfold.co](https://deepfold.co)

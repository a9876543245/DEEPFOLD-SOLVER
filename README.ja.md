# DEEPFOLD-SOLVER

> GPU アクセラレーション GTO ポーカーソルバー · CPU フォールバック対応 · 三言語 UI · ワンクリックインストーラー

**[English](README.md) · [中文](README.zh.md) · [日本語](README.ja.md)**

📘 **[User Guide (English)](USER_GUIDE.md)** · **[使用說明 (中文)](USER_GUIDE.zh.md)**

![Platform](https://img.shields.io/badge/platform-Windows%2010%2F11-blue)
![Version](https://img.shields.io/badge/version-1.1.0-green)
![Backend](https://img.shields.io/badge/backend-CUDA%20%2B%20CPU-orange)

DEEPFOLD-SOLVER は [DEEPFOLD](https://deepfold.co) のデスクトップ GTO ソルバーです。GPU アクセラレーション DCFR エンジン（CPU フォールバック完備）に **runout 集計、コンボ別ブロッカー解析、2,500+ プリフロップ チャート** を加え、Windows 用ワンクリックインストーラーに同梱しています。

## v1.1.0 のハイライト

> v1.1.0 は **市販ソルバーにない解析後インサイトツール** に焦点を当てた
> 機能リリースです。エンジン本体もリソース安全性を本格的に強化しました。

### 旗艦機能 3 つ

- **Runout Report** — ソルブ後ワンクリックで、列挙された全ての正準 turn
  カードを 13×4 グリッドに展開。dominant action で色分け。**By Class**
  ビューに切り替えると 23+ 枚の turn が **Pair / Flush / Straight /
  Overcard / Brick** の 5 つのテクスチャバケットに自動分類され、加重
  ストラテジー + EV を表示。Best EV / Worst EV / Most aggressive で
  ソート可能。CSV エクスポート対応。
- **1326 Combo Drill** — 任意の 169-class ラベルをクリックすると、その
  クラスの 4/6/12 specific combo を **per-combo ブロッカー解析** 付きで
  展開。各ハンドが相手レンジをどれだけブロックしているか + 最もブロック
  される相手クラスの Top-5 を表示。ポーカーで mixed strategy を選ぶ際の
  標準的なタイブレーカーが、ついに一級 UI として実装。
- **Memory Profile** — `safe / balanced / performance` プロファイルで
  host RAM / JSON / strategy-tree-node の予算をソルブ前に決定。ソルバー
  全体が予算を守る — もうサイレント OOM はありません。

### エンジンのリソース安全性

- **Pre-backend budget gate** — CPU host / GPU VRAM / AUTO fallback を
  アロケーション**前**に評価。OOM シナリオは構造化エラー + UI バッジに
  なり、クラッシュしません。
- **CUDA 例外ベースのエラー処理** — `CUDA_CHECK` が `exit()` でなく
  `CudaError` を throw。失敗時は partial allocation がきれいに rollback。
- **Chunked GPU マッチアップ アップロード** — host 側の `flat_ev` /
  `flat_valid` 重複を排除。runout 単位の `cudaMemset + cudaMemcpy` に変更。
  GPU prep 時のピーク host RAM を低減。
- **Strategy tree EV emission モード** — `none | visible | full` で JSON
  出力をトリム可能（headless ベンチマーク用途等）。
- **テスト階層化** — ctest にラベル付きスイート: `smoke`（~13s）、
  `correctness`（~106s）、`stress`（nightly）、`gpu` / `memory`。

### v1.0.4–1.0.11 から継続

- Phase 2 スート同型（モノトーン・3-of-suit ボードで 3〜7 倍高速化）
- GPU per-runout マッチアップテーブル（iso 有効時 CPU 比 6〜10 倍）
- Route A ナビゲーションキャッシュ（O(1) アクション切替、再ソルブ不要）
- Path B runout セレクタ（PioSolver スタイル chance-aware ナビゲーション）
- GameContextSelector — Cash 6max/8max + MTT + スタックピッカー

## ダウンロード

**Windows 10 / 11 (x64)** — [最新インストーラー](https://github.com/a9876543245/DEEPFOLD-SOLVER/releases/latest)

インストール後、アプリは自動更新されます。新バージョンが利用可能になると左上にバナーが表示され、ワンクリックでインストール＆再起動。

## システム要件

| | 最小 | 推奨 |
|---|---|---|
| OS | Windows 10 64-bit | Windows 11 64-bit |
| CPU | デュアルコア | クアッドコア以上 |
| RAM | 4 GB | 8 GB+ |
| GPU | — (CPU フォールバック可) | NVIDIA RTX 2000 シリーズ以上、4 GB+ VRAM |
| ディスク | 200 MB | 200 MB |

GPU は自動検出。右上のステータスピルで **CUDA** / **CPU** が一目瞭然。

## 始め方

1. インストールしてアプリを起動
2. **Google でサインイン** をクリック — システムブラウザが OAuth 用に開きます
3. DEEPFOLD PRO メンバーはそのままソルバーへ

まだメンバーでない方は [deepfold.co](https://deepfold.co) でアップグレード。

## 機能一覧

| 機能 | 説明 |
|---|---|
| **GTO ソルバー** | ベクトル化 GPU カーネル付き Discounted CFR。一般的な turn スポットを数秒で sub-percent exploitability まで収束 |
| **コンボ別戦略グリッド** | 13×13 グリッドが現在の決定ノードでアクションミックスごとに色分け。ホバーでスーテッド変種の内訳 |
| **アクション側 ↔ 相手視点切替** | 同じノードで自分の戦略 vs 相手の reach 加重レンジを切替 |
| **🆕 Runout Report (v1.1.0)** | ソルブ後ワンクリックで全 turn を 13×4 グリッド + テクスチャバケットビュー + 4 ソートモード + CSV エクスポート。詳細は [User Guide](USER_GUIDE.md#2-runout-report--see-every-turn-at-once) |
| **🆕 1326 Combo Drill (v1.1.0)** | 任意の 169-class を 4/6/12 specific combo に展開、相手レンジ vs per-combo ブロッカー解析付き。詳細は [User Guide](USER_GUIDE.md#3-combo-drill--break-169-classes-into-specific-combos) |
| **🆕 Memory Profile (v1.1.0)** | `safe / balanced / performance` プリセットで host RAM / JSON / strategy-tree-node 予算を境界決定。サイレント OOM なし |
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

エンジンは独立した CLI（`deepsolver_core.exe`）で、Tauri サイドカーとして同梱されます。Tauri はソルブごとにプロセスを起動し JSON 結果をパース。クライアント側ナビゲーション用の完全な戦略ツリーを含みます。

## ソースからビルド

必要なもの：
- **Node.js 20+** + **npm**
- **Rust 1.78+**（`rustup`）
- **CMake 3.20+** + **MSVC 2022**（Windows）
- **CUDA Toolkit 12.x**（オプション — CPU ビルドは不要）

```sh
git clone https://github.com/a9876543245/DEEPFOLD-SOLVER.git
cd DEEPFOLD-SOLVER
npm install

# C++ エンジンをビルド
cd core && mkdir -p build && cd build
cmake .. -DBUILD_TESTS=ON
cmake --build . --config Release
ctest

# 開発モード（../core/build/Release/ のエンジンバイナリを自動検出）
cd ../..
npm run tauri dev
```

> **注意**：同梱の GTO プリフロップチャートデータ（`gto_output/`、約31MB）と
> プリビルド済みエンジンサイドカー（`src-tauri/binaries/`）は **このリポジトリ
> には含まれていません**。公式インストーラーにのみ同梱されています。ソースから
> ビルドした版にはチャートデータが入らないため、必要なら同じ JSON スキーマで
> `gto_output/` ディレクトリをリポジトリルートに自分で用意してください。サインインは
> ビルド環境に `DEEPFOLD_GOOGLE_CLIENT_SECRET` の env var を設定する必要があります
> （未設定の場合、OAuth は実行時に失敗します）。

## サポート

- **バグ報告 / 機能リクエスト**：[Issue を開く](https://github.com/a9876543245/DEEPFOLD-SOLVER/issues)
- **会員関連**：[contact@deepfold.co](mailto:contact@deepfold.co)

バグ報告には以下を添えてください：
- アプリのバージョン（ウィンドウ右上 / **About**）
- そのときの backend ピル：**CUDA** / **CPU**
- Windows のバージョン（設定 → 詳細情報）
- UI の問題の場合はスクリーンショットまたは画面録画

## よくある質問

**GPU なしでも動きますか？**
はい。自動検出して CPU にフォールバックします。遅くなりますが戦略は完全に同一。

**macOS / Linux 対応は？**
v1.0.x ではなし。ロードマップにあります。

**ソルバーの戦略はどこかにアップロードされますか？**
いいえ。すべてローカルで実行されます。ネットワーク通信は deepfold.co への会員資格チェック（サインイン時）のみ。

**アップデートはどう動きますか？**
起動時にアプリが GitHub の最新リリースをチェック。署名が有効で新しいものがあればバナーを表示、ワンクリックでインストール。

## ライセンス

DEEPFOLD-SOLVER のソースコードは透明性のために公開されています。インストーラーは DEEPFOLD PRO メンバー向けです。© DEEPFOLD — All rights reserved.

[deepfold.co](https://deepfold.co)

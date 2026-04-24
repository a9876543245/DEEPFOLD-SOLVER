# DEEPFOLD-SOLVER

> GPU アクセラレーション GTO ポーカーソルバー · CPU フォールバック対応 · 三言語 UI · ワンクリックインストーラー

**[English](README.md) · [中文](README.zh.md) · [日本語](README.ja.md)**

![Platform](https://img.shields.io/badge/platform-Windows%2010%2F11-blue)
![Version](https://img.shields.io/badge/version-1.0.4-green)
![Backend](https://img.shields.io/badge/backend-CUDA%20%2B%20CPU-orange)

DEEPFOLD-SOLVER は [DEEPFOLD](https://deepfold.co) のデスクトップ GTO ソルバーです。GPU アクセラレーション DCFR エンジン（CPU フォールバック完備）と **2,500 件以上のプリフロップシナリオ** が、Windows 用ワンクリックインストーラーに同梱されています。

## v1.0.4 のハイライト

- **Phase 2 スート同型** — runout 列挙が PioSolver / GTO+ 流の対称圧縮を採用。モノトーンや 3-of-suit ボードは戦略の質を落とさず **3〜7 倍高速化**
- **GPU per-runout マッチアップテーブル** — chance ノード列挙が CUDA で実行。iso 有効時 **CPU 比 6〜10 倍高速**
- **Route A ナビゲーションキャッシュ** — 一度ソルブすれば、どこをクリックしても即時。UI のアクション切替は **O(1) キャッシュ参照** に。クリック毎に 8 秒待つ必要なし
- **Path B runout セレクタ** — iso が複数の正準 river カードを列挙する際、UI に runout ピッカーが表示。任意の正準カードをクリックしてその枝に切替。PioSolver スタイルの chance-aware ナビゲーション
- **GTO チャートライブラリ** — 2,550 件以上のプリフロップシナリオ（Cash 6max/8max + MTT）をアプリ内でブラウズ。ワンクリックで IP / OOP レンジに適用可能

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
| **Runout ピッカー (v1.0.4)** | iso 列挙有効時、任意の正準 river カードをクリックして枝切替 |
| **GTO チャートライブラリ (v1.0.4)** | 2,550 件以上のプリフロップシナリオをアプリ内ブラウズ。ワンクリックで IP / OOP レンジに適用 |
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

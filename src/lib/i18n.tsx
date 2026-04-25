import React, { createContext, useContext, useState, useCallback, type ReactNode } from 'react';

// ============================================================================
// Types
// ============================================================================

export type Language = 'en' | 'zh' | 'ja';

export const LANGUAGES: { code: Language; label: string; flag: string }[] = [
  { code: 'en', label: 'English', flag: 'EN' },
  { code: 'zh', label: '中文', flag: '中' },
  { code: 'ja', label: '日本語', flag: '日' },
];

type Dict = Record<string, Record<Language, string>>;

// ============================================================================
// Dictionary — all translatable UI strings
// ============================================================================

const dict: Dict = {
  // ---- App / Header ----
  'app.title': { en: 'DEEPFOLD - SOLVER', zh: 'DEEPFOLD - 求解器', ja: 'DEEPFOLD - ソルバー' },
  'app.subtitle': { en: 'GPU-Accelerated GTO Solver', zh: 'GPU 加速 GTO 求解器', ja: 'GPU高速GTOソルバー' },
  'app.subtitleNote': {
    en: 'CPU fallback when GPU unavailable',
    zh: '無 GPU 時自動切換 CPU',
    ja: 'GPU非対応時はCPUで動作',
  },
  'app.help': { en: 'How to use', zh: '使用說明', ja: '使い方' },

  // ---- Backend indicator ----
  'backend.gpu': { en: 'GPU', zh: 'GPU', ja: 'GPU' },
  'backend.cpu': { en: 'CPU', zh: 'CPU', ja: 'CPU' },
  'backend.detecting': { en: 'Detecting backend...', zh: '偵測後端中...', ja: 'バックエンド検出中...' },
  'backend.active': { en: 'Active backend', zh: '當前後端', ja: '使用中のバックエンド' },
  'backend.fallback': { en: 'CPU fallback active', zh: 'CPU 後備模式', ja: 'CPUフォールバック' },
  'backend.gpuUnavailable': {
    en: 'No CUDA GPU detected. Running on CPU.',
    zh: '未偵測到 CUDA GPU，正在使用 CPU 運算。',
    ja: 'CUDA GPUが検出されません。CPUで実行中。',
  },
  'backend.gpuReady': {
    en: 'GPU ready',
    zh: 'GPU 就緒',
    ja: 'GPU準備完了',
  },
  'backend.settings': { en: 'Backend', zh: '運算後端', ja: 'バックエンド' },
  'backend.auto': { en: 'Auto (prefer GPU)', zh: '自動（優先 GPU）', ja: '自動（GPU優先）' },
  'backend.forceCpu': { en: 'Force CPU', zh: '強制 CPU', ja: 'CPU強制' },
  'backend.forceGpu': { en: 'Force GPU', zh: '強制 GPU', ja: 'GPU強制' },
  'app.placeholder': {
    en: 'Select your position, set board cards, then click Solve to compute the GTO strategy',
    zh: '選擇你的位置，設定公共牌，然後點擊求解來計算 GTO 策略',
    ja: 'ポジションを選択し、ボードカードを設定して、求解をクリックしてGTO戦略を計算します',
  },
  'error': { en: 'Error', zh: '錯誤', ja: 'エラー' },

  // ---- Position Selector ----
  'position': { en: 'Position', zh: '位置', ja: 'ポジション' },
  'position.6max': { en: '6-Max Cash', zh: '6人現金桌', ja: '6人キャッシュ' },
  'position.srp': { en: 'Single Raise', zh: '單次加注', ja: 'シングルレイズ' },
  'position.3bp': { en: '3-Bet Pot', zh: '3-Bet 底池', ja: '3ベットポット' },
  'position.select': { en: 'Select your position', zh: '選擇你的位置', ja: 'ポジションを選択' },
  'position.selectOpp': { en: 'Select opponent position', zh: '選擇對手位置', ja: '相手のポジションを選択' },
  'position.hero': { en: 'Hero:', zh: '主角：', ja: 'ヒーロー：' },
  'position.heroIP': { en: 'Hero IP', zh: '主角 IP', ja: 'ヒーロー IP' },
  'position.heroOOP': { en: 'Hero OOP', zh: '主角 OOP', ja: 'ヒーロー OOP' },
  'position.srpPot': { en: 'Single Raise Pot', zh: '單次加注底池', ja: 'シングルレイズポット' },
  'position.3bpPot': { en: '3-Bet Pot', zh: '3-Bet 底池', ja: '3ベットポット' },
  'position.ipRaiser': { en: 'IP · Raiser', zh: 'IP · 加注者', ja: 'IP · レイザー' },
  'position.oopCaller': { en: 'OOP · Caller', zh: 'OOP · 跟注者', ja: 'OOP · コーラー' },
  'position.oop3bettor': { en: 'OOP · 3-Bettor', zh: 'OOP · 3-Bet 者', ja: 'OOP · 3ベッター' },
  'reset': { en: 'Reset', zh: '重置', ja: 'リセット' },

  // ---- Game Context Selector (game type / scenario / stack depth) ----
  'gctx': { en: 'Game / Stack', zh: '牌局 / 籌碼', ja: 'ゲーム / スタック' },
  'gctx.gameType': { en: 'Game type', zh: '牌局類型', ja: 'ゲーム種別' },
  'gctx.cash': { en: 'Cash', zh: '現金桌', ja: 'キャッシュ' },
  'gctx.mtt': { en: 'MTT', zh: '錦標賽', ja: 'トーナメント' },
  'gctx.format': { en: 'Format', zh: '場景', ja: 'フォーマット' },
  'gctx.stack': { en: 'Effective stack', zh: '有效籌碼', ja: '有効スタック' },
  'gctx.bb': { en: 'BB', zh: 'BB', ja: 'BB' },
  'gctx.any': { en: 'Any', zh: '任意', ja: 'すべて' },
  'gctx.noScenarios': {
    en: 'GTO chart library not available — using built-in TexasSolver ranges',
    zh: 'GTO 圖庫未載入 — 使用內建 TexasSolver 範圍',
    ja: 'GTOチャートライブラリが利用できません — 内蔵TexasSolverレンジを使用中',
  },
  // MTT scenario tooltips (folder names from gto_output/mtt/)
  'gctx.mtt.rfi':            { en: 'Raise First In', zh: '首位加注 (RFI)', ja: 'ファーストイン' },
  'gctx.mtt.vs_rfi':         { en: 'Facing an open', zh: '面對首位加注', ja: 'オープンに対する' },
  'gctx.mtt.vs_3b':          { en: 'We open, facing 3-bet', zh: '我方加注被 3-Bet', ja: '当方オープン後3ベット対応' },
  'gctx.mtt.vs_4b':          { en: 'Facing a 4-bet', zh: '面對 4-Bet', ja: '4ベット対応' },
  'gctx.mtt.vs_open_3b':     { en: 'Open 3-bet (3-bet vs open)', zh: '對開局加注 3-Bet', ja: 'オープンに3ベット' },
  'gctx.mtt.vs_open_call':   { en: 'Open call (call vs open)', zh: '對開局加注跟注', ja: 'オープンにコール' },
  'gctx.mtt.bb_defense_bvb': { en: 'BB defense (Blind vs Blind)', zh: 'BB vs SB 盲注大戰', ja: 'BB vs SB ブラインド戦' },
  'gctx.mtt.hu':             { en: 'Heads-up', zh: '一對一', ja: 'ヘッズアップ' },

  // ---- Board Selector ----
  'board': { en: 'Board', zh: '公共牌', ja: 'ボード' },
  'board.selectFlop': { en: 'Select Flop', zh: '選擇翻牌', ja: 'フロップを選択' },
  'board.flop': { en: 'Flop', zh: '翻牌', ja: 'フロップ' },
  'board.turn': { en: 'Turn', zh: '轉牌', ja: 'ターン' },
  'board.river': { en: 'River', zh: '河牌', ja: 'リバー' },
  'board.random': { en: 'Random', zh: '隨機', ja: 'ランダム' },
  'board.clear': { en: 'Clear', zh: '清除', ja: 'クリア' },
  'board.selectMore': { en: 'Select {n} more cards', zh: '還需選擇 {n} 張牌', ja: 'あと{n}枚選択' },
  'board.selectTurn': { en: 'Select Turn card', zh: '選擇轉牌', ja: 'ターンカードを選択' },
  'board.selectRiver': { en: 'Select River card', zh: '選擇河牌', ja: 'リバーカードを選択' },
  'board.complete': { en: 'Board complete', zh: '公共牌已完成', ja: 'ボード完成' },
  'board.clickToRemove': { en: 'Click a board card to remove it', zh: '點擊公共牌可以移除', ja: 'ボードカードをクリックして削除' },

  // ---- Solver Controls ----
  'config': { en: 'Configuration', zh: '設定', ja: '設定' },
  'config.potSize': { en: 'Pot Size', zh: '底池大小', ja: 'ポットサイズ' },
  'config.effStack': { en: 'Effective Stack', zh: '有效籌碼', ja: '有効スタック' },
  'config.betSizing': { en: 'Bet Sizing', zh: '下注尺寸', ja: 'ベットサイジング' },
  'config.standard': { en: 'Standard', zh: '標準', ja: 'スタンダード' },
  'config.polar': { en: 'Polar', zh: '極化', ja: 'ポーラー' },
  'config.smallBall': { en: 'Small Ball', zh: '小球', ja: 'スモールボール' },
  'config.advanced': { en: 'Advanced', zh: '進階', ja: '詳細設定' },
  'config.show': { en: 'Show', zh: '顯示', ja: '表示' },
  'config.hide': { en: 'Hide', zh: '隱藏', ja: '非表示' },
  'config.maxIter': { en: 'Max Iterations', zh: '最大疊代次數', ja: '最大反復回数' },
  'solve': { en: 'Solve', zh: '求解', ja: '求解' },
  'solving': { en: 'Solving...', zh: '求解中...', ja: '求解中...' },

  // ---- Strategy Panel ----
  'panel.solverResult': { en: 'Solver Result', zh: '求解結果', ja: '求解結果' },
  'panel.iterations': { en: 'Iterations', zh: '疊代次數', ja: '反復回数' },
  'panel.exploitability': { en: 'Exploitability', zh: '可剝削度', ja: '搾取可能性' },
  'panel.time': { en: 'Time', zh: '耗時', ja: '所要時間' },
  'panel.status': { en: 'Status', zh: '狀態', ja: 'ステータス' },
  'panel.converged': { en: 'Converged', zh: '已收斂', ja: '収束済み' },
  'panel.running': { en: 'Running', zh: '運行中', ja: '実行中' },
  'panel.progress': { en: 'Progress', zh: '進度', ja: '進捗' },
  'panel.elapsed': { en: 'Elapsed', zh: '已用時間', ja: '経過時間' },
  'panel.globalStrategy': { en: 'Global Strategy', zh: '整體策略', ja: 'グローバル戦略' },
  'panel.comboVariants': { en: 'Combo Variants', zh: '組合變體', ja: 'コンボ変種' },
  'panel.live': { en: 'live', zh: '存活', ja: '有効' },
  'panel.dead': { en: 'DEAD', zh: '已死', ja: '無効' },
  'panel.out': { en: 'OUT', zh: '範圍外', ja: '範囲外' },
  'panel.offRange': { en: 'Off-Range', zh: '範圍外', ja: '範囲外' },
  'panel.offRangeWarning': {
    en: 'Not in your preflop range. Below is a hypothetical GTO response if you happen to hold this hand.',
    zh: '不在你的翻前範圍內。以下是如果你持有這手牌的假設 GTO 回應。',
    ja: 'プリフロップレンジに含まれていません。このハンドを持っている場合の仮想GTO対応を以下に示します。',
  },
  'panel.best': { en: 'Best: ', zh: '最佳：', ja: '最善：' },
  'panel.suggested': { en: 'Suggested: ', zh: '建議：', ja: '推奨：' },
  'panel.clickToAnalyze': { en: 'Click to see combo analysis', zh: '點擊查看組合分析', ja: 'クリックしてコンボ分析を表示' },
  'panel.emptyHint': {
    en: 'Configure a board and click Solve to see the GTO strategy',
    zh: '設定公共牌並點擊求解來查看 GTO 策略',
    ja: 'ボードを設定して求解をクリックしてGTO戦略を表示',
  },

  // Quick Read
  'qr.primarily': { en: 'Primarily', zh: '主要', ja: '主に' },
  'qr.passive': { en: 'Passive spot — low board interaction', zh: '被動場景 — 與公共牌互動低', ja: 'パッシブスポット — ボードとの関連性が低い' },
  'qr.strongPref': { en: 'Strong preference for this action', zh: '強烈偏好此行動', ja: 'このアクションを強く推奨' },
  'qr.mixed': { en: 'Mixed:', zh: '混合：', ja: 'ミックス：' },
  'qr.or': { en: 'or', zh: '或', ja: 'または' },
  'qr.useBoth': { en: 'Use both actions frequently', zh: '兩種行動都經常使用', ja: '両方のアクションを頻繁に使用' },
  'qr.aggressive': { en: 'Aggressive spot', zh: '激進場景', ja: 'アグレッシブスポット' },
  'qr.leanAgg': { en: 'Lean towards betting and raising', zh: '傾向於下注和加注', ja: 'ベットとレイズに傾く' },
  'qr.defensive': { en: 'Passive / Defensive', zh: '被動 / 防守', ja: 'パッシブ / ディフェンシブ' },
  'qr.checkCall': { en: 'Check and call are preferred', zh: '過牌和跟注為主', ja: 'チェックとコールが推奨' },
  'qr.lean': { en: 'Lean', zh: '傾向', ja: '傾向' },
  'qr.mixedStrategy': { en: 'Mixed strategy across multiple actions', zh: '多種行動的混合策略', ja: '複数アクションにわたる混合戦略' },

  // ---- Grid View Modes ----
  'grid.view': { en: 'View', zh: '檢視', ja: '表示' },
  'grid.strategyMix': { en: 'Strategy Mix', zh: '策略混合', ja: '戦略ミックス' },
  'grid.heatmap': { en: 'Heatmap', zh: '熱力圖', ja: 'ヒートマップ' },
  'grid.actingLabel': { en: "Acting player's decision", zh: '當前行動者決策', ja: '行動中プレイヤーの判断' },
  'grid.opponentLabel': { en: "Opponent's range at node", zh: '對手在此節點的範圍', ja: 'このノードでの相手のレンジ' },
  'grid.viewActing': { en: 'Acting', zh: '行動者', ja: '行動中' },
  'grid.viewOpponent': { en: 'Opponent', zh: '對手', ja: '相手' },

  // ---- Action Bar ----
  'action.folds': { en: 'folds', zh: '棄牌', ja: 'フォールド' },
  'action.wins': { en: 'wins pot of', zh: '贏得底池', ja: 'ポット獲得' },
  'action.showdown': { en: 'Showdown', zh: '攤牌', ja: 'ショーダウン' },
  'action.toAct': { en: 'to act', zh: '行動中', ja: 'アクション' },
  'action.awaitDeal': { en: 'awaiting deal', zh: '等待發牌', ja: 'ディール待ち' },

  // ---- Range Editor Modal ----
  'range.editTitle': { en: 'Edit Preflop Range', zh: '編輯翻前範圍', ja: 'プリフロップレンジを編集' },
  'range.grid': { en: 'Grid', zh: '格子', ja: 'グリッド' },
  'range.text': { en: 'Text', zh: '文字', ja: 'テキスト' },
  'range.brushWeight': { en: 'Brush Weight', zh: '畫筆權重', ja: 'ブラシ重み' },
  'range.clearAll': { en: 'Clear All', zh: '全部清除', ja: 'すべてクリア' },
  'range.allFull': { en: '100% All', zh: '全部100%', ja: '全て100%' },
  'range.pasteHint': {
    en: 'Paste a range string. Supports formats:',
    zh: '貼上範圍字串。支援格式：',
    ja: 'レンジ文字列を貼り付け。対応形式：',
  },
  'range.parsed': { en: 'Parsed {n} combos', zh: '已解析 {n} 個組合', ja: '{n}コンボを解析' },
  'range.noValid': { en: 'No valid combos found', zh: '未找到有效組合', ja: '有効なコンボが見つかりません' },
  'range.applyGrid': { en: 'Apply to Grid', zh: '套用到格子', ja: 'グリッドに適用' },
  'range.apply': { en: 'Apply Range', zh: '套用範圍', ja: 'レンジを適用' },
  'cancel': { en: 'Cancel', zh: '取消', ja: 'キャンセル' },
  'edit': { en: 'EDIT', zh: '編輯', ja: '編集' },

  // ---- Node Lock Editor ----
  'lock.title': { en: 'Node Lock:', zh: '節點鎖定：', ja: 'ノードロック：' },
  'lock.desc': {
    en: 'Force specific action frequencies for this combo at the current node. The engine will skip strategy updates for this combo during iteration.',
    zh: '在當前節點強制設定此組合的行動頻率。引擎在疊代過程中將跳過此組合的策略更新。',
    ja: '現在のノードでこのコンボのアクション頻度を強制設定します。反復中、このコンボの戦略更新はスキップされます。',
  },
  'lock.total': { en: 'Total:', zh: '合計：', ja: '合計：' },
  'lock.autoBalance': { en: 'Auto-Balance', zh: '自動平衡', ja: '自動バランス' },
  'lock.save': { en: 'Lock Strategy', zh: '鎖定策略', ja: '戦略をロック' },
  'lock': { en: 'Lock', zh: '鎖定', ja: 'ロック' },
  'lock.tooltip': { en: 'Lock Strategy for this Combo', zh: '鎖定此組合的策略', ja: 'このコンボの戦略をロック' },

  // ---- Turn/River Card Selector ----
  'deal': { en: 'Deal', zh: '發牌', ja: 'ディール' },
  'deal.selectCard': {
    en: 'Select a card for the {street}. Cards already on the board are disabled.',
    zh: '為{street}選擇一張牌。已在公共牌上的牌已被禁用。',
    ja: '{street}のカードを選択してください。ボード上のカードは無効です。',
  },
  'deal.board': { en: 'Board:', zh: '公共牌：', ja: 'ボード：' },

  // ---- Action Navigator ----
  'nav.root': { en: 'Root', zh: '根節點', ja: 'ルート' },

  // ---- Guide Modal ----
  'guide.title': { en: 'How to Use DEEPFOLD', zh: '如何使用 DEEPFOLD', ja: 'DEEPFOLDの使い方' },
  'guide.subtitle': { en: 'GPU-Accelerated GTO Poker Solver', zh: 'GPU 加速 GTO 撲克求解器', ja: 'GPU高速GTOポーカーソルバー' },
  'guide.quickStart': { en: 'Quick Start', zh: '快速開始', ja: 'クイックスタート' },
  'guide.quickStartDesc': {
    en: 'DEEPFOLD computes Game Theory Optimal (GTO) strategies for Texas Hold\'em postflop play. Follow the steps below: pick positions, set a board, solve, then explore the results.',
    zh: 'DEEPFOLD 計算德州撲克翻後的博弈論最優 (GTO) 策略。按照以下步驟操作：選擇位置、設定公共牌、求解、然後探索結果。',
    ja: 'DEEPFOLDはテキサスホールデムのポストフロッププレイにおけるゲーム理論最適（GTO）戦略を計算します。以下の手順に従ってください：ポジションを選択、ボードを設定、求解、結果を探索。',
  },
  'guide.step1.title': { en: 'Select Position', zh: '選擇位置', ja: 'ポジション選択' },
  'guide.step1.desc': {
    en: 'Choose Single Raise or 3-Bet Pot. Then click your hero position (e.g., BTN), and choose your opponent (e.g., BB). This auto-loads precomputed GTO ranges and sets default pot/stack sizes.',
    zh: '選擇單次加注或 3-Bet 底池。然後點擊你的位置（如 BTN），再選擇對手（如 BB）。系統會自動載入預計算的 GTO 範圍並設定預設底池/籌碼。',
    ja: 'シングルレイズまたは3ベットポットを選択。ヒーローポジション（例：BTN）をクリックし、相手（例：BB）を選択。プリコンピュートGTOレンジが自動ロードされ、デフォルトのポット/スタックが設定されます。',
  },
  'guide.step2.title': { en: 'Set Board Cards', zh: '設定公共牌', ja: 'ボードカード設定' },
  'guide.step2.desc': {
    en: 'Click cards in the picker grid to select the flop (3 cards). Or click Random for a random flop. You can set 4 cards (starts from Turn) or 5 cards (River).',
    zh: '在選牌格子中點擊選擇翻牌（3 張牌）。或點擊隨機產生隨機翻牌。也可以設定 4 張牌（從轉牌開始）或 5 張牌（河牌）。',
    ja: 'ピッカーグリッドでカードをクリックしてフロップ（3枚）を選択。ランダムでランダムフロップも可能。4枚（ターンから）や5枚（リバー）も設定可能。',
  },
  'guide.step3.title': { en: 'Configure & Solve', zh: '設定並求解', ja: '設定して求解' },
  'guide.step3.desc': {
    en: 'Adjust Pot Size and Effective Stack if needed. Choose a bet sizing preset. Click the blue Solve button. The right panel shows progress with iteration count and elapsed time.',
    zh: '根據需要調整底池大小和有效籌碼。選擇下注尺寸預設。點擊藍色求解按鈕。右側面板顯示進度、疊代次數和耗時。',
    ja: 'ポットサイズと有効スタックを必要に応じて調整。ベットサイジングプリセットを選択。青い求解ボタンをクリック。右パネルに反復回数と経過時間が表示されます。',
  },
  'guide.step4.title': { en: 'Read the Results', zh: '閱讀結果', ja: '結果を読む' },
  'guide.step4.desc': {
    en: 'The center 13x13 grid shows color-coded strategies. Right panel shows a Quick Read (simplified recommendation) plus Global Strategy bars. Hover any cell for action breakdown tooltip.',
    zh: '中央 13x13 格子顯示色彩編碼的策略。右側面板顯示快速解讀（簡化建議）和整體策略條形圖。懸停任何格子可查看行動分解提示。',
    ja: '中央の13x13グリッドにカラーコード戦略が表示されます。右パネルにクイックリード（簡易推奨）とグローバル戦略バーが表示されます。セルにホバーでアクション内訳ツールチップ。',
  },
  'guide.step5.title': { en: 'Navigate the Game Tree', zh: '導航遊戲樹', ja: 'ゲームツリーの操作' },
  'guide.step5.desc': {
    en: 'Action Bar at the bottom shows available actions. Click one to advance — the solver re-computes. Breadcrumb in the header tracks your path. Click any breadcrumb to jump back.',
    zh: '底部行動欄顯示可用行動。點擊一個行動前進 — 求解器重新計算。頂部麵包屑追蹤你的路徑。點擊任何麵包屑可跳回。',
    ja: '下部のアクションバーに利用可能なアクションが表示されます。クリックして進む — ソルバーが再計算。ヘッダーのパンくずリストでパスを追跡。パンくずをクリックで戻れます。',
  },
  'guide.step6.title': { en: 'Analyze Specific Combos', zh: '分析特定組合', ja: '特定コンボの分析' },
  'guide.step6.desc': {
    en: 'Click any cell for detailed breakdown. In-range: instant strategy + EV. Off-range: real re-solve with GTO suggestion. Combo Variants shows all suited/offsuit combos with individual strategies.',
    zh: '點擊任何格子查看詳細分解。範圍內：即時策略 + EV。範圍外：真實重新求解 + GTO 建議。組合變體顯示所有同花/不同花組合的個別策略。',
    ja: '任意のセルをクリックで詳細分析。レンジ内：即時戦略+EV。レンジ外：実際の再求解+GTO提案。コンボ変種はスーテッド/オフスーテッド全コンボの個別戦略を表示。',
  },
  'guide.step7.title': { en: 'Advanced Features', zh: '進階功能', ja: '高度な機能' },
  'guide.step7.range': {
    en: 'Range Editor: Click EDIT → visual grid editor with drag-to-paint. Use quick templates or switch to Text mode to paste ranges.',
    zh: '範圍編輯器：點擊編輯 → 可拖動繪製的視覺格子編輯器。使用快速模板或切換到文字模式貼上範圍。',
    ja: 'レンジエディタ：編集をクリック → ドラッグペイント可能なビジュアルグリッド。クイックテンプレートまたはテキストモードでレンジを貼り付け。',
  },
  'guide.step7.lock': {
    en: 'Node Locking: Click a combo → click Lock → set forced action frequencies with sliders → solver respects your locked strategy.',
    zh: '節點鎖定：點擊組合 → 點擊鎖定 → 用滑桿設定強制行動頻率 → 求解器遵守你鎖定的策略。',
    ja: 'ノードロック：コンボをクリック → ロックをクリック → スライダーで強制アクション頻度を設定 → ソルバーがロック戦略を遵守。',
  },
  'guide.step7.heatmap': {
    en: 'Heatmap Mode: Click Heatmap above grid → select an action → see frequency intensity map with percentages.',
    zh: '熱力圖模式：點擊格子上方的熱力圖 → 選擇行動 → 查看帶百分比的頻率強度圖。',
    ja: 'ヒートマップモード：グリッド上のヒートマップをクリック → アクションを選択 → パーセンテージ付き頻度強度マップを表示。',
  },
  'guide.gotIt': { en: 'Got it', zh: '了解了', ja: '了解' },

  // ---- Auth Gate ----
  'auth.loginRequired': { en: 'Sign in to continue', zh: '請登入以繼續', ja: 'サインインして続行' },
  'auth.loginDesc': {
    en: 'DEEPFOLD-SOLVER requires a DEEPFOLD PRO membership. Sign in with the same Google account you use on deepfold.co.',
    zh: 'DEEPFOLD-SOLVER 需要 DEEPFOLD PRO 會員資格。請使用你在 deepfold.co 的 Google 帳號登入。',
    ja: 'DEEPFOLD-SOLVERはDEEPFOLD PROメンバーシップが必要です。deepfold.coで使用しているGoogleアカウントでサインインしてください。',
  },
  'auth.googleSignIn': { en: 'Sign in with Google', zh: '使用 Google 帳號登入', ja: 'Googleでサインイン' },
  'auth.proRequired': { en: 'PRO Membership Required', zh: '需要 PRO 會員', ja: 'PROメンバーシップが必要' },
  'auth.proDesc': {
    en: 'Your account does not have PRO access. Upgrade to DEEPFOLD PRO to unlock the GTO Solver.',
    zh: '你的帳號沒有 PRO 權限。升級到 DEEPFOLD PRO 以解鎖 GTO 求解器。',
    ja: 'お使いのアカウントにはPROアクセスがありません。GTOソルバーを解除するにはDEEPFOLD PROにアップグレードしてください。',
  },
  'auth.upgradePro': { en: 'Upgrade to PRO', zh: '升級到 PRO', ja: 'PROにアップグレード' },
  'auth.logout': { en: 'Sign out', zh: '登出', ja: 'サインアウト' },
  'auth.waitingForBrowser': {
    en: 'Complete sign-in in your browser window…',
    zh: '請在瀏覽器視窗完成登入…',
    ja: 'ブラウザウィンドウでサインインを完了してください…',
  },
  'auth.cancelSignIn': { en: 'Cancel', zh: '取消', ja: 'キャンセル' },

  // ---- Updater ----
  'update.available': { en: 'Update available', zh: '有可用更新', ja: 'アップデートあり' },
  'update.install':   { en: 'Install now',      zh: '立即安裝',     ja: '今すぐインストール' },
  'update.downloading': {
    en: 'Downloading update…',
    zh: '下載更新中…',
    ja: 'アップデートをダウンロード中…',
  },
  'update.installed': {
    en: 'Update installed — restarting…',
    zh: '更新已安裝 — 重新啟動中…',
    ja: 'アップデートをインストールしました — 再起動中…',
  },
  'update.failed': {
    en: 'Update failed',
    zh: '更新失敗',
    ja: 'アップデート失敗',
  },

  // ---- Spot Library ----
  'spots.title': { en: 'Spot Library', zh: '方案庫', ja: 'スポットライブラリ' },
  'spots.subtitle': {
    en: 'Common GTO spots — click any to solve and load',
    zh: '常見 GTO 方案 — 點擊任一以解算並載入',
    ja: '一般的なGTOスポット — クリックで解算・ロード',
  },
  'spots.filter': { en: 'Filter by matchup', zh: '按對局篩選', ja: 'マッチアップでフィルター' },
  'spots.allMatchups': { en: 'All Matchups', zh: '所有對局', ja: 'すべてのマッチアップ' },
  'spots.texture': { en: 'Board Texture', zh: '牌面質地', ja: 'ボードテクスチャ' },
  'spots.boards': { en: '{n} boards', zh: '{n} 個牌面', ja: '{n}ボード' },
  'spots.demoMode': {
    en: 'DEMO MODE — heuristic preview only. Install desktop app for real solver.',
    zh: '示範模式 — 僅啟發式預覽。請安裝桌面版以使用真實求解器。',
    ja: 'デモモード — ヒューリスティックプレビューのみ。リアルソルバーはデスクトップアプリで。',
  },
  'spots.clickToSolve': {
    en: 'Click any spot to solve with the real GTO solver (~10-30s first time, cached afterwards)',
    zh: '點擊任一方案以用真實 GTO 求解器解算（首次約 10-30 秒，之後快取）',
    ja: 'スポットをクリックで実際のGTOソルバーで解算（初回10-30秒、以降キャッシュ）',
  },
  'spots.preview': { en: 'Preview', zh: '預覽', ja: 'プレビュー' },
  'spots.solvingSpot': { en: 'Solving...', zh: '解算中...', ja: '解算中...' },

  // ---- Drill Mode ----
  'drill.title': { en: 'Drill Mode', zh: '訓練模式', ja: 'ドリルモード' },
  'drill.start': { en: 'Drill', zh: '訓練', ja: 'ドリル' },
  'drill.question': { en: 'What should you do?', zh: '你應該怎麼做？', ja: 'どうすべき？' },
  'drill.yourHand': { en: 'Your hand', zh: '你的手牌', ja: 'あなたのハンド' },
  'drill.correct': { en: 'Correct!', zh: '正確！', ja: '正解！' },
  'drill.suboptimal': { en: 'Suboptimal', zh: '次優', ja: '最善ではない' },
  'drill.gtoStrategy': { en: 'GTO Strategy for', zh: '的 GTO 策略', ja: 'のGTO戦略' },
  'drill.next': { en: 'Next', zh: '下一題', ja: '次へ' },
  'drill.complete': { en: 'Session Complete!', zh: '訓練完成！', ja: 'セッション完了！' },
  'drill.score': { en: 'Score', zh: '得分', ja: 'スコア' },
  'drill.optimal': { en: 'Optimal', zh: '最佳', ja: '最適' },
  'drill.good': { en: 'Good', zh: '良好', ja: '良い' },
  'drill.acceptable': { en: 'Acceptable', zh: '尚可', ja: '許容範囲' },
  'drill.mistake': { en: 'Mistake', zh: '錯誤', ja: 'ミス' },
  'drill.playAgain': { en: 'Play Again', zh: '再玩一次', ja: 'もう一度' },
  'drill.backToSolver': { en: 'Back to Solver', zh: '返回求解器', ja: 'ソルバーに戻る' },
};

// ============================================================================
// Context & Hook
// ============================================================================

interface I18nContext {
  lang: Language;
  setLang: (l: Language) => void;
  t: (key: string, params?: Record<string, string | number>) => string;
}

const Ctx = createContext<I18nContext>({
  lang: 'en',
  setLang: () => {},
  t: (key) => key,
});

export function LanguageProvider({ children }: { children: ReactNode }) {
  const [lang, setLang] = useState<Language>(() => {
    // Try to restore from localStorage
    const saved = typeof localStorage !== 'undefined' ? localStorage.getItem('deepfold-lang') : null;
    if (saved === 'zh' || saved === 'ja') return saved;
    return 'en';
  });

  const changeLang = useCallback((l: Language) => {
    setLang(l);
    try { localStorage.setItem('deepfold-lang', l); } catch {}
  }, []);

  const t = useCallback((key: string, params?: Record<string, string | number>): string => {
    const entry = dict[key];
    if (!entry) return key; // fallback: show key
    let text = entry[lang] || entry.en || key;
    if (params) {
      for (const [k, v] of Object.entries(params)) {
        text = text.replace(`{${k}}`, String(v));
      }
    }
    return text;
  }, [lang]);

  return (
    <Ctx.Provider value={{ lang, setLang: changeLang, t }}>
      {children}
    </Ctx.Provider>
  );
}

/** Hook to get the translation function */
export function useT() {
  const { t } = useContext(Ctx);
  return t;
}

/** Hook to get language + setter */
export function useLanguage() {
  const { lang, setLang } = useContext(Ctx);
  return { lang, setLang };
}

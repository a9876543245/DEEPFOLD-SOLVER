/// Shared type definitions used across the Rust backend.
/// Mirror the JSON schema from the C++ engine.
use serde::{Deserialize, Serialize};

/// Solver request configuration — sent to the C++ engine via CLI args
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolverRequest {
    pub board: String,
    pub pot_size: f64,
    pub effective_stack: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub history: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target_combo: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ip_range: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub oop_range: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub node_locks: Option<String>,
    #[serde(default = "default_iterations")]
    pub iterations: i32,
    #[serde(default = "default_exploitability")]
    pub exploitability: f64,
    /// Backend override: "auto" | "cpu" | "gpu". Defaults to "auto".
    #[serde(skip_serializing_if = "Option::is_none")]
    pub backend: Option<String>,
    /// OOP has initiative at root (can bet). Defaults to true if unset.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub oop_has_initiative: Option<bool>,
    /// Allow OOP donk bets even without initiative.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allow_donk_bet: Option<bool>,
    /// Bet sizes per street (fractions of pot). Must match UI's chosen
    /// sizing config or the solver tree won't contain the actions the user
    /// clicks, leading to silent fuzzy-match substitution.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub flop_sizes: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub turn_sizes: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub river_sizes: Option<Vec<f64>>,
}

fn default_iterations() -> i32 { 500 }
fn default_exploitability() -> f64 { 0.5 }

/// Strategy mix entry
#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyEntry {
    pub action: String,
    pub frequency: f64,
}

/// Target combo analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComboAnalysis {
    pub combo: String,
    pub best_action: String,
    pub ev: f64,
    pub strategy_mix: std::collections::HashMap<String, String>,
}

/// Full solver response — parsed from C++ engine JSON output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolverResponse {
    pub status: String,
    pub iterations_run: i32,
    pub exploitability_pct: f64,
    pub global_strategy: std::collections::HashMap<String, String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target_combo_analysis: Option<ComboAnalysis>,
    /// Active backend name (e.g. "CPU-DCFR", "CUDA (RTX 4060, 8GB)")
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backend: Option<String>,
    /// Per-grid-label strategies (e.g. "AA" → {"Check": 0.62, "Bet_33": 0.38}).
    /// Frequencies in [0, 1] — matches the UI's ComboStrategy contract.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub combo_strategies: Option<std::collections::HashMap<String,
        std::collections::HashMap<String, f64>>>,
    /// Player acting at the current node ("OOP" | "IP").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub acting_player: Option<String>,
    /// The opponent (non-acting) player side for the current node.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub opponent_side: Option<String>,
    /// Opponent's reach-weighted range at this node, keyed by grid label.
    /// Values normalized to [0, 1] where 1.0 is the heaviest label at the node.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub opponent_range: Option<std::collections::HashMap<String, f64>>,
    /// Route A: client-side navigation cache. Map of player-action history
    /// path → strategies at that node. Frontend can navigate by lookup
    /// instead of re-invoking the engine. Populated on every solve.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strategy_tree: Option<std::collections::HashMap<String, StrategyTreeEntry>>,
}

/// Per-node strategy bundle, keyed by player-action history in `strategy_tree`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyTreeEntry {
    pub acting: String,                 // "OOP" | "IP"
    pub action_labels: Vec<String>,
    /// global_strategy values are "<float>%" strings (matches existing schema)
    pub global_strategy: std::collections::HashMap<String, String>,
    /// Per-grid-label per-action frequency in [0, 1].
    pub combo_strategies: std::collections::HashMap<String,
        std::collections::HashMap<String, f64>>,
    pub opponent_side: String,
    pub opponent_range: std::collections::HashMap<String, f64>,
    /// Per-grid-label EV at this node (chips, from acting player's view).
    /// Empty for nodes the acting player doesn't reach.
    #[serde(default)]
    pub combo_evs: std::collections::HashMap<String, f64>,
    /// Path B: cumulative runout cards from root to this node (empty for
    /// nodes before any chance). Format: ["2c", "Jd"]. Lets the UI disclose
    /// "this strategy is for runout: 2♣ + J♦" instead of silently showing
    /// lex-min canonical strategies.
    #[serde(default)]
    pub dealt_cards: Vec<String>,
    /// Path B: canonical runout reps available at the IMMEDIATE PRIOR
    /// chance level. UI uses this to render a runout picker. Empty when no
    /// chance preceded this node (root, or both root + first action are
    /// pre-chance). The currently-shown runout is the LAST entry of
    /// `dealt_cards` matched against this list's `card`.
    #[serde(default)]
    pub runout_options: Vec<RunoutOption>,
}

/// One canonical runout class for the Path B runout picker.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunoutOption {
    pub card: String,    // "2c", "As", etc.
    pub weight: u8,      // orbit size
}

/// GPU detection info — returned by the `get_gpu_info` command.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuInfo {
    pub has_cuda_gpu: bool,
    pub gpu_description: String,
    pub gpu_backend_functional: bool,
}

/// Error response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorResponse {
    pub status: String,
    pub message: String,
}

/// Solver progress update (from stderr)
#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolverProgress {
    pub iteration: i32,
    pub exploitability: f64,
    pub elapsed_ms: f64,
}

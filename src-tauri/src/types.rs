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

    // ---- Sprint 3 (resource policy guide): memory budget controls ----
    //
    // memory_profile: high-level preset. "safe" | "balanced" | "performance".
    // Default = "balanced". Each preset maps to a (host, gpu, json,
    // strategy_tree_max_nodes) tuple in `resolve_memory_profile()`. Manual
    // numeric fields override the profile's default for that field only.
    /// "safe" | "balanced" | "performance". Defaults to "balanced".
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_profile: Option<String>,
    /// Override host RAM cap in MB. 0/None = use profile default.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub host_memory_mb: Option<u64>,
    /// Override GPU VRAM cap in MB. 0/None = let backend probe at runtime.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu_memory_mb: Option<u64>,
    /// Override JSON response cap in MB. 0/None = use profile default.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub json_memory_mb: Option<u64>,
    /// Override emitted strategy-tree node cap. 0/None = use profile default.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strategy_tree_max_nodes: Option<u32>,
}

/// Sprint 3 (resource policy guide): resolved memory budget tuple after
/// applying profile defaults + manual overrides. Used by engine.rs to
/// build CLI args.
#[derive(Debug, Clone, Copy)]
pub struct ResolvedMemoryBudget {
    pub host_mb: u64,
    pub gpu_mb: u64,
    pub json_mb: u64,
    pub strategy_tree_max_nodes: u32,
}

impl ResolvedMemoryBudget {
    /// Built-in presets. Match the GUIDE.md table exactly. GPU=0 means
    /// "let the backend probe at runtime"; we don't gate on a hard VRAM
    /// number for arbitrary devices.
    fn from_profile(profile: &str) -> Self {
        match profile {
            "safe" => Self {
                host_mb: 2048,
                gpu_mb: 0,
                json_mb: 50,
                strategy_tree_max_nodes: 500,
            },
            "performance" => Self {
                host_mb: 12288,
                gpu_mb: 0,
                json_mb: 150,
                strategy_tree_max_nodes: 5000,
            },
            // "balanced" is the default and applies for any unknown profile.
            _ => Self {
                host_mb: 6144,
                gpu_mb: 0,
                json_mb: 100,
                strategy_tree_max_nodes: 2000,
            },
        }
    }

    /// Apply per-field overrides on top of a profile.
    pub fn resolve(req: &SolverRequest) -> Self {
        let profile = req.memory_profile.as_deref().unwrap_or("balanced");
        let mut b = Self::from_profile(profile);
        if let Some(v) = req.host_memory_mb { if v > 0 { b.host_mb = v; } }
        if let Some(v) = req.gpu_memory_mb  { b.gpu_mb = v; } // 0 is meaningful
        if let Some(v) = req.json_memory_mb { if v > 0 { b.json_mb = v; } }
        if let Some(v) = req.strategy_tree_max_nodes {
            if v > 0 { b.strategy_tree_max_nodes = v; }
        }
        b
    }
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
    /// Sprint 2 (resource policy guide): per-solve resource estimate +
    /// budget decision. Populated by the C++ engine on every solve. Lets
    /// the UI show "tree truncated", "fell back to CPU because …",
    /// "estimated 12 MB host, 4 MB JSON" without re-running the solve.
    /// `default` so older saved .dsolver files (no `resources`) keep loading.
    #[serde(default)]
    pub resources: SolveResources,
}

/// Sprint 2: per-solve resource estimate. Mirrors C++ `SolveResources`
/// (core/include/types.h). Every field has a sane default so older save
/// files / older engine versions parse without error.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SolveResources {
    #[serde(default)]
    pub canonical_combos: u32,
    #[serde(default)]
    pub player_nodes: u32,
    #[serde(default)]
    pub estimated_matchup_bytes: u64,
    #[serde(default)]
    pub estimated_cpu_state_bytes: u64,
    #[serde(default)]
    pub estimated_gpu_state_bytes: u64,
    #[serde(default)]
    pub estimated_strategy_tree_bytes: u64,
    #[serde(default)]
    pub estimated_json_bytes: u64,
    #[serde(default)]
    pub host_budget_bytes: u64,
    #[serde(default)]
    pub gpu_budget_bytes: u64,
    #[serde(default)]
    pub strategy_tree_max_nodes: u32,
    #[serde(default)]
    pub strategy_tree_emitted_nodes: u32,
    #[serde(default)]
    pub strategy_tree_truncated: bool,
    #[serde(default)]
    pub budget_decision: String,
    #[serde(default)]
    pub diagnostic: String,
    #[serde(default)]
    pub fallback_reason: String,
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

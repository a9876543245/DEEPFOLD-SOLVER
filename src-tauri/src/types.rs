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
    /// v1.3.0: hard wall-clock cap on the iteration phase. None / 0 = no
    /// cap (legacy). UI mode presets pick the right value:
    ///   Quick=60, Standard=300, Deep=900.
    /// CFR is anytime, so stopping early just means "we ran for the budget,
    /// here's the running average so far". Better than letting CPU users
    /// stare at "estimated 5 hours" and walk away.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub time_budget_seconds: Option<u32>,
    /// Stage 5: runout decomposition mode. "off" (default) keeps the legacy
    /// collapse gate (turn/river equity approximated on boards too large to
    /// enumerate). "auto" routes rainbow/collapsed boards through flop-trunk +
    /// per-turn-card subgame decomposition (real runout equity,
    /// runout_approximated=false). "on" forces it even on enumerable boards
    /// (debug). None / "off" ⇒ arg omitted, sidecar keeps its default (off).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decompose_runouts: Option<String>,
    /// Roadmap ④: decomposition iteration presets (engine --decompose-*
    /// CLI flags). None = engine dev defaults. The UI fills these from
    /// DECOMPOSE_PRESETS (poker.ts) keyed on solveMode when Exact is on.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decompose_outer: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decompose_inner: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decompose_trunk_iters: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decompose_warm_start: Option<bool>,
    /// v1.4.0 Phase 2: CPU SIMD policy override. None / "auto" = CPUID picks.
    /// "scalar" forces scalar kernels (parity test / debugging).
    /// "avx2" requires AVX2 CPU (engine aborts on detection failure).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cpu_simd: Option<String>,
    /// v1.4.0 Phase 2: CPU CFR thread count. None / 0 = auto. 1 = serial
    /// traversers, 2 = OOP||IP via OMP. Higher values become useful in v1.5+.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cpu_threads: Option<u32>,
    /// v1.5.0 Phase 4: CPU backend variant — "reference" or "levelized".
    /// Default reference (parity oracle). Levelized scales to all cores.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cpu_backend: Option<String>,
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
    /// v1.3.0: which stop condition ended the iteration loop.
    /// "iter_cap" / "time_budget" / "exploit_target" / "" (legacy).
    #[serde(default)]
    pub early_stop_reason: String,
    /// True when the flop runout enumeration collapsed to the single-child
    /// fallback (memory gate): turn/river equity is approximated from the
    /// stale flop matchup. UI shows a warning. `default` so older saved
    /// .dsolver files (no field) keep loading.
    #[serde(default)]
    pub runout_approximated: bool,
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
    /// Roadmap ④ (estimate-only path): builder already collapsed the runout
    /// enumeration at estimate time ⇒ Fast shows the amber banner and Exact
    /// ('auto') would engage decomposition.
    #[serde(default)]
    pub runout_approximated: bool,
    // v1.2.2: pre-iteration solve cost prediction. ops_per_iteration ≈
    // player_nodes × MAX_ACTIONS × nc² (dominant cost per CFR iteration).
    // estimated_solve_seconds is ops × max_iterations / backend_throughput.
    // backend_for_estimate names the backend the estimate was computed for.
    #[serde(default)]
    pub ops_per_iteration: u64,
    #[serde(default)]
    pub backend_for_estimate: String,
    #[serde(default)]
    pub estimated_solve_seconds: f64,
    // v1.4.0 Phase 2: CPU mode diagnostics. Empty/0 on GPU solves.
    #[serde(default)]
    pub cpu_simd: String,
    #[serde(default)]
    pub cpu_threads_effective: u32,
    /// v1.5.0 Phase 4: "reference" or "levelized". Empty on GPU solves.
    #[serde(default)]
    pub cpu_backend_kind: String,
}

/// v1.2.2: Lightweight pre-solve estimate response. Returned by the
/// `estimate_solve` Tauri command which calls `deepsolver_core
/// --estimate-only` (sub-second on most spots). Frontend uses this to
/// show an ETA banner before the user commits to a long solve.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EstimateResponse {
    /// Engine status: "estimate" on success, "error" on failure.
    pub status: String,
    /// All the SolveResources fields (memory + ETA + budget decision).
    pub resources: SolveResources,
    /// Roadmap ④: Exact-mode feasibility pre-flight. Present when the
    /// request had decompose_runouts auto/on — prices the decomposed run
    /// so the UI can show "Exact ≈ N min, expected accuracy ~X" before the
    /// user commits. `default` keeps older engines parseable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decompose: Option<DecomposeEstimate>,
}

/// Roadmap ④: mirrors the C++ "decompose" block of the --estimate-only
/// JSON (see main.cpp estimate emit + solver_decomposed.h
/// DecompositionEstimate). Every field defaulted so schema drift degrades
/// gracefully instead of failing the whole estimate parse.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DecomposeEstimate {
    #[serde(default)]
    pub ok: bool,
    #[serde(default)]
    pub would_engage: bool,
    #[serde(default)]
    pub leaves: u32,
    #[serde(default)]
    pub lines: u32,
    #[serde(default)]
    pub trunk_nodes: u32,
    #[serde(default)]
    pub sweeps: i32,
    #[serde(default)]
    pub outer: i32,
    #[serde(default)]
    pub inner: i32,
    #[serde(default)]
    pub trunk_iters_per_sweep: i32,
    #[serde(default)]
    pub warm_start: bool,
    #[serde(default)]
    pub per_sweep_seconds: f64,
    /// GPU route: predicted VRAM-resident subgame count (0 = CPU route).
    /// Leaves beyond this stream and pay a per-revisit rebuild surcharge
    /// already folded into total_seconds.
    #[serde(default)]
    pub pinned_leaves_predicted: i32,
    #[serde(default)]
    pub total_seconds: f64,
    #[serde(default)]
    pub spr: f64,
    /// "high" | "medium" | "navigation"
    #[serde(default)]
    pub quality_tier: String,
    #[serde(default)]
    pub expected_exploit_lo_pct: f64,
    #[serde(default)]
    pub expected_exploit_hi_pct: f64,
    #[serde(default)]
    pub backend: String,
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

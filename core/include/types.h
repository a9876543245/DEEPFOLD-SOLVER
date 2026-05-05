/**
 * @file types.h
 * @brief Core type definitions and SoA data structures for DeepSolver.
 *
 * Clean-room design. All types used across the engine are defined here
 * to ensure consistency between CPU and GPU code paths.
 */

#pragma once

#include "memory_budget.h"

#include <cstdint>
#include <vector>
#include <array>
#include <string>

namespace deepsolver {

// ============================================================================
// Card Representation
// ============================================================================

/// Card is a uint8_t in [0, 51]. rank = card / 4, suit = card % 4.
using Card = uint8_t;

/// Rank constants (0-based: 2=0, 3=1, ..., A=12)
enum Rank : uint8_t {
    RANK_2 = 0, RANK_3, RANK_4, RANK_5, RANK_6,
    RANK_7, RANK_8, RANK_9, RANK_T, RANK_J,
    RANK_Q, RANK_K, RANK_A,
    NUM_RANKS = 13
};

/// Suit constants
enum Suit : uint8_t {
    CLUBS = 0, DIAMONDS = 1, HEARTS = 2, SPADES = 3,
    NUM_SUITS = 4
};

/// Total cards in a standard deck
constexpr uint8_t NUM_CARDS = 52;

/// Total starting hand combinations for Hold'em (52 choose 2)
constexpr uint16_t NUM_COMBOS = 1326;

/// Maximum board cards (flop=3, turn=4, river=5)
constexpr uint8_t MAX_BOARD_CARDS = 5;

/// Maximum actions per node (configurable, but bounded)
constexpr uint8_t MAX_ACTIONS = 6;

/// Inline card construction
constexpr Card make_card(Rank r, Suit s) { return static_cast<Card>(r * 4 + s); }

/// Extract rank from card
constexpr Rank card_rank(Card c) { return static_cast<Rank>(c / 4); }

/// Extract suit from card
constexpr Suit card_suit(Card c) { return static_cast<Suit>(c % 4); }

/// 64-bit bitmask for card sets (bits 0-51 used)
using CardMask = uint64_t;

/// Set a card in a bitmask
constexpr CardMask card_to_mask(Card c) { return 1ULL << c; }

// ============================================================================
// Game Tree Node Types
// ============================================================================

/// Node type enumeration for game tree
enum class NodeType : uint8_t {
    PLAYER_OOP = 0,   ///< Out-of-position player decision
    PLAYER_IP  = 1,   ///< In-position player decision
    CHANCE     = 2,   ///< Deal card (turn/river)
    TERMINAL   = 3,   ///< Showdown or fold
};

/// Action type enumeration
enum class ActionType : uint8_t {
    FOLD    = 0,
    CHECK   = 1,
    CALL    = 2,
    BET     = 3,   ///< Opening bet (no prior bet in this round)
    RAISE   = 4,   ///< Raise over existing bet
    ALLIN   = 5,
};

/// Represents a single action edge in the game tree
struct Action {
    ActionType type;
    float amount;       ///< Bet/raise amount (0 for fold/check/call)

    bool operator==(const Action& o) const {
        return type == o.type && amount == o.amount;
    }
};

// ============================================================================
// Terminal node types
// ============================================================================

enum class TerminalType : uint8_t {
    FOLD_OOP = 0,     ///< OOP player folded → IP wins
    FOLD_IP  = 1,     ///< IP player folded → OOP wins
    SHOWDOWN = 2,     ///< Both players go to showdown
};

// ============================================================================
// Flattened Game Tree (SoA layout for GPU)
// ============================================================================

/**
 * @brief BFS-flattened game tree in Structure-of-Arrays format.
 *
 * Absolutely NO pointers inside. All node relationships are expressed
 * via integer indices. This is the format uploaded to GPU global memory.
 */
struct FlatGameTree {
    // Per-node arrays (indexed by node_id)
    std::vector<uint8_t>   node_types;        ///< NodeType enum value
    std::vector<float>     pots;              ///< Pot size at this node
    std::vector<float>     stacks;            ///< Remaining effective stack
    std::vector<uint32_t>  parent_indices;    ///< Index of parent node (0 = root)
    std::vector<uint32_t>  children_offset;   ///< Start index in children[] array
    std::vector<uint8_t>   num_children;      ///< Number of child actions
    std::vector<uint8_t>   street;            ///< 0=flop, 1=turn, 2=river
    std::vector<uint8_t>   terminal_types;    ///< TerminalType for TERMINAL nodes
    std::vector<uint8_t>   active_player;     ///< 0=OOP, 1=IP for player nodes
    std::vector<float>     bet_into;          ///< Amount bet into this node

    // ---- Runout enumeration (Phase 1: chance-node enumeration) ----
    /// Card dealt by the chance node that produced THIS node (only meaningful
    /// for nodes whose parent is CHANCE; UINT8_MAX = "no card / not a runout
    /// child"). Used to reconstruct the full board at terminal evaluation.
    std::vector<uint8_t>   dealt_card;
    /// How many physical cards this iso-runout child represents (default 1).
    /// Allows the chance handler to weight children correctly when later we
    /// add suit isomorphism that collapses equivalent runouts.
    std::vector<uint8_t>   runout_weight;
    /// Index into matchup_ev_per_runout for terminal evaluation. -1 / 0 means
    /// "use the root matchup" (legacy behavior). Populated during precompute.
    std::vector<int32_t>   matchup_idx;

    // Flattened child indices (indexed by children_offset + child_idx)
    std::vector<uint32_t>  children;          ///< Child node indices
    std::vector<uint8_t>   child_action_types;///< ActionType for each child edge
    std::vector<float>     child_action_amts; ///< Bet amounts for each child edge

    uint32_t total_nodes = 0;
    uint32_t total_edges = 0;

    void reserve(uint32_t est_nodes, uint32_t est_edges) {
        node_types.reserve(est_nodes);
        pots.reserve(est_nodes);
        stacks.reserve(est_nodes);
        parent_indices.reserve(est_nodes);
        children_offset.reserve(est_nodes);
        num_children.reserve(est_nodes);
        street.reserve(est_nodes);
        terminal_types.reserve(est_nodes);
        active_player.reserve(est_nodes);
        bet_into.reserve(est_nodes);
        dealt_card.reserve(est_nodes);
        runout_weight.reserve(est_nodes);
        matchup_idx.reserve(est_nodes);
        children.reserve(est_edges);
        child_action_types.reserve(est_edges);
        child_action_amts.reserve(est_edges);
    }
};

// ============================================================================
// Solver Configuration
// ============================================================================

/// Bet sizing preset
struct BetSizingConfig {
    std::vector<float> flop_sizes  = {0.33f, 0.75f};    ///< Fraction of pot
    std::vector<float> turn_sizes  = {0.75f};
    std::vector<float> river_sizes = {0.75f};
    bool flop_allin  = true;
    bool turn_allin  = true;
    bool river_allin = true;
};

/// Node lock entry: force strategy at a specific node for a specific combo
struct NodeLockEntry {
    std::string history;               ///< Action path to the target node, e.g. "Check,Bet_33"
    uint16_t combo_idx = UINT16_MAX;   ///< Index into 1326 combos (populated after parsing)
    std::string combo_str;             ///< Original combo string, e.g. "Ad4s"
    std::vector<float> strategy;       ///< Forced action frequencies, size = num_actions at node
};

/// Full solver configuration
struct SolverConfig {
    float pot = 0.0f;
    float effective_stack = 0.0f;

    std::array<Card, MAX_BOARD_CARDS> board = {};
    uint8_t board_size = 0;    ///< 3=flop, 4=turn, 5=river

    BetSizingConfig bet_sizing;

    int max_iterations = 500;
    float target_exploitability = 0.005f;   ///< 0.5%
    int exploitability_check_interval = 50;

    // Tree pruning
    int raise_cap = 3;                      ///< Max raises per street
    float allin_threshold = 0.12f;          ///< SPR below this → force all-in
    bool allow_donk_bet = false;            ///< Allow OOP donk bets
    float donk_bet_size = 0.20f;            ///< If donk allowed, this size
    /// Whether OOP is assumed to have initiative at the root (flop).
    /// True  → OOP can bet/donk at root (e.g. 3-bet pot, OOP was preflop aggressor).
    /// False → OOP can only check at root (e.g. SRP, IP was preflop raiser).
    /// The engine defaults to true so the root offers meaningful action to the UI.
    bool oop_has_initiative = true;

    // DCFR hyperparameters (used when dcfr_schedule == STANDARD)
    float dcfr_alpha = 1.5f;
    float dcfr_beta  = 0.5f;
    float dcfr_gamma = 2.0f;

    // DCFR discount schedule selector. STANDARD uses (t/(t+1))^alpha-style
    // exponent decay + accumulative reach-weighted strategy_sum (textbook
    // DCFR). POSTFLOP_STYLE matches wasm-postflop / postflop-solver:
    //   alpha_t = t^1.5 / (t^1.5 + 1), beta_t = 0.5 (constant), gamma_t
    //   resets every power-of-4 iteration. strategy_sum is decay-and-add
    //   (no reach weighting), giving an "average over recent iterations of
    //   the current epoch" instead of "average over all iterations". This
    //   converges to less-mixed equilibria on deep trees, which appears to
    //   match published Pio outputs better than our STANDARD schedule on
    //   over-checking-prone spots.
    enum class DcfrSchedule : uint8_t {
        STANDARD = 0,
        POSTFLOP_STYLE = 1,
    };
    DcfrSchedule dcfr_schedule = DcfrSchedule::STANDARD;

    // Cash-game rake. Applied at every fold/showdown terminal:
    //   rake = min(pot_at_terminal * rake_rate, rake_cap)
    // Winner takes (pot - rake), losers take pot's share unchanged. On a
    // showdown tie each player loses rake/2 (split rake equally).
    // Default 0/0 = rake-free (matches solver-textbook / Pio cash mode).
    // NL25 typical: 0.05 / 2.0 (5% capped at 2bb). NL100+: lower cap usually.
    float rake_rate = 0.0f;
    float rake_cap  = 0.0f;

    // Preflop range weights: [1326] floats, 0.0 = not in range, 1.0 = full weight
    // Default: all 1.0 (no filtering)
    std::array<float, NUM_COMBOS> ip_range_weights;
    std::array<float, NUM_COMBOS> oop_range_weights;
    bool has_custom_ranges = false;    ///< True if user provided custom ranges

    // Node locks: force strategy at specific (node, combo) pairs
    std::vector<NodeLockEntry> node_locks;

    // Post-solve reporting passes. Full UI solves keep both enabled; batch
    // benchmarks can disable them to isolate pure CFR iteration throughput.
    bool compute_combo_evs = true;
    bool compute_exploitability = true;
    bool parallel_postsolve = true;
    // 0 = auto (hardware concurrency). Used by the CPU postsolve thread pool.
    uint32_t postsolve_threads = 0;
    // Skip the GPU-postsolve fast path even if the backend supports it.
    // Used by tests that want to validate CPU postsolve on a GPU-solved
    // strategy (or vice-versa).
    bool force_cpu_postsolve = false;

    /// Sprint 1 (market-beating plan): the host RAM / GPU VRAM / JSON /
    /// strategy-tree budget that gates every large allocation in the solve
    /// pipeline. Defaults are sane (6 GB host, 100 MB JSON, 2000 emitted
    /// nodes); CLI flags override per-solve.
    MemoryBudget memory_budget = MemoryBudget::defaults();

    /// Hard cap on emitted strategy-tree nodes. 0 = use memory_budget value.
    /// Surfaced as `--strategy-tree-max-nodes`. When tree generation hits
    /// this cap, remaining branches are pruned and SolverResult.resources
    /// records `strategy_tree_truncated=true`.
    uint32_t strategy_tree_max_nodes = 0;

    SolverConfig() {
        ip_range_weights.fill(1.0f);
        oop_range_weights.fill(1.0f);
    }
};

// ============================================================================
// Solver Result
// ============================================================================

/// Per-combo strategy analysis
struct ComboAnalysis {
    std::array<Card, 2> combo;
    std::string combo_str;
    std::string best_action;
    float ev = 0.0f;
    std::vector<std::pair<std::string, float>> strategy_mix;
};

struct SolverTiming {
    float tree_build_ms = 0.0f;
    float isomorphism_ms = 0.0f;
    float precompute_matchups_ms = 0.0f;
    float reach_init_ms = 0.0f;
    float node_locks_ms = 0.0f;
    float backend_prepare_ms = 0.0f;
    float iterations_ms = 0.0f;
    float finalize_ms = 0.0f;
    float combo_evs_ms = 0.0f;
    float exploitability_ms = 0.0f;
    float postsolve_ms = 0.0f;
    float total_ms = 0.0f;

    uint32_t tree_nodes = 0;
    uint32_t tree_edges = 0;
    uint32_t matchup_tables = 0;
    uint32_t postsolve_threads = 1;
};

/// Sprint 1 (market-beating plan): per-solve resource report. Every solve
/// fills this with byte-level estimates and the budget decision so the UI
/// (and benchmark harness) can show "this run cost X MB host, Y MB GPU,
/// Z MB JSON; reduced runouts? truncated tree? fell back to CPU?".
struct SolveResources {
    uint32_t canonical_combos          = 0;
    uint32_t player_nodes              = 0;
    uint64_t estimated_matchup_bytes   = 0;
    uint64_t estimated_cpu_state_bytes = 0;
    uint64_t estimated_gpu_state_bytes = 0;
    uint64_t estimated_strategy_tree_bytes = 0;
    uint64_t estimated_json_bytes      = 0;
    uint64_t host_budget_bytes         = 0;
    uint64_t gpu_budget_bytes          = 0;
    uint32_t strategy_tree_max_nodes   = 0;
    uint32_t strategy_tree_emitted_nodes = 0;
    bool     strategy_tree_truncated   = false;
    /// Stringified BudgetDecision (see memory_budget.h).
    std::string budget_decision        = "ok";
    /// Free-text describing why we fell back / downgraded, if any.
    std::string fallback_reason;
    /// Human-readable diagnostic when budget_decision != "ok".
    std::string diagnostic;
};

/// Full solver result
struct SolverResult {
    int iterations_run = 0;
    float exploitability_pct = 0.0f;
    bool combo_evs_computed = false;
    bool exploitability_computed = false;

    /// Action labels for the current node (e.g., "Check", "Bet_33", "Bet_75")
    std::vector<std::string> action_labels;

    /// Global strategy: aggregated action frequencies across all combos
    std::vector<std::pair<std::string, float>> global_strategy;

    /// Per-grid-label strategy (label → list of (action, frequency_0_to_1)).
    /// Populated by Solver::extract_combo_strategies_at. Powers the UI grid.
    std::vector<std::pair<std::string,
        std::vector<std::pair<std::string, float>>>> combo_strategies;

    /// Per-combo strategy matrix: strategy[combo_idx][action_idx] = frequency
    /// Only populated for full solve, NOT for API responses (context protection)
    std::vector<std::vector<float>> strategy_matrix;

    /// Per-combo EV
    std::vector<float> ev_vector;

    /// Optional: targeted combo analysis
    ComboAnalysis target_analysis;
    bool has_target = false;

    /// Acting player at the current node ("OOP" | "IP" | ""). UI uses this
    /// for the "當前行動者" header so it doesn't have to re-derive from the
    /// action_path.
    std::string acting_player;

    /// Opponent's reach-weighted range at the current node. Keyed by grid
    /// label (e.g. "AKs"), values normalized to [0, 1] where 1.0 is the
    /// heaviest label at this node. Used for the "view opponent" toggle.
    std::string opponent_side;  // "OOP" | "IP" (who the opponent is)
    std::vector<std::pair<std::string, float>> opponent_range;

    SolverTiming timing;

    /// Sprint 1: per-solve resource estimate + budget decision.
    SolveResources resources;
};

} // namespace deepsolver

/**
 * @file main.cpp
 * @brief CLI entry point for DeepSolver core engine.
 *
 * Usage:
 *   deepsolver_core.exe --pot 100 --stack 500 --board AsKd7c
 *                       [--history "Check,Bet33"]
 *                       [--target AhKh]
 *                       [--iterations 300]
 *                       [--exploitability 0.5]
 *
 * Output: JSON result to stdout.
 */

#include "types.h"
#include "card.h"
#include "hand_evaluator.h"
#include "solver.h"
#include "solver_backend.h"
#include "solver_decomposed.h"   // Stage 5: runout decomposition (opt-in route)
#include "cpu_simd.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <sstream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <cstring>
#include <map>
#include <stdexcept>

using namespace deepsolver;

// ============================================================================
// Argument Parsing
// ============================================================================

struct CLIArgs {
    float pot = 0;
    float stack = 0;
    std::string board_str;
    std::string history;
    std::string target_combo;
    int iterations = 500;
    float exploitability = 0.5f;    // percentage
    /// v1.3.0: hard wall-clock cap on the iteration phase. 0 = no cap.
    /// UI presets: Quick=60, Standard=300, Deep=900.
    int time_budget_seconds = 0;
    bool help = false;
    bool gpu_info = false;

    // Polish #2: benchmark mode. Empty = normal solve. Recognized presets:
    //   "standard" — AsKd7c rainbow, full ranges, 100 iter. Emits benchmark JSON
    //                 instead of normal solver JSON. Useful for perf regression
    //                 tracking and CI smoke timing.
    std::string benchmark;

    // v1.2.2: --estimate-only runs iso + tree build only, then emits a
    // SolveResources JSON block (memory + ETA estimates) and exits. Frontend
    // calls this before kicking off the real solve so the UI can show
    // "Estimated 12 minutes on CPU" before the user commits.
    bool estimate_only = false;

    // Backend selection: "auto" | "cpu" | "gpu"
    std::string backend = "auto";

    // Bet sizing
    std::vector<float> flop_sizes = {0.33f, 0.75f};
    std::vector<float> turn_sizes = {0.75f};
    std::vector<float> river_sizes = {0.75f};

    // Custom ranges (range-string format)
    std::string ip_range_str;
    std::string oop_range_str;

    // Node locks (JSON string)
    std::string node_locks_str;
    bool emit_strategy_tree = true;
    bool emit_strategy_tree_evs = true;
    /// Phase 3 (10-point plan): "visible" caches EVs only for emitted nodes
    /// (typical: hundreds of MB → few MB). "none" skips EV computation
    /// entirely. "full" is the legacy behavior (cache everything). Set via
    /// --strategy-tree-evs <mode>; the older --no-strategy-tree-evs flag
    /// still works and maps to "none".
    std::string strategy_tree_ev_mode = "visible";
    std::string postsolve = "full";  // full | ev | exploitability | none
    bool parallel_postsolve = true;
    uint32_t postsolve_threads = 0;
    bool force_cpu_postsolve = false;  // skip GPU postsolve fast path
    std::string dcfr_schedule = "postflop";  // "standard" | "postflop" — postflop
                                             // converges ~3-4× faster (2026-06-25)
    /// Stage 5: runout decomposition route. "off" (default) = legacy behavior.
    /// "auto" = decompose only when the builder would collapse the turn
    /// (rainbow/huge) → real turn/river under bounded memory, runout_approximated
    /// becomes false. "on" = always decompose (turn subgames) even on enumerable
    /// boards. Default off until the full UI slice + a rainbow end-to-end is green.
    std::string decompose_runouts = "off";   // off | auto | on
    /// Roadmap ④ (post-v1.9.0): decomposition iteration presets, promoted
    /// from the DEEPSOLVER_DECOMP_* env knobs to CLI flags so the UI's
    /// solveMode presets (poker.ts DECOMPOSE_PRESETS) can drive them.
    /// 0 / -1 = unset → keep the engine's dev defaults; env still wins over
    /// CLI as the dev override.
    int decompose_outer       = 0;    // --decompose-outer <sweeps>
    int decompose_inner       = 0;    // --decompose-inner <per-subgame iters>
    int decompose_trunk_iters = 0;    // --decompose-trunk-iters <K per sweep>
    int decompose_warmstart   = -1;   // --decompose-warmstart 0|1
    float rake_rate = 0.0f;
    float rake_cap  = 0.0f;

    // CPU SIMD policy + thread count overrides.
    //   --cpu-simd auto|scalar|avx2  default auto (CPUID picks)
    //   --cpu-threads N              default 0 (auto = hardware concurrency for
    //                                levelized backend; reference still caps at 2)
    std::string cpu_simd = "auto";
    uint32_t cpu_threads = 0;

    // v1.5.0 Phase 4: CPU CFR backend variant.
    //   levelized (default, v1.8.0+) — BFS-flat traversal, scales past 2 threads
    //   reference                    — recursive scratch-arena CpuBackend
    //                                  kept as parity oracle / debug escape hatch
    std::string cpu_backend_kind = "levelized";

    // v1.8.0 Sprint 3: persistent OpenMP team for the levelized backend's
    // forward / backward passes. Default 1 for production CPU solves; pass
    // 0 to A/B against the old per-level parallel-region path.
    int cpu_persistent_omp = 1;
    int cpu_showdown_batch = 0;

    // Sprint 1 (market-beating plan): per-solve memory budget overrides.
    // Default 0 = use the SolverConfig defaults (6 GB host, 100 MB JSON,
    // 2000 emitted strategy-tree nodes). Any non-zero value supersedes
    // memory_budget.h defaults for THIS solve only.
    uint64_t host_memory_mb        = 0;
    uint64_t gpu_memory_mb         = 0;
    uint64_t json_memory_mb        = 0;
    uint32_t strategy_tree_max_nodes = 0;

    // Tree construction flags (expose SolverConfig defaults)
    int  oop_has_initiative = 1;     // 0/1 (default 1: OOP can bet at root)
    int  allow_donk_bet     = 0;     // 0/1 (adds an extra small donk size when OOP has no initiative)

    // v1.8.0 paired benchmark mode (`--benchmark paired`):
    //   --benchmark-case standard | monotone   (which fixture)
    //   --runs N                                (paired N-of-each, BCBC...)
    //   --benchmark-threads N                   (cpu_threads override; 0=auto)
    std::string benchmark_case = "standard";
    int benchmark_runs = 3;
    int benchmark_threads_override = -1;   // -1 = use args.cpu_threads / 0=auto

    // v1.8.0 P3-9 — `--benchmark matrix --include-scalar` adds scalar-SIMD
    // rows (otherwise we only emit the AVX2 ones since shipping binaries
    // pick AVX2 by default).
    int  include_scalar = 0;
};

CLIArgs parse_args(int argc, char* argv[]) {
    CLIArgs args;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            args.help = true;
        } else if (arg == "--pot" && i + 1 < argc) {
            args.pot = std::stof(argv[++i]);
        } else if (arg == "--stack" && i + 1 < argc) {
            args.stack = std::stof(argv[++i]);
        } else if (arg == "--board" && i + 1 < argc) {
            args.board_str = argv[++i];
        } else if (arg == "--history" && i + 1 < argc) {
            args.history = argv[++i];
        } else if (arg == "--target" && i + 1 < argc) {
            args.target_combo = argv[++i];
        } else if (arg == "--iterations" && i + 1 < argc) {
            args.iterations = std::stoi(argv[++i]);
        } else if (arg == "--exploitability" && i + 1 < argc) {
            args.exploitability = std::stof(argv[++i]);
        } else if (arg == "--ip-range" && i + 1 < argc) {
            args.ip_range_str = argv[++i];
        } else if (arg == "--oop-range" && i + 1 < argc) {
            args.oop_range_str = argv[++i];
        } else if (arg == "--node-locks" && i + 1 < argc) {
            args.node_locks_str = argv[++i];
        } else if (arg == "--no-strategy-tree") {
            args.emit_strategy_tree = false;
        } else if (arg == "--no-strategy-tree-evs") {
            args.emit_strategy_tree_evs = false;
            args.strategy_tree_ev_mode = "none";
        } else if (arg == "--strategy-tree-evs" && i + 1 < argc) {
            args.strategy_tree_ev_mode = argv[++i];
            if (args.strategy_tree_ev_mode == "none") {
                args.emit_strategy_tree_evs = false;
            } else {
                args.emit_strategy_tree_evs = true;
            }
        } else if (arg == "--postsolve" && i + 1 < argc) {
            args.postsolve = argv[++i];
        } else if (arg == "--fast-postsolve") {
            args.postsolve = "none";
        } else if (arg == "--single-thread-postsolve") {
            args.parallel_postsolve = false;
        } else if (arg == "--parallel-postsolve") {
            args.parallel_postsolve = true;
        } else if (arg == "--postsolve-threads" && i + 1 < argc) {
            int v = std::stoi(argv[++i]);
            args.postsolve_threads = static_cast<uint32_t>(std::max(0, v));
        } else if (arg == "--force-cpu-postsolve") {
            args.force_cpu_postsolve = true;
        } else if (arg == "--gpu-postsolve") {
            args.force_cpu_postsolve = false;
        } else if (arg == "--dcfr-schedule" && i + 1 < argc) {
            args.dcfr_schedule = argv[++i];
        } else if (arg == "--decompose-runouts" && i + 1 < argc) {
            args.decompose_runouts = argv[++i];
        } else if (arg == "--decompose-outer" && i + 1 < argc) {
            args.decompose_outer = std::max(0, std::stoi(argv[++i]));
        } else if (arg == "--decompose-inner" && i + 1 < argc) {
            args.decompose_inner = std::max(0, std::stoi(argv[++i]));
        } else if (arg == "--decompose-trunk-iters" && i + 1 < argc) {
            args.decompose_trunk_iters = std::max(0, std::stoi(argv[++i]));
        } else if (arg == "--decompose-warmstart" && i + 1 < argc) {
            std::string v = argv[++i];
            args.decompose_warmstart = (v == "1" || v == "true" || v == "True") ? 1 : 0;
        } else if (arg == "--rake-rate" && i + 1 < argc) {
            args.rake_rate = std::stof(argv[++i]);
        } else if (arg == "--rake-cap" && i + 1 < argc) {
            args.rake_cap = std::stof(argv[++i]);
        } else if (arg == "--rake-nl25") {
            // NL25 standard: 5% capped at 2bb. Caller is expected to use
            // pot/stack in the SAME unit (e.g. 1bb=1 → cap=2.0,
            // 1bb=10 → cap=20, 1bb=100 → cap=200).
            args.rake_rate = 0.05f;
            args.rake_cap  = 2.0f;
        } else if (arg == "--host-memory-mb" && i + 1 < argc) {
            args.host_memory_mb = static_cast<uint64_t>(std::max(0, std::stoi(argv[++i])));
        } else if (arg == "--gpu-memory-mb" && i + 1 < argc) {
            args.gpu_memory_mb = static_cast<uint64_t>(std::max(0, std::stoi(argv[++i])));
        } else if (arg == "--json-memory-mb" && i + 1 < argc) {
            args.json_memory_mb = static_cast<uint64_t>(std::max(0, std::stoi(argv[++i])));
        } else if (arg == "--strategy-tree-max-nodes" && i + 1 < argc) {
            args.strategy_tree_max_nodes = static_cast<uint32_t>(std::max(0, std::stoi(argv[++i])));
        } else if (arg == "--backend" && i + 1 < argc) {
            args.backend = argv[++i];
        } else if (arg == "--oop-initiative" && i + 1 < argc) {
            std::string v = argv[++i];
            args.oop_has_initiative = (v == "1" || v == "true" || v == "True") ? 1 : 0;
        } else if (arg == "--allow-donk-bet" && i + 1 < argc) {
            std::string v = argv[++i];
            args.allow_donk_bet = (v == "1" || v == "true" || v == "True") ? 1 : 0;
        } else if (arg == "--flop-sizes" && i + 1 < argc) {
            args.flop_sizes.clear();
            std::string s = argv[++i];
            std::stringstream ss(s); std::string tok;
            while (std::getline(ss, tok, ',')) {
                try { args.flop_sizes.push_back(std::stof(tok)); } catch (...) {}
            }
        } else if (arg == "--turn-sizes" && i + 1 < argc) {
            args.turn_sizes.clear();
            std::string s = argv[++i];
            std::stringstream ss(s); std::string tok;
            while (std::getline(ss, tok, ',')) {
                try { args.turn_sizes.push_back(std::stof(tok)); } catch (...) {}
            }
        } else if (arg == "--river-sizes" && i + 1 < argc) {
            args.river_sizes.clear();
            std::string s = argv[++i];
            std::stringstream ss(s); std::string tok;
            while (std::getline(ss, tok, ',')) {
                try { args.river_sizes.push_back(std::stof(tok)); } catch (...) {}
            }
        } else if (arg == "--gpu-info") {
            args.gpu_info = true;
        } else if (arg == "--benchmark" && i + 1 < argc) {
            args.benchmark = argv[++i];
        } else if (arg == "--estimate-only") {
            args.estimate_only = true;
        } else if (arg == "--time-budget-seconds" && i + 1 < argc) {
            args.time_budget_seconds = std::max(0, std::stoi(argv[++i]));
        } else if (arg == "--cpu-simd" && i + 1 < argc) {
            args.cpu_simd = argv[++i];
        } else if (arg == "--cpu-threads" && i + 1 < argc) {
            int v = std::stoi(argv[++i]);
            args.cpu_threads = static_cast<uint32_t>(std::max(0, v));
        } else if (arg == "--cpu-backend" && i + 1 < argc) {
            args.cpu_backend_kind = argv[++i];
        } else if (arg == "--cpu-persistent-omp" && i + 1 < argc) {
            std::string v = argv[++i];
            args.cpu_persistent_omp = (v == "1" || v == "true" || v == "True") ? 1 : 0;
        } else if (arg == "--cpu-showdown-batch" && i + 1 < argc) {
            std::string v = argv[++i];
            args.cpu_showdown_batch = (v == "1" || v == "true" || v == "True") ? 1 : 0;
        } else if (arg == "--include-scalar") {
            args.include_scalar = 1;
        } else if (arg == "--benchmark-case" && i + 1 < argc) {
            args.benchmark_case = argv[++i];
        } else if (arg == "--runs" && i + 1 < argc) {
            args.benchmark_runs = std::max(1, std::stoi(argv[++i]));
        } else if (arg == "--benchmark-threads" && i + 1 < argc) {
            args.benchmark_threads_override = std::max(0, std::stoi(argv[++i]));
        }
    }

    return args;
}

void print_help() {
    std::cerr << R"(
DeepSolver Core Engine — GPU-Accelerated GTO Poker Solver
=========================================================

Usage:
  deepsolver_core.exe --pot <pot_size> --stack <effective_stack> --board <board_cards>
                      [--history <action_history>]
                      [--target <combo>]
                      [--iterations <max_iterations>]
                      [--exploitability <target_pct>]

Arguments:
  --pot <float>            Pot size (required)
  --stack <float>          Effective stack size (required)
  --board <string>         Board cards, e.g. "AsKd7c" (required)
  --history <string>       Action history, e.g. "Check,Bet33" (optional)
  --target <string>        Target combo to analyze, e.g. "AhKh" (optional)
  --iterations <int>       Max DCFR iterations (default: 500)
  --exploitability <float> Target exploitability % of pot; the solve stops early
                           once the running-average reaches it (default: 0.5,
                           checked every 50 iters; 0 = run all iterations)
  --backend <string>       Execution backend: auto | cpu | gpu (default: auto)
  --postsolve <string>     Reporting pass: full | ev | exploitability | none
  --fast-postsolve         Alias for --postsolve none
  --single-thread-postsolve Disable parallel postsolve passes
  --postsolve-threads <int> CPU postsolve worker cap (0 = auto)
  --force-cpu-postsolve    Skip the GPU postsolve fast path (CPU traversal)
  --gpu-postsolve          Re-enable GPU postsolve (default when supported)
  --no-strategy-tree       Omit the client navigation cache from JSON output
  --no-strategy-tree-evs   Keep strategy tree but omit per-node EV cache
  --gpu-info               Print detected GPU info and exit
  --benchmark <preset>     Run a fixed scenario for perf tracking, emit benchmark JSON.
                           Presets:
                             narrow   - AsKd7c, AA/KK/QQ/AK only, 100 iter
                             medium_sparse - AsKd7c, 169 combos/player, 100 iter
                             monotone - AsKsQs monotone, full ranges, 20 iter
                             standard — AsKd7c rainbow, full ranges, 100 iter
                             matrix   — sweep (scenario × backend × threads),
                                        emits aggregated JSON. Add --include-scalar
                                        to also benchmark scalar-SIMD variants.
  --estimate-only          Build tree + estimate solve time/memory, emit JSON, exit.
                           Skips precompute_matchups + iterations. Sub-second on most
                           spots — used by the UI to show ETA before committing.
                           With --decompose-runouts auto|on the JSON also carries a
                           "decompose" block (leaves, per-sweep/total seconds, SPR-keyed
                           expected-accuracy band) so the UI can price Exact mode.
  --decompose-runouts <m>  off (default) | auto | on. auto re-solves collapsed/
                           rainbow boards via flop-trunk + per-turn-card subgames
                           (real runout equity); on forces it (debug).
  --decompose-outer <n>    Outer sweeps (leaf value refreshes). Unset = engine default.
  --decompose-inner <n>    Per-subgame CFR iterations per solve. Unset = engine default.
  --decompose-trunk-iters <k>
                           Trunk CFR iterations per sweep (K). Unset = engine default.
  --decompose-warmstart <0|1>
                           Warm-start persistent subgames across sweeps.
                           DEEPSOLVER_DECOMP_* env vars override all four (dev knob).
  --time-budget-seconds <s> Hard wall-clock cap on the iteration phase. Stops at
                           min(iter cap, time budget). 0 = no time cap (default).
                           UI presets: Quick=60, Standard=300, Deep=900.
  --cpu-simd <mode>        SIMD mode for CPU CFR kernels: auto | scalar | avx2.
                           auto (default) picks AVX2 when CPUID + OS support it,
                           else scalar. Use "scalar" for parity tests / debugging.
  --cpu-threads <int>      CPU CFR thread count. 0 = auto (hardware concurrency
                           for levelized; capped at 2 for reference, the only
                           limit the recursive backend can use). Use 1 to force
                           serial. Larger values are honored on levelized via
                           num_threads(...) on each parallel-for.
  --cpu-backend <kind>     CPU CFR variant: levelized | reference.
                           levelized (default, v1.8.0+): BFS-flat traversal,
                             scales linearly to physical cores; production CPU.
                           reference: recursive scratch-arena; kept as parity
                             oracle and CLI-only debug escape hatch.
  --cpu-persistent-omp <0|1>
                           Levelized only. Wraps forward_pass / backward_pass in
                           a single `omp parallel` region per pass instead of
                           one per level. Default 1; use 0 only for A/B profiling.
  --cpu-showdown-batch <0|1>
                           Experimental levelized CPU A/B knob. Default 0.
                           Forces OOP level-0 showdown group batching when 1.
  --help, -h               Show this help message

Output:
  JSON result to stdout.

Example:
  deepsolver_core.exe --pot 100 --stack 500 --board AsKd7c --target AhKh
)"  << std::endl;
}

// ============================================================================
// Range Parsing: range-string format -> 1326-float weight array
// ============================================================================

/// Parse range string, e.g. "AA:1.0,AKs:0.5,A4o:1.0"
/// Returns a map of grid_label -> weight
void apply_range_string(const std::string& range_str, std::array<float, NUM_COMBOS>& weights) {
    if (range_str.empty()) return;

    // First, set all weights to 0 (custom range mode)
    weights.fill(0.0f);

    // Build reverse map: grid_label -> list of combo indices
    const auto& combo_table = get_combo_table();
    std::map<std::string, std::vector<uint16_t>> label_to_combos;
    for (uint16_t i = 0; i < NUM_COMBOS; ++i) {
        std::string label = combo_to_grid_label(combo_table[i]);
        label_to_combos[label].push_back(i);
    }

    // Parse "combo:freq,combo:freq,..."
    std::istringstream iss(range_str);
    std::string token;
    while (std::getline(iss, token, ',')) {
        // Trim whitespace
        size_t start = token.find_first_not_of(" \t");
        size_t end = token.find_last_not_of(" \t");
        if (start == std::string::npos) continue;
        token = token.substr(start, end - start + 1);

        size_t colon = token.find(':');
        if (colon == std::string::npos) continue;

        std::string label = token.substr(0, colon);
        float freq = std::stof(token.substr(colon + 1));

        auto it = label_to_combos.find(label);
        if (it != label_to_combos.end()) {
            for (uint16_t idx : it->second) {
                weights[idx] = freq;
            }
        }
    }
}

/// Parse a combo string like "Ad4s" into a combo index
uint16_t parse_combo_to_index(const std::string& combo_str) {
    if (combo_str.size() != 4) return UINT16_MAX;
    Card c0 = parse_card(combo_str.substr(0, 2));
    Card c1 = parse_card(combo_str.substr(2, 2));
    Combo combo(c0, c1);
    return combo.index();
}

/// Parse node locks from JSON string
/// Format: [{"history":"Check,Bet_33","combo":"Ad4s","strategy":[0,1,0,0]}, ...]
std::vector<NodeLockEntry> parse_node_locks(const std::string& json_str) {
    std::vector<NodeLockEntry> locks;
    if (json_str.empty()) return locks;

    // Manual JSON parser with brace-depth tracking for robustness
    size_t pos = 0;
    while (pos < json_str.size()) {
        size_t obj_start = json_str.find('{', pos);
        if (obj_start == std::string::npos) break;

        // Find matching '}' with depth tracking (handles nested objects)
        int depth = 1;
        size_t obj_end = obj_start + 1;
        bool in_string = false;
        while (obj_end < json_str.size() && depth > 0) {
            char ch = json_str[obj_end];
            if (ch == '\\' && in_string) { obj_end += 2; continue; } // skip escaped chars
            if (ch == '"') in_string = !in_string;
            else if (!in_string) {
                if (ch == '{') ++depth;
                else if (ch == '}') --depth;
            }
            if (depth > 0) ++obj_end;
        }
        if (depth != 0) break; // malformed JSON

        std::string obj = json_str.substr(obj_start + 1, obj_end - obj_start - 1);
        pos = obj_end + 1;

        NodeLockEntry entry;

        // Extract "key":"value" with escaped-quote awareness
        auto extract_str = [&](const std::string& key) -> std::string {
            std::string search = "\"" + key + "\":\"";
            size_t k = obj.find(search);
            if (k == std::string::npos) return "";
            size_t vstart = k + search.size();
            // Find closing quote, skipping escaped quotes
            size_t vend = vstart;
            while (vend < obj.size()) {
                if (obj[vend] == '\\') { vend += 2; continue; }
                if (obj[vend] == '"') break;
                ++vend;
            }
            if (vend >= obj.size()) return "";
            return obj.substr(vstart, vend - vstart);
        };

        entry.history = extract_str("history");
        entry.combo_str = extract_str("combo");
        entry.combo_idx = parse_combo_to_index(entry.combo_str);

        // Extract "strategy":[...]
        size_t strat_key = obj.find("\"strategy\":");
        if (strat_key != std::string::npos) {
            size_t bracket_start = obj.find('[', strat_key);
            size_t bracket_end = obj.find(']', bracket_start);
            if (bracket_start != std::string::npos && bracket_end != std::string::npos) {
                std::string arr = obj.substr(bracket_start + 1, bracket_end - bracket_start - 1);
                std::istringstream arr_stream(arr);
                std::string num;
                while (std::getline(arr_stream, num, ',')) {
                    // Trim whitespace
                    size_t s = num.find_first_not_of(" \t");
                    if (s == std::string::npos) continue;
                    num = num.substr(s);
                    try {
                        entry.strategy.push_back(std::stof(num));
                    } catch (...) {
                        // Skip malformed numbers
                    }
                }
            }
        }

        if (entry.combo_idx != UINT16_MAX && !entry.strategy.empty()) {
            locks.push_back(entry);
        }
    }
    return locks;
}

// ============================================================================
// JSON Output
// ============================================================================

std::string escape_json(const std::string& s) {
    std::string result;
    result.reserve(s.size());
    for (char c : s) {
        switch (c) {
            case '"': result += "\\\""; break;
            case '\\': result += "\\\\"; break;
            case '\n': result += "\\n"; break;
            case '\r': result += "\\r"; break;
            case '\t': result += "\\t"; break;
            default: result += c; break;
        }
    }
    return result;
}

const char* json_bool(bool value) {
    return value ? "true" : "false";
}

// JSON has no inf/nan tokens — a single non-finite float invalidates the
// whole response and wastes an otherwise-successful solve (seen once in a
// bulk run: `"98s":inf` inside a combo_evs map, GPU run-to-run noise).
// Sources guard their divisions; this is the emit-boundary backstop for
// every solver-derived float.
double jsafe(double v) {
    return std::isfinite(v) ? v : 0.0;
}

// Roadmap ④: one resolver for the decomposition iteration budget, shared by
// the solve route and the --estimate-only pre-flight so the ETA prices
// exactly the run the user would get. Precedence: engine dev defaults →
// CLI presets (--decompose-*) → DEEPSOLVER_DECOMP_* env (dev override wins).
deepsolver::DecompositionOptions resolve_decomposition_options(const CLIArgs& args) {
    deepsolver::DecompositionOptions o;
    // Dev defaults, unchanged from the pre-preset era (keeps the existing
    // test surface byte-identical when no flag/env is present).
    o.outer_iterations = 20;
    o.inner_iterations = 120;

    if (args.decompose_outer > 0)       o.outer_iterations = args.decompose_outer;
    if (args.decompose_inner > 0)       o.inner_iterations = args.decompose_inner;
    if (args.decompose_trunk_iters > 0) o.trunk_iterations_per_sweep = args.decompose_trunk_iters;
    if (args.decompose_warmstart >= 0)  o.warm_start_subgames = (args.decompose_warmstart != 0);

    if (const char* e = std::getenv("DEEPSOLVER_DECOMP_OUTER")) o.outer_iterations = std::max(1, std::atoi(e));
    if (const char* e = std::getenv("DEEPSOLVER_DECOMP_INNER")) o.inner_iterations = std::max(1, std::atoi(e));
    if (const char* e = std::getenv("DEEPSOLVER_DECOMP_WARMSTART"))
        o.warm_start_subgames = (std::atoi(e) != 0);
    if (const char* e = std::getenv("DEEPSOLVER_DECOMP_TRUNK_ITERS"))
        o.trunk_iterations_per_sweep = std::max(1, std::atoi(e));
    return o;
}

void write_cpu_diagnostics_json(
    std::ostream& out,
    const CpuBackendDiagnostics& d,
    const std::string& indent) {
    const std::ios::fmtflags old_flags = out.flags();
    const std::streamsize old_precision = out.precision();
    const std::string inner = indent + "  ";

    out << std::fixed << std::setprecision(4);
    out << indent << "{\n";
    out << inner << "\"available\": " << json_bool(d.available) << ",\n";
    out << inner << "\"canonical_combos\": " << d.canonical_combos << ",\n";
    out << inner << "\"oop_active_count\": " << d.oop_active_count << ",\n";
    out << inner << "\"ip_active_count\": " << d.ip_active_count << ",\n";
    out << inner << "\"oop_active_density\": " << d.oop_active_density << ",\n";
    out << inner << "\"ip_active_density\": " << d.ip_active_density << ",\n";
    out << inner << "\"oop_active_run_count\": "
        << d.oop_active_run_count << ",\n";
    out << inner << "\"ip_active_run_count\": "
        << d.ip_active_run_count << ",\n";
    out << inner << "\"oop_active_block_count\": "
        << d.oop_active_block_count << ",\n";
    out << inner << "\"ip_active_block_count\": "
        << d.ip_active_block_count << ",\n";
    out << inner << "\"oop_active_block_span\": "
        << d.oop_active_block_span << ",\n";
    out << inner << "\"ip_active_block_span\": "
        << d.ip_active_block_span << ",\n";
    out << inner << "\"oop_avg_run_length\": "
        << d.oop_avg_run_length << ",\n";
    out << inner << "\"ip_avg_run_length\": "
        << d.ip_avg_run_length << ",\n";
    out << inner << "\"oop_terminal_active_list\": "
        << json_bool(d.oop_terminal_active_list) << ",\n";
    out << inner << "\"ip_terminal_active_list\": "
        << json_bool(d.ip_terminal_active_list) << ",\n";
    out << inner << "\"oop_active_runs\": "
        << json_bool(d.oop_active_runs) << ",\n";
    out << inner << "\"ip_active_runs\": "
        << json_bool(d.ip_active_runs) << ",\n";
    out << inner << "\"oop_active_blocks\": "
        << json_bool(d.oop_active_blocks) << ",\n";
    out << inner << "\"ip_active_blocks\": "
        << json_bool(d.ip_active_blocks) << ",\n";
    out << inner << "\"oop_sparse_traversal\": "
        << json_bool(d.oop_sparse_traversal) << ",\n";
    out << inner << "\"ip_sparse_traversal\": "
        << json_bool(d.ip_sparse_traversal) << ",\n";
    out << inner << "\"oop_block_strategy\": "
        << json_bool(d.oop_block_strategy) << ",\n";
    out << inner << "\"ip_block_strategy\": "
        << json_bool(d.ip_block_strategy) << ",\n";
    out << inner << "\"oop_block_strategy_sum\": "
        << json_bool(d.oop_block_strategy_sum) << ",\n";
    out << inner << "\"ip_block_strategy_sum\": "
        << json_bool(d.ip_block_strategy_sum) << ",\n";
    out << inner << "\"oop_block_traversal\": "
        << json_bool(d.oop_block_traversal) << ",\n";
    out << inner << "\"ip_block_traversal\": "
        << json_bool(d.ip_block_traversal) << ",\n";
    out << inner << "\"oop_terminal_active2\": "
        << json_bool(d.oop_terminal_active2) << ",\n";
    out << inner << "\"ip_terminal_active2\": "
        << json_bool(d.ip_terminal_active2) << ",\n";
    out << inner << "\"ip_terminal_output_skip\": "
        << json_bool(d.ip_terminal_output_skip) << ",\n";
    out << inner << "\"oop_sparse_opp_reach_build\": "
        << json_bool(d.oop_sparse_opp_reach_build) << ",\n";
    out << inner << "\"ip_sparse_opp_reach_build\": "
        << json_bool(d.ip_sparse_opp_reach_build) << ",\n";
    out << inner << "\"fold_blocker_shortcut\": "
        << json_bool(d.fold_blocker_shortcut) << ",\n";
    out << inner << "\"fold_blocker_precomputed\": "
        << json_bool(d.fold_blocker_precomputed) << ",\n";
    out << inner << "\"showdown_rank_blocker_shortcut\": "
        << json_bool(d.showdown_rank_blocker_shortcut) << ",\n";
    out << inner << "\"showdown_signed_coeff_shortcut\": "
        << json_bool(d.showdown_signed_coeff_shortcut) << ",\n";
    out << inner << "\"sparse_terminal_no_full_clear_enabled\": "
        << json_bool(d.sparse_terminal_no_full_clear_enabled) << ",\n";
    out << inner << "\"matchup_category\": {\n";
    out << inner << "  \"tables\": "
        << d.matchup_category_table_count << ",\n";
    out << inner << "  \"rows\": "
        << d.matchup_category_rows << ",\n";
    out << inner << "  \"cells\": "
        << d.matchup_category_cells << ",\n";
    out << inner << "  \"invalid_cells\": "
        << d.matchup_category_invalid_cells << ",\n";
    out << inner << "  \"win_cells\": "
        << d.matchup_category_win_cells << ",\n";
    out << inner << "  \"lose_cells\": "
        << d.matchup_category_lose_cells << ",\n";
    out << inner << "  \"tie_cells\": "
        << d.matchup_category_tie_cells << ",\n";
    out << inner << "  \"zero_rake_payoff_cells\": "
        << d.matchup_zero_rake_payoff_cells << ",\n";
    out << inner << "  \"invalid_density\": "
        << d.matchup_category_invalid_density << ",\n";
    out << inner << "  \"win_density\": "
        << d.matchup_category_win_density << ",\n";
    out << inner << "  \"lose_density\": "
        << d.matchup_category_lose_density << ",\n";
    out << inner << "  \"tie_density\": "
        << d.matchup_category_tie_density << ",\n";
    out << inner << "  \"zero_rake_payoff_density\": "
        << d.matchup_zero_rake_payoff_density << ",\n";
    out << inner << "  \"row_payoff_nonzero_min\": "
        << d.matchup_payoff_row_nonzero_min << ",\n";
    out << inner << "  \"row_payoff_nonzero_p50\": "
        << d.matchup_payoff_row_nonzero_p50 << ",\n";
    out << inner << "  \"row_payoff_nonzero_p95\": "
        << d.matchup_payoff_row_nonzero_p95 << ",\n";
    out << inner << "  \"row_payoff_nonzero_max\": "
        << d.matchup_payoff_row_nonzero_max << ",\n";
    out << inner << "  \"row_payoff_nonzero_avg\": "
        << d.matchup_payoff_row_nonzero_avg << ",\n";
    out << inner << "  \"row_payoff_density_avg\": "
        << d.matchup_payoff_row_density_avg << "\n";
    out << inner << "}\n";
    out << indent << "}";

    out.flags(old_flags);
    out.precision(old_precision);
}

std::string result_to_json(
    const SolverResult& result, const CLIArgs& args,
    const std::string& backend_name,
    const std::map<std::string, deepsolver::Solver::StrategyTreeEntry>*
        strategy_tree = nullptr) {
    (void)args;
    std::ostringstream json;
    json << std::fixed << std::setprecision(2);

    json << "{\n";
    json << "  \"status\": \"success\",\n";
    json << "  \"backend\": \"" << escape_json(backend_name) << "\",\n";
    json << "  \"postsolve_mode\": \"" << escape_json(args.postsolve) << "\",\n";
    json << "  \"parallel_postsolve\": "
         << (args.parallel_postsolve ? "true" : "false") << ",\n";
    json << "  \"postsolve_threads_requested\": " << args.postsolve_threads << ",\n";
    json << "  \"force_cpu_postsolve\": "
         << (args.force_cpu_postsolve ? "true" : "false") << ",\n";
    json << "  \"dcfr_schedule\": \"" << escape_json(args.dcfr_schedule) << "\",\n";
    json << "  \"decompose_runouts\": \"" << escape_json(args.decompose_runouts) << "\",\n";
    json << "  \"rake_rate\": " << args.rake_rate << ",\n";
    json << "  \"rake_cap\": " << args.rake_cap << ",\n";
    json << "  \"iterations_run\": " << result.iterations_run << ",\n";
    json << "  \"exploitability_pct\": " << jsafe(result.exploitability_pct) << ",\n";
    // v1.3.0: which stop condition fired ("iter_cap" / "time_budget").
    json << "  \"early_stop_reason\": \"" << escape_json(result.early_stop_reason) << "\",\n";
    json << "  \"combo_evs_computed\": "
         << (result.combo_evs_computed ? "true" : "false") << ",\n";
    json << "  \"exploitability_computed\": "
         << (result.exploitability_computed ? "true" : "false") << ",\n";
    // Loud warning: flop runout collapsed to the stale-equity fallback, so
    // turn/river equity is approximated. UI surfaces this so the solve isn't
    // trusted as exact on later streets.
    json << "  \"runout_approximated\": "
         << (result.runout_approximated ? "true" : "false") << ",\n";
    json << "  \"timing\": {\n";
    json << std::setprecision(3);
    json << "    \"tree_build_ms\": " << result.timing.tree_build_ms << ",\n";
    json << "    \"isomorphism_ms\": " << result.timing.isomorphism_ms << ",\n";
    json << "    \"precompute_matchups_ms\": " << result.timing.precompute_matchups_ms << ",\n";
    json << "    \"reach_init_ms\": " << result.timing.reach_init_ms << ",\n";
    json << "    \"node_locks_ms\": " << result.timing.node_locks_ms << ",\n";
    json << "    \"backend_prepare_ms\": " << result.timing.backend_prepare_ms << ",\n";
    json << "    \"iterations_ms\": " << result.timing.iterations_ms << ",\n";
    json << "    \"phase_compute_strategy_ms\": "
         << result.timing.phase_compute_strategy_ms << ",\n";
    json << "    \"phase_apply_discount_ms\": "
         << result.timing.phase_apply_discount_ms << ",\n";
    json << "    \"phase_forward_pass_ms\": "
         << result.timing.phase_forward_pass_ms << ",\n";
    json << "    \"phase_backward_pass_oop_ms\": "
         << result.timing.phase_backward_pass_oop_ms << ",\n";
    json << "    \"phase_backward_pass_ip_ms\": "
         << result.timing.phase_backward_pass_ip_ms << ",\n";
    json << "    \"phase_backward_showdown_ms\": "
         << result.timing.phase_backward_showdown_ms << ",\n";
    json << "    \"phase_backward_fold_ms\": "
         << result.timing.phase_backward_fold_ms << ",\n";
    json << "    \"finalize_ms\": " << result.timing.finalize_ms << ",\n";
    json << "    \"combo_evs_ms\": " << result.timing.combo_evs_ms << ",\n";
    json << "    \"exploitability_ms\": " << result.timing.exploitability_ms << ",\n";
    json << "    \"postsolve_ms\": " << result.timing.postsolve_ms << ",\n";
    json << "    \"total_ms\": " << result.timing.total_ms << ",\n";
    json << "    \"tree_nodes\": " << result.timing.tree_nodes << ",\n";
    json << "    \"tree_edges\": " << result.timing.tree_edges << ",\n";
    json << "    \"matchup_tables\": " << result.timing.matchup_tables << ",\n";
    json << "    \"postsolve_threads\": " << result.timing.postsolve_threads << "\n";
    json << "  },\n";
    json << std::setprecision(2);

    json << "  \"cpu_diagnostics\": ";
    write_cpu_diagnostics_json(json, result.cpu_diagnostics, "  ");
    json << ",\n";

    // Sprint 1 (market-beating plan): per-solve resource estimate. Lets the
    // UI / benchmark show "this run was 12 MB host, 4 MB JSON, tree truncated
    // at 2000 nodes, fell back to CPU because of …" without re-running.
    {
        const auto& r = result.resources;
        json << "  \"resources\": {\n";
        json << "    \"canonical_combos\": " << r.canonical_combos << ",\n";
        json << "    \"player_nodes\": " << r.player_nodes << ",\n";
        json << "    \"estimated_matchup_bytes\": " << r.estimated_matchup_bytes << ",\n";
        json << "    \"estimated_cpu_state_bytes\": " << r.estimated_cpu_state_bytes << ",\n";
        json << "    \"estimated_gpu_state_bytes\": " << r.estimated_gpu_state_bytes << ",\n";
        json << "    \"estimated_strategy_tree_bytes\": " << r.estimated_strategy_tree_bytes << ",\n";
        json << "    \"estimated_json_bytes\": " << r.estimated_json_bytes << ",\n";
        json << "    \"host_budget_bytes\": " << r.host_budget_bytes << ",\n";
        json << "    \"gpu_budget_bytes\": " << r.gpu_budget_bytes << ",\n";
        json << "    \"strategy_tree_max_nodes\": " << r.strategy_tree_max_nodes << ",\n";
        json << "    \"strategy_tree_emitted_nodes\": " << r.strategy_tree_emitted_nodes << ",\n";
        json << "    \"strategy_tree_truncated\": " << (r.strategy_tree_truncated ? "true" : "false") << ",\n";
        json << "    \"budget_decision\": \"" << escape_json(r.budget_decision) << "\",\n";
        json << "    \"diagnostic\": \"" << escape_json(r.diagnostic) << "\",\n";
        json << "    \"fallback_reason\": \"" << escape_json(r.fallback_reason) << "\",\n";
        // v1.2.2: pre-iteration solve-time estimate
        json << "    \"ops_per_iteration\": " << r.ops_per_iteration << ",\n";
        json << "    \"backend_for_estimate\": \"" << escape_json(r.backend_for_estimate) << "\",\n";
        json << "    \"estimated_solve_seconds\": " << r.estimated_solve_seconds << ",\n";
        // v1.4.0 Phase 2: CPU mode diagnostics (empty/0 on GPU solves).
        json << "    \"cpu_simd\": \"" << escape_json(r.cpu_simd) << "\",\n";
        json << "    \"cpu_threads_effective\": " << r.cpu_threads_effective << ",\n";
        json << "    \"cpu_backend_kind\": \"" << escape_json(r.cpu_backend_kind) << "\"\n";
        json << "  },\n";
    }

    // Global strategy
    json << "  \"global_strategy\": {\n";
    for (size_t i = 0; i < result.global_strategy.size(); ++i) {
        json << "    \"" << escape_json(result.global_strategy[i].first) << "\": \""
             << std::setprecision(1) << jsafe(result.global_strategy[i].second) << "%\"";
        if (i + 1 < result.global_strategy.size()) json << ",";
        json << "\n";
    }
    json << "  }";

    // Per-combo strategies keyed by grid label. Frontend's ComboStrategy is
    // Record<string, number> with frequencies in [0, 1] — emit as floats, not %.
    if (!result.combo_strategies.empty()) {
        json << ",\n  \"combo_strategies\": {\n";
        for (size_t i = 0; i < result.combo_strategies.size(); ++i) {
            const auto& [label, mix] = result.combo_strategies[i];
            json << "    \"" << escape_json(label) << "\": {";
            for (size_t j = 0; j < mix.size(); ++j) {
                json << "\"" << escape_json(mix[j].first) << "\": "
                     << std::setprecision(4) << jsafe(mix[j].second);
                if (j + 1 < mix.size()) json << ", ";
            }
            json << "}";
            if (i + 1 < result.combo_strategies.size()) json << ",";
            json << "\n";
        }
        json << "  }";
        // Restore precision for downstream fields.
        json << std::setprecision(2);
    }

    // Acting player (for UI header "當前行動者")
    if (!result.acting_player.empty()) {
        json << ",\n  \"acting_player\": \"" << escape_json(result.acting_player) << "\"";
    }

    // Opponent's reach-weighted range at this node (for "view opponent" toggle).
    // Values in [0, 1] normalized to the heaviest label.
    if (!result.opponent_range.empty()) {
        json << ",\n  \"opponent_side\": \"" << escape_json(result.opponent_side) << "\"";
        json << ",\n  \"opponent_range\": {\n";
        json << std::setprecision(4);
        for (size_t i = 0; i < result.opponent_range.size(); ++i) {
            json << "    \"" << escape_json(result.opponent_range[i].first)
                 << "\": " << jsafe(result.opponent_range[i].second);
            if (i + 1 < result.opponent_range.size()) json << ",";
            json << "\n";
        }
        json << "  }";
        json << std::setprecision(2);
    }

    // Target combo analysis
    if (result.has_target) {
        json << ",\n  \"target_combo_analysis\": {\n";
        json << "    \"combo\": \"" << escape_json(result.target_analysis.combo_str) << "\",\n";
        json << "    \"best_action\": \"" << escape_json(result.target_analysis.best_action)
             << "\",\n";
        json << "    \"ev\": " << std::setprecision(2) << jsafe(result.target_analysis.ev) << ",\n";
        json << "    \"strategy_mix\": {\n";
        for (size_t i = 0; i < result.target_analysis.strategy_mix.size(); ++i) {
            json << "      \"" << escape_json(result.target_analysis.strategy_mix[i].first)
                 << "\": \"" << std::setprecision(0)
                 << jsafe(result.target_analysis.strategy_mix[i].second) << "%\"";
            if (i + 1 < result.target_analysis.strategy_mix.size()) json << ",";
            json << "\n";
        }
        json << "    }\n";
        json << "  }";
    }

    // Strategy tree (Route A: client-side navigation cache).
    // Format: { "<history_path>": { acting, action_labels[], global_strategy{},
    //                                combo_strategies{label:{action:freq}},
    //                                opponent_side, opponent_range{} } }
    // history_path uses comma-separated player action labels, "" for root.
    if (strategy_tree && !strategy_tree->empty()) {
        json << ",\n  \"strategy_tree\": {\n";
        size_t entry_i = 0;
        for (const auto& [path, entry] : *strategy_tree) {
            json << "    \"" << escape_json(path) << "\": {";
            json << "\"acting\":\"" << escape_json(entry.acting) << "\",";

            // action_labels
            json << "\"action_labels\":[";
            for (size_t i = 0; i < entry.action_labels.size(); ++i) {
                json << "\"" << escape_json(entry.action_labels[i]) << "\"";
                if (i + 1 < entry.action_labels.size()) json << ",";
            }
            json << "],";

            // global_strategy: action -> "%" string (matches existing schema)
            json << "\"global_strategy\":{";
            for (size_t i = 0; i < entry.global_strategy.size(); ++i) {
                json << "\"" << escape_json(entry.global_strategy[i].first) << "\":\""
                     << std::setprecision(1) << std::fixed
                     << jsafe(entry.global_strategy[i].second) << "%\"";
                if (i + 1 < entry.global_strategy.size()) json << ",";
            }
            json << "},";

            // combo_strategies: label -> {action -> freq}
            json << "\"combo_strategies\":{";
            for (size_t i = 0; i < entry.combo_strategies.size(); ++i) {
                const auto& [label, mix] = entry.combo_strategies[i];
                json << "\"" << escape_json(label) << "\":{";
                for (size_t j = 0; j < mix.size(); ++j) {
                    json << "\"" << escape_json(mix[j].first) << "\":"
                         << std::setprecision(4) << jsafe(mix[j].second);
                    if (j + 1 < mix.size()) json << ",";
                }
                json << "}";
                if (i + 1 < entry.combo_strategies.size()) json << ",";
            }
            json << "},";

            // opponent_side + opponent_range
            json << "\"opponent_side\":\"" << escape_json(entry.opponent_side) << "\",";
            json << "\"opponent_range\":{";
            for (size_t i = 0; i < entry.opponent_range.size(); ++i) {
                json << "\"" << escape_json(entry.opponent_range[i].first) << "\":"
                     << std::setprecision(4) << jsafe(entry.opponent_range[i].second);
                if (i + 1 < entry.opponent_range.size()) json << ",";
            }
            json << "}";

            // Per-grid-label EV (chips, from acting player's perspective).
            json << ",\"combo_evs\":{";
            for (size_t i = 0; i < entry.combo_evs.size(); ++i) {
                json << "\"" << escape_json(entry.combo_evs[i].first) << "\":"
                     << std::setprecision(2) << jsafe(entry.combo_evs[i].second);
                if (i + 1 < entry.combo_evs.size()) json << ",";
            }
            json << "}";

            // Path B fields: dealt_cards + runout_options. dealt_cards is the
            // cumulative card string for UI disclosure ("2c,Jd"). runout_options
            // lists every canonical card the immediate prior chance had (so
            // the UI can render a runout picker).
            static const char RANK_CH[] = "23456789TJQKA";
            static const char SUIT_CH[] = "cdhs";
            auto card_str = [&](uint8_t c) -> std::string {
                std::string s; s += RANK_CH[c / 4]; s += SUIT_CH[c % 4]; return s;
            };
            json << ",\"dealt_cards\":[";
            for (size_t i = 0; i < entry.dealt_cards.size(); ++i) {
                json << "\"" << card_str(entry.dealt_cards[i]) << "\"";
                if (i + 1 < entry.dealt_cards.size()) json << ",";
            }
            json << "],";
            json << "\"runout_options\":[";
            for (size_t i = 0; i < entry.runout_options.size(); ++i) {
                json << "{\"card\":\"" << card_str(entry.runout_options[i].card)
                     << "\",\"weight\":" << int(entry.runout_options[i].weight) << "}";
                if (i + 1 < entry.runout_options.size()) json << ",";
            }
            json << "]";

            json << "}";
            if (entry_i + 1 < strategy_tree->size()) json << ",";
            json << "\n";
            ++entry_i;
        }
        json << "  }";
        json << std::setprecision(2);  // restore
    }

    json << "\n}\n";
    return json.str();
}

// ============================================================================
// v1.8.0 P3-9: --benchmark matrix
//
// Sweeps a fixed cross-product of (scenario × backend × simd × threads) and
// emits one consolidated JSON document with throughput/timing metrics per
// row. Goal is regression detection — "v1.8 was 137 iter/s on AsKd7c
// levelized 8T, v1.9 is 142, that's a 4% gain we can ship" — without
// repeatedly re-running the single-solve binary by hand.
//
// Design choices:
//   * Single binary, single process: a Python wrapper that calls
//     deepsolver_core 60 times pays ~50ms × 60 = 3s of process overhead and
//     re-builds the tree every time. Doing it in-process amortizes the
//     setup cost when iter counts are small.
//   * Skip turn / river by default: those scenarios add 30-60s each and
//     mostly characterize the same kernels as flop spots. Add a flag if the
//     caller wants them.
//   * Force CPU backend only: GPU benchmarks are a separate question (need
//     to enumerate the device, handle missing CUDA gracefully).
//   * Fresh Solver per row: Solver doesn't expose a `reset()` and resetting
//     SIMD policy mid-process is supported by cpu_simd::set_policy. Tree
//     build is fast (<10ms on flop, <100ms on monotone iso-engaged).
// ============================================================================

namespace {

struct MatrixScenario {
    const char* name;
    const char* board;
    int iterations;     // small enough to keep the whole matrix under ~3 min
};

struct MatrixConfig {
    const char* backend;   // "reference" or "levelized"
    const char* simd;      // "scalar" or "avx2"
    uint32_t threads;
};

// POST_OPTIMIZATION_REVIEW Sec 4.3 Phase 2: collect terminal-by-matchup_idx
// reuse statistics. The Phase 2 ROI question is: when many terminals share
// the same matchup table, can we batch their showdown calls so the table
// streams from DRAM once per group instead of once per call? This function
// just measures the reuse pattern; it does not change any solver behavior.
struct TerminalReuseStats {
    std::uint64_t total_terminals    = 0;
    std::uint64_t distinct_tables    = 0;
    std::uint64_t max_per_table      = 0;
    double        mean_per_table     = 0.0;
    std::uint64_t p50_per_table      = 0;
    std::uint64_t p95_per_table      = 0;
};

static TerminalReuseStats compute_terminal_reuse(const deepsolver::FlatGameTree& tree) {
    TerminalReuseStats s;
    std::map<std::int32_t, std::uint64_t> counts;
    for (std::uint32_t n = 0; n < tree.total_nodes; ++n) {
        if (static_cast<deepsolver::NodeType>(tree.node_types[n])
                != deepsolver::NodeType::TERMINAL) continue;
        ++s.total_terminals;
        std::int32_t mi = (n < tree.matchup_idx.size()) ? tree.matchup_idx[n] : 0;
        counts[mi]++;
    }
    s.distinct_tables = counts.size();
    if (counts.empty()) return s;
    std::vector<std::uint64_t> per_table;
    per_table.reserve(counts.size());
    for (auto& kv : counts) per_table.push_back(kv.second);
    std::sort(per_table.begin(), per_table.end());
    s.max_per_table = per_table.back();
    s.mean_per_table = static_cast<double>(s.total_terminals)
                     / static_cast<double>(s.distinct_tables);
    auto pct = [&](double p) {
        const std::size_t idx = std::min(per_table.size() - 1,
            static_cast<std::size_t>(p * per_table.size()));
        return per_table[idx];
    };
    s.p50_per_table = pct(0.50);
    s.p95_per_table = pct(0.95);
    return s;
}

void run_benchmark_matrix(std::ostream& out, bool include_scalar) {
    // (deepsolver:: is already brought in by the file-level `using namespace
    // deepsolver` at the top of this TU.)

    // Scenarios chosen to cover the dominant code paths:
    //   rainbow_flop: hits the no-iso fast path (matchup table per runout = 1)
    //   two_tone_flop: 2 chance-suit symmetries collapse → smaller iso
    //   monotone_flop: most aggressive iso — heavy precompute, tiny trees
    static const MatrixScenario kScenarios[] = {
        { "rainbow_flop",  "AsKd7c", 100 },
        { "two_tone_flop", "AsKs7c",  60 },
        { "monotone_flop", "AsKsQs",  20 },  // huge tree, fewer iters
    };

    // Configs cover the (backend × thread) cross-product. SIMD is "avx2" by
    // default; --include-scalar adds scalar variants to validate the dispatch
    // table on the same machine. Reference backend tops out at 2 threads
    // (parallel-sections cap), so we don't bother running it at 4/8 — they
    // produce the same throughput.
    std::vector<MatrixConfig> configs = {
        { "reference", "avx2", 1 },
        { "reference", "avx2", 2 },
        { "levelized", "avx2", 1 },
        { "levelized", "avx2", 2 },
        { "levelized", "avx2", 4 },
        { "levelized", "avx2", 8 },
    };
    if (include_scalar) {
        configs.push_back({ "reference", "scalar", 1 });
        configs.push_back({ "reference", "scalar", 2 });
        configs.push_back({ "levelized", "scalar", 1 });
        configs.push_back({ "levelized", "scalar", 8 });
    }

    out << "{\n";
    out << "  \"benchmark\": \"matrix\",\n";
    out << "  \"hardware_concurrency\": "
        << std::thread::hardware_concurrency() << ",\n";
    out << "  \"results\": [\n";

    bool first_row = true;
    for (const auto& scen : kScenarios) {
        for (const auto& cfg : configs) {
            // Apply SIMD policy. Skip silently if the host can't service
            // ForceAvx2 — the parity test already covers that case, and
            // adding scalar fallback rows here would muddy the JSON.
            if (std::string(cfg.simd) == "avx2") {
                if (!cpu_simd::cpuid_supports_avx2_fma_os()) continue;
                cpu_simd::set_policy(cpu_simd::SimdPolicy::ForceAvx2);
            } else {
                cpu_simd::set_policy(cpu_simd::SimdPolicy::ForceScalar);
            }

            // Build config for this row.
            SolverConfig sc;
            auto board = parse_board(scen.board);
            sc.board_size = static_cast<uint8_t>(board.size());
            for (size_t i = 0; i < board.size() && i < MAX_BOARD_CARDS; ++i) {
                sc.board[i] = board[i];
            }
            sc.pot = 100.0f;
            sc.effective_stack = 500.0f;
            sc.max_iterations = scen.iterations;
            sc.target_exploitability = 0.0f;        // run all iters
            sc.exploitability_check_interval = 100000;  // disable interim checks
            sc.cpu_threads = cfg.threads;
            sc.cpu_backend_kind = (std::string(cfg.backend) == "levelized")
                ? SolverConfig::CpuBackendKind::LEVELIZED
                : SolverConfig::CpuBackendKind::REFERENCE;
            // Skip strategy_tree + postsolve to isolate solve throughput.
            sc.compute_combo_evs = false;
            sc.compute_exploitability = false;
            sc.parallel_postsolve = false;

            Solver solver(sc, BackendType::CPU);
            SolverResult res;
            try {
                res = solver.solve();
            } catch (const std::exception& e) {
                std::cerr << "matrix row failed: " << scen.name << " "
                          << cfg.backend << " " << cfg.simd << " " << cfg.threads
                          << ": " << e.what() << "\n";
                continue;
            }

            const double iter_ms = res.timing.iterations_ms;
            const double iter_per_sec = (iter_ms > 0.0)
                ? (static_cast<double>(res.iterations_run) * 1000.0 / iter_ms)
                : 0.0;
            const double nodes_per_sec = iter_per_sec
                * static_cast<double>(solver.tree_node_count());

            if (!first_row) out << ",\n";
            first_row = false;
            out << std::fixed << std::setprecision(3);
            out << "    {\n";
            out << "      \"scenario\": \""           << scen.name  << "\",\n";
            out << "      \"board\": \""              << scen.board << "\",\n";
            out << "      \"backend\": \""            << cfg.backend << "\",\n";
            out << "      \"simd\": \""               << cfg.simd << "\",\n";
            out << "      \"threads_requested\": "    << cfg.threads << ",\n";
            out << "      \"threads_effective\": "
                << res.resources.cpu_threads_effective << ",\n";
            out << "      \"iterations\": "           << res.iterations_run << ",\n";
            out << "      \"tree_nodes\": "           << solver.tree_node_count() << ",\n";
            out << "      \"matchup_tables\": "       << res.timing.matchup_tables << ",\n";
            // Phase 2 ROI signal: how often does each matchup table get reused
            // across distinct terminal nodes? High mean/p50 means a batching
            // pass (group terminals by matchup_idx) could amortize matrix
            // streaming across many calls.
            {
                const auto rs = compute_terminal_reuse(solver.tree());
                out << "      \"reuse_stats\": {\n";
                out << "        \"total_terminals\": "    << rs.total_terminals << ",\n";
                out << "        \"distinct_tables\": "    << rs.distinct_tables << ",\n";
                out << "        \"max_per_table\": "      << rs.max_per_table << ",\n";
                out << "        \"mean_per_table\": "     << rs.mean_per_table << ",\n";
                out << "        \"p50_per_table\": "      << rs.p50_per_table << ",\n";
                out << "        \"p95_per_table\": "      << rs.p95_per_table << "\n";
                out << "      },\n";
            }
            out << "      \"timing_ms\": {\n";
            out << "        \"tree_build\": "         << res.timing.tree_build_ms << ",\n";
            out << "        \"precompute_matchups\": " << res.timing.precompute_matchups_ms << ",\n";
            out << "        \"backend_prepare\": "    << res.timing.backend_prepare_ms << ",\n";
            out << "        \"iterations\": "         << res.timing.iterations_ms << ",\n";
            out << "        \"phase_compute_strategy\": "
                << res.timing.phase_compute_strategy_ms << ",\n";
            out << "        \"phase_apply_discount\": "
                << res.timing.phase_apply_discount_ms << ",\n";
            out << "        \"phase_forward_pass\": "
                << res.timing.phase_forward_pass_ms << ",\n";
            out << "        \"phase_backward_pass_oop\": "
                << res.timing.phase_backward_pass_oop_ms << ",\n";
            out << "        \"phase_backward_pass_ip\": "
                << res.timing.phase_backward_pass_ip_ms << ",\n";
            out << "        \"phase_backward_showdown\": "
                << res.timing.phase_backward_showdown_ms << ",\n";
            out << "        \"phase_backward_fold\": "
                << res.timing.phase_backward_fold_ms << ",\n";
            out << "        \"total\": "              << res.timing.total_ms << "\n";
            out << "      },\n";
            out << "      \"iterations_per_sec\": "   << iter_per_sec << ",\n";
            out << "      \"nodes_per_sec\": "        << nodes_per_sec << ",\n";
            out << "      \"memory_estimate_mb\": "
                << (res.resources.estimated_cpu_state_bytes / (1024.0 * 1024.0)) << "\n";
            out << "    }";
        }
    }

    out << "\n  ]\n";
    out << "}\n";

    // Reset policy for any subsequent code paths.
    cpu_simd::set_policy(cpu_simd::SimdPolicy::Auto);
}

}  // anonymous namespace (matrix)

// ============================================================================
// v1.8.0 paired benchmark
//
// Runs `baseline` and `candidate` solves in alternating order so thermal
// drift hits both labels equally instead of stacking on whichever gets
// run last. Reports best/median per label and the candidate/baseline
// ratio. Designed for comparing one boolean knob at a time (currently
// `cpu_persistent_omp` 0 vs 1) on the same fixture × thread count.
//
// Why alternation matters on a laptop:
//   The matrix benchmark runs 18 configs sequentially over 2-3 minutes
//   on a thermally-constrained laptop. CPU base clock can drop 15-20%
//   over that window, so the LAST config measured has lower
//   throughput regardless of code quality. A fixed-order baseline-then-
//   candidate run hands the candidate a "hot CPU" handicap that swamps
//   any 5% real difference. Alternating BCBC... interleaves them so
//   any monotonic thermal trend cancels in the ratio.
// ============================================================================

namespace {

struct PairedScenario {
    const char* name;
    const char* board;
    int iterations;
    const char* range;
};

static constexpr const char* kMediumSparseRange =
    "AA:1,KK:1,QQ:1,JJ:1,TT:1,99:1,88:1,77:1,66:1,55:1,44:1,33:1,22:1,"
    "AKs:1,AQs:1,AJs:1,ATs:1,A9s:1,A8s:1,A7s:1,A6s:1,A5s:1,A4s:1,A3s:1,A2s:1,"
    "KQs:1,KJs:1,KTs:1,QJs:1,QTs:1,JTs:1,T9s:1,98s:1,87s:1,"
    "AKo:1,AQo:1,AJo:1,KQo:1";

static const PairedScenario* lookup_paired_case(const std::string& name) {
    static const PairedScenario kStandard = { "standard", "AsKd7c", 100, "" };
    static const PairedScenario kMonotone = { "monotone", "AsKsQs",  20, "" };
    static const PairedScenario kNarrow   = {
        "narrow", "AsKd7c", 100, "AA:1,KK:1,QQ:1,AKs:1,AKo:1"
    };
    static const PairedScenario kMediumSparse = {
        "medium_sparse", "AsKd7c", 100, kMediumSparseRange
    };
    if (name == "standard") return &kStandard;
    if (name == "monotone") return &kMonotone;
    if (name == "narrow")   return &kNarrow;
    if (name == "medium_sparse") return &kMediumSparse;
    return nullptr;
}

// Run one solve with the requested config, return iter/s.
struct PairedRun {
    double iter_per_sec;
    double iterations_ms;
    double total_ms;
    int iterations_run;
};
static PairedRun run_one_paired(const PairedScenario& scen,
                                  uint32_t threads,
                                  bool persistent_omp)
{
    using namespace deepsolver;
    SolverConfig sc;
    auto board = parse_board(scen.board);
    sc.board_size = static_cast<uint8_t>(board.size());
    for (size_t i = 0; i < board.size() && i < MAX_BOARD_CARDS; ++i) {
        sc.board[i] = board[i];
    }
    sc.pot = 100.0f;
    sc.effective_stack = 500.0f;
    sc.max_iterations = scen.iterations;
    sc.target_exploitability = 0.0f;
    sc.exploitability_check_interval = 100000;
    sc.cpu_threads = threads;
    sc.cpu_backend_kind = SolverConfig::CpuBackendKind::LEVELIZED;
    sc.cpu_persistent_omp = persistent_omp;
    sc.compute_combo_evs = false;
    sc.compute_exploitability = false;
    sc.parallel_postsolve = false;
    if (scen.range != nullptr && scen.range[0] != '\0') {
        apply_range_string(scen.range, sc.ip_range_weights);
        apply_range_string(scen.range, sc.oop_range_weights);
        sc.has_custom_ranges = true;
    }

    Solver solver(sc, BackendType::CPU);
    auto result = solver.solve();
    PairedRun out{};
    out.iterations_run = result.iterations_run;
    out.iterations_ms = result.timing.iterations_ms;
    out.total_ms      = result.timing.total_ms;
    out.iter_per_sec  = (out.iterations_ms > 0.0)
        ? static_cast<double>(out.iterations_run) * 1000.0 / out.iterations_ms
        : 0.0;
    return out;
}

// Returns false on invalid input so main() can surface a non-zero exit code
// (POST_OPTIMIZATION_REVIEW Finding 4: invalid --benchmark-case used to print
// an error and return success, hiding bad invocations from automation).
bool run_benchmark_paired(std::ostream& out,
                           const std::string& case_name,
                           int runs_per_label,
                           int threads_override)
{
    using namespace deepsolver;
    const PairedScenario* scen = lookup_paired_case(case_name);
    if (!scen) {
        std::cerr << "{\"status\":\"error\",\"message\":\"unknown --benchmark-case: "
                  << case_name
                  << " (valid: standard | monotone | narrow | medium_sparse)\"}\n";
        return false;
    }
    const uint32_t threads = (threads_override >= 0)
        ? static_cast<uint32_t>(threads_override)
        : 0u;   // 0 = backend auto

    cpu_simd::set_policy(cpu_simd::SimdPolicy::Auto);

    // Warmup — paid once per label before measurement so neither run's first
    // measured iteration eats the OS-page-fault / first-touch tax. Both
    // baseline AND candidate get warmed so the candidate's first measured run
    // doesn't carry persistent-OMP team-spawn cost (POST_OPTIMIZATION_REVIEW
    // Finding 3: previously only baseline was warmed).
    (void)run_one_paired(*scen, threads, /*persistent_omp=*/false);
    (void)run_one_paired(*scen, threads, /*persistent_omp=*/true);

    // Alternating BCBC... order. With runs_per_label=3 we do 6 total runs:
    // B C B C B C. Stops thermal drift from hitting only one label.
    std::vector<PairedRun> baseline_runs, candidate_runs;
    baseline_runs.reserve(runs_per_label);
    candidate_runs.reserve(runs_per_label);
    for (int k = 0; k < runs_per_label; ++k) {
        baseline_runs.push_back(run_one_paired(*scen, threads, false));
        candidate_runs.push_back(run_one_paired(*scen, threads, true));
    }

    auto best = [](const std::vector<PairedRun>& v) {
        double m = 0.0;
        for (auto& r : v) if (r.iter_per_sec > m) m = r.iter_per_sec;
        return m;
    };
    auto worst = [](const std::vector<PairedRun>& v) {
        double m = 1e18;
        for (auto& r : v) if (r.iter_per_sec < m) m = r.iter_per_sec;
        return m;
    };
    auto median = [](std::vector<PairedRun> v) {
        if (v.empty()) return 0.0;
        std::sort(v.begin(), v.end(),
                  [](const PairedRun& a, const PairedRun& b){
                      return a.iter_per_sec < b.iter_per_sec;
                  });
        const std::size_t mid = v.size() / 2;
        return (v.size() % 2 == 1)
            ? v[mid].iter_per_sec
            : (v[mid - 1].iter_per_sec + v[mid].iter_per_sec) * 0.5;
    };

    const double base_best = best(baseline_runs);
    const double cand_best = best(candidate_runs);
    const double base_med  = median(baseline_runs);
    const double cand_med  = median(candidate_runs);
    const double ratio_best = (base_best > 0.0) ? (cand_best / base_best) : 0.0;
    const double ratio_med  = (base_med  > 0.0) ? (cand_med  / base_med ) : 0.0;

    out << std::fixed << std::setprecision(4);
    out << "{\n";
    out << "  \"benchmark\": \"paired\",\n";
    out << "  \"case\": \""               << scen->name  << "\",\n";
    out << "  \"board\": \""              << scen->board << "\",\n";
    out << "  \"range\": \""              << scen->range << "\",\n";
    out << "  \"iterations\": "           << scen->iterations << ",\n";
    out << "  \"runs_per_label\": "       << runs_per_label << ",\n";
    out << "  \"order\": \"alternating BCBC\",\n";
    out << "  \"threads_requested\": "    << threads << ",\n";
    out << "  \"hardware_concurrency\": " << std::thread::hardware_concurrency() << ",\n";

    auto emit_label = [&](const char* label, const std::vector<PairedRun>& v,
                           bool persistent) {
        out << "  \"" << label << "\": {\n";
        out << "    \"cpu_persistent_omp\": " << (persistent ? "true" : "false") << ",\n";
        out << "    \"runs_iter_per_sec\": [";
        for (std::size_t i = 0; i < v.size(); ++i) {
            if (i) out << ", ";
            out << v[i].iter_per_sec;
        }
        out << "],\n";
        out << "    \"best\": "   << best(v)   << ",\n";
        out << "    \"median\": " << median(v) << ",\n";
        out << "    \"min\": "    << worst(v)  << "\n";
        out << "  },\n";
    };
    emit_label("baseline",  baseline_runs,  false);
    emit_label("candidate", candidate_runs, true);

    out << "  \"ratio_best\": "   << ratio_best << ",\n";
    out << "  \"ratio_median\": " << ratio_med  << "\n";
    out << "}\n";

    cpu_simd::set_policy(cpu_simd::SimdPolicy::Auto);
    return true;
}

}  // anonymous namespace

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    auto args = parse_args(argc, argv);

    if (args.help) {
        print_help();
        return 0;
    }

    // v1.8.0 P3-9: --benchmark matrix runs an independent multi-row sweep and
    // exits before the single-solve path takes over. Handled before "standard"
    // because matrix mode emits its own JSON and doesn't share state with the
    // post-solve writer.
    if (args.benchmark == "matrix") {
        run_benchmark_matrix(std::cout, args.include_scalar != 0);
        return 0;
    }

    // v1.8.0: --benchmark paired runs alternating baseline-vs-candidate
    // iterations on a single fixture × thread count, reporting best / median
    // / ratio. Use this for A/B comparing one knob (currently
    // cpu_persistent_omp) without thermal drift contaminating the result.
    if (args.benchmark == "paired") {
        const bool ok = run_benchmark_paired(std::cout,
                                              args.benchmark_case,
                                              args.benchmark_runs,
                                              args.benchmark_threads_override);
        return ok ? 0 : 1;
    }

    // Polish #2: --benchmark <preset> overrides scenario inputs with a known
    // fixed setup so perf regression tracking is reproducible across runs.
    // Output is replaced with a benchmark-only JSON (see end of main).
    bool benchmark_mode = false;
    if (!args.benchmark.empty()) {
        benchmark_mode = true;
        if (args.benchmark == "standard") {
            // AsKd7c rainbow flop, both players defaulting to full 1326 range,
            // standard bet sizing, 100 DCFR iterations. Picked because:
            //  - rainbow flop hits the default (no-iso-fallback) code path;
            //  - 100 iter is enough to measure throughput without dominating
            //    setup costs (tree build / matchups).
            args.board_str = "AsKd7c";
            args.pot       = 100.0f;
            args.stack     = 500.0f;
            args.iterations = 100;
            args.ip_range_str.clear();   // empty = default full range
            args.oop_range_str.clear();
            // Keep postsolve full so combo_evs + exploitability are measured too.
        } else if (args.benchmark == "narrow") {
            args.board_str = "AsKd7c";
            args.pot       = 100.0f;
            args.stack     = 500.0f;
            args.iterations = 100;
            args.ip_range_str  = "AA:1,KK:1,QQ:1,AKs:1,AKo:1";
            args.oop_range_str = "AA:1,KK:1,QQ:1,AKs:1,AKo:1";
        } else if (args.benchmark == "medium_sparse") {
            args.board_str = "AsKd7c";
            args.pot       = 100.0f;
            args.stack     = 500.0f;
            args.iterations = 100;
            args.ip_range_str  = kMediumSparseRange;
            args.oop_range_str = kMediumSparseRange;
        } else if (args.benchmark == "monotone") {
            // AsKsQs exercises canonical-compressed matchup tables and
            // chance-runout grouping; keep it shorter than standard.
            args.board_str = "AsKsQs";
            args.pot       = 100.0f;
            args.stack     = 500.0f;
            args.iterations = 20;
            args.ip_range_str.clear();
            args.oop_range_str.clear();
        } else {
            std::cerr << "{\"status\":\"error\",\"message\":\"unknown --benchmark preset: "
                      << args.benchmark
                      << " (valid: standard | monotone | narrow | medium_sparse | matrix)\"}\n";
            return 1;
        }
    }

    // Handle --gpu-info: print detected GPU info and exit (for Tauri diagnostics)
    if (args.gpu_info) {
        bool has = has_cuda_gpu();
        std::string desc = describe_cuda_gpu();
        std::cout << "{\n"
                  << "  \"has_cuda_gpu\": " << (has ? "true" : "false") << ",\n"
                  << "  \"gpu_description\": \"" << escape_json(desc) << "\",\n"
                  << "  \"gpu_backend_functional\": "
                  << (GPU_BACKEND_FUNCTIONAL ? "true" : "false") << "\n"
                  << "}\n";
        return 0;
    }

    // Validate required arguments
    if (args.pot <= 0 || args.stack <= 0 || args.board_str.empty()) {
        std::cerr << "{\"status\": \"error\", \"message\": \"Missing required arguments: --pot, --stack, --board\"}\n";
        return 1;
    }

    try {
        // Parse board
        auto board_cards = parse_board(args.board_str);

        // Build solver config
        SolverConfig config;
        config.pot = args.pot;
        config.effective_stack = args.stack;
        config.board_size = static_cast<uint8_t>(board_cards.size());
        for (size_t i = 0; i < board_cards.size() && i < MAX_BOARD_CARDS; ++i) {
            config.board[i] = board_cards[i];
        }
        config.max_iterations = args.iterations;
        config.target_exploitability = args.exploitability / 100.0f;
        config.time_budget_seconds = args.time_budget_seconds;
        config.bet_sizing.flop_sizes = args.flop_sizes;
        config.bet_sizing.turn_sizes = args.turn_sizes;
        config.bet_sizing.river_sizes = args.river_sizes;
        config.oop_has_initiative = (args.oop_has_initiative != 0);
        config.allow_donk_bet = (args.allow_donk_bet != 0);
        config.parallel_postsolve = args.parallel_postsolve;
        config.postsolve_threads = args.postsolve_threads;
        config.force_cpu_postsolve = args.force_cpu_postsolve;
        config.cpu_threads = args.cpu_threads;

        // v1.5.0 Phase 4: --cpu-backend reference|levelized
        {
            std::string kind = args.cpu_backend_kind;
            for (char& ch : kind)
                ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
            if (kind == "levelized") {
                config.cpu_backend_kind = SolverConfig::CpuBackendKind::LEVELIZED;
            } else {
                config.cpu_backend_kind = SolverConfig::CpuBackendKind::REFERENCE;
            }
        }

        // v1.8.0 Sprint 3 gated rollout: forward the persistent-OMP request
        // to the backend. Field stays default-false in SolverConfig — only
        // explicit --cpu-persistent-omp 1 enables it.
        // Production default is on; --cpu-persistent-omp 0 keeps an A/B escape hatch.
        config.cpu_persistent_omp = (args.cpu_persistent_omp != 0);
        config.cpu_showdown_batch = (args.cpu_showdown_batch != 0);

        // v1.4.0 Phase 2: apply --cpu-simd policy. set_policy() is idempotent
        // and re-resolves the kernel table on next call to kernels(). Done
        // before backend creation so the CpuBackend's first prepare() picks
        // up the right kernels.
        {
            std::string mode = args.cpu_simd;
            for (char& ch : mode)
                ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
            if (mode == "scalar") {
                cpu_simd::set_policy(cpu_simd::SimdPolicy::ForceScalar);
            } else if (mode == "avx2") {
                cpu_simd::set_policy(cpu_simd::SimdPolicy::ForceAvx2);
            } else {
                cpu_simd::set_policy(cpu_simd::SimdPolicy::Auto);
            }
        }
        {
            std::string sched = args.dcfr_schedule;
            for (char& ch : sched) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
            if (sched == "postflop" || sched == "postflop_style" || sched == "wasm") {
                config.dcfr_schedule = SolverConfig::DcfrSchedule::POSTFLOP_STYLE;
            } else {
                config.dcfr_schedule = SolverConfig::DcfrSchedule::STANDARD;
            }
        }
        config.rake_rate = args.rake_rate;
        config.rake_cap  = args.rake_cap;

        // Sprint 1: per-run memory budget overrides. Only apply non-zero values
        // so callers that don't care fall back to MemoryBudget defaults.
        if (args.host_memory_mb > 0) {
            config.memory_budget.host_bytes = args.host_memory_mb * 1024ULL * 1024ULL;
        }
        if (args.gpu_memory_mb > 0) {
            config.memory_budget.gpu_bytes = args.gpu_memory_mb * 1024ULL * 1024ULL;
        }
        if (args.json_memory_mb > 0) {
            config.memory_budget.json_bytes = args.json_memory_mb * 1024ULL * 1024ULL;
        }
        if (args.strategy_tree_max_nodes > 0) {
            config.memory_budget.strategy_tree_max_nodes = args.strategy_tree_max_nodes;
            config.strategy_tree_max_nodes = args.strategy_tree_max_nodes;
        }

        std::string postsolve_mode = args.postsolve;
        for (char& ch : postsolve_mode) {
            if (ch >= 'A' && ch <= 'Z') ch = static_cast<char>(ch - 'A' + 'a');
        }
        if (postsolve_mode == "full") {
            config.compute_combo_evs = true;
            config.compute_exploitability = true;
        } else if (postsolve_mode == "ev" || postsolve_mode == "combo-evs") {
            config.compute_combo_evs = true;
            config.compute_exploitability = false;
        } else if (postsolve_mode == "exploitability") {
            config.compute_combo_evs = false;
            config.compute_exploitability = true;
        } else if (postsolve_mode == "none" || postsolve_mode == "off") {
            config.compute_combo_evs = false;
            config.compute_exploitability = false;
        } else {
            throw std::invalid_argument("Invalid --postsolve value: " + args.postsolve);
        }

        // Target combo analysis reads Solver::ev_, so keep the EV pass enabled
        // even when a batch caller requested a lighter postsolve mode.
        if (!args.target_combo.empty()) {
            config.compute_combo_evs = true;
        }

        // Apply custom ranges
        if (!args.ip_range_str.empty()) {
            apply_range_string(args.ip_range_str, config.ip_range_weights);
            config.has_custom_ranges = true;
        }
        if (!args.oop_range_str.empty()) {
            apply_range_string(args.oop_range_str, config.oop_range_weights);
            config.has_custom_ranges = true;
        }

        // Parse node locks
        if (!args.node_locks_str.empty()) {
            config.node_locks = parse_node_locks(args.node_locks_str);
        }

        // Progress callback: output to stderr
        auto progress = [](int iter, float exploit, float elapsed_ms) {
            std::cerr << "[Iter " << iter << "] Exploitability: "
                      << std::fixed << std::setprecision(2) << exploit << "%"
                      << " (" << elapsed_ms << "ms)\n";
        };

        // Initialize hand evaluator
        auto& eval = get_evaluator();
        eval.initialize();

        // Parse backend selection
        BackendType backend_type = parse_backend_type(args.backend);

        Solver solver(config, backend_type);

        // v1.2.2: --estimate-only — build tree + estimate, emit JSON, exit.
        // Used by frontend to show ETA banner before committing to the
        // real solve. Sub-second on most spots, ~100ms on monotone iso.
        if (args.estimate_only) {
            SolveResources r = solver.estimate_only();
            std::cout << "{\n"
                      << "  \"status\": \"estimate\",\n"
                      << "  \"resources\": {\n"
                      << "    \"canonical_combos\": " << r.canonical_combos << ",\n"
                      << "    \"player_nodes\": " << r.player_nodes << ",\n"
                      << "    \"estimated_matchup_bytes\": " << r.estimated_matchup_bytes << ",\n"
                      << "    \"estimated_cpu_state_bytes\": " << r.estimated_cpu_state_bytes << ",\n"
                      << "    \"estimated_gpu_state_bytes\": " << r.estimated_gpu_state_bytes << ",\n"
                      << "    \"estimated_strategy_tree_bytes\": " << r.estimated_strategy_tree_bytes << ",\n"
                      << "    \"estimated_json_bytes\": " << r.estimated_json_bytes << ",\n"
                      << "    \"host_budget_bytes\": " << r.host_budget_bytes << ",\n"
                      << "    \"gpu_budget_bytes\": " << r.gpu_budget_bytes << ",\n"
                      << "    \"strategy_tree_max_nodes\": " << r.strategy_tree_max_nodes << ",\n"
                      << "    \"budget_decision\": \"" << escape_json(r.budget_decision) << "\",\n"
                      << "    \"diagnostic\": \"" << escape_json(r.diagnostic) << "\",\n"
                      << "    \"fallback_reason\": \"" << escape_json(r.fallback_reason) << "\",\n"
                      << "    \"runout_approximated\": " << json_bool(r.runout_approximated) << ",\n"
                      << "    \"ops_per_iteration\": " << r.ops_per_iteration << ",\n"
                      << "    \"backend_for_estimate\": \"" << escape_json(r.backend_for_estimate) << "\",\n"
                      << "    \"estimated_solve_seconds\": " << r.estimated_solve_seconds << ",\n"
                      << "    \"cpu_simd\": \"" << escape_json(r.cpu_simd) << "\",\n"
                      << "    \"cpu_threads_effective\": " << r.cpu_threads_effective << ",\n"
                      << "    \"cpu_backend_kind\": \"" << escape_json(r.cpu_backend_kind) << "\"\n"
                      << "  }";

            // Roadmap ④: Exact-mode feasibility pre-flight. When the caller
            // plans to decompose, price the decomposed run (trunk build +
            // per-line subgame sampling — sub-second) and predict whether
            // "auto" would actually engage, so the UI can show
            // "Exact ≈ N min, expected accuracy ~X" BEFORE the user commits.
            if (args.decompose_runouts == "auto" || args.decompose_runouts == "on") {
                // Would "auto" engage? Builder-level collapse is the common
                // Exact trigger (rainbow/huge boards) and is exact at
                // estimate time. On top, mirror the ① state gate with the
                // quantities the estimate has — since the compact-formula
                // recalibration both terms are near-exact: gpu_state uses
                // the same B1a compact formula as the gate, and the matchup
                // count is the same dedup rule precompute_matchups applies
                // (the old chance-child proxy over-counted ~30× on iso
                // boards). Residual bias stays conservative: host
                // bytes-per-cell (EV+valid+category ≥9 B) over the GPU's
                // uploaded 8 B/cell. A false "will decompose" shows a scary
                // ETA that turns out fast; a false "won't" would mean an
                // unwarned 90-minute wait.
                bool engage_via_collapse = r.runout_approximated;
                if (!engage_via_collapse) {
                    const bool gpu_planned =
                        r.backend_for_estimate.rfind("CUDA", 0) == 0;
                    if (gpu_planned) {
                        uint64_t budget = r.gpu_budget_bytes > 0
                            ? r.gpu_budget_bytes
                            : static_cast<uint64_t>(
                                  static_cast<double>(gpu_free_bytes()) * 0.80);
                        engage_via_collapse = budget > 0 &&
                            (r.estimated_gpu_state_bytes +
                             r.estimated_matchup_bytes) > budget;
                    } else {
                        engage_via_collapse = r.host_budget_bytes > 0 &&
                            (r.estimated_matchup_bytes +
                             r.estimated_cpu_state_bytes) > r.host_budget_bytes;
                    }
                }
                const bool would_engage =
                    (args.decompose_runouts == "on") || engage_via_collapse;

                deepsolver::DecompositionOptions dopts =
                    resolve_decomposition_options(args);
                // Mirror the solve route's backend pick: collapse-driven
                // decomposition streams subgames on the GPU; forced-on-
                // enumerable runs parallel CPU subgames. (The estimator
                // itself falls back to CPU pricing when no CUDA device.)
                dopts.subgame_backend = engage_via_collapse
                    ? BackendType::GPU : BackendType::CPU;
                // Mirror the solve route's nav flag — extraction is a real
                // per-leaf cost the ETA must include for app solves.
                dopts.build_nav = args.emit_strategy_tree;
                deepsolver::DecompositionEstimate de =
                    deepsolver::estimate_decomposition(config, dopts);

                std::cout << ",\n  \"decompose\": {\n"
                          << "    \"ok\": " << json_bool(de.ok) << ",\n"
                          << "    \"would_engage\": " << json_bool(would_engage) << ",\n"
                          << "    \"leaves\": " << de.leaf_count << ",\n"
                          << "    \"lines\": " << de.line_count << ",\n"
                          << "    \"trunk_nodes\": " << de.trunk_nodes << ",\n"
                          << "    \"sweeps\": " << de.sweeps << ",\n"
                          << "    \"outer\": " << dopts.outer_iterations << ",\n"
                          << "    \"inner\": " << dopts.inner_iterations << ",\n"
                          << "    \"trunk_iters_per_sweep\": " << dopts.trunk_iterations_per_sweep << ",\n"
                          << "    \"warm_start\": " << json_bool(dopts.warm_start_subgames) << ",\n"
                          << "    \"per_sweep_seconds\": " << jsafe(de.per_sweep_seconds) << ",\n"
                          << "    \"pinned_leaves_predicted\": " << de.pinned_leaves_predicted << ",\n"
                          << "    \"total_seconds\": " << jsafe(de.total_seconds) << ",\n"
                          << "    \"spr\": " << jsafe(de.spr) << ",\n"
                          << "    \"quality_tier\": \"" << escape_json(de.quality_tier) << "\",\n"
                          << "    \"expected_exploit_lo_pct\": " << jsafe(de.expected_exploit_lo_pct) << ",\n"
                          << "    \"expected_exploit_hi_pct\": " << jsafe(de.expected_exploit_hi_pct) << ",\n"
                          << "    \"backend\": \"" << escape_json(de.backend_label) << "\"\n"
                          << "  }";
            }
            std::cout << "\n}\n";
            return 0;
        }

        // In --benchmark mode, suppress the per-iter stderr progress callback —
        // it sits inside the timed iteration loop and otherwise contaminates the
        // throughput measurement with logging overhead (POST_OPTIMIZATION_REVIEW
        // Finding 1: standard benchmark collapsed to ~60 iter/s vs ~110+ silent).
        SolverResult result;
        bool monolithic_ok = true;
        std::string monolithic_err;
        try {
            result = solver.solve(benchmark_mode ? ProgressCallback{} : progress);
        } catch (const std::exception& e) {
            // Roadmap ① (post-v1.9.0): a backend OOM on an enumerated tree
            // used to kill the whole run before --decompose-runouts could
            // engage (the routing below reads result.runout_approximated
            // AFTER the solve). The decomposed route manages its own bounded
            // memory, so let it recover spots the monolithic path cannot
            // allocate. Decompose off — or the decomposed solve also
            // failing — rethrows the original error unchanged.
            if (args.decompose_runouts != "auto" && args.decompose_runouts != "on") {
                throw;
            }
            monolithic_ok = false;
            monolithic_err = e.what();
            std::cerr << "[decompose] monolithic solve failed ("
                      << monolithic_err << "); attempting decomposed fallback\n";
        }
        std::string backend_name = monolithic_ok ? solver.backend_name()
                                                 : std::string("none");

        uint32_t target_node = 0;
        if (monolithic_ok) {
            // Process history navigation
            if (!args.history.empty()) {
                target_node = solver.navigate_to_node(args.history);
                result.action_labels = solver.get_action_labels_at(target_node);
                result.global_strategy = solver.extract_global_strategy_at(target_node);
                result.combo_strategies = solver.extract_combo_strategies_at(target_node);
            }

            // Acting player + opponent range (for UI view switcher)
            result.acting_player = solver.acting_player_at(target_node);
            auto opp = solver.extract_opponent_range_at(target_node);
            result.opponent_side = opp.opponent;
            result.opponent_range = std::move(opp.labels);

            // Target combo analysis
            if (!args.target_combo.empty()) {
                result.target_analysis = solver.analyze_combo(args.target_combo);
                result.has_target = true;
            }
        }

        // ── Stage 5: runout decomposition route ──────────────────────────────
        // When the monolithic build collapsed the turn (rainbow/huge → the amber
        // "runout_approximated" path) OR the caller forced it, re-solve via the
        // flop trunk + per-turn-card subgames so turn/river get REAL equity under
        // bounded memory, and stitch the per-node strategies into the SAME nav
        // cache the UI already consumes. The collapsed solve above is cheap (it
        // collapsed to a tiny tree), so reading result.runout_approximated to
        // decide costs almost nothing. Default-off ⇒ legacy path byte-identical.
        deepsolver::DecomposedResult decomp;
        bool decomposed = false;
        // A failed monolithic solve counts as "approximated": the board could
        // not enumerate within budget, which is exactly the case decomposition
        // exists for (and the only way we reach here with !monolithic_ok).
        const bool treat_as_approximated =
            result.runout_approximated || !monolithic_ok;
        const bool want_decompose =
            (args.decompose_runouts == "on") ||
            (args.decompose_runouts == "auto" && treat_as_approximated);
        if (want_decompose) {
            // Roadmap ④: iteration budget resolved by the shared helper
            // (dev defaults → --decompose-* CLI presets → DEEPSOLVER_DECOMP_*
            // env). Same resolver the --estimate-only pre-flight prices.
            deepsolver::DecompositionOptions o = resolve_decomposition_options(args);
            o.build_nav            = args.emit_strategy_tree;
            o.nav_max_player_depth = 8;
            o.nav_max_nodes        = config.memory_budget.strategy_tree_max_nodes;  // 0 = unlimited
            if (treat_as_approximated) {
                // Rainbow/huge: adaptive GPU streaming. cap = -1 ⇒ pin as many
                // turn subgames resident as the user's FREE VRAM allows (measured
                // at runtime so it never OOMs — not every GPU has 32 GB), skipping
                // the matchup re-upload for pinned leaves; stream the rest. (No
                // CUDA → subgames fall back to CPU, still correct.)
                o.subgame_backend  = BackendType::GPU;
                o.gpu_resident_cap = -1;
            } else {
                // Forced on an enumerable board: parallel CPU subgames (better
                // quality, already fast — no GPU benefit there).
                o.subgame_backend = BackendType::CPU;
                o.cache_subgames  = true;
            }
            // Exception-guarded: a mid-decompose throw (typical: another app
            // claims VRAM mid-run and a streamed leaf's cudaMalloc fails) must
            // degrade to the monolithic result we already hold — not kill the
            // run. Leaving `decomposed=false` reuses the decomp.ok=false
            // handling below, including the !monolithic_ok rethrow.
            try {
                decomp = deepsolver::solve_decomposed(config, o);
            } catch (const std::exception& e) {
                decomp.ok = false;
                std::cerr << "[decompose] solve_decomposed threw ("
                          << e.what() << "); keeping the monolithic result.\n";
            }
            if (decomp.ok) {
                // Roadmap ③ instrumentation: which leaves actually warm-started
                // (streamed leaves past the VRAM pin budget stay cold), so a
                // partial warm-start never reads as a full one.
                std::cerr << "[decompose] outer=" << decomp.outer_iterations_run
                          << " inner=" << o.inner_iterations
                          << " trunk_iters=" << decomp.trunk_iterations_run
                          << " leaves=" << decomp.leaf_count
                          << " pinned=" << decomp.gpu_cap_used
                          << " warm_leaves=" << decomp.warm_start_leaves
                          << " warm_start=" << (o.warm_start_subgames ? "on" : "off")
                          << " exploit=" << decomp.exploitability_pct << "%\n";
                decomposed = true;
                result.runout_approximated     = false;  // the whole point.
                result.exploitability_pct      = decomp.exploitability_pct;
                result.exploitability_computed = true;
                // Roadmap ④ (desktop e2e finding): the UI badge read
                // "81.91% in 2 iter" — outer sweep count undersells the run.
                // trunk_iterations_run (outer × K) is the number comparable
                // to a monolithic solve's iteration count (see
                // DecomposedResult docs), so report that.
                result.iterations_run          = decomp.trunk_iterations_run;
                backend_name = (o.subgame_backend == BackendType::GPU)
                                   ? "decomposed-gpu" : "decomposed-cpu";
                // Align the top-level "current node" view with the stitched nav
                // cache (root unless a history path was requested).
                auto it = decomp.strategy_tree.find(args.history);
                if (it != decomp.strategy_tree.end()) {
                    const auto& e = it->second;
                    if (!e.acting.empty())           result.acting_player    = e.acting;
                    if (!e.action_labels.empty())    result.action_labels    = e.action_labels;
                    if (!e.global_strategy.empty())  result.global_strategy  = e.global_strategy;
                    if (!e.combo_strategies.empty()) result.combo_strategies = e.combo_strategies;
                    result.opponent_side  = e.opponent_side;
                    result.opponent_range = e.opponent_range;
                }
            } else {
                std::cerr << "[decompose] solve_decomposed failed (ok=false); "
                             "keeping the monolithic result.\n";
            }
        }

        // No monolithic result to fall back on: the solve threw and the
        // decomposed route didn't rescue it. Surface the original error.
        if (!monolithic_ok && !decomposed) {
            throw std::runtime_error(monolithic_err);
        }
        if (!monolithic_ok) {
            result.resources.fallback_reason =
                "Monolithic solve failed (" + monolithic_err +
                "); recovered via runout decomposition.";
        }

        // Build strategy tree for client-side navigation cache (Route A).
        // Benchmarks can disable this to avoid measuring large JSON output.
        std::map<std::string, deepsolver::Solver::StrategyTreeEntry> strategy_tree;
        const std::map<std::string, deepsolver::Solver::StrategyTreeEntry>* strategy_tree_ptr = nullptr;
        if (decomposed && args.emit_strategy_tree) {
            // Stitched trunk + per-turn-card subgames (built in the route above).
            strategy_tree = std::move(decomp.strategy_tree);
            strategy_tree_ptr = &strategy_tree;
            result.resources.strategy_tree_emitted_nodes =
                static_cast<uint32_t>(strategy_tree.size());
            result.resources.strategy_tree_truncated = decomp.strategy_tree_truncated;
        } else if (args.emit_strategy_tree) {
            // Phase 3: pick the explicit mode if --strategy-tree-evs was set,
            // otherwise honor --no-strategy-tree-evs (NONE) or default to
            // VISIBLE (only emitted nodes cache an EV vector).
            using Mode = deepsolver::Solver::StrategyTreeEvMode;
            Mode ev_mode = Mode::VISIBLE;
            if (args.strategy_tree_ev_mode == "none" || !args.emit_strategy_tree_evs) {
                ev_mode = Mode::NONE;
            } else if (args.strategy_tree_ev_mode == "full") {
                ev_mode = Mode::FULL;
            } else {
                ev_mode = Mode::VISIBLE;
            }
            // Sprint 1: surface truncation back into result.resources so the
            // UI can show a "tree truncated" badge instead of silently
            // dropping branches.
            //
            // Polish #4: read max_nodes from result.resources (NOT args)
            // because the JSON-cap auto-action inside solver.solve() may
            // have lowered it. Using args here would build the full tree
            // and silently exceed the JSON budget the user set.
            bool tree_truncated = false;
            const uint32_t effective_max_nodes =
                result.resources.strategy_tree_max_nodes > 0
                    ? result.resources.strategy_tree_max_nodes
                    : args.strategy_tree_max_nodes;
            strategy_tree = solver.build_strategy_tree(
                /*max_player_depth=*/8, ev_mode,
                /*max_nodes=*/effective_max_nodes, &tree_truncated);
            strategy_tree_ptr = &strategy_tree;
            result.resources.strategy_tree_emitted_nodes =
                static_cast<uint32_t>(strategy_tree.size());
            result.resources.strategy_tree_truncated = tree_truncated;
        }

        // Polish #2: in --benchmark mode, replace the standard JSON output
        // with a compact benchmark report so callers can grep perf metrics
        // without parsing the full 24 MB strategy tree blob.
        if (benchmark_mode) {
            const double iter_ms  = static_cast<double>(result.timing.iterations_ms);
            const uint64_t tree_nodes = result.timing.tree_nodes;
            const double iters_per_sec = iter_ms > 0
                ? (static_cast<double>(result.iterations_run) / (iter_ms / 1000.0))
                : 0.0;
            const double nodes_per_sec = iter_ms > 0
                ? (static_cast<double>(tree_nodes) * static_cast<double>(result.iterations_run) / (iter_ms / 1000.0))
                : 0.0;
            // Sum the estimated memory components for a single "what does this
            // solve cost" number. Matchup tables dominate on iso-engaged trees;
            // CPU state dominates on rainbow flops.
            const uint64_t total_est_bytes =
                result.resources.estimated_matchup_bytes +
                result.resources.estimated_cpu_state_bytes +
                result.resources.estimated_strategy_tree_bytes +
                result.resources.estimated_json_bytes;
            const double mem_est_mb = static_cast<double>(total_est_bytes) / (1024.0 * 1024.0);

            std::cout << "{\n"
                      << "  \"benchmark\": \"" << escape_json(args.benchmark) << "\",\n"
                      << "  \"board\": \"" << escape_json(args.board_str) << "\",\n"
                      << "  \"backend\": \"" << escape_json(backend_name) << "\",\n"
                      << "  \"iterations_run\": " << result.iterations_run << ",\n"
                      << "  \"exploitability_pct\": " << result.exploitability_pct << ",\n"
                      << "  \"iterations_per_sec\": " << iters_per_sec << ",\n"
                      << "  \"nodes_per_sec\": " << nodes_per_sec << ",\n"
                      << "  \"memory_estimate_mb\": " << mem_est_mb << ",\n"
                      << "  \"tree_nodes\": " << tree_nodes << ",\n"
                      << "  \"cpu\": {\n"
                      << "    \"simd\": \"" << escape_json(result.resources.cpu_simd) << "\",\n"
                      << "    \"threads\": " << result.resources.cpu_threads_effective << ",\n"
                      << "    \"backend_kind\": \"" << escape_json(result.resources.cpu_backend_kind) << "\"\n"
                      << "  },\n";
            std::cout << "  \"cpu_diagnostics\": ";
            write_cpu_diagnostics_json(
                std::cout, result.cpu_diagnostics, "  ");
            std::cout << ",\n"
                      << "  \"timing\": {\n"
                      << "    \"total_ms\": " << result.timing.total_ms << ",\n"
                      << "    \"tree_build_ms\": " << result.timing.tree_build_ms << ",\n"
                      << "    \"backend_prepare_ms\": " << result.timing.backend_prepare_ms << ",\n"
                      << "    \"iterations_ms\": " << result.timing.iterations_ms << ",\n"
                      << "    \"phase_compute_strategy_ms\": "
                      << result.timing.phase_compute_strategy_ms << ",\n"
                      << "    \"phase_apply_discount_ms\": "
                      << result.timing.phase_apply_discount_ms << ",\n"
                      << "    \"phase_forward_pass_ms\": "
                      << result.timing.phase_forward_pass_ms << ",\n"
                      << "    \"phase_backward_pass_oop_ms\": "
                      << result.timing.phase_backward_pass_oop_ms << ",\n"
                      << "    \"phase_backward_pass_ip_ms\": "
                      << result.timing.phase_backward_pass_ip_ms << ",\n"
                      << "    \"phase_backward_showdown_ms\": "
                      << result.timing.phase_backward_showdown_ms << ",\n"
                      << "    \"phase_backward_fold_ms\": "
                      << result.timing.phase_backward_fold_ms << ",\n"
                      << "    \"finalize_ms\": " << result.timing.finalize_ms << ",\n"
                      << "    \"postsolve_ms\": " << result.timing.postsolve_ms << "\n"
                      << "  }\n"
                      << "}\n";
        } else {
            // Output JSON to stdout (with the strategy tree appended).
            std::cout << result_to_json(result, args, backend_name, strategy_tree_ptr);
        }

    } catch (const std::exception& e) {
        std::cerr << "{\"status\": \"error\", \"message\": \""
                  << escape_json(e.what()) << "\"}\n";
        return 1;
    }

    return 0;
}

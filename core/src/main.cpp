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

#include <iostream>
#include <string>
#include <sstream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <cstring>
#include <map>

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
    bool help = false;
    bool gpu_info = false;

    // Backend selection: "auto" | "cpu" | "gpu"
    std::string backend = "auto";

    // Bet sizing
    std::vector<float> flop_sizes = {0.33f, 0.75f};
    std::vector<float> turn_sizes = {0.75f};
    std::vector<float> river_sizes = {0.75f};

    // Custom ranges (TexasSolver format)
    std::string ip_range_str;
    std::string oop_range_str;

    // Node locks (JSON string)
    std::string node_locks_str;

    // Tree construction flags (expose SolverConfig defaults)
    int  oop_has_initiative = 1;     // 0/1 (default 1: OOP can bet at root)
    int  allow_donk_bet     = 0;     // 0/1 (adds an extra small donk size when OOP has no initiative)
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
  --exploitability <float> Target exploitability % (default: 0.5)
  --backend <string>       Execution backend: auto | cpu | gpu (default: auto)
  --gpu-info               Print detected GPU info and exit
  --help, -h               Show this help message

Output:
  JSON result to stdout.

Example:
  deepsolver_core.exe --pot 100 --stack 500 --board AsKd7c --target AhKh
)"  << std::endl;
}

// ============================================================================
// Range Parsing: TexasSolver format -> 1326-float weight array
// ============================================================================

/// Parse TexasSolver range string, e.g. "AA:1.0,AKs:0.5,A4o:1.0"
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

std::string result_to_json(
    const SolverResult& result, const CLIArgs& args,
    const std::string& backend_name,
    const std::map<std::string, deepsolver::Solver::StrategyTreeEntry>*
        strategy_tree = nullptr) {
    std::ostringstream json;
    json << std::fixed << std::setprecision(2);

    json << "{\n";
    json << "  \"status\": \"success\",\n";
    json << "  \"backend\": \"" << escape_json(backend_name) << "\",\n";
    json << "  \"iterations_run\": " << result.iterations_run << ",\n";
    json << "  \"exploitability_pct\": " << result.exploitability_pct << ",\n";

    // Global strategy
    json << "  \"global_strategy\": {\n";
    for (size_t i = 0; i < result.global_strategy.size(); ++i) {
        json << "    \"" << escape_json(result.global_strategy[i].first) << "\": \""
             << std::setprecision(1) << result.global_strategy[i].second << "%\"";
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
                     << std::setprecision(4) << mix[j].second;
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
                 << "\": " << result.opponent_range[i].second;
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
        json << "    \"ev\": " << std::setprecision(2) << result.target_analysis.ev << ",\n";
        json << "    \"strategy_mix\": {\n";
        for (size_t i = 0; i < result.target_analysis.strategy_mix.size(); ++i) {
            json << "      \"" << escape_json(result.target_analysis.strategy_mix[i].first)
                 << "\": \"" << std::setprecision(0)
                 << result.target_analysis.strategy_mix[i].second << "%\"";
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
                     << entry.global_strategy[i].second << "%\"";
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
                         << std::setprecision(4) << mix[j].second;
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
                     << std::setprecision(4) << entry.opponent_range[i].second;
                if (i + 1 < entry.opponent_range.size()) json << ",";
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
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    auto args = parse_args(argc, argv);

    if (args.help) {
        print_help();
        return 0;
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
        config.bet_sizing.flop_sizes = args.flop_sizes;
        config.bet_sizing.turn_sizes = args.turn_sizes;
        config.bet_sizing.river_sizes = args.river_sizes;
        config.oop_has_initiative = (args.oop_has_initiative != 0);
        config.allow_donk_bet = (args.allow_donk_bet != 0);

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

        // Run solver
        Solver solver(config, backend_type);
        SolverResult result = solver.solve(progress);
        std::string backend_name = solver.backend_name();

        // Process history navigation
        uint32_t target_node = 0;
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

        // Build strategy tree for client-side navigation cache (Route A).
        // The frontend caches this and walks it on every action click,
        // skipping the engine subprocess entirely until a board change.
        auto strategy_tree = solver.build_strategy_tree(/*max_player_depth=*/8);

        // Output JSON to stdout (with the strategy tree appended).
        std::cout << result_to_json(result, args, backend_name, &strategy_tree);

    } catch (const std::exception& e) {
        std::cerr << "{\"status\": \"error\", \"message\": \""
                  << escape_json(e.what()) << "\"}\n";
        return 1;
    }

    return 0;
}

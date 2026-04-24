/**
 * @file game_tree_builder.h
 * @brief Dynamic game tree construction with action abstraction and pruning.
 *
 * Builds a decision tree from solver configuration, applying industry-standard
 * pruning rules to prevent exponential node explosion:
 *   - Raise cap (max 3-4 actions per street)
 *   - All-in threshold (SPR < 12% → force all-in)
 *   - Street-by-street bet size simplification
 *   - Donk bet pruning
 *   - Geometric sizing for deep stacks
 */

#pragma once

#include "types.h"
#include "isomorphism.h"
#include <vector>
#include <cmath>

namespace deepsolver {

// ============================================================================
// Internal Tree Node (CPU-side, pointer-based during construction only)
// ============================================================================

struct TreeNode {
    NodeType type;
    uint8_t  street;          ///< 0=flop, 1=turn, 2=river
    uint8_t  active_player;   ///< 0=OOP, 1=IP
    float    pot;
    float    stack;           ///< Remaining effective stack
    float    bet_into;        ///< Current bet to call (0 if no bet)
    int      raise_count;     ///< Number of raises so far in this street
    bool     oop_has_initiative; ///< Did OOP make the last aggressive action?

    TerminalType terminal_type = TerminalType::SHOWDOWN;

    /// Child edges: (action, child_node_index)
    std::vector<std::pair<Action, uint32_t>> children;

    uint32_t node_id = 0;    ///< Index in the flat arrays

    /// Card dealt by the parent CHANCE node to reach this node, or
    /// UINT8_MAX if this node was not produced by a chance deal. Used by
    /// runout enumeration so terminal evaluation knows which board to use.
    uint8_t  dealt_card = 0xFFu;
    /// Iso multiplicity of this dealt card (1 = no iso, k = represents k
    /// equivalent cards under suit isomorphism). Stays 1 until Phase 2.
    uint8_t  runout_weight = 1;
    /// Cumulative runout cards dealt from root to this node. The full board
    /// at any node = config.board ∪ runout_cards. Built up during recursion.
    std::vector<uint8_t> runout_cards;
};

// ============================================================================
// Game Tree Builder
// ============================================================================

class GameTreeBuilder {
public:
    explicit GameTreeBuilder(const SolverConfig& config);

    /// Build the full game tree and return it in SoA format.
    FlatGameTree build();

    /// Get the total node count (valid after build)
    uint32_t node_count() const { return static_cast<uint32_t>(nodes_.size()); }

private:
    const SolverConfig& config_;
    std::vector<TreeNode> nodes_;

    /// Add a node to the tree and return its index
    uint32_t add_node(TreeNode node);

    /// Recursively build the tree from a given node
    void build_subtree(uint32_t node_idx);

    /// Generate available actions for a player node
    std::vector<Action> generate_actions(const TreeNode& node) const;

    /// Get bet sizes for the current street
    const std::vector<float>& get_bet_sizes(uint8_t street) const;

    /// Check if all-in should be the current street's option
    bool has_allin(uint8_t street) const;

    /// Compute geometric bet sizing:
    /// Calculate the bet size that, if repeated each street, reaches all-in.
    float compute_geometric_size(float pot, float stack, int streets_remaining) const;

    /// Check if a raise would trigger the all-in threshold
    bool should_force_allin(float pot, float stack, float proposed_bet) const;

    /// Flatten the pointer-based tree into SoA format
    FlatGameTree flatten() const;
};

// ============================================================================
// Implementation
// ============================================================================

inline GameTreeBuilder::GameTreeBuilder(const SolverConfig& config)
    : config_(config)
{}

inline uint32_t GameTreeBuilder::add_node(TreeNode node) {
    node.node_id = static_cast<uint32_t>(nodes_.size());
    nodes_.push_back(std::move(node));
    return nodes_.back().node_id;
}

inline const std::vector<float>& GameTreeBuilder::get_bet_sizes(uint8_t street) const {
    switch (street) {
        case 0: return config_.bet_sizing.flop_sizes;
        case 1: return config_.bet_sizing.turn_sizes;
        case 2: return config_.bet_sizing.river_sizes;
        default: return config_.bet_sizing.river_sizes;
    }
}

inline bool GameTreeBuilder::has_allin(uint8_t street) const {
    switch (street) {
        case 0: return config_.bet_sizing.flop_allin;
        case 1: return config_.bet_sizing.turn_allin;
        case 2: return config_.bet_sizing.river_allin;
        default: return true;
    }
}

inline float GameTreeBuilder::compute_geometric_size(float pot, float stack,
                                                      int streets_remaining) const {
    if (streets_remaining <= 0) return stack;
    // Solve: pot * ((1+x)^n - 1) / x = stack, approximately:
    // x = (stack / pot)^(1/n) - 1 ... simplified geometric ratio
    // We want the pot-fraction bet b such that betting b*pot each street
    // exhausts the stack. Using the geometric series:
    // stack = b*pot * (1 + (1+2b) + (1+2b)^2 + ... )
    // Approximate: b ≈ ((stack/pot + 1)^(1/n) - 1) / 2
    float ratio = std::pow((stack + pot) / pot, 1.0f / streets_remaining);
    float geo_fraction = (ratio - 1.0f) / 2.0f;
    return std::max(0.2f, std::min(geo_fraction, 3.0f));  // Clamp to [20%, 300%]
}

inline bool GameTreeBuilder::should_force_allin(float pot, float stack,
                                                 float proposed_bet) const {
    float remaining = stack - proposed_bet;
    float new_pot = pot + proposed_bet * 2; // Both players contribute
    if (remaining <= 0) return true;
    float spr = remaining / new_pot;
    return spr < config_.allin_threshold;
}

inline std::vector<Action> GameTreeBuilder::generate_actions(const TreeNode& node) const {
    std::vector<Action> actions;

    if (node.bet_into > 0) {
        // Facing a bet: Fold, Call, Raise options
        actions.push_back({ActionType::FOLD, 0.0f});
        actions.push_back({ActionType::CALL, node.bet_into});

        // Raise options (if under raise cap)
        if (node.raise_count < config_.raise_cap) {
            const auto& sizes = get_bet_sizes(node.street);
            float current_pot = node.pot + node.bet_into; // pot when facing bet

            for (float frac : sizes) {
                float raise_to = node.bet_into + current_pot * frac;
                raise_to = std::min(raise_to, node.stack);

                // Check all-in threshold
                if (should_force_allin(current_pot, node.stack, raise_to)) {
                    // Force all-in instead
                    if (has_allin(node.street)) {
                        actions.push_back({ActionType::ALLIN, node.stack});
                    }
                    break;  // No point adding smaller raises
                }

                actions.push_back({ActionType::RAISE, raise_to});
            }

            // Add explicit all-in if not already forced
            if (has_allin(node.street) && actions.back().type != ActionType::ALLIN) {
                actions.push_back({ActionType::ALLIN, node.stack});
            }
        }
    } else {
        // No bet to face: Check or Bet options
        actions.push_back({ActionType::CHECK, 0.0f});

        // Donk bet pruning: OOP without initiative can't donk by default
        bool can_bet = true;
        if (node.active_player == 0 && !node.oop_has_initiative && !config_.allow_donk_bet) {
            can_bet = false;
        }

        if (can_bet) {
            const auto& sizes = get_bet_sizes(node.street);
            int streets_remaining = 2 - node.street; // flop=2, turn=1, river=0

            for (float frac : sizes) {
                float bet_amount = node.pot * frac;
                bet_amount = std::min(bet_amount, node.stack);

                // Geometric sizing for deep stacks
                if (streets_remaining > 0 && node.stack > node.pot * 2.0f) {
                    float geo = compute_geometric_size(node.pot, node.stack, streets_remaining + 1);
                    // If geometric size is significantly different, consider adding it
                    if (std::abs(geo - frac) > 0.1f && geo > frac) {
                        // The geometric size fills a gap — we'll add it only if
                        // it doesn't duplicate existing sizes
                        float geo_bet = node.pot * geo;
                        geo_bet = std::min(geo_bet, node.stack);
                        if (should_force_allin(node.pot, node.stack, geo_bet)) {
                            actions.push_back({ActionType::ALLIN, node.stack});
                            break;
                        } else {
                            // Add geometric as a virtual size between existing ones
                            // (handled by the caller if needed)
                        }
                    }
                }

                if (should_force_allin(node.pot, node.stack, bet_amount)) {
                    if (has_allin(node.street)) {
                        actions.push_back({ActionType::ALLIN, node.stack});
                    }
                    break;
                }

                actions.push_back({ActionType::BET, bet_amount});
            }

            // Donk bet (restricted size): if allowed but only one small size
            if (node.active_player == 0 && !node.oop_has_initiative && config_.allow_donk_bet) {
                float donk = node.pot * config_.donk_bet_size;
                donk = std::min(donk, node.stack);
                actions.push_back({ActionType::BET, donk});
            }

            // Explicit all-in option
            if (has_allin(node.street) && !actions.empty() &&
                actions.back().type != ActionType::ALLIN) {
                actions.push_back({ActionType::ALLIN, node.stack});
            }
        }
    }

    // Deduplicate: remove duplicate all-in entries
    auto last = std::unique(actions.begin(), actions.end());
    actions.erase(last, actions.end());

    // Cap total actions at MAX_ACTIONS
    if (actions.size() > MAX_ACTIONS) {
        actions.resize(MAX_ACTIONS);
    }

    return actions;
}

inline void GameTreeBuilder::build_subtree(uint32_t node_idx) {
    // CRITICAL: copy node fields into a local snapshot. We CANNOT hold a
    // reference into nodes_[] across calls to add_node() — push_back may
    // reallocate the vector and invalidate the reference. The previous code
    // (TreeNode& node = nodes_[node_idx]) silently corrupted memory once the
    // tree was big enough to trigger a reallocation. The bug only became
    // reproducible after the oop_has_initiative fix grew the tree past the
    // initial reservation.
    NodeType n_type;
    uint8_t  n_street;
    uint8_t  n_active_player;
    float    n_pot;
    float    n_stack;
    float    n_bet_into;
    int      n_raise_count;
    bool     n_oop_has_initiative;
    std::vector<uint8_t> n_runout_cards;
    {
        const TreeNode& node = nodes_[node_idx];
        n_type               = node.type;
        n_street             = node.street;
        n_active_player      = node.active_player;
        n_pot                = node.pot;
        n_stack              = node.stack;
        n_bet_into           = node.bet_into;
        n_raise_count        = node.raise_count;
        n_oop_has_initiative = node.oop_has_initiative;
        n_runout_cards       = node.runout_cards;
    }

    // Terminal checks
    if (n_type == NodeType::TERMINAL) return;
    if (n_type == NodeType::CHANCE) {
        uint8_t next_street = n_street + 1;
        if (next_street > 2) return; // Past river = terminal

        // PHASE 2 RUNOUT ENUMERATION (with suit isomorphism):
        //   - Compute the current full board (config flop ∪ runout cards
        //     dealt so far) and its suit-permutation group G.
        //   - Quotient the undealt deck into G-orbits. One representative per
        //     orbit becomes a child, with runout_weight = orbit size.
        //   - Sum of weights = number of undealt cards (so the chance-node
        //     weighted average remains an unbiased EV estimator).
        //
        // Speedup vs full enumeration depends on board texture:
        //   - Rainbow (4 distinct suits): G={id} → 1.0x (no compression)
        //   - Two-tone: 2x;  Monotone flop / 3-of-suit turn: 3x;
        //   - Paired flop with two suits on the pair: 2x.
        //
        // Memory safety: even with iso, a rainbow flop's 49 turn × 47 river
        // = 2300 leaf matchup tables (~640KB each = ~1.5GB) is too much. We
        // gate flop-level enumeration on the iso-compressed child count.
        std::vector<Card> full_board;
        full_board.reserve(MAX_BOARD_CARDS);
        for (uint8_t i = 0; i < config_.board_size; ++i)
            full_board.push_back(config_.board[i]);
        for (uint8_t c : n_runout_cards) full_board.push_back(c);

        CanonicalRunouts cr = enumerate_canonical_runouts(
            full_board.data(), static_cast<uint8_t>(full_board.size()));

        uint8_t cards_already = static_cast<uint8_t>(full_board.size());
        // Always enumerate at turn level (one chance step left). At flop
        // level enumerate only if iso keeps the projected leaf count under
        // ~2000 matchup tables (~1.3GB at 640KB/table).
        bool enumerate;
        if (cards_already >= 4) {
            enumerate = true;
        } else {
            // Estimate level-2 (river) child count = ~47 minus iso. We don't
            // know exact iso compression at the next level without recursing,
            // so use canonical_turns * 47 as a worst-case bound.
            size_t projected = cr.reps.size() * 47u;
            enumerate = (projected <= 2000);
        }

        if (!enumerate) {
            // Legacy single-child fallback: emit one child without dealing a
            // card. Terminal eval uses the root matchup. Correct CFR-wise,
            // just no per-runout equity. Used only when iso can't tame the
            // memory blowup (typically rainbow flops).
            TreeNode child;
            child.type = NodeType::PLAYER_OOP;
            child.street = next_street;
            child.active_player = 0;
            child.pot = n_pot;
            child.stack = n_stack;
            child.bet_into = 0;
            child.raise_count = 0;
            child.oop_has_initiative = true;
            child.dealt_card = 0xFFu;
            child.runout_weight = 1;
            child.runout_cards = n_runout_cards;
            uint32_t child_idx = add_node(std::move(child));
            nodes_[node_idx].children.push_back({{ActionType::CHECK, 0}, child_idx});
            build_subtree(child_idx);
            return;
        }

        for (const auto& rep : cr.reps) {
            TreeNode child;
            child.type = NodeType::PLAYER_OOP;
            child.street = next_street;
            child.active_player = 0;
            child.pot = n_pot;
            child.stack = n_stack;
            child.bet_into = 0;
            child.raise_count = 0;
            child.oop_has_initiative = true;
            child.dealt_card = rep.card;
            child.runout_weight = rep.weight;  // Phase 2: orbit size
            child.runout_cards = n_runout_cards;
            child.runout_cards.push_back(rep.card);

            uint32_t child_idx = add_node(std::move(child));
            nodes_[node_idx].children.push_back(
                {{ActionType::CHECK, static_cast<float>(rep.card)}, child_idx});
            build_subtree(child_idx);
        }
        return;
    }

    // Player decision node — generate_actions needs a TreeNode-like view.
    // Reconstruct one from the snapshot so generate_actions can read it
    // safely even if nodes_ later reallocates.
    TreeNode node_view{};
    node_view.type               = n_type;
    node_view.street             = n_street;
    node_view.active_player      = n_active_player;
    node_view.pot                = n_pot;
    node_view.stack              = n_stack;
    node_view.bet_into           = n_bet_into;
    node_view.raise_count        = n_raise_count;
    node_view.oop_has_initiative = n_oop_has_initiative;
    auto actions = generate_actions(node_view);

    for (const auto& action : actions) {
        TreeNode child;
        child.street = n_street;
        child.oop_has_initiative = n_oop_has_initiative;
        child.raise_count = n_raise_count;
        // Inherit cumulative runout cards from parent so terminal evaluation
        // along this branch knows the full board.
        child.runout_cards = n_runout_cards;
        child.dealt_card = 0xFFu;  // not a chance child

        switch (action.type) {
            case ActionType::FOLD: {
                child.type = NodeType::TERMINAL;
                child.pot = n_pot;
                child.stack = n_stack;
                child.terminal_type = (n_active_player == 0)
                    ? TerminalType::FOLD_OOP : TerminalType::FOLD_IP;
                break;
            }
            case ActionType::CHECK: {
                if (n_active_player == 0) {
                    // OOP checks → IP acts
                    child.type = NodeType::PLAYER_IP;
                    child.active_player = 1;
                    child.pot = n_pot;
                    child.stack = n_stack;
                    child.bet_into = 0;
                } else {
                    // IP checks → end of street (both checked)
                    if (n_street < 2) {
                        // Move to next street (chance node)
                        child.type = NodeType::CHANCE;
                        child.street = n_street;
                    } else {
                        // River → showdown
                        child.type = NodeType::TERMINAL;
                        child.terminal_type = TerminalType::SHOWDOWN;
                    }
                    child.pot = n_pot;
                    child.stack = n_stack;
                }
                break;
            }
            case ActionType::CALL: {
                float call_amount = n_bet_into;
                float new_pot = n_pot + call_amount;
                float new_stack = n_stack - call_amount;

                if (new_stack <= 0.01f) {
                    child.type = NodeType::TERMINAL;
                    child.terminal_type = TerminalType::SHOWDOWN;
                    child.pot = new_pot;
                    child.stack = 0;
                } else if (n_street < 2) {
                    child.type = NodeType::CHANCE;
                    child.pot = new_pot;
                    child.stack = new_stack;
                } else {
                    child.type = NodeType::TERMINAL;
                    child.terminal_type = TerminalType::SHOWDOWN;
                    child.pot = new_pot;
                    child.stack = new_stack;
                }
                break;
            }
            case ActionType::BET:
            case ActionType::RAISE:
            case ActionType::ALLIN: {
                float bet_amount = action.amount;
                float new_pot = n_pot + bet_amount;
                float new_stack = n_stack - bet_amount;

                if (new_stack <= 0.01f) new_stack = 0;

                child.type = (n_active_player == 0)
                    ? NodeType::PLAYER_IP : NodeType::PLAYER_OOP;
                child.active_player = 1 - n_active_player;
                child.pot = new_pot;
                child.stack = new_stack;
                child.bet_into = bet_amount;
                child.raise_count = (action.type == ActionType::RAISE)
                    ? n_raise_count + 1 : n_raise_count;

                // Track initiative
                child.oop_has_initiative = (n_active_player == 0);

                if (new_stack <= 0.01f) {
                    // All-in: opponent can only call or fold
                    child.raise_count = config_.raise_cap;
                }
                break;
            }
        }

        uint32_t child_idx = add_node(std::move(child));
        nodes_[node_idx].children.push_back({action, child_idx});

        build_subtree(child_idx);
    }
}

inline FlatGameTree GameTreeBuilder::build() {
    nodes_.clear();
    // Pre-reserve generously. Even with the snapshot fix in build_subtree,
    // avoiding reallocations during construction keeps pointers from prior
    // siblings' children vectors valid and reduces allocator pressure for
    // big trees (e.g. SPR>5 with raise_cap=3 across three streets).
    nodes_.reserve(200000);

    // Root: OOP acts first on the CURRENT street. Board size determines which
    // street we're actually on: 3 cards = flop (street 0), 4 cards = turn
    // (street 1), 5 cards = river (street 2). Previously this was hardcoded
    // to 0 which forced a 3-street tree even when analyzing a turn, wasting
    // ~2-3x the CFR compute on virtual "flop" betting rounds that the real
    // game already passed.
    uint8_t cur_street = 0;
    if (config_.board_size >= 5)      cur_street = 2;  // river
    else if (config_.board_size == 4) cur_street = 1;  // turn
    else                               cur_street = 0;  // flop (or less)

    TreeNode root;
    root.type = NodeType::PLAYER_OOP;
    root.street = cur_street;
    root.active_player = 0;
    root.pot = config_.pot;
    root.stack = config_.effective_stack;
    root.bet_into = 0;
    root.raise_count = 0;
    // OOP has initiative by default so bet options are available at the root.
    // The UI/CLI can override via SolverConfig.oop_has_initiative (used for
    // analyzing single-raised pots where OOP would only check to IP). See
    // --oop-initiative CLI flag.
    root.oop_has_initiative = config_.oop_has_initiative;

    add_node(std::move(root));
    build_subtree(0);

    return flatten();
}

inline FlatGameTree GameTreeBuilder::flatten() const {
    FlatGameTree flat;
    uint32_t n = static_cast<uint32_t>(nodes_.size());
    flat.reserve(n, n * 3);
    flat.total_nodes = n;

    uint32_t edge_offset = 0;

    for (uint32_t i = 0; i < n; ++i) {
        const auto& node = nodes_[i];
        flat.node_types.push_back(static_cast<uint8_t>(node.type));
        flat.pots.push_back(node.pot);
        flat.stacks.push_back(node.stack);
        flat.parent_indices.push_back(0); // Will fix below
        flat.children_offset.push_back(edge_offset);
        flat.num_children.push_back(static_cast<uint8_t>(node.children.size()));
        flat.street.push_back(node.street);
        flat.terminal_types.push_back(static_cast<uint8_t>(node.terminal_type));
        flat.active_player.push_back(node.active_player);
        flat.bet_into.push_back(node.bet_into);
        flat.dealt_card.push_back(node.dealt_card);
        flat.runout_weight.push_back(node.runout_weight);
        flat.matchup_idx.push_back(0);  // populated post-build by Solver::precompute_matchups

        for (const auto& [action, child_idx] : node.children) {
            flat.children.push_back(child_idx);
            flat.child_action_types.push_back(static_cast<uint8_t>(action.type));
            flat.child_action_amts.push_back(action.amount);
            edge_offset++;
        }
    }

    flat.total_edges = edge_offset;

    // Fix parent indices
    for (uint32_t i = 0; i < n; ++i) {
        for (const auto& [action, child_idx] : nodes_[i].children) {
            flat.parent_indices[child_idx] = i;
        }
    }

    return flat;
}

} // namespace deepsolver

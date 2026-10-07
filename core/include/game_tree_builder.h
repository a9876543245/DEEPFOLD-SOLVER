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
#include "memory_budget.h"
#include "terminal_plan.h"   // bytes_for_matchup_rank_tables (A4-host inc 4)
#include <vector>
#include <cmath>

namespace deepsolver {

// ============================================================================
// Internal Tree Node (CPU-side, pointer-based during construction only)
// ============================================================================

struct TreeNode {
    NodeType type = NodeType::PLAYER_OOP;
    uint8_t  street = 0;          ///< 0=flop, 1=turn, 2=river
    uint8_t  active_player = 0;   ///< 0=OOP, 1=IP
    float    pot = 0.0f;
    float    stack = 0.0f;        ///< Remaining stack of the player TO ACT
    float    bet_into = 0.0f;     ///< Outstanding contribution difference to call
    int      raise_count = 0;     ///< Number of raises so far in this street
    /// Chips the player TO ACT has put in on this street (the opponent has
    /// street_contrib + bet_into). Needed for "x" raises: raise TO x times the
    /// opponent's street wager.
    float    street_contrib = 0.0f;
    /// Last bettor/raiser on this street (0 = OOP, 1 = IP, 2 = none yet).
    uint8_t  aggressor = 2;
    /// The previous street's last aggressor — on the root street, IP when OOP
    /// has no initiative. OOP's opening bet into an IP aggressor is a donk bet
    /// and takes the donk menu.
    uint8_t  prev_aggressor = 2;

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
    /// Chance children only: the suit permutations mapping dealt_card onto
    /// the other members of its runout orbit (CanonicalRunout::member_perms).
    std::vector<std::array<uint8_t, 4>> runout_member_perms;
};

// ============================================================================
// Bet-sizing menus
// ============================================================================

using SizingMenus = std::array<std::array<StreetSizing, 3>, 2>;   ///< [player][street]

/// Street index of a board size (3 = flop 0, 4 = turn 1, 5 = river 2).
inline uint8_t street_of_board_size(uint8_t board_size) {
    return board_size >= 5 ? 2 : (board_size == 4 ? 1 : 0);
}

/// The per-player, per-street menus the builder plays from (2026-10-06).
/// A custom config (--bet-sizing) is used as given. A legacy config derives
/// them so the legacy trees come out unchanged: each street's one list serves
/// bets and raises of both players; OOP may lead every later street with it;
/// on the root street OOP without initiative leads only with
/// allow_donk_bet, adding donk_bet_size to the list.
inline SizingMenus resolve_sizing_menus(const SolverConfig& cfg) {
    SizingMenus m;
    if (cfg.bet_sizing.custom) {
        for (int p = 0; p < 2; ++p)
            for (int st = 0; st < 3; ++st) m[p][st] = cfg.bet_sizing.player[p][st];
        return m;
    }
    const std::vector<float>* lists[3] = {
        &cfg.bet_sizing.flop_sizes, &cfg.bet_sizing.turn_sizes, &cfg.bet_sizing.river_sizes};
    const bool allin[3] = {
        cfg.bet_sizing.flop_allin, cfg.bet_sizing.turn_allin, cfg.bet_sizing.river_allin};
    const uint8_t root_street = street_of_board_size(cfg.board_size);
    for (int st = 0; st < 3; ++st) {
        std::vector<BetSize> sizes;
        for (float f : *lists[st]) sizes.push_back({BetSize::Kind::PotFraction, f});
        for (int p = 0; p < 2; ++p) {
            m[p][st].bet   = sizes;
            m[p][st].raise = sizes;
            m[p][st].allin = allin[st];
        }
        if (st == root_street && !cfg.oop_has_initiative) {
            if (cfg.allow_donk_bet) {
                m[0][st].donk = sizes;
                m[0][st].donk.push_back({BetSize::Kind::PotFraction, cfg.donk_bet_size});
            }
        } else {
            m[0][st].donk = sizes;
        }
    }
    return m;
}

// ============================================================================
// Game Tree Builder
// ============================================================================

class GameTreeBuilder {
public:
    explicit GameTreeBuilder(const SolverConfig& config);

    /// Phase 2 (10-point plan): callers can hand in the canonical-combo
    /// count and a memory budget so the runout-enumeration gate is
    /// byte-based instead of an arbitrary "projected <= 2000". When
    /// `nc_canonical_estimate == 0` the builder falls back to the legacy
    /// runout-count heuristic.
    /// `host_dense_matchup` (A4-host inc 4): false when the solve will take
    /// the rank/fold blockers and therefore never materializes the dense nc²
    /// tables — the gate must then price the per-runout RANK tables instead.
    /// Charging the dense price there is a ~500× over-estimate and is what
    /// used to collapse every rainbow flop (28.6 GB projected vs 6.2 MB
    /// real on AsKd7c). Callers derive it from
    /// `host_dense_matchup_required()` so builder, precompute and the
    /// estimators cannot disagree.
    void set_memory_policy(
        uint16_t nc_canonical_estimate,
        const MemoryBudget& budget,
        uint64_t matchup_bytes_per_cell =
            memory_budget::kMatchupBytesPerCell,
        bool host_dense_matchup = true) {
        nc_estimate_   = nc_canonical_estimate;
        budget_        = budget;
        matchup_bytes_per_cell_ = matchup_bytes_per_cell;
        host_dense_matchup_ = host_dense_matchup;
        budget_set_    = true;
    }

    /// Streaming/subgame decomposition (opt-in): when set, the flop→turn
    /// chance node emits one subgame-leaf placeholder per canonical turn card
    /// (preserving dealt_card / runout_weight / entering pot+stack) but does
    /// NOT recurse into turn betting. The result is a small "trunk" — flop
    /// betting whose turn chance children are leaves — that the decomposition
    /// orchestrator (solver_decomposed.h) couples to per-turn-card subgames.
    /// Default false ⇒ the builder behaves exactly as before.
    void set_truncate_at_chance(bool on) { truncate_at_chance_ = on; }

    /// Post-v1.9.0 roadmap ①: force the single-child runout fallback at every
    /// chance gate. The matchup-byte gate below models matchup tables ONLY —
    /// a monotone flop's tiny nc slips under it, the tree fully enumerates,
    /// and the un-modeled per-node CFR state (total_nodes × nc) then kills
    /// the backend at prepare. Solver::solve() measures the enumerated
    /// tree's state footprint against the planned backend's budget and
    /// re-builds with this set when it cannot fit. Default false ⇒ builder
    /// behaves exactly as before.
    void set_force_runout_collapse(bool on) { force_runout_collapse_ = on; }

    /// Build the full game tree and return it in SoA format.
    FlatGameTree build();

    /// Get the total node count (valid after build)
    uint32_t node_count() const { return static_cast<uint32_t>(nodes_.size()); }

    /// The suit permutations every chance node's runout orbits are taken
    /// under (see range_perms_). Solver::project_enumerated_tree()
    /// re-enumerates the canonical runouts with exactly this set, so its
    /// per-chance counts are the builder's own.
    const std::vector<std::array<uint8_t, 4>>& range_perms() const {
        return range_perms_;
    }

    /// The group the runout orbits are taken under
    /// (IsomorphismMapping::runout_perms). The constructor derives the same
    /// group from the config; a solve whose classes come from elsewhere (a
    /// forced iso — decomposition subgames share the flop trunk's) must pass
    /// its own.
    void set_runout_group(const std::vector<std::array<uint8_t, 4>>& perms) {
        range_perms_ = perms;
    }

private:
    const SolverConfig& config_;
    std::vector<TreeNode> nodes_;
    /// The hand-class group (hand_class_suit_group: fixes the ROOT board,
    /// preserves ranges and locks, moves only never-flush suits). Every
    /// chance node's runout orbits are taken under (perms fixing that node's
    /// board) ∩ this set, so two runout cards share one child only when the
    /// two subgames are isomorphic AND the hand classes are invariant under
    /// the map. 2026-10-06 audit: this used to be the range/lock group alone,
    /// which on a flop root admitted perms fixing flop ∪ turn but not the
    /// flop (AcKd7h + 7s turn → (h s)) — the river orbits then merged Xh with
    /// Xs while the hand classes were singletons.
    std::vector<std::array<uint8_t, 4>> range_perms_;
    uint16_t      nc_estimate_ = 0;
    MemoryBudget  budget_      = MemoryBudget::defaults();
    uint64_t      matchup_bytes_per_cell_ =
        memory_budget::kMatchupBytesPerCell;
    /// See set_memory_policy(). Default true = pre-A4-inc-4 behavior.
    bool          host_dense_matchup_ = true;
    bool          budget_set_  = false;
    /// Set true when the memory gate forces the single-child runout fallback
    /// (no card dealt → turn/river solved on the stale flop matchup). Surfaced
    /// so the result can warn that runout equity is approximated rather than
    /// degrading silently.
    bool          runout_collapsed_ = false;
    /// See set_truncate_at_chance(). Opt-in; default off.
    bool          truncate_at_chance_ = false;
    /// See set_force_runout_collapse(). Opt-in; default off.
    bool          force_runout_collapse_ = false;

    /// Add a node to the tree and return its index
    uint32_t add_node(TreeNode node);

    /// Recursively build the tree from a given node
    void build_subtree(uint32_t node_idx);

    /// Generate available actions for a player node
    std::vector<Action> generate_actions(const TreeNode& node) const;

    /// Does putting `additional` chips in (bet, or call + raise) leave the
    /// actor's remaining stack below allin_threshold × the pot once the
    /// opponent calls? Then the action is played as all-in instead.
    bool should_force_allin(float pot, float bet_into, float stack,
                            float additional) const;

    /// resolve_sizing_menus(config_), resolved once per builder.
    SizingMenus menus_;

    /// Flatten the pointer-based tree into SoA format
    FlatGameTree flatten() const;
};

// ============================================================================
// Implementation
// ============================================================================

inline GameTreeBuilder::GameTreeBuilder(const SolverConfig& config)
    : config_(config)
{
    IsoConstraints c;
    c.oop_weights = &config.oop_range_weights;
    c.ip_weights  = &config.ip_range_weights;
    c.node_locks  = &config.node_locks;
    range_perms_ = (config.iso_mode == IsoMode::Fast)
        ? hand_class_suit_group(config.board.data(), config.board_size, &c)
        : board_symmetry_group(config.board.data(), config.board_size, &c);
    menus_ = resolve_sizing_menus(config);
}

inline uint32_t GameTreeBuilder::add_node(TreeNode node) {
    node.node_id = static_cast<uint32_t>(nodes_.size());
    nodes_.push_back(std::move(node));
    return nodes_.back().node_id;
}

inline bool GameTreeBuilder::should_force_allin(float pot, float bet_into,
                                                 float stack, float additional) const {
    const float remaining = stack - additional;
    if (remaining <= 0.0f) return true;
    // Pot once the opponent calls: the actor adds `additional`, the opponent
    // the part of it they have not matched yet. 2026-10-06 audit: a raise was
    // priced against pot + bet_into + 2·additional — 2·bet_into too much —
    // which forced all-in earlier than the documented threshold.
    const float new_pot = pot + 2.0f * additional - bet_into;
    return remaining / new_pot < config_.allin_threshold;
}

inline std::vector<Action> GameTreeBuilder::generate_actions(const TreeNode& node) const {
    // 2026-10-06 (audit + custom sizing): every menu size becomes a candidate
    // amount; candidates are sorted and de-duplicated, and an all-in — the
    // menu's own or one a size was forced into — is appended exactly once, so
    // menu order cannot drop a size (a `break` after the first forced all-in
    // used to discard every later, smaller one) and no node carries two
    // all-ins (the donk branch used to append a second one).
    const StreetSizing& menu = menus_[node.active_player][node.street];
    std::vector<Action> actions;
    std::vector<float> amounts;
    bool add_allin = false;
    auto consider = [&](float additional) {
        if (!(additional > 0.0f)) return;
        if (additional >= node.stack ||
            should_force_allin(node.pot, node.bet_into, node.stack, additional)) {
            if (menu.allin) add_allin = true;   // without all-in the size is dropped
            return;
        }
        amounts.push_back(additional);
    };

    ActionType sized_type = ActionType::BET;
    if (node.bet_into > 0) {
        // Facing a bet: Fold, Call, Raise options
        actions.push_back({ActionType::FOLD, 0.0f});
        actions.push_back({ActionType::CALL, node.bet_into});
        sized_type = ActionType::RAISE;

        if (node.raise_count < config_.raise_cap && node.stack > node.bet_into) {
            const float mine = node.street_contrib;
            const float theirs = mine + node.bet_into;
            for (const BetSize& b : menu.raise) {
                // Amount is the actor's ADDITIONAL investment (call + raise).
                // A % raise is Pio's: the raise on top of the call is that
                // fraction of the pot after calling. An "x" raise is TO x
                // times the opponent's street wager. Heads-up, bet_into is the
                // last wager increment, so a full raise invests at least 2x it;
                // a short all-in is legal even when it cannot meet that.
                float additional = (b.kind == BetSize::Kind::Multiplier)
                    ? b.value * theirs - mine
                    : node.bet_into + b.value * (node.pot + node.bet_into);
                additional = std::max(additional, 2.0f * node.bet_into);
                consider(additional);
            }
            if (menu.allin) add_allin = true;
        }
    } else {
        // No bet to face: Check or Bet options
        actions.push_back({ActionType::CHECK, 0.0f});
        // OOP acts first on every street, so OOP without a bet to face is
        // opening the street; into the previous street's IP aggressor that is
        // a donk bet. An empty donk menu means check only.
        const bool donk_spot = (node.active_player == 0 && node.prev_aggressor == 1);
        const std::vector<BetSize>& opens = donk_spot ? menu.donk : menu.bet;
        if (!(donk_spot && opens.empty())) {
            for (const BetSize& b : opens) {
                if (b.kind == BetSize::Kind::PotFraction) consider(node.pot * b.value);
            }
            if (menu.allin) add_allin = true;
        }
    }

    std::sort(amounts.begin(), amounts.end());
    float last = -1.0f;
    for (float a : amounts) {
        if (last >= 0.0f && a - last <= 1e-4f * std::max(1.0f, a)) continue;
        actions.push_back({sized_type, a});
        last = a;
    }
    // Defensive cap (validated menus never reach it): drop the largest sized
    // options, never the all-in.
    const std::size_t room = MAX_ACTIONS - (add_allin ? 1u : 0u);
    if (actions.size() > room) actions.resize(room);
    if (add_allin) actions.push_back({ActionType::ALLIN, node.stack});
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
    float    n_street_contrib;
    uint8_t  n_aggressor;
    uint8_t  n_prev_aggressor;
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
        n_street_contrib     = node.street_contrib;
        n_aggressor          = node.aggressor;
        n_prev_aggressor     = node.prev_aggressor;
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
            full_board.data(), static_cast<uint8_t>(full_board.size()),
            &range_perms_);

        uint8_t cards_already = static_cast<uint8_t>(full_board.size());

        // Streaming/subgame decomposition truncation (opt-in). At the
        // flop→turn chance (next_street == 1), emit one subgame-leaf
        // placeholder per canonical turn card but DO NOT recurse into turn
        // betting. The decomposition orchestrator solves each leaf as an
        // independent turn subgame and injects its per-combo root value here.
        // Taken BEFORE the memory gate, so runout_collapsed_ is never set on
        // this path and the OOM-prone full enumeration never happens.
        if (truncate_at_chance_ && next_street == 1) {
            for (const auto& rep : cr.reps) {
                TreeNode child;
                child.type = NodeType::PLAYER_OOP;   // subgame root acts first
                child.street = next_street;
                child.active_player = 0;
                child.pot = n_pot;
                child.stack = n_stack;
                child.bet_into = 0;
                child.raise_count = 0;
                child.prev_aggressor = n_aggressor;   // street_contrib 0, aggressor none
                child.dealt_card = rep.card;
                child.runout_weight = rep.weight;
                child.runout_cards = n_runout_cards;
                child.runout_cards.push_back(rep.card);
                child.runout_member_perms = rep.member_perms;
                uint32_t child_idx = add_node(std::move(child));
                nodes_[node_idx].children.push_back(
                    {{ActionType::CHECK, static_cast<float>(rep.card)}, child_idx});
                // No build_subtree(child_idx): leaf placeholder (0 children).
            }
            return;
        }

        // Always enumerate at turn level (one chance step left). At flop
        // level enumerate only if matchup tables fit the host RAM budget.
        //
        // Phase 2 (10-point plan): the previous gate was `projected <= 2000`
        // which only counted runout tables, not bytes. A monotone flop with
        // `nc` ≈ 200 produces ~12 runouts × 47 ≈ 564 leaves, well under 2000,
        // but the matchup tables are 564 × 200² × 8 B ≈ 180 MB. A rainbow
        // flop with `nc` ≈ 1300 hits ~14 GB at the same projected leaf count
        // — which silently OOMs. The byte-based gate refuses to enumerate
        // when the matchup tables alone would blow the host budget.
        bool enumerate;
        if (force_runout_collapse_) {
            // Solver-driven rebuild: the enumerated tree's CFR state was
            // measured over-budget. Collapse every chance level (matches the
            // topology a rainbow flop's matchup gate produces naturally).
            enumerate = false;
        } else if (cards_already >= 4) {
            enumerate = true;
        } else {
            // Worst-case: river enumerates one child per remaining card
            // (47 cards undealt at flop). We don't know exact iso compression
            // at the next level without recursing, so use canonical_turns × 47
            // as the upper bound.
            const size_t projected_leaves = cr.reps.size() * 47u;
            if (budget_set_ && nc_estimate_ > 0) {
                // Reserve ~half the host budget for matchup tables — the rest
                // is needed for CPU CFR state, strategy-tree EV cache, and
                // the JSON response. This is the single decision point the
                // 10-point plan calls out.
                // A4-host inc 4: price what precompute will REALLY build.
                // On a blocker solve that is one rank vector + board mask per
                // runout (nc-independent), not an nc² matrix — the difference
                // between 28.6 GB and 6.2 MB on a rainbow flop, i.e. between
                // "always collapse" and "enumerate if the CFR state fits".
                // The state itself is gated after precompute by solver.h's
                // ① collapse gate, which measures the built tree.
                const uint64_t matchup_bytes = host_dense_matchup_
                    ? bytes_for_matchup_tables(
                          static_cast<uint64_t>(projected_leaves),
                          static_cast<uint64_t>(nc_estimate_),
                          matchup_bytes_per_cell_)
                    : bytes_for_matchup_rank_tables(
                          static_cast<uint64_t>(projected_leaves));
                const uint64_t matchup_cap = (budget_.host_bytes > 0)
                    ? (budget_.host_bytes / 2ULL)
                    : (3ULL * 1024 * 1024 * 1024); // 3 GB safety floor
                enumerate = (matchup_bytes <= matchup_cap);
            } else {
                // Legacy fallback for callers that didn't set a budget.
                enumerate = (projected_leaves <= 2000);
            }
        }

        if (!enumerate) {
            // Legacy single-child fallback: emit one child without dealing a
            // card. Terminal eval uses the root matchup. Correct CFR-wise,
            // just no per-runout equity. Used only when iso can't tame the
            // memory blowup (typically rainbow flops). Flag it so the result
            // can tell the user the turn/river equity is approximated.
            runout_collapsed_ = true;
            TreeNode child;
            child.type = NodeType::PLAYER_OOP;
            child.street = next_street;
            child.active_player = 0;
            child.pot = n_pot;
            child.stack = n_stack;
            child.bet_into = 0;
            child.raise_count = 0;
            child.prev_aggressor = n_aggressor;   // street_contrib 0, aggressor none
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
            child.prev_aggressor = n_aggressor;   // street_contrib 0, aggressor none
            child.dealt_card = rep.card;
            child.runout_weight = rep.weight;  // Phase 2: orbit size
            child.runout_cards = n_runout_cards;
            child.runout_cards.push_back(rep.card);
            child.runout_member_perms = rep.member_perms;

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
    node_view.street_contrib     = n_street_contrib;
    node_view.aggressor          = n_aggressor;
    node_view.prev_aggressor     = n_prev_aggressor;
    auto actions = generate_actions(node_view);

    for (const auto& action : actions) {
        TreeNode child;
        child.street = n_street;
        child.raise_count = n_raise_count;
        // The opponent acts next on every non-terminal child of a decision
        // (or the street ends): its street wager is the actor's + bet_into.
        child.street_contrib = n_street_contrib + n_bet_into;
        child.aggressor = n_aggressor;
        child.prev_aggressor = n_prev_aggressor;
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
                // The opponent has already invested bet_into more than this
                // actor. Switching turns must switch stack ownership too; it
                // must NOT subtract this actor's new investment twice.
                const float opponent_stack = n_stack - n_bet_into;
                const float outstanding = bet_amount - n_bet_into;
                const float bettor_remaining = n_stack - bet_amount;

                child.type = (n_active_player == 0)
                    ? NodeType::PLAYER_IP : NodeType::PLAYER_OOP;
                child.active_player = 1 - n_active_player;
                child.pot = new_pot;
                child.stack = opponent_stack;
                child.bet_into = outstanding;
                child.raise_count = (action.type == ActionType::RAISE)
                    ? n_raise_count + 1 : n_raise_count;

                // This actor is now the street's last aggressor.
                child.aggressor = n_active_player;

                if (bettor_remaining <= 0.01f) {
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
    // SolverConfig.oop_has_initiative (--oop-initiative) says who was the
    // last aggressor before the root street: OOP (3-bet pot, OOP raised last
    // preflop) or IP (single-raised pot, IP opened) — OOP's opening bet into
    // IP then takes the donk menu.
    root.aggressor = 2;
    root.prev_aggressor = config_.oop_has_initiative ? 0 : 1;

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
        uint16_t perm_set = kNoRunoutPerms;
        if (!node.runout_member_perms.empty()) {
            auto it = std::find(flat.runout_perm_sets.begin(), flat.runout_perm_sets.end(),
                                node.runout_member_perms);
            perm_set = static_cast<uint16_t>(it - flat.runout_perm_sets.begin());
            if (it == flat.runout_perm_sets.end()) {
                flat.runout_perm_sets.push_back(node.runout_member_perms);
            }
        }
        flat.runout_perm_set.push_back(perm_set);

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

    flat.runout_approximated = runout_collapsed_;
    return flat;
}

} // namespace deepsolver

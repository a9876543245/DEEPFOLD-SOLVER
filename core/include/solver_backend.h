/**
 * @file solver_backend.h
 * @brief Abstract backend interface for DCFR execution.
 *
 * The Solver class is an orchestrator that:
 *   - Builds the game tree (CPU)
 *   - Computes suit isomorphism (CPU)
 *   - Precomputes the showdown matchup matrix (CPU)
 *   - Initializes per-combo reach probabilities (CPU)
 *   - Delegates the iteration hot path to an ISolverBackend
 *
 * Concrete backends:
 *   - CpuBackend  — recursive CFR on CPU (reference implementation)
 *   - GpuBackend  — iterative CFR on CUDA GPU (5-15x faster on flop trees)
 *
 * The backend ONLY handles the DCFR iteration loop. Post-solve passes
 * (EV computation, best-response / exploitability) run on the CPU using
 * the finalized strategy — this keeps the backend surface small.
 */

#pragma once

#include "types.h"
#include "isomorphism.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace deepsolver {

// ============================================================================
// DCFR discount-factor computation (shared by CPU + GPU backends)
// ============================================================================
//
// STANDARD: textbook DCFR exponent decay. (t/(t+1))^alpha, ((t+1)/(t+2))^gamma.
// POSTFLOP_STYLE: matches wasm-postflop / postflop-solver schedule —
//   pos_disc = t^1.5 / (t^1.5 + 1)
//   neg_disc = 0.5 (constant)
//   strat_weight = (t' / (t' + 1))^3   where t' = iteration - nearest_lower_pow_4(iteration)
//   strategy_sum update is decay-and-add (no reach), not accumulative.
inline void compute_dcfr_factors(
    int iteration, const SolverConfig& config,
    float& pos_disc, float& neg_disc, float& strat_weight)
{
    if (config.dcfr_schedule == SolverConfig::DcfrSchedule::POSTFLOP_STYLE) {
        double t_alpha = static_cast<double>(std::max(0, iteration - 1));
        double pow_alpha = t_alpha * std::sqrt(t_alpha);
        pos_disc = static_cast<float>(pow_alpha / (pow_alpha + 1.0));
        neg_disc = 0.5f;

        unsigned t = static_cast<unsigned>(std::max(0, iteration));
        unsigned nearest_lower_pow4 = 0;
        if (t > 0) {
            int hi = 0;
            for (unsigned x = t; x > 1; x >>= 1) ++hi;
            nearest_lower_pow4 = 1u << (hi & ~1u);
        }
        double t_gamma = static_cast<double>(t - nearest_lower_pow4);
        double r = t_gamma / (t_gamma + 1.0);
        strat_weight = static_cast<float>(r * r * r);
    } else {
        if (iteration <= 0) {
            pos_disc = neg_disc = 0.0f;
            strat_weight = 1.0f;
            return;
        }
        float t = static_cast<float>(iteration);
        pos_disc = std::pow(t / (t + 1.0f), config.dcfr_alpha);
        neg_disc = std::pow(t / (t + 1.0f), config.dcfr_beta);
        strat_weight = std::pow((t + 1.0f) / (t + 2.0f), config.dcfr_gamma);
    }
}

inline bool dcfr_decay_and_add(const SolverConfig& config) {
    return config.dcfr_schedule == SolverConfig::DcfrSchedule::POSTFLOP_STYLE;
}

// ============================================================================
// Context passed from Solver orchestrator → backend
// ============================================================================

/**
 * @brief All precomputed inputs a backend needs to run DCFR.
 *
 * All pointers are borrowed: the Solver orchestrator owns the underlying data
 * and guarantees it stays alive for the backend's lifetime.
 */
struct SolverContext {
    const FlatGameTree*          tree          = nullptr;
    const IsomorphismMapping*    iso           = nullptr;
    const SolverConfig*          config        = nullptr;

    /// Precomputed showdown matchup matrix for the ROOT board. Indexed
    /// [ci * nc + cj]. Kept for backward-compat / shortcut when no chance
    /// node is in the tree (e.g. solving a river spot).
    const std::vector<float>*    matchup_ev    = nullptr;
    const std::vector<float>*    matchup_valid = nullptr;
    /// Phase 1 chance enumeration: per-runout matchup tables. The terminal
    /// handler should pick the table by tree.matchup_idx[node_idx].
    /// matchup_ev_per_runout[k] is the matchup-ev table for runout k.
    const std::vector<std::vector<float>>* matchup_ev_per_runout    = nullptr;
    const std::vector<std::vector<float>>* matchup_valid_per_runout = nullptr;

    /// Per-canonical-combo reach probabilities derived from range weights.
    const std::vector<float>*    ip_reach      = nullptr;
    const std::vector<float>*    oop_reach     = nullptr;

    /// Resolved node locks: (node_idx, canonical_combo) → forced strategy.
    const std::map<std::pair<uint32_t, uint16_t>, std::vector<float>>*
        resolved_locks = nullptr;
};

// ============================================================================
// Backend interface
// ============================================================================

class ISolverBackend {
public:
    virtual ~ISolverBackend() = default;

    /// One-time initialization: allocate solver state (regrets, strategy_sum,
    /// current_strategy). GPU implementations also upload tree + reach +
    /// matchup matrix to device memory.
    ///
    /// Must be called before any iterate() or finalize() call.
    virtual void prepare(const SolverContext& ctx) = 0;

    /// Run ONE DCFR iteration. Expected sequence:
    ///   1. Compute current_strategy from regrets (regret matching)
    ///   2. Apply DCFR discount to regrets
    ///   3. Traverse for OOP — accumulate OOP regrets AND OOP strategy_sum
    ///   4. Traverse for IP  — accumulate IP  regrets AND IP  strategy_sum
    ///
    /// @param iteration 0-based iteration index (for DCFR weight formulas).
    virtual void iterate(int iteration) = 0;

    /// Finalize the solve: normalize strategy_sum → averaged strategy.
    /// For GPU backends this also downloads the result to host memory.
    /// After this call, strategy() returns valid data.
    virtual void finalize() = 0;

    /// Averaged final strategy per node, per (action, canonical_combo).
    /// Layout: strategy[node_idx][action * num_canonical + canonical_combo]
    ///         ∈ [0, 1], rows sum to 1 over actions for each combo.
    virtual const std::vector<std::vector<float>>& strategy() const = 0;

    /// Human-readable backend name, e.g. "CPU-DCFR", "CUDA (RTX 4060, 8GB)".
    /// Shown in UI as the active backend indicator.
    virtual const char* name() const = 0;

    // ------------------------------------------------------------------------
    // Optional GPU postsolve hooks
    // ------------------------------------------------------------------------
    //
    // Backends that can reuse their device-resident state (tree, matchup
    // tables, averaged strategy, root reach) to compute per-combo EV and
    // best-response values on-device override these. The CPU backend leaves
    // the defaults — the Solver orchestrator falls back to its own
    // CPU parallel postsolve loop in that case.
    //
    // Vectors are returned in canonical-combo order, length = num_canonical.
    // Empty return means "not supported / fall back to CPU"; callers must
    // be prepared for that.

    /// Returns true if compute_combo_evs_gpu() / compute_best_response_gpu()
    /// produce meaningful results. Solver checks this before dispatching.
    virtual bool supports_gpu_postsolve() const { return false; }

    /// OOP-perspective per-combo EV at root, before the 1/total_ip_weight
    /// scaling applied by Solver::compute_combo_evs(). Empty on backends
    /// that don't support GPU postsolve.
    virtual std::vector<float> compute_combo_evs_gpu() { return {}; }

    /// Per-combo best-response value at root for the given player
    /// (0 = OOP, 1 = IP). Empty on backends that don't support GPU postsolve.
    virtual std::vector<float> compute_best_response_gpu(int /*player*/) { return {}; }
};

// ============================================================================
// Backend selection
// ============================================================================

enum class BackendType {
    AUTO = 0,   ///< Prefer GPU, fall back to CPU if no CUDA device.
    CPU  = 1,   ///< Force CPU backend.
    GPU  = 2,   ///< Force GPU backend (throws if unavailable).
};

/// System-level flag: GPU backend is functional as of Phase 4.9, with
/// Phase 2 (iso runout enumeration) parity restored. Validated:
///   - 3-of-suit turn solve: GPU matches CPU within 1-3% per-combo and
///     0.1% global (DCFR FP32 oscillation), and is ~6.5x faster.
///   - Rainbow turn solve: GPU matches CPU and is ~10.4x faster.
/// AUTO mode prefers GPU when CUDA hardware (CC 7.0+) is present.
constexpr bool GPU_BACKEND_FUNCTIONAL = true;

/// Parse a backend type from a string ("auto" | "cpu" | "gpu"). Case-insensitive.
/// Defaults to AUTO on unrecognized input.
BackendType parse_backend_type(const std::string& s);

/// Returns true if a CUDA-capable GPU (compute capability 7.0+) is available.
/// Safe to call even on builds without CUDA support (returns false).
bool has_cuda_gpu();

/// Returns a human-readable description of the detected GPU, or empty string
/// if no GPU is available. E.g. "NVIDIA GeForce RTX 4060 (8192 MB, CC 8.9)".
std::string describe_cuda_gpu();

/// Factory: construct a backend of the requested type.
///
/// - BackendType::AUTO: returns GPU if available, else CPU.
/// - BackendType::CPU:  always returns CpuBackend.
/// - BackendType::GPU:  returns GpuBackend or nullptr if unavailable.
///
/// Never throws. Check the return value is non-null before use.
std::unique_ptr<ISolverBackend> create_backend(BackendType type);

} // namespace deepsolver

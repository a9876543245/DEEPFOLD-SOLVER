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
/// Shared CPU-thread resolution. Returns the effective worker count for a
/// `config.cpu_threads` request:
///   0 → auto: hardware concurrency (typically every core/HT)
///   1 → serial: forced single thread
///   N → clamp to [1, hw_concurrency]
///
/// Used both by the levelized backend's prepare() (to pin its OMP team
/// size) and by Solver::estimate_only() (so the pre-solve resource preview
/// reports the same number the backend will actually use). Keeping it in
/// one place avoids "estimate said 4, backend ran with 8" drift.
inline uint32_t resolve_cpu_threads(uint32_t requested, uint32_t hw_concurrency) {
    if (hw_concurrency == 0u) hw_concurrency = 1u;
    if (requested == 0u) return hw_concurrency;
    if (requested == 1u) return 1u;
    return (requested < hw_concurrency) ? requested : hw_concurrency;
}

struct SolverContext {
    const FlatGameTree*          tree          = nullptr;
    const IsomorphismMapping*    iso           = nullptr;
    const SolverConfig*          config        = nullptr;

    /// Precomputed showdown matchup matrix for the ROOT board. Indexed
    /// [ci * nc + cj]. Kept for backward-compat / shortcut when no chance
    /// node is in the tree (e.g. solving a river spot).
    const std::vector<float>*    matchup_ev    = nullptr;
    const std::vector<float>*    matchup_valid = nullptr;
    /// Per-original-combo hand ranks for the root matchup board. Entries for
    /// combos dead on the board are UINT16_MAX. Dense singleton showdown
    /// shortcuts use this to avoid scanning the full category/valid matrix.
    const std::vector<uint16_t>* matchup_original_ranks = nullptr;
    /// v1.8.2 A2 encoding (POST_OPTIMIZATION_REVIEW Sec 4.3): pre-thresholded
    /// per-cell category in {0=invalid, 1=win, 2=lose, 3=tie}. Built once at
    /// precompute time so the showdown hot-loop reads 1 byte/cell instead of
    /// re-thresholding ev_f32 (4 bytes/cell). Bench: ~1.97x on cold-cache
    /// regimes (monotone-shape workloads), neutral on hot.
    /// `matchup_ev`/`matchup_valid` are still kept around for postsolve
    /// (combo_evs / exploitability), which uses the full continuous ev.
    const std::vector<uint8_t>*  matchup_category = nullptr;
    /// Zero-rake showdown coefficient matrix: +valid for OOP-win, -valid
    /// for OOP-lose, 0 for tie/invalid. Present only for rake-free solves.
    /// CPU showdown can use this to replace category decode + valid multiply
    /// with a single signed dot product. Postsolve still uses matchup_ev.
    const std::vector<float>*    matchup_showdown_coeff = nullptr;
    /// Equivalent zero-rake coefficient encoded as signed valid pair counts.
    /// For canonical buckets (i,j), signed_count = +/- valid_original_pairs.
    /// During terminal evaluation the opponent canonical weight cancels one
    /// denominator, so kernels multiply by the unweighted opponent reach and
    /// the output row's inverse canonical weight.
    const std::vector<int8_t>*    matchup_showdown_count = nullptr;
    /// Phase 1 chance enumeration: per-runout matchup tables. The terminal
    /// handler should pick the table by tree.matchup_idx[node_idx].
    /// matchup_ev_per_runout[k] is the matchup-ev table for runout k.
    const std::vector<std::vector<float>>* matchup_ev_per_runout    = nullptr;
    const std::vector<std::vector<float>>* matchup_valid_per_runout = nullptr;
    const std::vector<std::vector<uint8_t>>* matchup_category_per_runout = nullptr;
    const std::vector<std::vector<float>>* matchup_showdown_coeff_per_runout = nullptr;
    const std::vector<std::vector<int8_t>>* matchup_showdown_count_per_runout = nullptr;
    const std::vector<std::vector<uint16_t>>* matchup_original_ranks_per_runout = nullptr;
    /// Full-board mask for each matchup table. Fold terminal shortcuts use
    /// this to apply turn/river dead-card filtering without reading the dense
    /// matchup_valid matrix.
    const std::vector<CardMask>* matchup_board_masks = nullptr;

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

    /// Effective worker count for diagnostics. CPU backends should resolve
    /// `config.cpu_threads` (0=auto, 1=serial, N=clamped) inside prepare()
    /// and return that here. Reported in `SolveResources.cpu_threads_effective`
    /// — important so users can verify their `--cpu-threads` cap actually
    /// took effect rather than silently falling back to `omp_get_max_threads()`.
    /// Default 0 means "unknown / use env"; the orchestrator falls back to
    /// the legacy heuristic in that case (e.g. for GPU backends).
    virtual uint32_t cpu_threads_effective() const { return 0; }

    // v1.8.1+ per-iteration-phase timing in ms (cumulative across the
    // whole solve). Default 0.0 — only the levelized CPU backend
    // currently instruments these. Solver pulls them into SolverTiming
    // after iterate() loop completes.
    virtual double phase_compute_strategy_ms()  const { return 0.0; }
    virtual double phase_apply_discount_ms()    const { return 0.0; }
    virtual double phase_forward_pass_ms()      const { return 0.0; }
    virtual double phase_backward_pass_oop_ms() const { return 0.0; }
    virtual double phase_backward_pass_ip_ms()  const { return 0.0; }
    // CPU-seconds (sum across threads) inside terminal evaluation, split by
    // terminal type. Diagnostic only — used to decide whether to attack
    // showdown matrix kernels or fold accumulation next.
    virtual double phase_backward_showdown_ms() const { return 0.0; }
    virtual double phase_backward_fold_ms()     const { return 0.0; }

    // CPU-only sparse traversal / terminal gate diagnostics. GPU backends and
    // backends that do not expose these choices return an unavailable block.
    virtual CpuBackendDiagnostics cpu_diagnostics() const { return {}; }

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

/// Returns true if a CUDA-capable GPU is available AND its compute
/// capability is at or above the binary's minimum SASS arch. The threshold
/// is set in backend_factory.cpp::kMinCudaMajor/kMinCudaMinor and
/// currently matches the CUDA toolkit floor (13.x → CC 7.5 / Turing).
/// Safe to call on builds without CUDA support (returns false).
bool has_cuda_gpu();

/// Returns a human-readable description of the detected GPU, or empty string
/// if no GPU is available. E.g. "NVIDIA GeForce RTX 4060 (8192 MB, CC 8.9)".
std::string describe_cuda_gpu();

/// Returns the reason the CUDA backend was rejected, or empty string when
/// the GPU IS usable. Reasons:
///   - "No CUDA-capable GPU detected." (no device)
///   - "GPU '<name>' has compute capability X.Y, below required N.M ..."
///     For Pascal/Volta hardware the message specifically calls out that
///     these archs need a CUDA-12.x build (planned for v1.3.0).
///   - "Built without CUDA support." (CPU-only build)
///   - Driver query failures with the cudaError_t string
/// Used to populate resources.fallback_reason in solve(), so the UI can
/// distinguish "no GPU at all" from "GPU detected but excluded".
std::string cuda_gpu_reject_reason();

/// Returns CUDA compute capability of device 0 as major*10 + minor
/// (Ada=89, Pascal=61, Blackwell=120). Returns 0 when no CUDA device.
/// v1.2.2: used by ETA estimator since Pascal is ~10× slower than Ada at
/// fp32+atomic CFR workloads.
int cuda_compute_capability();

/// Factory: construct a backend of the requested type.
///
/// - BackendType::AUTO: returns GPU if available, else CPU.
/// - BackendType::CPU:  always returns CpuBackend.
/// - BackendType::GPU:  returns GpuBackend or nullptr if unavailable.
///
/// Never throws. Check the return value is non-null before use.
std::unique_ptr<ISolverBackend> create_backend(BackendType type);

} // namespace deepsolver

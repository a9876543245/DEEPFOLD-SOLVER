/**
 * @file backend_factory.cpp
 * @brief Backend selection + GPU detection implementation.
 *
 * Kept in a .cpp (not header) because CUDA runtime API calls depend on
 * whether DEEPSOLVER_USE_CUDA is defined at compile time.
 */

#include "solver_backend.h"
#include "cpu_backend.h"

#ifdef DEEPSOLVER_USE_CUDA
  #include "gpu_backend.h"
  #include <cuda_runtime.h>
  #include <sstream>
#endif

#include <algorithm>
#include <cctype>
#include <memory>
#include <string>

namespace deepsolver {

// ============================================================================
// BackendType parsing
// ============================================================================

BackendType parse_backend_type(const std::string& s) {
    std::string lower;
    lower.reserve(s.size());
    for (char c : s) lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));

    if (lower == "cpu")  return BackendType::CPU;
    if (lower == "gpu" || lower == "cuda") return BackendType::GPU;
    return BackendType::AUTO;
}

// ============================================================================
// CUDA detection
// ============================================================================

#ifdef DEEPSOLVER_USE_CUDA

// Minimum CUDA compute capability the CFR kernels need.
//
// In principle the kernels work on Pascal (6.1) — they only need basic fp32
// + atomicAdd + warp shuffle, no Tensor Cores. BUT the bundled binary's
// minimum SASS arch is whatever CUDA toolkit we build with: CUDA 13.x
// dropped Pascal (sm_61) and Volta (sm_70) entirely, so the released
// installer's floor is Turing (sm_75 → CC 7.5).
//
// Keeping this gate aligned with the SASS list (NOT with what the hardware
// could theoretically run) avoids the failure mode where we'd say "GPU
// available, switching to GPU" and then crash on first kernel launch with
// CUDA_ERROR_NO_BINARY_FOR_GPU.
//
// v1.3.0+ may ship a separate Pascal/Volta-friendly build via CUDA 12.x.
// When that lands, lower this floor to 6.0 to match.
constexpr int kMinCudaMajor = 7;
constexpr int kMinCudaMinor = 5;

bool has_cuda_gpu() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess || count == 0) return false;

    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) return false;

    return (prop.major > kMinCudaMajor) ||
           (prop.major == kMinCudaMajor && prop.minor >= kMinCudaMinor);
}

std::string describe_cuda_gpu() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return "";

    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) return "";

    std::ostringstream oss;
    oss << prop.name
        << " (" << (prop.totalGlobalMem / (1024ull * 1024ull)) << " MB, "
        << "CC " << prop.major << "." << prop.minor << ")";
    return oss.str();
}

// Returns CUDA compute capability of device 0 as `major*10 + minor`
// (e.g. 89 for Ada, 61 for Pascal, 120 for Blackwell). Returns 0 when no
// CUDA device is detected. Used by solve-time estimator to pick the right
// per-arch throughput rate (Pascal is ~10× slower than Ada at fp32+atomic).
int cuda_compute_capability() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return 0;
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) return 0;
    return prop.major * 10 + prop.minor;
}

// Why was the CUDA backend skipped? Returns an empty string when the GPU
// IS usable (caller should not surface anything in that case). Otherwise
// returns a short human-readable reason — used to populate
// resources.fallback_reason when AUTO downgrades to CPU, so users know
// the difference between "no GPU" and "GPU detected but excluded".
std::string cuda_gpu_reject_reason() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        return std::string("CUDA driver query failed: ") + cudaGetErrorString(err);
    }
    if (count == 0) {
        return "No CUDA-capable GPU detected.";
    }
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
        return "CUDA device 0 detected but properties query failed.";
    }
    const bool ok = (prop.major > kMinCudaMajor) ||
                    (prop.major == kMinCudaMajor && prop.minor >= kMinCudaMinor);
    if (ok) return "";

    std::ostringstream oss;
    oss << "GPU '" << prop.name << "' has compute capability "
        << prop.major << "." << prop.minor << ", below required "
        << kMinCudaMajor << "." << kMinCudaMinor << ". ";
    // Pascal (CC 6.x) and Volta (CC 7.0) hardware can run the kernels in
    // principle — the limit comes from the CUDA toolkit version we built
    // with (13.x dropped these archs). Steer those users to the right
    // place instead of just saying "below required".
    if (prop.major == 6 || (prop.major == 7 && prop.minor == 0)) {
        oss << "Pascal/Volta GPUs need a CUDA-12.x build (planned for "
            << "v1.3.0); current bundled binary is CUDA-13.x. Falling back "
            << "to CPU for now.";
    } else {
        oss << "Falling back to CPU.";
    }
    return oss.str();
}

#else  // Build without CUDA — GPU path unavailable

bool has_cuda_gpu() { return false; }
std::string describe_cuda_gpu() { return ""; }
std::string cuda_gpu_reject_reason() { return "Built without CUDA support."; }
int cuda_compute_capability() { return 0; }

#endif

// ============================================================================
// Factory
// ============================================================================

std::unique_ptr<ISolverBackend> create_backend(BackendType type) {
    switch (type) {
        case BackendType::GPU:
#ifdef DEEPSOLVER_USE_CUDA
            if (has_cuda_gpu()) {
                // Explicit GPU request: honor it even if skeleton (user sees throw).
                // Real kernels land in Phase 4.
                return std::make_unique<GpuBackend>();
            }
#endif
            return nullptr;  // caller handles explicit-GPU unavailable

        case BackendType::AUTO:
#ifdef DEEPSOLVER_USE_CUDA
            // Only auto-pick GPU once it's validated (Phase 5). Until then,
            // AUTO stays on CPU even if CUDA hardware is present so users
            // don't get crashes from an incomplete GPU path.
            if (has_cuda_gpu() && GPU_BACKEND_FUNCTIONAL) {
                return std::make_unique<GpuBackend>();
            }
#endif
            return std::make_unique<CpuBackend>();

        case BackendType::CPU:
        default:
            return std::make_unique<CpuBackend>();
    }
}

} // namespace deepsolver

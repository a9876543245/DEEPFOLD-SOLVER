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

bool has_cuda_gpu() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess || count == 0) return false;

    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) return false;

    // Require compute capability 7.0+ (Volta / Turing / Ampere / Ada / Blackwell)
    return prop.major >= 7;
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

#else  // Build without CUDA — GPU path unavailable

bool has_cuda_gpu() { return false; }
std::string describe_cuda_gpu() { return ""; }

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

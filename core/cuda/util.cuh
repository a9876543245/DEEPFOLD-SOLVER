/**
 * @file util.cuh
 * @brief CUDA utility helpers, half-precision operations, error checking.
 */

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

// ============================================================================
// Error Checking — throws CudaError instead of exit()
// ============================================================================
//
// Phase 4 of the 10-point maturity plan: CUDA failures must be recoverable.
// Calling exit(EXIT_FAILURE) skips all destructors, leaks device memory,
// and prevents the CLI's structured JSON error path (and the Tauri/Rust
// CPU-fallback detector) from running.
//
// Now: every cudaError_t failure throws a CudaError. main.cpp catches
// std::exception and emits {"status":"error","message":"..."} on stderr.
// Rust's engine.rs scans that message for "CUDA"/"out of memory" to decide
// whether to retry on CPU.
//
// Non-zero return codes inside __device__ code are not affected — these
// macros only run on the host side (cudaMalloc / cudaMemcpy / kernel launch
// errors) where exception unwinding is well-defined.

namespace deepsolver {
namespace gpu {

class CudaError : public std::runtime_error {
public:
    CudaError(const char* file, int line, cudaError_t err)
        : std::runtime_error(build_message(file, line, err)),
          file_(file), line_(line), code_(err) {}

    cudaError_t code() const noexcept { return code_; }
    const char* file() const noexcept { return file_; }
    int         line() const noexcept { return line_; }

private:
    static std::string build_message(const char* file, int line, cudaError_t err) {
        // Keep the literal substrings "CUDA" and the cudaGetErrorString text so
        // Rust's GPU-error sniffer (engine.rs:looks_like_gpu_err) and humans
        // can see what happened. cudaGetErrorString includes "out of memory"
        // for cudaErrorMemoryAllocation, which the Rust side also matches on.
        std::string msg = "CUDA error at ";
        msg += file ? file : "<unknown>";
        msg += ":";
        msg += std::to_string(line);
        msg += ": ";
        const char* es = cudaGetErrorString(err);
        msg += es ? es : "<no error string>";
        msg += " (code=";
        msg += std::to_string(static_cast<int>(err));
        msg += ")";
        return msg;
    }

    const char* file_;
    int         line_;
    cudaError_t code_;
};

} // namespace gpu
} // namespace deepsolver

#define CUDA_CHECK(call)                                                        \
    do {                                                                         \
        cudaError_t _cuda_err = (call);                                          \
        if (_cuda_err != cudaSuccess) {                                          \
            throw ::deepsolver::gpu::CudaError(__FILE__, __LINE__, _cuda_err);   \
        }                                                                        \
    } while (0)

// ============================================================================
// Half-precision helpers
// ============================================================================

namespace deepsolver {
namespace gpu {

/// Clamp a half value to [0, max_val]
__device__ __forceinline__ __half hclamp_pos(__half x) {
    return __hmax(x, __float2half(0.0f));
}

/// Half to float
__device__ __forceinline__ float h2f(__half x) {
    return __half2float(x);
}

/// Float to half
__device__ __forceinline__ __half f2h(float x) {
    return __float2half(x);
}

/// Safe division (returns 0 if denominator is 0)
__device__ __forceinline__ __half hdiv_safe(__half num, __half den) {
    if (__heq(den, __float2half(0.0f))) return __float2half(0.0f);
    return __hdiv(num, den);
}

/// Regret matching: compute strategy from positive regrets
/// Input: regrets[] of length num_actions
/// Output: strategy[] normalized probabilities
__device__ __forceinline__ void regret_matching(__half* regrets, __half* strategy, int num_actions) {
    __half sum = __float2half(0.0f);
    for (int a = 0; a < num_actions; ++a) {
        __half pos = hclamp_pos(regrets[a]);
        strategy[a] = pos;
        sum = __hadd(sum, pos);
    }

    if (__hgt(sum, __float2half(0.0f))) {
        for (int a = 0; a < num_actions; ++a) {
            strategy[a] = __hdiv(strategy[a], sum);
        }
    } else {
        // Uniform strategy if all regrets are non-positive
        __half uniform = __float2half(1.0f / static_cast<float>(num_actions));
        for (int a = 0; a < num_actions; ++a) {
            strategy[a] = uniform;
        }
    }
}

/// DCFR discount factor for positive regrets: (t/(t+1))^alpha
__device__ __forceinline__ float dcfr_pos_discount(int t, float alpha) {
    float ratio = static_cast<float>(t) / static_cast<float>(t + 1);
    return powf(ratio, alpha);
}

/// DCFR discount factor for negative regrets: (t/(t+1))^beta
__device__ __forceinline__ float dcfr_neg_discount(int t, float beta) {
    float ratio = static_cast<float>(t) / static_cast<float>(t + 1);
    return powf(ratio, beta);
}

/// DCFR strategy contribution weight: ((t+1)/(t+2))^gamma
__device__ __forceinline__ float dcfr_strat_weight(int t, float gamma) {
    float ratio = static_cast<float>(t + 1) / static_cast<float>(t + 2);
    return powf(ratio, gamma);
}

// ============================================================================
// Memory Management Helpers
// ============================================================================

/// Allocate and copy host array to device (half precision)
inline __half* upload_to_device_half(const float* host_data, size_t count) {
    __half* d_ptr;
    CUDA_CHECK(cudaMalloc(&d_ptr, count * sizeof(__half)));

    // Convert float → half on host, then upload
    std::vector<__half> h_data(count);
    for (size_t i = 0; i < count; ++i) {
        h_data[i] = __float2half(host_data[i]);
    }
    CUDA_CHECK(cudaMemcpy(d_ptr, h_data.data(), count * sizeof(__half),
                           cudaMemcpyHostToDevice));
    return d_ptr;
}

/// Download device half array to host floats
inline void download_from_device_half(float* host_data, const __half* d_ptr, size_t count) {
    std::vector<__half> h_data(count);
    CUDA_CHECK(cudaMemcpy(h_data.data(), d_ptr, count * sizeof(__half),
                           cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < count; ++i) {
        host_data[i] = __half2float(h_data[i]);
    }
}

/// Allocate and zero-fill device memory
template<typename T>
inline T* alloc_device_zeroed(size_t count) {
    T* d_ptr;
    CUDA_CHECK(cudaMalloc(&d_ptr, count * sizeof(T)));
    CUDA_CHECK(cudaMemset(d_ptr, 0, count * sizeof(T)));
    return d_ptr;
}

/// Free device memory
template<typename T>
inline void free_device(T*& ptr) {
    if (ptr) {
        CUDA_CHECK(cudaFree(ptr));
        ptr = nullptr;
    }
}

} // namespace gpu
} // namespace deepsolver

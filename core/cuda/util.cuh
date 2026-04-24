/**
 * @file util.cuh
 * @brief CUDA utility helpers, half-precision operations, error checking.
 */

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

// ============================================================================
// Error Checking Macro
// ============================================================================

#define CUDA_CHECK(call)                                                        \
    do {                                                                         \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA Error at %s:%d: %s\n",                       \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
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

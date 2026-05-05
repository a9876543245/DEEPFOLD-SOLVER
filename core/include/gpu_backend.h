/**
 * @file gpu_backend.h
 * @brief CUDA GPU implementation of ISolverBackend (declarations only).
 *
 * All device memory handling + CUDA runtime calls live in gpu_backend.cu,
 * hidden behind a pimpl so this header stays CUDA-free.
 *
 * Phase 4 progress:
 *   4.1 prepare() uploads tree + matchup matrix + reach probabilities → device.
 *        iterate() and finalize() still throw "not yet implemented".
 *   4.2 topological level sort of tree
 *   4.3 fixed showdown kernel + new fold terminal kernel
 *   4.4 reach probability forward propagation kernel
 *   4.5 action value backward pass
 *   4.6 node value aggregation
 *   4.7 wire up compute_strategy / update_regrets / update_strategy_sum kernels
 *   4.8 node lock device support
 *   4.9 result download — iterate() / finalize() become fully functional.
 */

#pragma once

#include "solver_backend.h"

#include <memory>
#include <string>
#include <vector>

namespace deepsolver {

class GpuBackend final : public ISolverBackend {
public:
    GpuBackend();
    ~GpuBackend() override;

    // Non-copyable (owns device memory via pimpl)
    GpuBackend(const GpuBackend&) = delete;
    GpuBackend& operator=(const GpuBackend&) = delete;

    void prepare(const SolverContext& ctx) override;
    void iterate(int iteration) override;
    void finalize() override;
    const std::vector<std::vector<float>>& strategy() const override { return strategy_; }
    const char* name() const override { return name_.c_str(); }

    // GPU postsolve fast paths. Both reuse the device buffers already
    // populated by iterate()+finalize() (averaged strategy lives in
    // current_strategy after finalize, root reach in reach.{oop,ip}_reach).
    //   compute_combo_evs_gpu():    OOP-perspective per-combo EV at root,
    //                               returned in canonical-combo order, length nc.
    //                               Caller applies the same 1/total_ip_weight
    //                               scaling that compute_combo_evs() does on CPU.
    //   compute_best_response_gpu(player): per-combo BR value at root for
    //                               player ∈ {0=OOP, 1=IP}, length nc.
    bool supports_gpu_postsolve() const override { return true; }
    std::vector<float> compute_combo_evs_gpu() override;
    std::vector<float> compute_best_response_gpu(int player) override;

private:
    // pimpl hides CUDA types from this header
    struct Impl;
    std::unique_ptr<Impl> impl_;

    std::string name_;
    std::vector<std::vector<float>> strategy_;
};

} // namespace deepsolver

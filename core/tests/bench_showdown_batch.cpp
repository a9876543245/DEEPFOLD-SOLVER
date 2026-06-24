#include "cpu_simd.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

using deepsolver::cpu_simd::avx2_kernels;

namespace {

constexpr std::size_t kNc = 1326;

struct CaseData {
    std::vector<uint8_t> cat;
    std::vector<float> valid;
    std::vector<std::vector<float>> reach;
    std::vector<std::vector<float>> out_per;
    std::vector<std::vector<float>> out_batch;
    std::vector<const float*> reach_ptrs;
    std::vector<float*> out_ptrs;
    std::vector<float> win;
    std::vector<float> lose;
    std::vector<float> tie;
};

static CaseData make_case(std::size_t m) {
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist01(0.0f, 1.0f);
    std::uniform_real_distribution<float> dist_pay(-80.0f, 120.0f);

    CaseData d;
    d.cat.assign(kNc * kNc, 0);
    d.valid.assign(kNc * kNc, 0.0f);
    for (std::size_t i = 0; i < kNc * kNc; ++i) {
        const float v = dist01(rng);
        d.valid[i] = (v < 0.05f) ? 0.0f : v;
        if (d.valid[i] == 0.0f) {
            d.cat[i] = 0;
        } else {
            const float bucket = dist01(rng);
            d.cat[i] = (bucket < 0.34f) ? 1 : (bucket < 0.67f) ? 2 : 3;
        }
    }

    d.reach.resize(m);
    d.out_per.resize(m);
    d.out_batch.resize(m);
    d.reach_ptrs.resize(m);
    d.out_ptrs.resize(m);
    d.win.resize(m);
    d.lose.resize(m);
    d.tie.resize(m);
    for (std::size_t t = 0; t < m; ++t) {
        d.reach[t].resize(kNc);
        d.out_per[t].assign(kNc, 0.0f);
        d.out_batch[t].assign(kNc, 0.0f);
        for (float& x : d.reach[t]) x = dist01(rng);
        d.reach_ptrs[t] = d.reach[t].data();
        d.out_ptrs[t] = d.out_batch[t].data();
        d.win[t] = dist_pay(rng);
        d.lose[t] = dist_pay(rng);
        d.tie[t] = dist_pay(rng) * 0.1f;
    }
    return d;
}

template <typename Fn>
static double time_seconds(int repeats, Fn&& fn) {
    const auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < repeats; ++r) fn();
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(t1 - t0).count();
}

static double max_diff(const CaseData& d) {
    double md = 0.0;
    for (std::size_t t = 0; t < d.out_per.size(); ++t) {
        for (std::size_t i = 0; i < kNc; ++i) {
            md = std::max(md, static_cast<double>(
                std::fabs(d.out_per[t][i] - d.out_batch[t][i])));
        }
    }
    return md;
}

static void run_case(std::size_t m, int repeats) {
    CaseData d = make_case(m);

    const double per_s = time_seconds(repeats, [&] {
        for (std::size_t t = 0; t < m; ++t) {
            avx2_kernels.showdown_oop_full(
                d.cat.data(), d.valid.data(), d.reach[t].data(), nullptr,
                d.out_per[t].data(), kNc, d.win[t], d.lose[t], d.tie[t]);
        }
    });

    const double batch_s = time_seconds(repeats, [&] {
        avx2_kernels.showdown_oop_full_batch(
            d.cat.data(), d.valid.data(), kNc, m,
            d.reach_ptrs.data(), nullptr, d.out_ptrs.data(),
            d.win.data(), d.lose.data(), d.tie.data(), 0, kNc);
    });

    std::cout << "M=" << std::setw(3) << m
              << "  per-call=" << std::setw(8) << std::fixed << std::setprecision(3)
              << per_s * 1000.0
              << " ms  batch=" << std::setw(8) << batch_s * 1000.0
              << " ms  speedup=" << std::setprecision(3) << (per_s / batch_s)
              << "x  max_diff=" << std::scientific << max_diff(d) << "\n";
}

}  // namespace

int main() {
    std::cout << "=== showdown batch micro-bench (AVX2, nc=" << kNc << ") ===\n";
    run_case(1, 8);
    run_case(4, 8);
    run_case(8, 6);
    run_case(32, 3);
    run_case(128, 1);
    return 0;
}

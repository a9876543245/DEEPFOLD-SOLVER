/**
 * @file cpu_kernels_dispatch.cpp
 * @brief Pick AVX2 vs scalar kernels at runtime via CPUID.
 *
 * The detection is done lazily on first call to `kernels()`. The result is
 * cached in a function-local static so subsequent calls are a single load.
 *
 * What we check (per Intel SDM Vol. 2, CPUID leaves):
 *
 *   1. CPUID leaf 1, ECX bit 28 — AVX (basic 256-bit float support)
 *   2. CPUID leaf 1, ECX bit 27 — OSXSAVE (OS saves the YMM state on context switch)
 *   3. CPUID leaf 1, ECX bit 12 — FMA (fused multiply-add)
 *   4. CPUID leaf 7 sub-leaf 0, EBX bit 5 — AVX2
 *   5. XGETBV with XCR0, bits 1+2 set — OS has actually enabled XMM+YMM saving
 *
 * Step 5 is the gotcha that catches people: a hypervisor or weird OS can
 * report CPUID=AVX-supported but not actually save YMM on context switch,
 * which corrupts AVX state across threads. The XGETBV check is the
 * authoritative "yes you can really use AVX2" signal.
 */

#include "cpu_simd.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>

#if defined(_MSC_VER)
#  include <intrin.h>      // __cpuid, __cpuidex, _xgetbv
#  include <immintrin.h>   // _XCR_XFEATURE_ENABLED_MASK
#elif defined(__GNUC__) || defined(__clang__)
#  include <cpuid.h>
#endif

namespace deepsolver::cpu_simd {

namespace {

inline void cpuid_call(int leaf, int subleaf, int regs[4]) {
#if defined(_MSC_VER)
    __cpuidex(regs, leaf, subleaf);
#elif defined(__GNUC__) || defined(__clang__)
    unsigned int a, b, c, d;
    __cpuid_count(leaf, subleaf,
                  a, b, c, d);
    regs[0] = static_cast<int>(a);
    regs[1] = static_cast<int>(b);
    regs[2] = static_cast<int>(c);
    regs[3] = static_cast<int>(d);
#else
    regs[0] = regs[1] = regs[2] = regs[3] = 0;
#endif
}

inline unsigned long long xgetbv_call(unsigned int xcr) {
#if defined(_MSC_VER)
    return _xgetbv(xcr);
#elif defined(__GNUC__) || defined(__clang__)
    unsigned int eax, edx;
    __asm__ volatile (".byte 0x0f, 0x01, 0xd0"
                      : "=a"(eax), "=d"(edx)
                      : "c"(xcr));
    return (static_cast<unsigned long long>(edx) << 32) | eax;
#else
    (void)xcr;
    return 0;
#endif
}

bool detect_avx2_support() {
    int regs[4] = {0, 0, 0, 0};

    cpuid_call(0, 0, regs);
    int max_leaf = regs[0];
    if (max_leaf < 7) return false;

    cpuid_call(1, 0, regs);
    bool has_avx     = (regs[2] & (1 << 28)) != 0;
    bool has_osxsave = (regs[2] & (1 << 27)) != 0;
    bool has_fma     = (regs[2] & (1 << 12)) != 0;
    if (!has_avx || !has_osxsave || !has_fma) return false;

    cpuid_call(7, 0, regs);
    bool has_avx2 = (regs[1] & (1 << 5)) != 0;
    if (!has_avx2) return false;

    // OS must have YMM state save enabled — XCR0 bits 1 (XMM) + 2 (YMM).
    constexpr unsigned int xcr_xfeature = 0;  // _XCR_XFEATURE_ENABLED_MASK
    unsigned long long xcr0 = xgetbv_call(xcr_xfeature);
    constexpr unsigned long long xmm_ymm_mask = 0x6;
    if ((xcr0 & xmm_ymm_mask) != xmm_ymm_mask) return false;

    return true;
}

std::atomic<SimdPolicy> g_policy{SimdPolicy::Auto};
std::atomic<const Kernels*> g_kernels_cached{nullptr};

const Kernels& resolve_kernels() {
    SimdPolicy pol = g_policy.load(std::memory_order_acquire);
    bool hw_avx2 = detect_avx2_support();

    switch (pol) {
        case SimdPolicy::ForceScalar:
            return scalar_kernels;
        case SimdPolicy::ForceAvx2:
            if (!hw_avx2) {
                std::fprintf(stderr,
                    "[CPU] --cpu-simd avx2 requested but CPUID reports no "
                    "AVX2/FMA/OS-YMM support. Aborting.\n");
                std::exit(2);
            }
            return avx2_kernels;
        case SimdPolicy::Auto:
        default:
            return hw_avx2 ? avx2_kernels : scalar_kernels;
    }
}

}  // namespace

void set_policy(SimdPolicy policy) {
    g_policy.store(policy, std::memory_order_release);
    // Invalidate cached selection so next kernels() call re-resolves.
    g_kernels_cached.store(nullptr, std::memory_order_release);
}

const Kernels& kernels() {
    const Kernels* cached = g_kernels_cached.load(std::memory_order_acquire);
    if (cached) return *cached;
    const Kernels& fresh = resolve_kernels();
    g_kernels_cached.store(&fresh, std::memory_order_release);
    return fresh;
}

SimdMode active_mode() {
    return (&kernels() == &avx2_kernels) ? SimdMode::Avx2 : SimdMode::Scalar;
}

const char* mode_label() {
    return (active_mode() == SimdMode::Avx2) ? "avx2" : "scalar";
}

bool cpuid_supports_avx2_fma_os() {
    return detect_avx2_support();
}

}  // namespace deepsolver::cpu_simd

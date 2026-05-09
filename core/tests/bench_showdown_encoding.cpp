/**
 * @file bench_showdown_encoding.cpp
 * @brief Standalone micro-benchmark for showdown matrix-encoding alternatives.
 *
 * Goal: pick the right encoding for the showdown_oop_full matrix workload
 * before doing 71-touchpoint surgery on the production matchup pipeline.
 *
 * Candidates measured (all on the same synthetic-but-realistic matchup data):
 *   baseline — ev_f32 + valid_f32, 8 bytes/cell. Mirrors the production AVX2
 *              kernel in cpu_kernels_avx2.cpp.
 *   A2       — category_u8 + valid_f32, 5 bytes/cell. category encodes
 *              {0=invalid, 1=win, 2=lose, 3=tie} pre-thresholded at precompute.
 *              Loses no precision; valid stays exact float.
 *
 * For each candidate we report:
 *   ms / call          — wall time per terminal call
 *   GB/s effective     — bytes streamed through the kernel per second
 *   ratio vs baseline  — speedup factor (>1.0 means candidate is faster)
 *
 * Methodology:
 *   - nc = 1326 to mirror production canonical-combo count.
 *   - Synthetic data: ev uniform [-1, 1], valid uniform [0, 1] with ~5% set
 *     to 0 (matching the pair-conflict rate on a 3-card flop).
 *   - Two regimes:
 *       hot  — same matchup table reused for every call (best case for L3).
 *              Tells us whether we're compute-bound for a hot table.
 *       cold — N independent tables cycled through. With nc^2 cells per table
 *              (~7 MB at f32, ~1.7 MB at u8+f32 mixed), more tables push past
 *              L3 capacity and force DRAM reads. Tells us whether bandwidth
 *              actually dominates in production-like patterns.
 *   - 1T and 8T runs. 8T is what users see; 1T isolates per-thread cost.
 *
 * Build target: `bench_showdown_encoding`. NOT a ctest — run manually:
 *   cmake --build .\core\build_cpu --config Release --target bench_showdown_encoding
 *   .\core\build_cpu\Release\bench_showdown_encoding.exe
 */

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <thread>
#include <vector>

#include <immintrin.h>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

constexpr std::size_t kNc = 1326;          // canonical combos
constexpr std::size_t kCells = kNc * kNc;  // matrix cells

// ---------------------------------------------------------------------------
// hsum256 — copied from cpu_kernels_avx2.cpp so the micro-bench matches the
// production reduction tail.
// ---------------------------------------------------------------------------
static inline float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s  = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

// ---------------------------------------------------------------------------
// Baseline kernel — bit-identical to the production showdown_oop_full body.
// Skip mask omitted: we measure the dense path (no narrow-range short-circuit)
// because that's where the bandwidth bottleneck lives.
// ---------------------------------------------------------------------------
static void showdown_baseline(
    const float* ev_matrix, const float* valid_matrix,
    const float* opp_reach_w,
    float* out, std::size_t n,
    float win_p, float lose_p, float tie_p)
{
    __m256 vwin  = _mm256_set1_ps(win_p);
    __m256 vlose = _mm256_set1_ps(lose_p);
    __m256 vtie  = _mm256_set1_ps(tie_p);
    __m256 v05   = _mm256_set1_ps(0.5f);
    __m256 vn05  = _mm256_set1_ps(-0.5f);

    for (std::size_t c = 0; c < n; ++c) {
        const float* ev_row    = ev_matrix    + c * n;
        const float* valid_row = valid_matrix + c * n;
        __m256 acc = _mm256_setzero_ps();
        std::size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            __m256 ev    = _mm256_loadu_ps(ev_row + i);
            __m256 valid = _mm256_loadu_ps(valid_row + i);
            __m256 reach = _mm256_loadu_ps(opp_reach_w + i);
            __m256 m_win  = _mm256_cmp_ps(ev, v05,  _CMP_GT_OS);
            __m256 m_lose = _mm256_cmp_ps(ev, vn05, _CMP_LT_OS);
            __m256 payoff = _mm256_blendv_ps(vtie, vwin, m_win);
            payoff        = _mm256_blendv_ps(payoff, vlose, m_lose);
            __m256 rv     = _mm256_mul_ps(reach, valid);
            acc           = _mm256_fmadd_ps(rv, payoff, acc);
        }
        float sum = hsum256(acc);
        for (; i < n; ++i) {
            float ev = ev_row[i];
            float valid = valid_row[i];
            if (valid <= 0.0f) continue;
            float p;
            if (ev >  0.5f) p = win_p;
            else if (ev < -0.5f) p = lose_p;
            else                 p = tie_p;
            sum += opp_reach_w[i] * valid * p;
        }
        out[c] = sum;
    }
}

// ---------------------------------------------------------------------------
// A2 kernel — category_u8 + valid_f32. category is precomputed at table-build
// time from ev's threshold bucket; valid stays exact float.
//
// Per cell: 1 byte (category) + 4 bytes (valid) = 5 bytes vs baseline's 8.
// 37.5% bandwidth reduction.
//
// Inner loop:
//   load 8 bytes of category (1 cache line gets you 64 cells at 1B each)
//   load 8 floats of valid
//   load 8 floats of reach
//   spread cat bytes to i32 lanes; cmp to 1/2/3; blend payoff
//   acc += reach * valid * payoff
// ---------------------------------------------------------------------------
enum : uint8_t {
    CAT_INVALID = 0,
    CAT_WIN     = 1,
    CAT_LOSE    = 2,
    CAT_TIE     = 3,
};

static void showdown_a2(
    const uint8_t* cat_matrix, const float* valid_matrix,
    const float* opp_reach_w,
    float* out, std::size_t n,
    float win_p, float lose_p, float tie_p)
{
    __m256 vwin  = _mm256_set1_ps(win_p);
    __m256 vlose = _mm256_set1_ps(lose_p);
    __m256 vtie  = _mm256_set1_ps(tie_p);
    __m256i v_one  = _mm256_set1_epi32(CAT_WIN);
    __m256i v_two  = _mm256_set1_epi32(CAT_LOSE);
    __m256i v_three = _mm256_set1_epi32(CAT_TIE);

    for (std::size_t c = 0; c < n; ++c) {
        const uint8_t* cat_row   = cat_matrix   + c * n;
        const float*   valid_row = valid_matrix + c * n;
        __m256 acc = _mm256_setzero_ps();
        std::size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            // Load 8 bytes (low 64 bits of XMM) and zero-extend to 8x i32.
            __m128i cat8 = _mm_loadl_epi64(
                reinterpret_cast<const __m128i*>(cat_row + i));
            __m256i cat32 = _mm256_cvtepu8_epi32(cat8);

            // Build win/lose/tie masks on i32 lanes, reinterpret as ps mask.
            __m256i mw_i = _mm256_cmpeq_epi32(cat32, v_one);
            __m256i ml_i = _mm256_cmpeq_epi32(cat32, v_two);
            __m256i mt_i = _mm256_cmpeq_epi32(cat32, v_three);
            __m256 m_win  = _mm256_castsi256_ps(mw_i);
            __m256 m_lose = _mm256_castsi256_ps(ml_i);
            __m256 m_tie  = _mm256_castsi256_ps(mt_i);

            // payoff = invalid(0) -> 0; win -> win_p; lose -> lose_p; tie -> tie_p.
            __m256 payoff = _mm256_setzero_ps();
            payoff = _mm256_blendv_ps(payoff, vwin,  m_win);
            payoff = _mm256_blendv_ps(payoff, vlose, m_lose);
            payoff = _mm256_blendv_ps(payoff, vtie,  m_tie);

            __m256 valid = _mm256_loadu_ps(valid_row + i);
            __m256 reach = _mm256_loadu_ps(opp_reach_w + i);
            __m256 rv    = _mm256_mul_ps(reach, valid);
            acc          = _mm256_fmadd_ps(rv, payoff, acc);
        }
        float sum = hsum256(acc);
        for (; i < n; ++i) {
            uint8_t cat = cat_row[i];
            if (cat == CAT_INVALID) continue;
            float p;
            if      (cat == CAT_WIN)  p = win_p;
            else if (cat == CAT_LOSE) p = lose_p;
            else                      p = tie_p;
            sum += opp_reach_w[i] * valid_row[i] * p;
        }
        out[c] = sum;
    }
}

// ---------------------------------------------------------------------------
// Synthetic data generator. Mirrors the rough distribution of a real
// 3-card-flop matchup table:
//   ~5% invalid (card overlap) — both ev and valid set to 0.
//   ev uniform in [-1, 1] — kernel buckets via 0.5 / -0.5 thresholds.
//   valid uniform in (0, 1] for valid pairs.
// ---------------------------------------------------------------------------
struct MatchupTable {
    std::vector<float>   ev;
    std::vector<float>   valid;
    std::vector<uint8_t> category;       // pre-thresholded
};

static MatchupTable make_table(uint64_t seed) {
    MatchupTable t;
    t.ev.assign(kCells, 0.0f);
    t.valid.assign(kCells, 0.0f);
    t.category.assign(kCells, CAT_INVALID);

    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> uev(-1.0f, 1.0f);
    std::uniform_real_distribution<float> uval(0.05f, 1.0f);
    std::bernoulli_distribution           uinvalid(0.05);   // ~5% invalid

    for (std::size_t k = 0; k < kCells; ++k) {
        if (uinvalid(rng)) continue;   // leave invalid (zeros + CAT_INVALID)
        float ev = uev(rng);
        float vl = uval(rng);
        t.ev[k]    = ev;
        t.valid[k] = vl;
        if      (ev >  0.5f) t.category[k] = CAT_WIN;
        else if (ev < -0.5f) t.category[k] = CAT_LOSE;
        else                 t.category[k] = CAT_TIE;
    }
    return t;
}

static std::vector<float> make_reach(uint64_t seed) {
    std::vector<float> r(kNc);
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    for (std::size_t i = 0; i < kNc; ++i) r[i] = u(rng);
    return r;
}

// ---------------------------------------------------------------------------
// Driver: run K calls of `kernel` over `tables`, picking the next table for
// each call. Returns wall-clock seconds for the whole batch.
//
// Threading: when threads > 1, we partition the call count across threads.
// Each thread picks an independent reach buffer (so the writes don't false-
// share) and a private out buffer. Tables are READ shared.
// ---------------------------------------------------------------------------
template <typename CallFn>
static double run_calls(int threads, std::size_t calls, CallFn&& fn) {
    threads = (threads <= 0) ? 1 : threads;
    auto t0 = std::chrono::steady_clock::now();

    std::atomic<std::size_t> done{0};
    std::vector<std::thread> workers;
    workers.reserve(threads);
    const std::size_t per = (calls + threads - 1) / threads;

    for (int tid = 0; tid < threads; ++tid) {
        const std::size_t lo = std::min(calls, static_cast<std::size_t>(tid) * per);
        const std::size_t hi = std::min(calls, lo + per);
        workers.emplace_back([&fn, &done, lo, hi, tid]() {
            for (std::size_t k = lo; k < hi; ++k) {
                fn(k, tid);
                done.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }
    for (auto& w : workers) w.join();

    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(t1 - t0).count();
}

struct Result {
    double seconds;
    double calls_per_sec;
    double gb_per_sec_effective;
};

// Bytes per call streamed through the kernel from each candidate's matrix.
static double bytes_per_call_baseline() {
    // ev_f32 + valid_f32 + reach (already L1-resident, count or not?).
    // Count only the matrix portion — reach is reused across c iterations.
    return static_cast<double>(kCells) * 8.0;
}
static double bytes_per_call_a2() {
    // category_u8 + valid_f32
    return static_cast<double>(kCells) * 5.0;
}

static Result run_baseline(int threads, std::size_t calls,
                            const std::vector<MatchupTable>& tables,
                            const std::vector<std::vector<float>>& reaches,
                            std::vector<std::vector<float>>& out_per_thread) {
    const std::size_t T = tables.size();
    auto fn = [&](std::size_t k, int tid) {
        const MatchupTable& mt = tables[k % T];
        const std::vector<float>& rch = reaches[(k + tid) % reaches.size()];
        showdown_baseline(mt.ev.data(), mt.valid.data(), rch.data(),
                          out_per_thread[tid].data(), kNc,
                          1.0f, -1.0f, 0.0f);
    };
    double s = run_calls(threads, calls, fn);
    return Result{
        s,
        calls / s,
        (calls * bytes_per_call_baseline() / s) / 1e9,
    };
}

static Result run_a2(int threads, std::size_t calls,
                      const std::vector<MatchupTable>& tables,
                      const std::vector<std::vector<float>>& reaches,
                      std::vector<std::vector<float>>& out_per_thread) {
    const std::size_t T = tables.size();
    auto fn = [&](std::size_t k, int tid) {
        const MatchupTable& mt = tables[k % T];
        const std::vector<float>& rch = reaches[(k + tid) % reaches.size()];
        showdown_a2(mt.category.data(), mt.valid.data(), rch.data(),
                    out_per_thread[tid].data(), kNc,
                    1.0f, -1.0f, 0.0f);
    };
    double s = run_calls(threads, calls, fn);
    return Result{
        s,
        calls / s,
        (calls * bytes_per_call_a2() / s) / 1e9,
    };
}

// ---------------------------------------------------------------------------
// Sanity check: baseline and A2 must agree to within FP rounding on a
// representative call. If they diverge here we have a bug in A2's encoding.
// ---------------------------------------------------------------------------
static bool verify_parity(const MatchupTable& mt, const std::vector<float>& rch,
                          float win_p, float lose_p, float tie_p) {
    std::vector<float> out_b(kNc, 0.0f), out_a(kNc, 0.0f);
    showdown_baseline(mt.ev.data(), mt.valid.data(), rch.data(),
                      out_b.data(), kNc, win_p, lose_p, tie_p);
    showdown_a2(mt.category.data(), mt.valid.data(), rch.data(),
                out_a.data(), kNc, win_p, lose_p, tie_p);
    float max_diff = 0.0f;
    for (std::size_t c = 0; c < kNc; ++c) {
        float d = std::abs(out_b[c] - out_a[c]);
        if (d > max_diff) max_diff = d;
    }
    constexpr float kTol = 1e-4f;
    if (max_diff > kTol) {
        std::cerr << "[parity] FAIL  max|baseline - a2| = " << max_diff
                  << " (tol " << kTol << ")\n";
        return false;
    }
    std::cout << "[parity] PASS  max|baseline - a2| = " << max_diff << "\n";
    return true;
}

// ---------------------------------------------------------------------------
// Scenario runner. Two regimes:
//   hot  — 1 table, K calls. Same data reused → table sits in L2/L3 after
//          the first sweep. Compute-bound test.
//   cold — N tables, K calls. Tables cycled, K >> N so each table is touched
//          many times across the run. With N tables × ~7 MB each easily
//          exceeds L3 → DRAM-bound test.
// ---------------------------------------------------------------------------
static void run_scenario(const char* label, int threads,
                          std::size_t num_tables, std::size_t calls) {
    std::vector<MatchupTable> tables;
    tables.reserve(num_tables);
    for (std::size_t k = 0; k < num_tables; ++k) {
        tables.push_back(make_table(0xDEADBEEFULL + 17 * k));
    }
    std::vector<std::vector<float>> reaches;
    for (int i = 0; i < std::max(threads, 1) * 2; ++i) {
        reaches.push_back(make_reach(0xCAFEBABEULL + 31u * i));
    }
    std::vector<std::vector<float>> outs(std::max(threads, 1),
                                          std::vector<float>(kNc, 0.0f));

    // Warmup — touch every table once on each thread.
    {
        auto warm_b = [&](std::size_t k, int tid) {
            const MatchupTable& mt = tables[k % num_tables];
            showdown_baseline(mt.ev.data(), mt.valid.data(),
                              reaches[tid].data(), outs[tid].data(), kNc,
                              1.0f, -1.0f, 0.0f);
        };
        run_calls(threads, num_tables, warm_b);
        auto warm_a = [&](std::size_t k, int tid) {
            const MatchupTable& mt = tables[k % num_tables];
            showdown_a2(mt.category.data(), mt.valid.data(),
                        reaches[tid].data(), outs[tid].data(), kNc,
                        1.0f, -1.0f, 0.0f);
        };
        run_calls(threads, num_tables, warm_a);
    }

    Result rb = run_baseline(threads, calls, tables, reaches, outs);
    Result ra = run_a2(threads, calls, tables, reaches, outs);

    const double speedup = rb.seconds / ra.seconds;
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "--- " << label << "  threads=" << threads
              << "  tables=" << num_tables
              << "  calls=" << calls << " ---\n";
    std::cout << "  baseline:  " << rb.seconds * 1000.0 << " ms total | "
              << (rb.seconds / calls) * 1e6 << " us/call | "
              << rb.gb_per_sec_effective << " GB/s\n";
    std::cout << "  a2:        " << ra.seconds * 1000.0 << " ms total | "
              << (ra.seconds / calls) * 1e6 << " us/call | "
              << ra.gb_per_sec_effective << " GB/s\n";
    std::cout << "  speedup:   " << speedup << "x  (a2 is "
              << (speedup > 1.0 ? "faster" : "slower") << ")\n\n";
}

}  // anonymous namespace

int main() {
    std::cout << "=== showdown encoding micro-bench (nc=" << kNc << ") ===\n\n";

    // Parity gate first — abort if A2 diverges from baseline numerically.
    auto t0 = make_table(42);
    auto r0 = make_reach(7);
    if (!verify_parity(t0, r0, 1.0f, -1.0f, 0.0f)) return 1;
    if (!verify_parity(t0, r0, 0.85f, -1.05f, 0.10f)) return 1;
    std::cout << "\n";

    const int hw = std::max(1, (int)std::thread::hardware_concurrency());
    const int t1 = 1, t8 = std::min(8, hw);

    // Hot regime: 1 table, many calls. Should sit in L2/L3.
    run_scenario("hot, 1T",   t1, /*tables=*/1,   /*calls=*/500);
    run_scenario("hot, 8T",   t8, /*tables=*/1,   /*calls=*/500 * t8);

    // Production-like cold regimes. monotone has 446 tables; standard has ~1.
    // Pick 4 sizes to bracket the L3 boundary on a typical 8-MB L3.
    //   1   table  -> ~7 MB,  L3-resident
    //   4   tables -> ~28 MB, busts L3
    //   16  tables -> ~112 MB, fully DRAM
    //   64  tables -> ~448 MB, fully DRAM (bigger than monotone's 446 to extreme)
    for (std::size_t T : {std::size_t(4), std::size_t(16), std::size_t(64)}) {
        std::cout << "[cold T=" << T << "  ev/valid total = "
                  << (T * kCells * 8) / (1024 * 1024) << " MB]\n";
        run_scenario("cold 1T", t1, T, T * 50);
        run_scenario("cold 8T", t8, T, T * 50 * t8);
    }

    return 0;
}

cmake_minimum_required(VERSION 3.20)

# ============================================================================
# CUDA is optional — detect availability
# ============================================================================
option(ENABLE_CUDA "Enable CUDA GPU acceleration" ON)
# --use_fast_math trades precision for speed (fast div/sqrt, FMA, FTZ). Default
# ON (production). Turn OFF to test whether GPU's lower-precision math biases
# DCFR equilibria — e.g. the Stage-3 decomposition GPU-vs-CPU subgame quality
# gap (see memory next_session_runout_decomposition). Build a separate dir with
# -DCUDA_FAST_MATH=OFF so the production build dir stays untouched.
option(CUDA_FAST_MATH "Use CUDA --use_fast_math (faster GPU math, lower precision)" ON)

if(ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        project(deepsolver_core LANGUAGES CXX CUDA)
        set(HAVE_CUDA TRUE)
    else()
        message(STATUS "CUDA compiler not found — building CPU-only")
        project(deepsolver_core LANGUAGES CXX)
        set(HAVE_CUDA FALSE)
    endif()
else()
    project(deepsolver_core LANGUAGES CXX)
    set(HAVE_CUDA FALSE)
endif()

# ============================================================================
# C++ Standard
# ============================================================================
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ============================================================================
# Compiler Flags
# ============================================================================
if(MSVC)
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/W4>")
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/O2>")
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/EHsc>")
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/fp:fast>")
    # /Oi: prefer intrinsics; /Ot: favor speed over size in inner loops.
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/Oi>")
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/Ot>")
    # OpenMP for the two-traverser CFR parallelization.
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/openmp>")
    # NOTE: /arch:AVX2 is intentionally NOT global. Only cpu_kernels_avx2.cpp
    # gets it (set_source_files_properties below) so the rest of the binary
    # stays at the SSE2 baseline and runs on pre-Haswell CPUs. Runtime CPUID
    # dispatch in cpu_kernels_dispatch.cpp picks scalar vs AVX2 at startup.
else()
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-O3>")
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Wall>")
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Wextra>")
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-fopenmp>")
    add_link_options("-fopenmp")
    # AVX2 / FMA are set per-source on cpu_kernels_avx2.cpp; rest of TU stays
    # at the default x86-64 baseline so old hardware can still load the binary.
endif()

if(HAVE_CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    # v1.2.2: multi-arch fat binary so installed releases run natively on
    # all GPUs supported by the CUDA toolkit we build with.
    #
    # CUDA 13.x dropped Pascal (compute_61) AND Volta (compute_70) — minimum
    # supported arch is now Turing (compute_75). Owners of GTX 10-series
    # cards need a separate CUDA-12.x build (tracked for v1.3.0). For now
    # they auto-fall-back to CPU with a clear diagnostic instead of silent
    # routing.
    #
    #   75: Turing     (RTX 20-series, GTX 16-series)
    #   80: Ampere A100
    #   86: Ampere     (RTX 30-series)
    #   89: Ada        (RTX 40-series)
    #   90: Hopper     (H100); also serves as PTX-JIT source for Blackwell
    # Build size grows ~5× (one SASS per arch), but each device gets a
    # native binary instead of paying a JIT cost on first launch.
    set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;90")

    if(CUDA_FAST_MATH)
        add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>")
    else()
        message(STATUS "CUDA --use_fast_math DISABLED (precision-mode build)")
    endif()
    if(MSVC)
        add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/O2>")
        add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/EHsc>")
        add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W3>")
    endif()
endif()

# ============================================================================
# Source Files
# ============================================================================

# CPU SIMD kernels — required by every target that pulls in solver.h /
# cpu_backend.h / cpu_backend_levelized.h (any of those reference
# cpu_simd::kernels(), cpu_simd::active_mode(), cpu_simd::mode_label()).
# Centralized here so test executables can reuse the exact same list and we
# don't drift toward "solver test fails to link cpu_simd::*" again.
set(CPU_KERNEL_SOURCES
    src/cpu_kernels_scalar.cpp
    src/cpu_kernels_avx2.cpp
    src/cpu_kernels_dispatch.cpp
)

set(CORE_SOURCES
    src/hand_evaluator.cpp
    src/backend_factory.cpp
    src/main.cpp
    ${CPU_KERNEL_SOURCES}
)

# Per-source AVX2 flag — confines the AVX2 opcodes to one .obj. The rest of
# the binary stays at the platform's SSE2 baseline. CPUID dispatch at startup
# decides whether to call into the AVX2 functions or the scalar ones.
# Source-file properties are scoped to this directory, so all targets in this
# CMakeLists.txt that compile cpu_kernels_avx2.cpp (deepsolver_core, test_solver,
# test_gpu_postsolve, future parity tests) automatically pick up the flag.
if(MSVC)
    set_source_files_properties(src/cpu_kernels_avx2.cpp
        PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
else()
    set_source_files_properties(src/cpu_kernels_avx2.cpp
        PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")
endif()

# ============================================================================
# Executable Target
# ============================================================================
if(HAVE_CUDA)
    set(CUDA_SOURCES
        cuda/cfr_kernel.cu
        cuda/eval_kernel.cu
        cuda/gpu_backend.cu
    )
    add_executable(deepsolver_core ${CORE_SOURCES} ${CUDA_SOURCES})
    target_compile_definitions(deepsolver_core PRIVATE DEEPSOLVER_USE_CUDA)
    set_target_properties(deepsolver_core PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    target_link_libraries(deepsolver_core PRIVATE cudart)
    target_include_directories(deepsolver_core PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/cuda
    )
else()
    add_executable(deepsolver_core ${CORE_SOURCES})
    target_include_directories(deepsolver_core PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
endif()

# Increase stack size for deep recursive CFR traversal (16 MB)
if(MSVC)
    target_link_options(deepsolver_core PRIVATE "/STACK:16777216")
endif()

# ============================================================================
# Tests (optional, build with -DBUILD_TESTS=ON)
# ============================================================================
option(BUILD_TESTS "Build unit tests" OFF)

if(BUILD_TESTS)
    enable_testing()

    # ------------------------------------------------------------------
    # Sprint 2 (market-beating plan): test layering by ctest LABELS.
    #   smoke       — must finish under ~10s. The fast safety net every
    #                 commit / IDE save gets to run.
    #   correctness — under 2 min target. Convergence / EV sanity / strategy
    #                 shape. The pre-merge gate.
    #   stress      — manual / nightly. Big solves, OOM rejection, GPU OOM
    #                 fallback, JSON cap. Don't run in inner loop.
    #   gpu         — anything that REQUIRES a CUDA device.
    # Run with: ctest -L smoke   /  ctest -L correctness  /  ctest -L stress
    # ------------------------------------------------------------------

    add_executable(test_evaluator tests/test_evaluator.cpp src/hand_evaluator.cpp)
    target_include_directories(test_evaluator PRIVATE include)
    if(MSVC)
        target_compile_options(test_evaluator PRIVATE /W4 /O2 /EHsc)
    endif()
    add_test(NAME HandEvaluator COMMAND test_evaluator)
    set_tests_properties(HandEvaluator PROPERTIES LABELS "smoke" TIMEOUT 30)

    add_executable(test_tree tests/test_tree.cpp)
    target_include_directories(test_tree PRIVATE include)
    if(MSVC)
        target_compile_options(test_tree PRIVATE /W4 /O2 /EHsc)
    endif()
    add_test(NAME GameTree COMMAND test_tree)
    set_tests_properties(GameTree PROPERTIES LABELS "smoke" TIMEOUT 30)

    add_executable(test_solver
        tests/test_solver.cpp
        src/hand_evaluator.cpp
        src/backend_factory.cpp
        ${CPU_KERNEL_SOURCES}
    )
    target_include_directories(test_solver PRIVATE include)
    if(MSVC)
        target_compile_options(test_solver PRIVATE /W4 /O2 /EHsc)
        target_link_options(test_solver PRIVATE "/STACK:16777216")
    endif()
    # Sprint 5 (resource policy guide): the original SolverCorrectness was
    # one big binary running 6 tests serially in 240+ seconds, which made
    # `ctest -L correctness` useless as a daily gate. We now register two
    # ctest entries against the same exe but with `--suite=` filters:
    #
    #   SolverCorrectnessFast    → fast sanity tests, target <60s, label
    #                              `correctness` so daily runs include it.
    #   SolverConvergenceStress  → slow convergence/polar tests, label
    #                              `stress` so they only run on demand.
    #
    # Keep the legacy `SolverCorrectness` name aliased to `--suite=all` for
    # any external scripts that still invoke it by name.
    add_test(NAME SolverCorrectnessFast
        COMMAND test_solver --suite=fast)
    set_tests_properties(SolverCorrectnessFast PROPERTIES
        LABELS "correctness" TIMEOUT 120)

    add_test(NAME SolverConvergenceStress
        COMMAND test_solver --suite=stress)
    set_tests_properties(SolverConvergenceStress PROPERTIES
        LABELS "stress" TIMEOUT 600)

    add_test(NAME SolverCorrectness
        COMMAND test_solver --suite=all)
    set_tests_properties(SolverCorrectness PROPERTIES
        LABELS "stress" TIMEOUT 600)

    # ----------------------------------------------------------------
    # Cross-config parity (v1.6.0+): reference vs levelized, scalar vs
    # avx2, thread cap consistency, persistent-OMP toggle, plus tighter
    # extended fixtures (river / limited sizing / narrow range).
    #
    # Split into two ctest entries (POST_OPTIMIZATION_REVIEW Finding 2):
    #   CpuParityFast      — fast subset (~2s; all fixtures pin collapsed /
    #                        tiny trees) on smoke label so every CI run
    #                        gates on it.
    #   CpuParityExtended  — slower tight-tolerance fixtures on the
    #                        correctness label; kept off smoke so the
    #                        smoke loop stays <10s wall-clock. Includes the
    #                        enumerated-runout iso rank-blocker fixture
    #                        (~40s alone), hence the 180s timeout.
    # Both run the same binary with --suite=fast|extended.
    # ----------------------------------------------------------------
    add_executable(test_parity
        tests/test_parity.cpp
        src/hand_evaluator.cpp
        src/backend_factory.cpp
        ${CPU_KERNEL_SOURCES}
    )
    target_include_directories(test_parity PRIVATE include)
    if(MSVC)
        target_compile_options(test_parity PRIVATE /W4 /O2 /EHsc)
        target_link_options(test_parity PRIVATE "/STACK:16777216")
    endif()
    add_test(NAME CpuParityFast COMMAND test_parity --suite=fast)
    set_tests_properties(CpuParityFast PROPERTIES
        LABELS "smoke;parity" TIMEOUT 30)

    add_test(NAME CpuParityExtended COMMAND test_parity --suite=extended)
    set_tests_properties(CpuParityExtended PROPERTIES
        LABELS "parity;correctness" TIMEOUT 180)

    # ----------------------------------------------------------------
    # Streaming/subgame decomposition validation (Stage 1, opt-in).
    # Solves a monotone flop both monolithically (oracle) and decomposed,
    # printing an exploitability convergence sweep. Its OWN label so it stays
    # out of the smoke/parity/correctness gates until the approach is proven.
    # Run explicitly with: ctest -L decomposition --output-on-failure
    # ----------------------------------------------------------------
    add_executable(test_decomposition
        tests/test_decomposition.cpp
        src/hand_evaluator.cpp
        src/backend_factory.cpp
        ${CPU_KERNEL_SOURCES}
    )
    target_include_directories(test_decomposition PRIVATE include)
    if(MSVC)
        target_compile_options(test_decomposition PRIVATE /W4 /O2 /EHsc)
        target_link_options(test_decomposition PRIVATE "/STACK:16777216")
    endif()
    add_test(NAME Decomposition COMMAND test_decomposition)
    set_tests_properties(Decomposition PROPERTIES LABELS "decomposition" TIMEOUT 1200)

    # ----------------------------------------------------------------
    # Per-kernel scalar vs AVX2 parity (v1.8.0 Sprint 0.2). Loops every
    # entry in `cpu_simd::Kernels` across representative lengths to
    # localize any future divergence between the scalar and AVX2 TUs
    # before it shows up as a backend-level symptom.
    # ----------------------------------------------------------------
    # hand_evaluator: the iso rank-blocker test needs REAL hand ranks. Its
    # equivalence to the dense terminal holds because rank is constant within a
    # suit-isomorphism orbit, so synthetic ranks would break that invariant and
    # make the comparison vacuous.
    add_executable(test_cpu_kernels
        tests/test_cpu_kernels.cpp
        src/hand_evaluator.cpp
        ${CPU_KERNEL_SOURCES}
    )
    target_include_directories(test_cpu_kernels PRIVATE include)
    if(MSVC)
        target_compile_options(test_cpu_kernels PRIVATE /W4 /O2 /EHsc)
    endif()
    add_test(NAME CpuKernelParity COMMAND test_cpu_kernels)
    set_tests_properties(CpuKernelParity PROPERTIES
        LABELS "smoke;parity" TIMEOUT 30)

    # ----------------------------------------------------------------
    # Showdown encoding micro-benchmark (developer tool, NOT a ctest).
    # Run manually before doing matrix-format surgery — picks between
    # baseline f32+f32 and A2 u8+f32 encodings using realistic synthetic
    # data, with hot/cold cache regimes and 1T/8T thread sweeps.
    # ----------------------------------------------------------------
    add_executable(bench_showdown_encoding
        tests/bench_showdown_encoding.cpp)
    target_include_directories(bench_showdown_encoding PRIVATE include)
    if(MSVC)
        target_compile_options(bench_showdown_encoding PRIVATE
            /W4 /O2 /EHsc /arch:AVX2)
    endif()

    add_executable(bench_showdown_batch
        tests/bench_showdown_batch.cpp
        ${CPU_KERNEL_SOURCES})
    target_include_directories(bench_showdown_batch PRIVATE include)
    if(MSVC)
        target_compile_options(bench_showdown_batch PRIVATE /W4 /O2 /EHsc)
    endif()

    # ----------------------------------------------------------------
    # Smoke-via-CLI: tiny solves that exercise the new memory-budget
    # surface end-to-end. Each runs in well under 10 seconds.
    # Requires the deepsolver_core target so we depend on it explicitly.
    # ----------------------------------------------------------------
    add_test(NAME CliRainbowSmoke
        COMMAND deepsolver_core
            --pot 100 --stack 500 --board AsKd7c
            --iterations 5 --backend cpu
            --strategy-tree-evs none --postsolve none)
    set_tests_properties(CliRainbowSmoke PROPERTIES LABELS "smoke" TIMEOUT 60)

    # Hard-cap rejection: tiny --host-memory-mb forces a structured error
    # exit (return code 1, stderr JSON). PASS_REGULAR_EXPRESSION matches
    # the precompute_matchups diagnostic; WILL_FAIL flips the exit-code
    # interpretation so the test passes when the engine returns 1 with the
    # expected error text. (CTest evaluates regex against stderr by default
    # for FAIL_REGULAR_EXPRESSION; we use the inverse so that "would
    # allocate ... GB" being absent fails the test.)
    add_test(NAME CliMemoryBudgetReject
        COMMAND deepsolver_core
            --pot 100 --stack 500 --board AsKd7c
            --iterations 5 --backend cpu --host-memory-mb 8)
    set_tests_properties(CliMemoryBudgetReject PROPERTIES
        LABELS "smoke;memory" TIMEOUT 60
        PASS_REGULAR_EXPRESSION "precompute_matchups would allocate"
        WILL_FAIL FALSE)

    # Strategy-tree truncation: --strategy-tree-max-nodes 50 should emit
    # at most 50 entries and set strategy_tree_truncated=true.
    add_test(NAME CliStrategyTreeTruncate
        COMMAND deepsolver_core
            --pot 100 --stack 500 --board AsKd7c
            --iterations 5 --backend cpu
            --strategy-tree-max-nodes 50 --postsolve none)
    set_tests_properties(CliStrategyTreeTruncate PROPERTIES
        LABELS "smoke;memory" TIMEOUT 60
        PASS_REGULAR_EXPRESSION "\"strategy_tree_truncated\": true")

    # Roadmap ② (post-v1.9.0): no non-finite float may ever reach the JSON
    # output. A bulk run once emitted `"98s":inf` inside a combo_evs map
    # (QdJhTc, GPU noise) which invalidated the entire response. Board and
    # postsolve flags chosen to exercise every value-bearing emit path
    # (combo EVs, strategies, opponent ranges, exploitability).
    add_test(NAME CliJsonFiniteFloats
        COMMAND deepsolver_core
            --pot 100 --stack 500 --board QdJhTc
            --iterations 5 --backend cpu
            --postsolve full --strategy-tree-evs visible)
    set_tests_properties(CliJsonFiniteFloats PROPERTIES
        LABELS "smoke" TIMEOUT 120
        # Matches `": inf`, `":inf`, `": "inf%`, `":-nan(ind)` etc. — any
        # inf/nan token in a JSON value position, quoted or bare.
        FAIL_REGULAR_EXPRESSION "\": *\"?-?(inf|nan)")

    # Roadmap ① (post-v1.9.0): state-aware collapse gate. Monotone Ah9h4h at
    # 2.83 GB host: the builder's worst-case matchup projection (~1.4 GB)
    # passes the host/2 cap so the tree fully enumerates (~185k nodes), but
    # the REAL footprint (446 deduped tables + CPU state ≈ 2.92 GB) busts
    # the budget — pre-fix the CPU host gate REJECTED this solve. The state
    # gate must instead rebuild collapsed and complete with
    # runout_approximated=true. Deterministic on any machine: CPU branch,
    # no hardware probe.
    add_test(NAME CliMonoStateGateCollapse
        COMMAND deepsolver_core
            --pot 100 --stack 500 --board Ah9h4h
            --iterations 5 --backend cpu --postsolve none
            --host-memory-mb 2900)
    set_tests_properties(CliMonoStateGateCollapse PROPERTIES
        LABELS "smoke;memory" TIMEOUT 60
        # Emitted only when the gate fired AND the collapsed rebuild solved.
        PASS_REGULAR_EXPRESSION "Enumerated runout tree needs")

    # Roadmap ④ (post-v1.9.0): Exact-mode feasibility pre-flight. With
    # --decompose-runouts, --estimate-only must emit the "decompose" block
    # and predict engagement on a board whose enumerated state busts the
    # budget (same spot as the state-gate test — deterministic CPU branch).
    add_test(NAME CliExactPreflightEstimate
        COMMAND deepsolver_core
            --pot 100 --stack 500 --board Ah9h4h
            --backend cpu --host-memory-mb 2900
            --estimate-only --decompose-runouts auto)
    set_tests_properties(CliExactPreflightEstimate PROPERTIES
        LABELS "smoke;memory" TIMEOUT 60
        PASS_REGULAR_EXPRESSION "\"would_engage\": true")

    # Roadmap ④: the --decompose-* preset flags must reach the resolved
    # options the pre-flight prices (same resolver the real solve uses).
    add_test(NAME CliExactPreflightPresetFlags
        COMMAND deepsolver_core
            --pot 100 --stack 500 --board Ah9h4h
            --backend cpu --host-memory-mb 2900
            --estimate-only --decompose-runouts auto
            --decompose-outer 2 --decompose-inner 450
            --decompose-trunk-iters 300 --decompose-warmstart 1)
    set_tests_properties(CliExactPreflightPresetFlags PROPERTIES
        LABELS "smoke;memory" TIMEOUT 60
        PASS_REGULAR_EXPRESSION "\"inner\": 450")

    # Sprint 1 (resource policy guide): pre-backend budget gates.
    # CPU host gate: matchup fits but total host estimate exceeds budget.
    add_test(NAME CliCpuHostBudgetReject
        COMMAND deepsolver_core
            --pot 100 --stack 500 --board AsKd7c
            --iterations 5 --backend cpu --host-memory-mb 30)
    set_tests_properties(CliCpuHostBudgetReject PROPERTIES
        LABELS "smoke;memory" TIMEOUT 60
        # P1-2 (v1.2.1): regex matches both legacy "Host RAM estimate" and
        # new split forms ("Common host RAM estimate", "CPU total host RAM
        # estimate"). Lowercase 'h' is the common substring.
        PASS_REGULAR_EXPRESSION "host RAM estimate")

    # GPU explicit: insufficient VRAM budget rejects without backend alloc.
    add_test(NAME CliGpuVramBudgetReject
        COMMAND deepsolver_core
            --pot 100 --stack 500 --board AsKd7c
            --iterations 5 --backend gpu --gpu-memory-mb 1)
    set_tests_properties(CliGpuVramBudgetReject PROPERTIES
        LABELS "smoke;memory;gpu" TIMEOUT 60
        PASS_REGULAR_EXPRESSION "GPU state estimate")

    # AUTO fallback: insufficient VRAM should downgrade to CPU and
    # populate fallback_reason in the resources block.
    # v1.4.0+: backend label may be "CPU-DCFR", "CPU-DCFR-AVX2", or
    # "CPU-Levelized-AVX2" depending on runtime SIMD dispatch and the
    # selected cpu_backend_kind. The "CPU-" prefix is the stable invariant.
    add_test(NAME CliGpuAutoFallback
        COMMAND deepsolver_core
            --pot 100 --stack 500 --board AsKd7c
            --iterations 5 --backend auto --gpu-memory-mb 1)
    set_tests_properties(CliGpuAutoFallback PROPERTIES
        LABELS "smoke;memory;gpu" TIMEOUT 60
        PASS_REGULAR_EXPRESSION "\"backend\": \"CPU-")

    # P1-1 (v1.2.1): VISIBLE EV cache must respect strategy_tree_max_nodes.
    # With cap=3 the estimated_strategy_tree_bytes should land in the tens
    # of KB range (3 nodes * ~5.5 KB/node ≈ 16 KB), NOT the multi-MB range
    # the un-capped pre-walk would produce. Regex matches 1-5 digit values
    # which excludes the 7+ digit baseline (~11 MB on 2000 nodes).
    # AsKd7c rainbow keeps the test light — AsKsQs would trigger heavy iso
    # enumeration that pushes total runtime past 60s.
    add_test(NAME CliEvCacheRespectsCap
        COMMAND deepsolver_core
            --pot 100 --stack 500 --board AsKd7c
            --iterations 5 --backend cpu
            --strategy-tree-evs visible
            --strategy-tree-max-nodes 3
            --postsolve none)
    set_tests_properties(CliEvCacheRespectsCap PROPERTIES
        LABELS "smoke;memory" TIMEOUT 120
        # Match "estimated_strategy_tree_bytes": <1-5 digits>, — i.e., < 100,000
        # bytes. Baseline (un-capped) would be 7+ digits (~11 MB), so this
        # excludes the regression. Trailing comma closes the JSON line.
        PASS_REGULAR_EXPRESSION "\"estimated_strategy_tree_bytes\": [0-9][0-9]?[0-9]?[0-9]?[0-9]?,")

    # P1-2 (v1.2.1): GPU backend must NOT bypass the host budget gate for
    # buffers that live on host regardless of backend (matchup tables,
    # strategy_tree EV cache, JSON response).
    #
    # Scenario chosen to thread two needles: matchup small enough to pass
    # precompute_matchups (host_bytes/2 cap), but matchup+ev+json large
    # enough to fail the new common-host gate. AsKsQs at 5 MB host budget
    # produces matchup ≈ 0.9 MB (passes precompute's 2.5 MB cap) and common
    # ≈ 5.4 MB (fails the common-host gate at 5 MB). Pre-P1-2 this would
    # silently allocate ~5 MB on host during a "GPU solve".
    add_test(NAME CliGpuCommonHostBudgetReject
        COMMAND deepsolver_core
            --pot 100 --stack 500 --board AsKsQs
            --iterations 5 --backend gpu
            --host-memory-mb 5
            --strategy-tree-evs visible)
    set_tests_properties(CliGpuCommonHostBudgetReject PROPERTIES
        LABELS "smoke;memory;gpu" TIMEOUT 60
        PASS_REGULAR_EXPRESSION "Common host RAM estimate")

    # GPU postsolve parity test — only when CUDA is available, since the test
    # exercises the GPU backend path directly. Skip-marker inside the test
    # handles "CUDA compiled in but no device present" at runtime.
    if(HAVE_CUDA)
        add_executable(test_gpu_postsolve
            tests/test_gpu_postsolve.cpp
            src/hand_evaluator.cpp
            src/backend_factory.cpp
            ${CPU_KERNEL_SOURCES}
            cuda/cfr_kernel.cu
            cuda/eval_kernel.cu
            cuda/gpu_backend.cu
        )
        target_compile_definitions(test_gpu_postsolve PRIVATE DEEPSOLVER_USE_CUDA)
        target_include_directories(test_gpu_postsolve PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/cuda
        )
        set_target_properties(test_gpu_postsolve PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
        )
        target_link_libraries(test_gpu_postsolve PRIVATE cudart)
        if(MSVC)
            target_compile_options(test_gpu_postsolve PRIVATE
                "$<$<COMPILE_LANGUAGE:CXX>:/W4>"
                "$<$<COMPILE_LANGUAGE:CXX>:/O2>"
                "$<$<COMPILE_LANGUAGE:CXX>:/EHsc>")
            target_link_options(test_gpu_postsolve PRIVATE "/STACK:16777216")
        endif()
        add_test(NAME GpuPostsolveParity COMMAND test_gpu_postsolve)
        set_tests_properties(GpuPostsolveParity PROPERTIES
            LABELS "smoke;gpu" TIMEOUT 60)

        # Node-locked GPU solves must be bit-exact across runs (rank-blocker
        # bucket scatter uses order-independent fixed-point accumulation —
        # see tests/cli_lock_determinism.cmake for the full story).
        add_test(NAME CliGpuLockDeterminism
            COMMAND ${CMAKE_COMMAND}
                -DEXE=$<TARGET_FILE:deepsolver_core>
                -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/cli_lock_determinism.cmake)
        set_tests_properties(CliGpuLockDeterminism PROPERTIES
            LABELS "smoke;gpu" TIMEOUT 120)

        # Ship gate: GPU must reach the CPU equilibrium on an ENUMERATED tree
        # (the lift/normalize chance-row overflow corrupted every enumerated
        # GPU solve while collapsed trees stayed bit-exact — see
        # tests/cli_enumerated_parity.cmake for the full story).
        add_test(NAME CliEnumeratedGpuParity
            COMMAND ${CMAKE_COMMAND}
                -DEXE=$<TARGET_FILE:deepsolver_core>
                -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/cli_enumerated_parity.cmake)
        set_tests_properties(CliEnumeratedGpuParity PROPERTIES
            LABELS "smoke;gpu;parity" TIMEOUT 300)

        # ----------------------------------------------------------------
        # Stage 3: GPU subgame streaming validation. SAME source as the CPU
        # `Decomposition` test, but compiled WITH DEEPSOLVER_USE_CUDA + the
        # CUDA backend so create_backend(GPU) returns a real GpuBackend (the
        # CPU-only target silently falls back to CpuBackend). The extra GPU
        # block (guarded by #ifdef DEEPSOLVER_USE_CUDA in the test) streams
        # subgames through the GpuBackend and asserts exploit parity with the
        # monolithic oracle + bounded peak residency.
        # ----------------------------------------------------------------
        add_executable(test_decomposition_gpu
            tests/test_decomposition.cpp
            src/hand_evaluator.cpp
            src/backend_factory.cpp
            ${CPU_KERNEL_SOURCES}
            cuda/cfr_kernel.cu
            cuda/eval_kernel.cu
            cuda/gpu_backend.cu
        )
        target_compile_definitions(test_decomposition_gpu PRIVATE DEEPSOLVER_USE_CUDA)
        target_include_directories(test_decomposition_gpu PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/cuda
        )
        set_target_properties(test_decomposition_gpu PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
        )
        target_link_libraries(test_decomposition_gpu PRIVATE cudart)
        if(MSVC)
            target_compile_options(test_decomposition_gpu PRIVATE
                "$<$<COMPILE_LANGUAGE:CXX>:/W4>"
                "$<$<COMPILE_LANGUAGE:CXX>:/O2>"
                "$<$<COMPILE_LANGUAGE:CXX>:/EHsc>")
            target_link_options(test_decomposition_gpu PRIVATE "/STACK:16777216")
        endif()
        add_test(NAME DecompositionGpu COMMAND test_decomposition_gpu)
        set_tests_properties(DecompositionGpu PROPERTIES
            LABELS "decomposition;gpu" TIMEOUT 1800)
    endif()
endif()

# ============================================================================
# Install
# ============================================================================
install(TARGETS deepsolver_core RUNTIME DESTINATION bin)

message(STATUS "DeepSolver Core Configuration:")
message(STATUS "  CUDA Enabled: ${HAVE_CUDA}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")

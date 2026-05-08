cmake_minimum_required(VERSION 3.20)

# ============================================================================
# CUDA is optional — detect availability
# ============================================================================
option(ENABLE_CUDA "Enable CUDA GPU acceleration" ON)

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

    add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>")
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
    # Cross-config parity (v1.6.0): reference vs levelized, scalar vs
    # avx2, thread cap consistency. Each scenario runs 2 small solves
    # (60 iter on AsKd2c) and asserts root strategy diverges < 2pp and
    # AhKh EV diverges < 1 chip. Total runtime budget < 10s — fits the
    # `smoke` label so every CI run gates on it.
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
    add_test(NAME CpuParity COMMAND test_parity)
    set_tests_properties(CpuParity PROPERTIES LABELS "smoke" TIMEOUT 60)

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

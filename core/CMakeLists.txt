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
else()
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-O2>")
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Wall>")
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Wextra>")
endif()

if(HAVE_CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_ARCHITECTURES "89")

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
set(CORE_SOURCES
    src/hand_evaluator.cpp
    src/backend_factory.cpp
    src/main.cpp
)

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
        PASS_REGULAR_EXPRESSION "Host RAM estimate")

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
    add_test(NAME CliGpuAutoFallback
        COMMAND deepsolver_core
            --pot 100 --stack 500 --board AsKd7c
            --iterations 5 --backend auto --gpu-memory-mb 1)
    set_tests_properties(CliGpuAutoFallback PROPERTIES
        LABELS "smoke;memory;gpu" TIMEOUT 60
        PASS_REGULAR_EXPRESSION "\"backend\": \"CPU-DCFR\"")

    # GPU postsolve parity test — only when CUDA is available, since the test
    # exercises the GPU backend path directly. Skip-marker inside the test
    # handles "CUDA compiled in but no device present" at runtime.
    if(HAVE_CUDA)
        add_executable(test_gpu_postsolve
            tests/test_gpu_postsolve.cpp
            src/hand_evaluator.cpp
            src/backend_factory.cpp
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

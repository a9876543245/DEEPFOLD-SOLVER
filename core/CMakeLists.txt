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

    add_executable(test_evaluator tests/test_evaluator.cpp src/hand_evaluator.cpp)
    target_include_directories(test_evaluator PRIVATE include)
    if(MSVC)
        target_compile_options(test_evaluator PRIVATE /W4 /O2 /EHsc)
    endif()
    add_test(NAME HandEvaluator COMMAND test_evaluator)

    add_executable(test_tree tests/test_tree.cpp)
    target_include_directories(test_tree PRIVATE include)
    if(MSVC)
        target_compile_options(test_tree PRIVATE /W4 /O2 /EHsc)
    endif()
    add_test(NAME GameTree COMMAND test_tree)

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
    add_test(NAME SolverCorrectness COMMAND test_solver)
endif()

# ============================================================================
# Install
# ============================================================================
install(TARGETS deepsolver_core RUNTIME DESTINATION bin)

message(STATUS "DeepSolver Core Configuration:")
message(STATUS "  CUDA Enabled: ${HAVE_CUDA}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")

# Pre-solve ETA regression (2026-09-24).
#
# The GPU ETA used to price player_nodes × 6 × nc² ops per iteration over a
# per-generation rate. On enumerated GPU trees that was 4.5–39× pessimistic
# (the GPU's kernels are linear in nc) — 14× on this very fixture, a
# 1.45M-node monotone flop: 788.5 ms per iteration estimated, 56.5 measured —
# while the raked dense-terminal path went 1.5× OPTIMISTIC.
# memory_budget.h::estimate_gpu_iteration_seconds() replaced it with a
# measured model.
#
# This pins the property the user sees: on a large enumerated GPU tree the
# --estimate-only per-iteration cost lands within 2× of the measured one.
# The estimate side is derived model-agnostically — the slope of
# estimated_solve_seconds between two iteration caps — so a pre-fix binary
# fails on the ratio itself, not on a missing field. Then:
#   - estimated_iteration_ms must BE that slope (it is what the solve JSON
#     and bench-matrix report), and
#   - the solve's own resources block must price the same number (the
#     preview and the solve share Solver::estimated_iteration_seconds()).
#
# Calibrated on CC 12.0 (RTX 5090). Other GPU generations scale by an
# UNMEASURED per-CC ratio, so a red on another card means that ratio needs
# its own measurement. A GPU busy with other work reads as "optimistic" —
# check nvidia-smi before believing a low ratio.
#
# Usage: cmake -DEXE=<deepsolver_core path> -P cli_eta.cmake

if(NOT EXE)
  message(FATAL_ERROR "pass -DEXE=<deepsolver_core path>")
endif()

# Budgets pinned so neither the default 6 GiB host budget (the estimate peaks
# at 5.5) nor the free-VRAM probe can collapse the tree under the test.
set(SPOT --pot 100 --stack 500 --board AsKsQs
    --flop-sizes 0.33,0.75 --turn-sizes 0.33,0.75 --river-sizes 0.33,0.75
    --backend gpu --gpu-memory-mb 24000 --host-memory-mb 16000
    --exploitability 0 --postsolve none --no-strategy-tree --no-progress)

# "12.345" -> 12345: a decimal string times 1000, truncated (math() is
# integer-only). Bails out on anything else rather than misreading it.
function(to_milli value out)
  if(NOT value MATCHES "^[0-9]+(\\.[0-9]*)?$")
    message(FATAL_ERROR "unexpected number format '${value}'")
  endif()
  string(REGEX REPLACE "\\..*$" "" whole "${value}")
  set(frac "")
  if(value MATCHES "\\.([0-9]*)$")
    set(frac "${CMAKE_MATCH_1}")
  endif()
  string(APPEND frac "000")
  string(SUBSTRING "${frac}" 0 3 frac)
  string(REGEX REPLACE "^0+([0-9])" "\\1" whole "${whole}")
  string(REGEX REPLACE "^0+([0-9])" "\\1" frac "${frac}")
  math(EXPR v "${whole} * 1000 + ${frac}")
  set(${out} ${v} PARENT_SCOPE)
endfunction()

# ---- estimate: the ETA's slope over 1000 extra iterations ----------------
foreach(cap 1000 2000)
  execute_process(COMMAND ${EXE} ${SPOT} --iterations ${cap} --estimate-only
    OUTPUT_VARIABLE est_${cap} RESULT_VARIABLE rc ERROR_VARIABLE err)
  if(NOT rc EQUAL 0)
    message(FATAL_ERROR "--estimate-only (${cap} iterations) failed rc=${rc}: ${err}")
  endif()
  string(JSON approx GET "${est_${cap}}" resources runout_approximated)
  if(approx)
    message(FATAL_ERROR
      "fixture collapsed at estimate time - this pins the ENUMERATED GPU tree")
  endif()
  string(JSON s GET "${est_${cap}}" resources estimated_solve_seconds)
  to_milli("${s}" eta_ms_${cap})
endforeach()
# Seconds over 1000 iterations, ×1000 → µs per iteration.
math(EXPR est_us "${eta_ms_2000} - ${eta_ms_1000}")

# ---- measurement: the same spot, solved ----------------------------------
execute_process(COMMAND ${EXE} ${SPOT} --iterations 40
  OUTPUT_VARIABLE out RESULT_VARIABLE rc ERROR_VARIABLE err)
if(NOT rc EQUAL 0)
  message(FATAL_ERROR "solve failed rc=${rc}: ${err}")
endif()
string(JSON mode    GET "${out}" tree_mode)
string(JSON nodes   GET "${out}" timing tree_nodes)
string(JSON backend GET "${out}" backend)
string(JSON ran     GET "${out}" iterations_run)
string(JSON loop_ms GET "${out}" timing iterations_ms)
# Fixture validity first: a collapsed tree, a CPU fallback or a short run
# would each make the comparison below pass without testing anything.
if(NOT mode STREQUAL "enumerated")
  message(FATAL_ERROR "solve ran ${mode}, not the enumerated tree")
endif()
if(nodes LESS 1000000)
  message(FATAL_ERROR "tree has ${nodes} nodes - expected the 1.45M-node enumerated flop")
endif()
if(NOT backend MATCHES "^CUDA")
  message(FATAL_ERROR "solve ran on '${backend}', not the GPU")
endif()
if(NOT ran EQUAL 40)
  message(FATAL_ERROR "solve stopped at ${ran} of 40 iterations")
endif()
to_milli("${loop_ms}" loop_us)
math(EXPR meas_us "${loop_us} / ${ran}")

math(EXPR est_x100 "${est_us} * 100 / ${meas_us}")
math(EXPR est_x2 "${est_us} * 2")
math(EXPR meas_x2 "${meas_us} * 2")
if(est_us GREATER meas_x2)
  message(FATAL_ERROR
    "ETA is ${est_x100}% of the measured iteration (estimate ${est_us} us, "
    "measured ${meas_us} us) - more than 2x pessimistic on a large "
    "enumerated GPU tree")
endif()
if(est_x2 LESS meas_us)
  message(FATAL_ERROR
    "ETA is ${est_x100}% of the measured iteration (estimate ${est_us} us, "
    "measured ${meas_us} us) - more than 2x optimistic (or the GPU was busy "
    "with other work: check nvidia-smi)")
endif()

# ---- the reported field is the ETA's own number, on both paths ------------
string(JSON f ERROR_VARIABLE jerr GET "${est_1000}" resources estimated_iteration_ms)
if(jerr)
  message(FATAL_ERROR "--estimate-only emits no resources.estimated_iteration_ms")
endif()
to_milli("${f}" field_us)
string(JSON g GET "${out}" resources estimated_iteration_ms)
to_milli("${g}" solve_us)
foreach(pair "estimate-only field;${field_us}" "solve resources;${solve_us}")
  list(GET pair 0 label)
  list(GET pair 1 v)
  math(EXPR d "${v} - ${est_us}")
  if(d LESS 0)
    math(EXPR d "0 - ${d}")
  endif()
  # 1% + 20 us: the slope is read off two 6-digit prints, the solve's copy
  # off a 2-decimal one.
  math(EXPR tol "${est_us} / 100 + 20")
  if(d GREATER tol)
    message(FATAL_ERROR
      "${label} estimated_iteration_ms (${v} us) is not the ETA's slope "
      "(${est_us} us)")
  endif()
endforeach()

message(STATUS
  "gpu_eta: ${nodes} nodes, estimate ${est_us} us/iter vs measured "
  "${meas_us} us/iter (${est_x100}%)")

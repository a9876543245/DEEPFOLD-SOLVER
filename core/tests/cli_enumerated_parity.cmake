# Ship gate: GPU solves on ENUMERATED trees must converge to the CPU
# equilibrium.
#
# The lift/normalize kernels used to write na rows at CHANCE nodes while the
# dense layout only reserves MAX_ACTIONS=6 per node. Enumerated chance nodes
# (river deal on a turn board: na~46) overflowed into the action_values of
# neighboring nodes, so every GPU solve on an enumerated tree converged to a
# WRONG equilibrium (this spot: 3.78% exploitability vs CPU 2.98% at 300
# iters; combo-frequency deviations up to 58pp). Collapsed trees have
# single-child chance nodes and stayed correct, which is why no existing test
# caught it: parity suites are CPU-only and GPU tests run on collapsed or
# CPU-finalized paths.
#
# This script solves one turn board (enumerated by construction) on both
# backends and asserts converged exploitability agrees within 0.30pp
# (measured post-fix gap: 0.01pp).
#
# Usage: cmake -DEXE=<path-to-deepsolver_core> -P cli_enumerated_parity.cmake

if(NOT EXE)
  message(FATAL_ERROR "pass -DEXE=<deepsolver_core path>")
endif()

set(COMMON_ARGS
  --pot 100 --stack 500 --board AsKd7c2h
  --iterations 300 --dcfr-schedule standard
  --postsolve exploitability)

foreach(backend gpu cpu)
  execute_process(
    COMMAND ${EXE} ${COMMON_ARGS} --backend ${backend}
    OUTPUT_VARIABLE out_${backend}
    RESULT_VARIABLE rc_${backend}
    ERROR_VARIABLE err_${backend})
  if(NOT rc_${backend} EQUAL 0)
    message(FATAL_ERROR
      "${backend} run failed (rc=${rc_${backend}}): ${err_${backend}}")
  endif()

  # The spot must build the ENUMERATED tree — the topology this bug class
  # corrupts. A collapsed tree here would silently test nothing.
  if(NOT out_${backend} MATCHES "\"runout_approximated\": false")
    message(FATAL_ERROR
      "${backend} run is not on an enumerated tree (runout_approximated "
      "is not false) — the fixture no longer covers the GPU chance-row "
      "overflow path")
  endif()

  # exploitability_pct is emitted std::fixed setprecision(2); parse the two
  # fraction digits separately (math(EXPR) would read \"08\" as octal).
  if(out_${backend} MATCHES
      "\"exploitability_pct\": ([0-9]+)\\.([0-9])([0-9])")
    math(EXPR centi_${backend}
      "${CMAKE_MATCH_1} * 100 + ${CMAKE_MATCH_2} * 10 + ${CMAKE_MATCH_3}")
    set(pct_${backend} "${CMAKE_MATCH_1}.${CMAKE_MATCH_2}${CMAKE_MATCH_3}")
  else()
    message(FATAL_ERROR
      "${backend} run: could not parse exploitability_pct from stdout")
  endif()
endforeach()

if(centi_gpu GREATER centi_cpu)
  math(EXPR diff "${centi_gpu} - ${centi_cpu}")
else()
  math(EXPR diff "${centi_cpu} - ${centi_gpu}")
endif()

# 30 centi-pp = 0.30pp. Post-fix gap is 0.01pp; the pre-fix bug showed 0.80pp.
if(diff GREATER 30)
  message(FATAL_ERROR
    "GPU diverged from CPU on an enumerated tree: gpu=${pct_gpu}% "
    "cpu=${pct_cpu}% |diff|=${diff} centi-pp exceeds 30 — GPU chance-row "
    "overflow class regression")
endif()
message(STATUS
  "enumerated parity ok: gpu=${pct_gpu}% cpu=${pct_cpu}% "
  "|diff|=${diff} centi-pp (limit 30)")

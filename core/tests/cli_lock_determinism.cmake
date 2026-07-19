# Regression: GPU solves with --node-locks must be run-to-run deterministic.
#
# Locked solves on singleton-iso (rank-blocker) boards used to wobble at
# ~1e-4 strategy-frequency scale: rank_blocker_terminal_kernel scattered
# opponent reach into rank buckets with float shared-memory atomicAdd, whose
# same-bucket resolution order follows warp scheduling. Unlocked solves
# masked it (uniform iter-0 reach means equal addends, which sum identically
# in any order) — any lock perturbs that and the ULP noise amplifies through
# DCFR. Fixed by 64-bit fixed-point accumulation (integer adds are
# associative, so bucket sums are order-independent).
#
# This script runs the original repro spot twice and asserts byte-identical
# stdout JSON after stripping timing (_ms) lines.
#
# Usage: cmake -DEXE=<path-to-deepsolver_core> -P cli_lock_determinism.cmake

if(NOT EXE)
  message(FATAL_ERROR "pass -DEXE=<deepsolver_core path>")
endif()

set(ARGS
  --pot 100 --stack 500 --board AsKd2c
  --iterations 30 --postsolve none
  --backend gpu --dcfr-schedule standard
  --node-locks "[{\"history\":\"\",\"combo\":\"AhKh\",\"strategy\":[1,0,0,0]}]")

foreach(run 1 2)
  execute_process(
    COMMAND ${EXE} ${ARGS}
    OUTPUT_VARIABLE out_${run}
    RESULT_VARIABLE rc_${run}
    ERROR_VARIABLE err_${run})
  if(NOT rc_${run} EQUAL 0)
    message(FATAL_ERROR "run ${run} failed (rc=${rc_${run}}): ${err_${run}}")
  endif()
  string(REGEX REPLACE "[^\n]*_ms[^\n]*\n" "" out_${run} "${out_${run}}")
  if(out_${run} STREQUAL "")
    message(FATAL_ERROR "run ${run} produced no JSON output")
  endif()
endforeach()

if(NOT "${out_1}" STREQUAL "${out_2}")
  message(FATAL_ERROR
    "locked GPU solve is run-to-run NONDETERMINISTIC (filtered stdout "
    "differs between two identical runs)")
endif()
message(STATUS "locked GPU solve deterministic across 2 runs")

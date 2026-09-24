# Pre-solve ETA regression, CPU (2026-09-24).
#
# The CPU ETA used to price player_nodes × 6 × nc² ops per iteration over a
# per-thread rate table (v1.7.1). That shape only fits the dense dot-product
# terminals; on a rank-blocker board every showdown is O(nc) plus a fixed
# per-terminal sweep, and on this fixture — the AsKd7c2h turn, the singleton
# board every rainbow turn solve runs on — v3.2.0 read 11-13× pessimistic
# (332 ms per iteration estimated at 1 thread, 26 measured).
# memory_budget.h::estimate_cpu_iteration_seconds() replaced it with a
# measured model.
#
# Same method as cli_eta.cmake: the estimate is the SLOPE of
# estimated_solve_seconds between two iteration caps (model-agnostic, so a
# pre-fix binary fails on the ratio itself), it must land within 2× of the
# measured iteration, and estimated_iteration_ms must BE that slope on both
# the --estimate-only and the solve path. Checked at 1 thread and at 8
# (clamped to the machine; the model prices the effective team).
#
# The measurement is the FASTEST of three solves: other load on the machine
# only ever slows a run down, so the fastest run is the one closest to what
# the model describes. Calibrated on a Ryzen 9 9950X3D (8 cores, AVX2); a
# red "optimistic" on a busy machine usually means other processes held the
# cores — check the task manager before believing it.
#
# Usage: cmake -DEXE=<deepsolver_core path> -P cli_eta_cpu.cmake

if(NOT EXE)
  message(FATAL_ERROR "pass -DEXE=<deepsolver_core path>")
endif()

set(SPOT --pot 100 --stack 500 --board AsKd7c2h
    --backend cpu --cpu-backend levelized
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

# threads;iterations per solve (~1 s of iterations each)
foreach(case "1;40" "8;200")
  list(GET case 0 threads)
  list(GET case 1 iters)
  set(RUN ${SPOT} --cpu-threads ${threads})

  # ---- estimate: the ETA's slope over 1000 extra iterations --------------
  foreach(cap 1000 2000)
    execute_process(COMMAND ${EXE} ${RUN} --iterations ${cap} --estimate-only
      OUTPUT_VARIABLE est_${cap} RESULT_VARIABLE rc ERROR_VARIABLE err)
    if(NOT rc EQUAL 0)
      message(FATAL_ERROR "--estimate-only (${cap} iterations, ${threads}T) failed rc=${rc}: ${err}")
    endif()
    string(JSON s GET "${est_${cap}}" resources estimated_solve_seconds)
    to_milli("${s}" eta_ms_${cap})
  endforeach()
  # Seconds over 1000 iterations, ×1000 → µs per iteration.
  math(EXPR est_us "${eta_ms_2000} - ${eta_ms_1000}")

  # ---- measurement: the fastest of three solves ---------------------------
  set(meas_us "")
  set(runs "")
  foreach(k 1 2 3)
    execute_process(COMMAND ${EXE} ${RUN} --iterations ${iters}
      OUTPUT_VARIABLE out RESULT_VARIABLE rc ERROR_VARIABLE err)
    if(NOT rc EQUAL 0)
      message(FATAL_ERROR "solve (${threads}T) failed rc=${rc}: ${err}")
    endif()
    string(JSON mode    GET "${out}" tree_mode)
    string(JSON plan    GET "${out}" resources terminal_representation)
    string(JSON backend GET "${out}" backend)
    string(JSON ran     GET "${out}" iterations_run)
    string(JSON eff     GET "${out}" resources cpu_threads_effective)
    string(JSON loop_ms GET "${out}" timing iterations_ms)
    # Fixture validity first: each of these would make the comparison below
    # pass or fail without testing the model.
    if(NOT mode STREQUAL "enumerated")
      message(FATAL_ERROR "solve ran ${mode}, not the enumerated turn")
    endif()
    if(NOT plan STREQUAL "rank_blocker_only")
      message(FATAL_ERROR "terminal plan is ${plan}, not rank_blocker_only")
    endif()
    if(NOT backend MATCHES "^CPU-Levelized")
      message(FATAL_ERROR "solve ran on '${backend}', not the levelized CPU backend")
    endif()
    if(NOT ran EQUAL iters)
      message(FATAL_ERROR "solve stopped at ${ran} of ${iters} iterations")
    endif()
    if(threads EQUAL 1 AND NOT eff EQUAL 1)
      message(FATAL_ERROR "--cpu-threads 1 ran on ${eff} threads")
    endif()
    to_milli("${loop_ms}" loop_us)
    math(EXPR us "${loop_us} / ${ran}")
    list(APPEND runs ${us})
    if(meas_us STREQUAL "" OR us LESS meas_us)
      set(meas_us ${us})
    endif()
  endforeach()

  math(EXPR est_x100 "${est_us} * 100 / ${meas_us}")
  math(EXPR est_x2 "${est_us} * 2")
  math(EXPR meas_x2 "${meas_us} * 2")
  if(est_us GREATER meas_x2)
    message(FATAL_ERROR
      "${eff}T: ETA is ${est_x100}% of the measured iteration (estimate "
      "${est_us} us, measured ${meas_us} us, runs ${runs}) - more than 2x "
      "pessimistic on a rank-blocker turn")
  endif()
  if(est_x2 LESS meas_us)
    message(FATAL_ERROR
      "${eff}T: ETA is ${est_x100}% of the measured iteration (estimate "
      "${est_us} us, measured ${meas_us} us, runs ${runs}) - more than 2x "
      "optimistic (or other processes held the cores)")
  endif()

  # ---- the reported field is the ETA's own number, on both paths ----------
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
        "${eff}T: ${label} estimated_iteration_ms (${v} us) is not the ETA's "
        "slope (${est_us} us)")
    endif()
  endforeach()

  message(STATUS
    "cpu_eta ${eff}T: estimate ${est_us} us/iter vs measured ${meas_us} us/iter "
    "(fastest of ${runs}) = ${est_x100}%")
endforeach()

# T0/0b: the time-to-target instrumentation contract.
#
# A solver's speed is only meaningful as "wall clock to reach accuracy X"; a
# fixed-iteration iterations/sec figure is not comparable across engines and is
# not what the tool is bought for. `--convergence-log` records every
# exploitability probe so ONE solve to the tightest target yields time-to-
# accuracy for every looser threshold, and `--exploitability-interval` pins the
# probe cadence so the overshoot past the target is bounded and the measurement
# reproduces. scripts/bench-matrix.mjs's time_to_target fixtures read exactly
# this block.
#
# Three properties are pinned here:
#   1. --convergence-log emits a "convergence" block with probes.
#   2. Naming an interval PINS the cadence to it (the default cadence
#      self-calibrates against probe cost and coarsens without bound, which
#      would make the measured time-to-target depend on machine load).
#   3. WITHOUT the flag there is no "convergence" key at all — default solve
#      output stays byte-identical to previous builds, which is what the
#      bit-exact A/B in ROADMAP §5 depends on.
#
# Red on v2.5.0: the flags do not exist there, so property 1 finds no block.
#
# Usage: cmake -DEXE=<path-to-deepsolver_core> -P cli_convergence_log.cmake

if(NOT EXE)
  message(FATAL_ERROR "pass -DEXE=<deepsolver_core path>")
endif()

set(COMMON_ARGS
  --pot 100 --stack 500 --board AsKd7c
  --iterations 3000 --exploitability 0.5
  --backend cpu --postsolve exploitability
  --no-strategy-tree --no-progress)

set(INTERVAL 100)

# ---- 1 + 2: the instrumented run -------------------------------------------
execute_process(
  COMMAND ${EXE} ${COMMON_ARGS}
          --exploitability-interval ${INTERVAL} --convergence-log
  OUTPUT_VARIABLE out_on
  RESULT_VARIABLE rc_on
  ERROR_VARIABLE err_on)
if(NOT rc_on EQUAL 0)
  message(FATAL_ERROR "instrumented run failed (rc=${rc_on}): ${err_on}")
endif()

if(NOT out_on MATCHES "\"convergence\"")
  message(FATAL_ERROR
    "--convergence-log emitted no \"convergence\" block — the time-to-target "
    "harness (scripts/bench-matrix.mjs) has nothing to read")
endif()

string(REGEX MATCHALL "\"iteration\": [0-9]+" iters "${out_on}")
list(LENGTH iters n_probes)
if(n_probes LESS 3)
  message(FATAL_ERROR
    "expected at least 3 probes before reaching 0.5% at interval ${INTERVAL}, "
    "got ${n_probes} — the curve cannot support multi-threshold readings")
endif()

# Cadence: probe k must sit at exactly k*INTERVAL. A self-calibrated stride
# would drift to 2x/4x/16x the base interval and this loop would catch it.
set(k 0)
foreach(entry IN LISTS iters)
  math(EXPR k "${k} + 1")
  math(EXPR want "${k} * ${INTERVAL}")
  if(NOT entry STREQUAL "\"iteration\": ${want}")
    message(FATAL_ERROR
      "probe ${k} is at ${entry}, expected \"iteration\": ${want} — "
      "--exploitability-interval did not pin the cadence, so a measured "
      "time-to-target would depend on probe scheduling")
  endif()
endforeach()

# The run must actually reach the target through a probe, otherwise the
# early-stop path this instruments never executed.
if(NOT out_on MATCHES "\"early_stop_reason\": \"exploit_target\"")
  message(FATAL_ERROR
    "fixture did not stop on exploit_target — it no longer exercises the "
    "probe crossing this test exists to pin")
endif()

# ---- 3: default output carries no convergence key --------------------------
execute_process(
  COMMAND ${EXE} ${COMMON_ARGS}
  OUTPUT_VARIABLE out_off
  RESULT_VARIABLE rc_off
  ERROR_VARIABLE err_off)
if(NOT rc_off EQUAL 0)
  message(FATAL_ERROR "default run failed (rc=${rc_off}): ${err_off}")
endif()
if(out_off MATCHES "\"convergence\"")
  message(FATAL_ERROR
    "a default solve emitted a \"convergence\" block — this breaks the "
    "bit-exact A/B against previously shipped binaries")
endif()

message(STATUS
  "convergence log ok: ${n_probes} probes pinned at interval ${INTERVAL}, "
  "default output clean")

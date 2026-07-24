# Budget-gate regressions (review round 2, PR-0R3).
#
# The second Phase-0 review measured three reproducible budget bugs:
#   1. the host HARD gate summed old subtotals while the peak-host model fed
#      only the post-hoc budget_decision diagnostic — a solve could exceed
#      its host budget and still exit 0/success;
#   2. an AUTO GPU→CPU downgrade never rebuilt the footprint — the CPU solve
#      ran gated with cpu_state=0 and reported budget_decision
#      "gpu_oom_likely";
#   3. the VRAM ceiling compared solver STATE only — a dense turn solve
#      measurably exceeded the user's --gpu-memory-mb while reporting "ok".
#
# Each CASE asserts on PARSED JSON VALUES (string(JSON)), not just a
# provenance regex, per the review's test requirements.
#
# Usage: cmake -DEXE=<deepsolver_core> -DCASE=<name> -P cli_budget_gates.cmake

if(NOT EXE OR NOT CASE)
  message(FATAL_ERROR "pass -DEXE=<deepsolver_core path> -DCASE=<case name>")
endif()

set(COMMON --pot 100 --stack 500 --iterations 5 --exploitability 0
    --postsolve none --no-strategy-tree --no-progress)

# ---------------------------------------------------------------------------
if(CASE STREQUAL "host_reject_cpu")
  # 48 MiB host budget is below even the process floor — the peak-host gate
  # must reject BEFORE solving. Pre-fix binaries exit 0 with
  # budget_decision "host_oom_likely" written only as a diagnostic.
  execute_process(
    COMMAND ${EXE} --board AsKd7c --backend cpu --host-memory-mb 48 ${COMMON}
    OUTPUT_VARIABLE out RESULT_VARIABLE rc ERROR_VARIABLE err)
  if(rc EQUAL 0)
    message(FATAL_ERROR
      "expected hard reject at 48 MiB host budget, got exit 0 "
      "(peak-host model not wired into the hard gate)")
  endif()
  string(FIND "${err}${out}" "peak host RAM" hit)
  if(hit EQUAL -1)
    message(FATAL_ERROR
      "reject fired but not from the peak-host gate. stderr: ${err}")
  endif()
  message(STATUS "host_reject_cpu: peak-host gate rejected as required")

# ---------------------------------------------------------------------------
elseif(CASE STREQUAL "host_reject_gpu")
  # GPU solves still spend host RAM (matchup tables + 2x strategy + CUDA
  # context). A 128 MiB host budget must reject a GPU solve upfront with
  # the "host-side even on a GPU solve" advice.
  execute_process(
    COMMAND ${EXE} --board AsKd7c --backend gpu --host-memory-mb 128 ${COMMON}
    OUTPUT_VARIABLE out RESULT_VARIABLE rc ERROR_VARIABLE err)
  if(rc EQUAL 0)
    message(FATAL_ERROR
      "expected host reject for GPU solve at 128 MiB, got exit 0")
  endif()
  string(FIND "${err}${out}" "peak host RAM" hit)
  if(hit EQUAL -1)
    message(FATAL_ERROR
      "reject fired but not from the peak-host gate. stderr: ${err}")
  endif()
  message(STATUS "host_reject_gpu: peak-host gate covers GPU solves")

# ---------------------------------------------------------------------------
elseif(CASE STREQUAL "auto_downgrade_refit")
  # AUTO picks GPU for the turn tree at 100 iterations (work = N × iters
  # must clear kGpuAutoMinWork = 150k; at the COMMON 5 iters the heuristic
  # goes straight to CPU and the downgrade path is never exercised); the
  # 20 MiB VRAM budget then forces the downgrade. The FINAL footprint must
  # describe the CPU solve that actually runs: cpu_state > 0,
  # budget_decision "ok" (the GPU shortfall belongs in fallback_reason),
  # backend label CPU. Trailing --iterations overrides COMMON's 5.
  execute_process(
    COMMAND ${EXE} --board AsKd7c2h --backend auto --gpu-memory-mb 20
            ${COMMON} --iterations 100
    OUTPUT_VARIABLE out RESULT_VARIABLE rc ERROR_VARIABLE err)
  if(NOT rc EQUAL 0)
    message(FATAL_ERROR "AUTO downgrade should still solve; rc=${rc}: ${err}")
  endif()
  string(JSON status GET "${out}" status)
  if(NOT status STREQUAL "success")
    message(FATAL_ERROR "expected status success, got ${status}")
  endif()
  string(JSON backend GET "${out}" backend)
  if(NOT backend MATCHES "^CPU")
    message(FATAL_ERROR "expected CPU backend after downgrade, got ${backend}")
  endif()
  string(JSON cpu_state GET "${out}" resources estimated_cpu_state_bytes)
  if(cpu_state LESS_EQUAL 0)
    message(FATAL_ERROR
      "downgraded CPU solve reports estimated_cpu_state_bytes=${cpu_state} "
      "(footprint not rebuilt after AUTO GPU->CPU)")
  endif()
  string(JSON bd GET "${out}" resources budget_decision)
  if(NOT bd STREQUAL "ok")
    message(FATAL_ERROR
      "budget_decision must describe the FINAL backend; got \"${bd}\" "
      "(gpu_oom_likely belongs in fallback_reason)")
  endif()
  string(JSON fb GET "${out}" resources fallback_reason)
  string(FIND "${fb}" "Auto-downgraded" fbhit)
  if(fbhit EQUAL -1)
    message(FATAL_ERROR "fallback_reason must record the downgrade; got: ${fb}")
  endif()
  message(STATUS "auto_downgrade_refit: footprint rebuilt for the CPU solve")

# ---------------------------------------------------------------------------
elseif(CASE STREQUAL "gpu_device_total_reject")
  # Iso-engaged turn (mono flop + offsuit turn card): the rank-blocker CANNOT
  # activate, so the dense EV/valid upload really happens. 90 MiB sits
  # between the STATE estimate (~41 MiB — the old state-only check passes
  # and the solve then claims 182 MiB of device memory (measured
  # allocated_device_total 190,840,832 B), silently breaching the user's
  # ceiling) and the device TOTAL estimate (183.6 MiB, within 0.9% of
  # measured) — the honest ceiling rejects upfront. NOTE: singleton-iso
  # turns (e.g. AsKd7c2h) skip the dense upload and would make this fixture
  # meaningless — do not swap the board without re-checking iso engagement.
  execute_process(
    COMMAND ${EXE} --board Ah9h4h2c --backend gpu --gpu-memory-mb 90 ${COMMON}
    OUTPUT_VARIABLE out RESULT_VARIABLE rc ERROR_VARIABLE err)
  if(rc EQUAL 0)
    message(FATAL_ERROR
      "expected device-total reject at 90 MiB VRAM budget, got exit 0 "
      "(ceiling still compares solver state only)")
  endif()
  string(FIND "${err}${out}" "device-total" hit)
  if(hit EQUAL -1)
    message(FATAL_ERROR
      "reject fired but not from the device-total ceiling. stderr: ${err}")
  endif()
  message(STATUS "gpu_device_total_reject: VRAM ceiling gates on device total")

else()
  message(FATAL_ERROR "unknown CASE: ${CASE}")
endif()

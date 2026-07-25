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

elseif(CASE STREQUAL "singleton_device_estimate")
  # PR-4 TerminalRepresentationPlan: on a singleton-iso board the rank-
  # blocker serves every terminal and NO dense matchup uploads — the
  # device-total estimate must say so. Pre-plan binaries charged the dense
  # upload unconditionally: AsKd7c2h estimated 593 MB device total vs 98 MB
  # measured (6×), which could falsely downgrade AUTO solves under tight
  # VRAM budgets. Honest bound: device_total ≤ 1.5 × state (tree metadata +
  # reserve only on top). Also pins the plan label in the telemetry.
  execute_process(
    COMMAND ${EXE} --pot 100 --stack 500 --board AsKd7c2h
            --backend gpu --estimate-only
    OUTPUT_VARIABLE out RESULT_VARIABLE rc ERROR_VARIABLE err)
  if(NOT rc EQUAL 0)
    message(FATAL_ERROR "estimate-only failed rc=${rc}: ${err}")
  endif()
  string(JSON rep GET "${out}" resources terminal_representation)
  if(NOT rep STREQUAL "rank_blocker_only")
    message(FATAL_ERROR
      "expected terminal_representation rank_blocker_only on the singleton "
      "turn, got \"${rep}\"")
  endif()
  string(JSON dev GET "${out}" resources estimated_device_total_bytes)
  string(JSON gstate GET "${out}" resources estimated_gpu_state_bytes)
  math(EXPR bound "(${gstate} * 3) / 2")
  if(dev GREATER ${bound})
    message(FATAL_ERROR
      "device-total estimate ${dev} still prices the dense upload the "
      "rank-blocker skips (state ${gstate}, honest bound ${bound})")
  endif()
  string(JSON gmatch GET "${out}" resources estimated_gpu_matchup_bytes)
  if(NOT gmatch EQUAL 0)
    message(FATAL_ERROR
      "estimated_gpu_matchup_bytes must be 0 under rank_blocker_only, "
      "got ${gmatch}")
  endif()
  message(STATUS "singleton_device_estimate: plan-aware device estimate OK")

elseif(CASE STREQUAL "postsolve_rb_selfcheck")
  # A4-host inc 1+2: the blocker postsolve must reproduce the dense
  # postsolve numerically. Float summation order differs, so this is a
  # tolerance check, not a bit-exact diff — it is the arbiter for the
  # perspective-sign and payoff conventions of the port. The singleton
  # rainbow flop exercises showdown + both fold types through both BR
  # sweeps (exploitability engages OOP and IP traversers); the EV sweep
  # runs too (default postsolve) so a crash there would fail the run.
  # Increment 2 flipped the default: the PLAN drives the blocker path, so
  # the dense side of this A/B is now the DEEPSOLVER_POSTSOLVE_DENSE=1
  # escape hatch and the blocker side is the plain default.
  set(ARGS --pot 100 --stack 500 --board AsKd2c --iterations 60
      --exploitability 0 --backend cpu --no-strategy-tree --no-progress)
  set(ENV{DEEPSOLVER_POSTSOLVE_DENSE} "1")
  execute_process(
    COMMAND ${EXE} ${ARGS}
    OUTPUT_VARIABLE off_out RESULT_VARIABLE off_rc ERROR_VARIABLE off_err)
  unset(ENV{DEEPSOLVER_POSTSOLVE_DENSE})
  if(NOT off_rc EQUAL 0)
    message(FATAL_ERROR "dense postsolve run failed rc=${off_rc}: ${off_err}")
  endif()
  execute_process(
    COMMAND ${EXE} ${ARGS}
    OUTPUT_VARIABLE on_out RESULT_VARIABLE on_rc ERROR_VARIABLE on_err)
  if(NOT on_rc EQUAL 0)
    message(FATAL_ERROR "blocker postsolve run failed rc=${on_rc}: ${on_err}")
  endif()

  string(JSON off_exp GET "${off_out}" exploitability_pct)
  string(JSON on_exp  GET "${on_out}"  exploitability_pct)

  # |on - off| <= 0.05 pct. CMake math() is integer-only, so compare in
  # 1e-4 "micro-pct" units (the JSON prints fixed decimals, no exponents).
  function(to_micro out val)
    string(REGEX MATCH "^([0-9]+)" _ "${val}")
    set(int_part "${CMAKE_MATCH_1}")
    string(REGEX MATCH "\\.([0-9]+)" _ "${val}")
    set(frac "${CMAKE_MATCH_1}0000")
    string(SUBSTRING "${frac}" 0 4 frac)
    math(EXPR u "${int_part} * 10000 + ${frac}")
    set(${out} ${u} PARENT_SCOPE)
  endfunction()
  to_micro(off_u "${off_exp}")
  to_micro(on_u  "${on_exp}")
  math(EXPR diff_u "${on_u} - ${off_u}")
  if(diff_u LESS 0)
    math(EXPR diff_u "0 - ${diff_u}")
  endif()
  if(diff_u GREATER 500)
    message(FATAL_ERROR
      "blocker postsolve diverges from dense: dense=${off_exp} "
      "blocker=${on_exp} (|diff| ${diff_u} micro-pct > 500)")
  endif()
  message(STATUS
    "postsolve_rb_selfcheck: dense=${off_exp} blocker=${on_exp} "
    "(diff ${diff_u} micro-pct <= 500)")

else()
  message(FATAL_ERROR "unknown CASE: ${CASE}")
endif()

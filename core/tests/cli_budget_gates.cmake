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

elseif(CASE STREQUAL "host_dense_matchup_skip")
  # A4-host increment 3: on a RankBlockerOnly solve the host must not build
  # the dense nc² EV/valid/category tables at all — nothing reads them, and
  # they are the ~30 GB term on big spots. AsKd7c2h (singleton turn, 49
  # runout tables) priced 535 MB of host matchup tables on v2.4.0 and
  # measured ~1.02 GB peak RSS; after the skip it is 0.12 MB / ~235 MB.
  #
  # The SECOND half of this test is the one that keeps the first honest: the
  # skip is NOT valid whenever a player's range is narrow enough to engage
  # the CPU backends' active-list terminal kernels, which have no blocker
  # route. Such a solve must still report the dense tables as built. A test
  # that only checked the skip would pass just as happily on a build that
  # skipped them unconditionally — and that build reads empty tables.
  #
  # B1b inc 2 moved which fixture can prove that. The old one was the
  # medium_sparse preset — narrow on BOTH sides — and compaction dissolves
  # its premise: dropping the slots neither player holds leaves a space that
  # is 100% dense by construction, so the active-list kernels correctly stop
  # engaging and the tables are correctly skipped. What compaction cannot
  # remove is ASYMMETRY: a player whose range is sparse relative to the
  # UNION of both ranges still engages those kernels. So the guard now runs
  # a wide-IP / 3-combo-OOP solve, and a third case pins the new behaviour
  # (two-sided narrow ⇒ compaction engages ⇒ skip) so a regression that
  # silently stopped compacting would be caught here too.
  execute_process(
    COMMAND ${EXE} --pot 100 --stack 500 --board AsKd7c2h
            --iterations 5 --exploitability 0 --backend cpu
            --no-strategy-tree --no-progress
    OUTPUT_VARIABLE wide_out RESULT_VARIABLE wide_rc ERROR_VARIABLE wide_err)
  if(NOT wide_rc EQUAL 0)
    message(FATAL_ERROR "wide-range turn solve failed rc=${wide_rc}: ${wide_err}")
  endif()
  string(JSON rep GET "${wide_out}" resources terminal_representation)
  if(NOT rep STREQUAL "rank_blocker_only")
    message(FATAL_ERROR
      "fixture no longer exercises the blocker plan (got \"${rep}\")")
  endif()
  string(JSON dense GET "${wide_out}" resources host_dense_matchup)
  if(dense)
    message(FATAL_ERROR
      "dense host matchup tables were still materialized on a "
      "rank_blocker_only solve (A4-host inc 3 skip did not engage)")
  endif()
  string(JSON mbytes GET "${wide_out}" resources estimated_matchup_bytes)
  # 49 tables × nc² × 9 B is half a gigabyte; the rank tables are ~130 KB.
  # Anything above 1 MiB means the estimate still prices the dense tables.
  if(mbytes GREATER 1048576)
    message(FATAL_ERROR
      "estimated_matchup_bytes ${mbytes} still prices dense tables the "
      "solve no longer builds")
  endif()

  # Asymmetric: OOP holds 3 canonical combos, IP holds the full default
  # range. The union is therefore everything IP holds, so compaction cannot
  # engage, OOP stays sparse within it, and the active-list kernels must
  # still force the dense tables.
  execute_process(
    COMMAND ${EXE} --pot 100 --stack 500 --board AsKd7c2h
            --iterations 5 --exploitability 0 --backend cpu
            --no-strategy-tree --no-progress --oop-range "AA:1"
    OUTPUT_VARIABLE narrow_out RESULT_VARIABLE narrow_rc ERROR_VARIABLE narrow_err)
  if(NOT narrow_rc EQUAL 0)
    message(FATAL_ERROR "asymmetric-range solve failed rc=${narrow_rc}: ${narrow_err}")
  endif()
  string(JSON narrow_rep GET "${narrow_out}" resources terminal_representation)
  string(JSON narrow_dense GET "${narrow_out}" resources host_dense_matchup)
  string(JSON narrow_nc GET "${narrow_out}" resources canonical_combos)
  string(JSON narrow_live GET "${narrow_out}" resources live_combos)
  if(NOT narrow_rep STREQUAL "rank_blocker_only")
    message(FATAL_ERROR
      "asymmetric fixture no longer runs a blocker-plan board (got "
      "\"${narrow_rep}\") - it can no longer prove the guard")
  endif()
  if(NOT narrow_live EQUAL narrow_nc)
    message(FATAL_ERROR
      "asymmetric fixture compacted (${narrow_live} of ${narrow_nc}) - it no "
      "longer holds a player sparse within the union, so it cannot prove "
      "that the active-list kernels still force the dense tables")
  endif()
  if(NOT narrow_dense)
    message(FATAL_ERROR
      "asymmetric-range solve skipped the dense tables, but its active-list "
      "terminal kernels have no blocker route - they would read empty "
      "tables")
  endif()

  # B1b inc 2: narrow on BOTH sides. Compaction must engage, and with it the
  # active-list route retires — the space it would have walked IS the space.
  execute_process(
    COMMAND ${EXE} --pot 100 --stack 500 --board AsKd7c2h
            --iterations 5 --exploitability 0 --backend cpu
            --no-strategy-tree --no-progress
            --oop-range "AA:1,KK:1,QQ:1,AKs:1" --ip-range "JJ:1,TT:1,99:1,JTs:1"
    OUTPUT_VARIABLE b1b_out RESULT_VARIABLE b1b_rc ERROR_VARIABLE b1b_err)
  if(NOT b1b_rc EQUAL 0)
    message(FATAL_ERROR "two-sided narrow solve failed rc=${b1b_rc}: ${b1b_err}")
  endif()
  # Behaviour first, engagement second: a build that shipped the telemetry but
  # stopped compacting fails with the behavioural message rather than an
  # arithmetic one. (On a pre-B1b binary this case fails earlier still, on the
  # missing live_combos field — also red, just less informative.)
  string(JSON b1b_dense GET "${b1b_out}" resources host_dense_matchup)
  if(b1b_dense)
    message(FATAL_ERROR
      "two-sided narrow solve kept the dense tables: the compacted space is "
      "100% live, so no active-list kernel can engage and nothing reads them")
  endif()
  string(JSON b1b_nc GET "${b1b_out}" resources canonical_combos)
  string(JSON b1b_live GET "${b1b_out}" resources live_combos)
  if(NOT b1b_live LESS b1b_nc)
    message(FATAL_ERROR
      "B1b compaction did not engage on a two-sided narrow range "
      "(live_combos ${b1b_live} of ${b1b_nc})")
  endif()
  message(STATUS
    "host_dense_matchup_skip: wide=skipped (${mbytes} B), asymmetric=kept, "
    "two-sided-narrow=compacted ${b1b_live}/${b1b_nc} and skipped")

elseif(CASE STREQUAL "rainbow_gate_enumerates")
  # A4-host increment 4: the tree builder's runout gate used to charge the
  # DENSE matchup price on every board. On a rainbow flop that is 2303
  # projected leaves × 1176² × 9 B ≈ 28.6 GB, so the gate collapsed the tree
  # at ANY sane budget — even though the solve (rank/fold blockers, inc 3)
  # really needs 6.2 MB of rank tables there. Now the gate prices what
  # precompute will build, and the enumerated tree is decided by the CFR
  # state instead: solver.h's ① check, mirrored here by --estimate-only.
  #
  # Both directions are pinned, because a build that simply never collapses
  # would pass a one-sided test and then OOM:
  #   24 GB budget → enumerates (state 16.3 GB + 2× strategy fits)
  #    6 GB budget → still collapses (it does not)
  # On v2.4.0 BOTH report the collapsed tree's 216 player nodes.
  execute_process(
    COMMAND ${EXE} --pot 100 --stack 500 --board AsKd7c --iterations 200
            --estimate-only --backend cpu --host-memory-mb 24000
    OUTPUT_VARIABLE big_out RESULT_VARIABLE big_rc ERROR_VARIABLE big_err)
  if(NOT big_rc EQUAL 0)
    message(FATAL_ERROR "estimate at 24 GB failed rc=${big_rc}: ${big_err}")
  endif()
  string(JSON big_nodes GET "${big_out}" resources player_nodes)
  if(big_nodes LESS 10000)
    message(FATAL_ERROR
      "rainbow flop still collapses at a 24 GB budget (player_nodes "
      "${big_nodes}) - the builder is still charging the dense matchup "
      "price the blockers never pay")
  endif()

  execute_process(
    COMMAND ${EXE} --pot 100 --stack 500 --board AsKd7c --iterations 200
            --estimate-only --backend cpu --host-memory-mb 6000
    OUTPUT_VARIABLE small_out RESULT_VARIABLE small_rc ERROR_VARIABLE small_err)
  if(NOT small_rc EQUAL 0)
    message(FATAL_ERROR "estimate at 6 GB failed rc=${small_rc}: ${small_err}")
  endif()
  string(JSON small_nodes GET "${small_out}" resources player_nodes)
  if(small_nodes GREATER 10000)
    message(FATAL_ERROR
      "rainbow flop enumerated at a 6 GB budget (player_nodes "
      "${small_nodes}) - its CFR state needs ~21 GB, so the state gate is "
      "not holding")
  endif()
  message(STATUS
    "rainbow_gate_enumerates: 24 GB -> ${big_nodes} player nodes, "
    "6 GB -> ${small_nodes}")

elseif(CASE STREQUAL "peak_host_model")
  # Peak-host model recalibration (2026-07-27). Two measured corrections,
  # one assertion each.
  #
  # (a) The fused ev·valid postsolve matrix (tables × nc² × 4 B) is real host
  #     memory precompute builds on every dense-plan solve that will run a
  #     postsolve, and the model never charged it. On AsKsQs that is 211 MB
  #     on top of 503 MB of matchup tables — the whole of that board's -9.6%
  #     under-estimate. v2.4.0 reports ~503 MB here.
  execute_process(
    COMMAND ${EXE} --pot 100 --stack 500 --board AsKsQs --iterations 3
            --exploitability 0 --backend cpu --no-strategy-tree --no-progress
    OUTPUT_VARIABLE dense_out RESULT_VARIABLE dense_rc ERROR_VARIABLE dense_err)
  if(NOT dense_rc EQUAL 0)
    message(FATAL_ERROR "dense-board solve failed rc=${dense_rc}: ${dense_err}")
  endif()
  string(JSON dense_matchup GET "${dense_out}" resources estimated_matchup_bytes)
  # dense tables alone = 446 × 344² × 10 B = 528 MB; + fused = 739 MB.
  if(dense_matchup LESS 629145600)   # 600 MiB
    message(FATAL_ERROR
      "estimated_matchup_bytes ${dense_matchup} does not include the fused "
      "ev/valid postsolve table the solve really builds")
  endif()

  # (b) The finalized strategy is charged 2× on CPU but only ~1.25-1.36× on
  #     GPU (the flat download is released as the nested repack forms). With
  #     the flat 2× the enumerated rainbow priced 8.27 GB and got collapsed
  #     at an 8 GB budget; the honest 1.5× prices 6.96 GB against a 6.19 GB
  #     measured peak, so it must ENUMERATE there.
  execute_process(
    COMMAND ${EXE} --pot 100 --stack 500 --board AsKd7c --iterations 200
            --estimate-only --backend gpu --host-memory-mb 8000
    OUTPUT_VARIABLE gpu_out RESULT_VARIABLE gpu_rc ERROR_VARIABLE gpu_err)
  if(NOT gpu_rc EQUAL 0)
    message(FATAL_ERROR "gpu estimate failed rc=${gpu_rc}: ${gpu_err}")
  endif()
  string(JSON gpu_nodes GET "${gpu_out}" resources player_nodes)
  if(gpu_nodes LESS 10000)
    message(FATAL_ERROR
      "rainbow flop collapses on GPU at an 8 GB budget (player_nodes "
      "${gpu_nodes}) - the host model is still charging a finalized-strategy "
      "copy that a GPU solve never holds")
  endif()
  message(STATUS
    "peak_host_model: matchup ${dense_matchup} B incl. fused, GPU 8 GB "
    "budget enumerates ${gpu_nodes} player nodes")

elseif(CASE STREQUAL "gpu_strat_buffer")
  # B1a increment 3: the GPU keeps regrets + strategy_sum and derives the
  # regret-matched / averaged strategy inside each consumer, so a normal solve
  # allocates TWO strat-shaped buffers, not three. Node-LOCKED solves still
  # allocate the third — the lock override is a write into a materialized
  # strategy — so both directions are pinned here; a build that dropped the
  # buffer unconditionally would write locks into a null pointer.
  #
  # Uses allocated_state_bytes (backend truth, not an estimate). On AsKd7c2h
  # the strat term is ~5 MB/buffer against ~62 MB of full-tree buffers:
  # v2.4.0 allocates 91.9 MB, the unlocked solve now takes 76.9 MB.
  execute_process(
    COMMAND ${EXE} --pot 100 --stack 500 --board AsKd7c2h --iterations 3
            --exploitability 0 --backend gpu --postsolve none
            --no-strategy-tree --no-progress
    OUTPUT_VARIABLE free_out RESULT_VARIABLE free_rc ERROR_VARIABLE free_err)
  if(NOT free_rc EQUAL 0)
    message(FATAL_ERROR "unlocked GPU solve failed rc=${free_rc}: ${free_err}")
  endif()
  string(JSON free_state GET "${free_out}" resources allocated_state_bytes)

  execute_process(
    COMMAND ${EXE} --pot 100 --stack 500 --board AsKd7c2h --iterations 3
            --exploitability 0 --backend gpu --postsolve none
            --no-strategy-tree --no-progress
            --node-locks "[{\"history\":\"\",\"combo\":\"AhKh\",\"strategy\":[1,0,0,0]}]"
    OUTPUT_VARIABLE lock_out RESULT_VARIABLE lock_rc ERROR_VARIABLE lock_err)
  if(NOT lock_rc EQUAL 0)
    message(FATAL_ERROR "locked GPU solve failed rc=${lock_rc}: ${lock_err}")
  endif()
  string(JSON lock_state GET "${lock_out}" resources allocated_state_bytes)

  if(NOT lock_state GREATER free_state)
    message(FATAL_ERROR
      "locked solve (${lock_state} B) does not allocate more device state "
      "than the unlocked one (${free_state} B) - either the unlocked solve "
      "still keeps a current_strategy buffer, or the locked one lost the "
      "buffer its override writes into")
  endif()
  # One strat-shaped buffer is the difference. Anything larger means the two
  # solves differ in more than the strategy buffer and this stopped being a
  # test of increment 3.
  math(EXPR delta "${lock_state} - ${free_state}")
  math(EXPR bound "${free_state} / 4")
  if(delta GREATER ${bound})
    message(FATAL_ERROR
      "locked-vs-unlocked device state differs by ${delta} B, more than the "
      "single strat buffer this test is about (state ${free_state} B)")
  endif()
  message(STATUS
    "gpu_strat_buffer: unlocked ${free_state} B, locked ${lock_state} B "
    "(+${delta} B = the materialized strategy)")

elseif(CASE STREQUAL "peak_host_not_under")
  # The peak-host gate REJECTS solves whose predicted host footprint exceeds
  # the budget. It may be conservative; it must never be materially UNDER, or
  # it waves through solves that then exhaust host RAM.
  #
  # B1b inc 2 broke that property by fixing something else: compaction shrank
  # every nc-scaled term, and the terms that do NOT scale with nc — the flat
  # game tree, the GPU's host-side per-node index tables, and the strategy copy
  # the mid-loop exploitability probe materializes — had been hiding inside the
  # old over-charge. A narrow-range enumerated GPU solve read −17.9%.
  #
  # This runs exactly that shape: real ranges (so compaction engages), a
  # rainbow flop that enumerates, GPU, and the probe ON, then compares the
  # estimate the gate used against the measured peak RSS of the same run.
  # 120 iterations, not 40: the probe cadence means a short run never fires one,
  # and then the estimate's probe charge is compared against a peak that never
  # paid it — the test would pass for the wrong reason (measured at +51% margin
  # while the real regime reads +12%). The engagement assertion below pins it.
  execute_process(
    COMMAND ${EXE} --pot 100 --stack 500 --board AsKd7c
            --iterations 120 --backend gpu --no-progress
            --host-memory-mb 16384
            --oop-range "AA:1,KK:1,QQ:1,JJ:1,TT:1,99:1,88:1,77:1,66:1,55:1,AKs:1,AQs:1,AJs:1,ATs:1,A5s:1,A4s:1,KQs:1,KJs:1,KTs:1,QJs:1,QTs:1,JTs:1,T9s:1,98s:1,AKo:1,AQo:1,AJo:1,KQo:1"
            --ip-range "JJ:1,TT:1,99:1,88:1,77:1,66:1,55:1,44:1,33:1,22:1,AQs:1,AJs:1,ATs:1,A9s:1,A8s:1,KQs:1,KJs:1,KTs:1,QJs:1,QTs:1,JTs:1,J9s:1,T9s:1,98s:1,87s:1,76s:1,AQo:1,AJo:1,KQo:1"
    OUTPUT_VARIABLE ph_out RESULT_VARIABLE ph_rc ERROR_VARIABLE ph_err)
  if(NOT ph_rc EQUAL 0)
    message(FATAL_ERROR "peak-host fixture failed rc=${ph_rc}: ${ph_err}")
  endif()
  string(JSON ph_mode GET "${ph_out}" tree_mode)
  string(JSON ph_nc   GET "${ph_out}" resources canonical_combos)
  string(JSON ph_live GET "${ph_out}" resources live_combos)
  string(JSON ph_est  GET "${ph_out}" resources estimated_peak_host_bytes)
  string(JSON ph_meas GET "${ph_out}" resources measured_peak_rss_bytes)
  # Fixture validity first — every one of these is load-bearing. A collapsed
  # tree, an inert compaction or a missing measurement would each make the
  # comparison below pass without testing anything.
  if(NOT ph_mode STREQUAL "enumerated")
    message(FATAL_ERROR
      "fixture collapsed (tree_mode ${ph_mode}) - the under-estimate needs the "
      "enumerated tree's host terms")
  endif()
  if(NOT ph_live LESS ph_nc)
    message(FATAL_ERROR
      "compaction inert (${ph_live} of ${ph_nc}) - this fixture only exposes "
      "the gap once the nc-scaled over-charge is gone")
  endif()
  if(ph_meas LESS 100000000)
    message(FATAL_ERROR
      "measured peak RSS ${ph_meas} B is implausibly small - telemetry broken?")
  endif()
  # Prove the mid-loop probe actually fired: it calls finalize(), so host RSS
  # jumps by a whole strategy copy DURING iterations. Without that jump the run
  # never paid the cost the estimate charges, and the comparison is decoration.
  string(JSON ph_prep GET "${ph_out}" resources measured_rss_after_prepare_bytes)
  string(JSON ph_iter GET "${ph_out}" resources measured_rss_after_iterations_bytes)
  math(EXPR ph_prep_x2 "${ph_prep} * 2")
  if(ph_iter LESS ph_prep_x2)
    message(FATAL_ERROR
      "the exploitability probe never fired (RSS after prepare ${ph_prep} B, "
      "after iterations ${ph_iter} B) - raise --iterations until it does, or "
      "this fixture cannot test the probe's strategy copy")
  endif()
  if(ph_est LESS ph_meas)
    math(EXPR ph_short "(${ph_meas} - ${ph_est}) / 1048576")
    message(FATAL_ERROR
      "peak-host estimate is UNDER measured by ${ph_short} MiB "
      "(est ${ph_est} B vs measured ${ph_meas} B). The host gate must never "
      "under-charge - it exists to reject solves that do not fit.")
  endif()
  math(EXPR ph_margin "(${ph_est} - ${ph_meas}) * 100 / ${ph_meas}")
  message(STATUS
    "peak_host_not_under: enumerated, live ${ph_live}/${ph_nc}, "
    "estimate +${ph_margin}% over measured peak RSS")

else()
  message(FATAL_ERROR "unknown CASE: ${CASE}")
endif()

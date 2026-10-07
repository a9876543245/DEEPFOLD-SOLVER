# CLI input handling (2026-10-06 audit).
#
# Every case used to be either silently ignored or silently misread:
#   - node locks written with a space after a colon (Python's json.dumps)
#     read an empty history (the ROOT) or were dropped;
#   - a lock whose history names no node landed on the deepest node reached;
#   - unknown range tokens / out-of-range weights / duplicate board cards
#     solved "successfully";
#   - an unknown --history step navigated somewhere else;
#   - parse errors aborted with nothing on stdout or stderr.
#
# Usage: cmake -DEXE=<deepsolver_core> -DCASE=<name> -P cli_inputs.cmake

if(NOT EXE OR NOT CASE)
  message(FATAL_ERROR "pass -DEXE=<deepsolver_core path> -DCASE=<case name>")
endif()

set(COMMON --pot 100 --stack 200 --iterations 20 --exploitability 0
    --backend cpu --cpu-threads 2 --no-progress --strategy-tree-evs none)

function(expect_error label pattern)
  execute_process(COMMAND ${EXE} ${COMMON} ${ARGN}
    OUTPUT_VARIABLE out ERROR_VARIABLE err RESULT_VARIABLE rc)
  if(rc EQUAL 0)
    message(FATAL_ERROR "${label}: expected an error exit, got 0")
  endif()
  string(FIND "${err}" "\"status\": \"error\"" is_json)
  if(is_json EQUAL -1)
    message(FATAL_ERROR "${label}: no JSON error on stderr:\n${err}")
  endif()
  string(REGEX MATCH "${pattern}" hit "${err}")
  if(NOT hit)
    message(FATAL_ERROR "${label}: error does not match '${pattern}':\n${err}")
  endif()
endfunction()

function(run_ok label outvar)
  execute_process(COMMAND ${EXE} ${COMMON} ${ARGN}
    OUTPUT_VARIABLE out ERROR_VARIABLE err RESULT_VARIABLE rc)
  if(NOT rc EQUAL 0)
    message(FATAL_ERROR "${label}: expected success, got ${rc}:\n${err}")
  endif()
  set(${outvar} "${out}" PARENT_SCOPE)
endfunction()

if(CASE STREQUAL "bad_inputs")
  expect_error("duplicate board card" "duplicate card" --board AsKd7c2h2h)
  expect_error("unknown range token" "unknown hand" --board AsKd7c2h --ip-range "AA,ZZ")
  expect_error("weight above 1" "outside" --board AsKd7c2h --ip-range "AA:50")
  expect_error("bad iteration count" "Invalid arguments" --board AsKd7c2h --iterations abc)
  expect_error("unknown backend" "Invalid --backend" --board AsKd7c2h --backend gpuu)
  expect_error("unknown history step" "not a node" --board AsKd7c2h --history "Check,Raise_100")
  expect_error("lock on a missing node" "not a node"
    --board AsKd7c2h --node-locks "[{\"history\":\"Check,Raise_100\",\"combo\":\"AhAc\",\"strategy\":[0,0,1]}]")
  expect_error("lock with the wrong action count" "actions"
    --board AsKd7c2h --node-locks "[{\"history\":\"\",\"combo\":\"AhAc\",\"strategy\":[1]}]")
  expect_error("x size in a bet menu" "only applies to raises"
    --board AsKd7c2h --bet-sizing "{\"oop\":{\"turn\":{\"bet\":[\"2x\"]}}}")
  expect_error("zero size" "pot fractions" --board AsKd7c2h --river-sizes 0)
  expect_error("target player without target" "needs --target"
    --board AsKd7c2h --target-player oop)
  expect_error("bad target player" "Invalid --target-player"
    --board AsKd7c2h --target AA --target-player both)

elseif(CASE STREQUAL "lenient_inputs")
  # Range shorthands: "AK" (both shapes), a specific combo, no weight, a %.
  run_ok("range shorthands" out --board AsKd7c2h
    --ip-range "AK,QhJh,TT:50%" --oop-range "aa,KQs:0.5")
  string(JSON status GET "${out}" status)
  if(NOT status STREQUAL "success")
    message(FATAL_ERROR "range shorthands: status ${status}")
  endif()

  # A lock written with spaces (json.dumps style) must land on its node.
  # Lock every AA combo to check at the root: AA's root strategy is then
  # pure Check.
  run_ok("spaced lock" out --board AsKd7c2h
    --node-locks "[{\"history\": \"\", \"combo\": \"AA\", \"strategy\": [1, 0, 0]}]")
  string(JSON aa_check GET "${out}" combo_strategies AA Check)
  if(NOT aa_check GREATER_EQUAL 0.999)
    message(FATAL_ERROR "spaced lock: AA Check = ${aa_check}, expected 1")
  endif()

elseif(CASE STREQUAL "bet_sizing")
  # Pio-style menus: OOP bets 50% on the river root, IP may raise 2.5x.
  run_ok("custom menus" out --board AsKd7c2h3s
    --bet-sizing "{\"oop\":{\"river\":{\"bet\":[\"50%\"],\"allin\":false}},\"ip\":{\"river\":{\"raise\":[\"2.5x\"],\"allin\":false}}}")
  string(JSON root_labels GET "${out}" global_strategy)
  if(NOT root_labels MATCHES "Bet_50" OR root_labels MATCHES "All-in")
    message(FATAL_ERROR "custom menus: root actions ${root_labels}")
  endif()
  run_ok("custom menus, facing the bet" out --board AsKd7c2h3s --history Bet_50
    --bet-sizing "{\"oop\":{\"river\":{\"bet\":[\"50%\"],\"allin\":false}},\"ip\":{\"river\":{\"raise\":[\"2.5x\"],\"allin\":false}}}")
  string(JSON labels GET "${out}" global_strategy)
  # 2.5x of a 50 bet is a raise TO 125: 75 on top of the call, 37.5% of the
  # 200-chip pot after calling = Raise_38 (labels use the menus' "%").
  if(NOT labels MATCHES "Raise_38")
    message(FATAL_ERROR "custom menus: IP's actions facing the bet are ${labels}")
  endif()

elseif(CASE STREQUAL "serve")
  # --serve: the solve stays in memory and answers node / range queries.
  set(req "${CMAKE_CURRENT_BINARY_DIR}/cli_inputs_serve_requests.txt")
  file(WRITE "${req}"
    "{\"cmd\":\"node\",\"history\":\"Check\"}\n"
    "{\"cmd\":\"node\",\"history\":\"Check,Check\"}\n"
    "{\"cmd\": \"ranges\", \"history\": \"Check,Bet_75,Call\"}\n"
    "{\"cmd\":\"node\",\"history\":\"Nope\"}\n"
    "{\"cmd\":\"quit\"}\n")
  execute_process(COMMAND ${EXE} ${COMMON} --board AsKd7c2h --stack 300 --serve
    INPUT_FILE "${req}"
    OUTPUT_VARIABLE out ERROR_VARIABLE err RESULT_VARIABLE rc)
  if(NOT rc EQUAL 0)
    message(FATAL_ERROR "serve: exit ${rc}:\n${err}")
  endif()
  string(REPLACE "@@DEEPSOLVER_END@@" ";" parts "${out}")
  list(GET parts 0 result)
  list(GET parts 1 check)
  list(GET parts 2 chance)
  list(GET parts 3 ranges)
  list(GET parts 4 bad)
  string(JSON session GET "${result}" session)
  if(NOT session)
    message(FATAL_ERROR "serve: the result must announce a live session")
  endif()
  string(JSON acting GET "${check}" acting)
  string(JSON kind GET "${check}" kind)
  if(NOT acting STREQUAL "IP" OR NOT kind STREQUAL "player")
    message(FATAL_ERROR "serve: 'Check' must be IP's decision, got ${kind}/${acting}")
  endif()
  string(JSON kind GET "${chance}" kind)
  string(JSON nrun LENGTH "${chance}" runouts)
  if(NOT kind STREQUAL "chance" OR NOT nrun EQUAL 48)
    message(FATAL_ERROR "serve: 'Check,Check' must be the river deal (48 cards), got ${kind}/${nrun}")
  endif()
  string(JSON pot GET "${ranges}" pot)
  string(JSON init GET "${ranges}" oop_has_initiative)
  if(NOT pot EQUAL 250 OR init)
    message(FATAL_ERROR "serve: ranges after bet/call must be pot 250, IP initiative (got ${pot}/${init})")
  endif()
  string(JSON status GET "${bad}" status)
  if(NOT status STREQUAL "error")
    message(FATAL_ERROR "serve: an unknown history must be an error reply")
  endif()

elseif(CASE STREQUAL "target")
  # 2026-10-07: --target takes a grid label (the UI sends labels; they used
  # to come back empty), and --target-player gives an out-of-range hand a
  # small weight so the solve plays it instead of reading a zero-reach lane.
  set(R --board Td9d6h --oop-range "AA,KK,AKs,KQs" --ip-range "QQ,JJ,AQs,KJs")
  run_ok("label target" out ${R} --target AKs)
  string(JSON mix GET "${out}" target_combo_analysis strategy_mix)
  string(JSON n LENGTH "${mix}")
  if(n EQUAL 0)
    message(FATAL_ERROR "label target: AKs must get a strategy:
${out}")
  endif()
  run_ok("out of range, not added" out ${R} --target 72o)
  string(JSON mix GET "${out}" target_combo_analysis strategy_mix)
  string(JSON n LENGTH "${mix}")
  if(NOT n EQUAL 0)
    message(FATAL_ERROR "72o is in no range: no strategy without --target-player")
  endif()
  run_ok("out of range, added" out ${R} --target 72o --target-player oop)
  string(JSON best GET "${out}" target_combo_analysis best_action)
  if(best STREQUAL "")
    message(FATAL_ERROR "72o added to OOP's range must get a strategy:
${out}")
  endif()

elseif(CASE STREQUAL "serve_view")
  # 2026-10-07: on Td9d6h the 2c turn child stands for 2c and 2s. "#2s" must
  # show the 2s world (its board, its combos), not fail or show 2c's; a card
  # on the board is still an error.
  set(req "${CMAKE_CURRENT_BINARY_DIR}/cli_inputs_serve_view_requests.txt")
  file(WRITE "${req}"
    "{\"cmd\":\"node\",\"history\":\"Check,Check#2s\"}\n"
    "{\"cmd\":\"node\",\"history\":\"Check,Check#Td\"}\n"
    "{\"cmd\":\"quit\"}\n")
  execute_process(COMMAND ${EXE} ${COMMON} --board Td9d6h --serve
    --oop-range "22,AA,KK" --ip-range "QQ,JJ,AQs,KJs"
    --flop-sizes 0.5 --turn-sizes 0.5 --river-sizes 0.5
    INPUT_FILE "${req}"
    OUTPUT_VARIABLE out ERROR_VARIABLE err RESULT_VARIABLE rc)
  if(NOT rc EQUAL 0)
    message(FATAL_ERROR "serve_view: exit ${rc}:\n${err}")
  endif()
  string(REPLACE "@@DEEPSOLVER_END@@" ";" parts "${out}")
  list(GET parts 1 turn)
  list(GET parts 2 onboard)
  string(JSON status GET "${turn}" status)
  string(JSON board GET "${turn}" board)
  if(NOT status STREQUAL "ok" OR NOT board MATCHES "2s" OR board MATCHES "2c")
    message(FATAL_ERROR "serve_view: '#2s' must show the 2s board, got ${status} ${board}")
  endif()
  # The 2s is dealt: 2h2c is a live hand there and 2s2h is not (the 2c
  # child's own labels would say the opposite).
  string(JSON live ERROR_VARIABLE no_live GET "${turn}" combo_strategies 2h2c)
  string(JSON dead ERROR_VARIABLE no_dead GET "${turn}" combo_strategies 2s2h)
  if(no_live OR NOT no_dead)
    message(FATAL_ERROR "serve_view: the 2s world must list 2h2c and not 2s2h:\n${turn}")
  endif()
  string(JSON status GET "${onboard}" status)
  if(NOT status STREQUAL "error")
    message(FATAL_ERROR "serve_view: a turn card on the board must be an error reply")
  endif()

else()
  message(FATAL_ERROR "unknown CASE ${CASE}")
endif()

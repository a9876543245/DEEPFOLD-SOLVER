# The stitched Exact nav tree must keep ONE entry per (betting line × turn
# card), even when the strategy-tree node cap is smaller than the merge would
# otherwise fill.
#
# TrunkDecomposition::stitch_nav merges each turn subgame's extracted nav under
# its own base key ("<line>" for the lex-min runout, "<line>#<card>" for the
# rest). It used to stop merging the moment out.size() hit nav_max_nodes, so on
# a spot with many canonical runouts the FIRST subgame's deep entries consumed
# the whole cap and every later turn card was simply absent. The Runout Report
# keys off exactly those per-card root entries, so it rendered "n/a" for the
# whole board (observed on standard sizing at the 2000-node default cap).
#
# The fix admits each subgame's root entry unconditionally and lets the cap
# bound only the deeper entries, so the emitted count may exceed the cap by at
# most one entry per leaf. Both halves of that are asserted here:
#   - every canonical turn card still has its root entry
#   - the total exceeds the cap (impossible under the old first-come rule)
#
# Measured on this fixture: pre-fix 0 roots / 40 entries (exactly the cap),
# post-fix 22 roots / 62 entries. Deterministic CPU-subgame route (decompose
# "on" over an enumerable board), ~3 s.
#
# Usage: cmake -DEXE=<path-to-deepsolver_core> -P cli_stitched_nav_roots.cmake

if(NOT EXE)
  message(FATAL_ERROR "pass -DEXE=<deepsolver_core path>")
endif()

set(CAP 40)

execute_process(
  COMMAND ${EXE}
    --pot 55 --stack 165 --board Ah9h4h
    --ip-range "AA,KK,QQ,AKs" --oop-range "AA,KK,QQ,AKs"
    --flop-sizes 0.5 --turn-sizes 0.5 --river-sizes 0.5
    --iterations 4 --dcfr-schedule standard
    --decompose-runouts on --decompose-outer 1 --decompose-inner 4
    --postsolve exploitability --strategy-tree-evs visible
    --strategy-tree-max-nodes ${CAP}
  OUTPUT_VARIABLE out
  RESULT_VARIABLE rc
  ERROR_VARIABLE err)

if(NOT rc EQUAL 0)
  message(FATAL_ERROR "solve failed (rc=${rc}): ${err}")
endif()

# The stitched merge only runs on the decomposed route. A monolithic fallback
# here would silently test nothing.
if(NOT out MATCHES "\"backend\": \"decomposed")
  message(FATAL_ERROR
    "not a decomposed run — the fixture no longer exercises the stitched nav "
    "merge (check --decompose-runouts handling)")
endif()

# Root entries end right after the two-character turn token; deeper entries
# under the same runout continue with a comma, so they don't match.
string(REGEX MATCHALL "Check,Check#[2-9TJQKA][cdhs]\"" roots "${out}")
list(REMOVE_DUPLICATES roots)
list(LENGTH roots nroots)

# One "acting" per emitted strategy-tree entry.
string(REGEX MATCHALL "\"acting\":" acts "${out}")
list(LENGTH acts nacts)

if(nroots LESS 15)
  message(FATAL_ERROR
    "stitched nav kept only ${nroots} turn-card root entries under "
    "\"Check,Check\" (expected ~22) — per-runout roots are being starved by "
    "the node cap again, which blanks the Runout Report")
endif()

if(NOT nacts GREATER ${CAP})
  message(FATAL_ERROR
    "emitted ${nacts} entries with cap ${CAP} — the per-leaf root reserve is "
    "not in effect (the old first-come rule capped the total exactly)")
endif()

message(STATUS
  "stitched nav roots ok: ${nroots} runout roots, ${nacts} entries (cap ${CAP})")

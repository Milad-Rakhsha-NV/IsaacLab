#!/usr/bin/env bash
# Deferred, sequential, restart-safe continuation of the MASTER sweep after the
# master process (PID 144300) died unexpectedly mid-run.
#
# Context (2026-07-23 ~23:23Z): the master sweep was killed externally while
# running `g1 / newton_dvi_pspg / contact_15` at iteration 467/1000 -- the master
# log and that cell's run.log both freeze mid-iteration with NO `END ... exit=`
# marker, NO `Training time:`, NO Python traceback, and NO `FAILED` line. That
# signature is an external kill of the master bash process (resource/OOM-class),
# not a code/asset bug -- so NO code fix is warranted (unlike the earlier
# dr_legs.usda symlink asset fix). The kill left the contact_15 phase partly
# done: g1 Jacobi + APGD c15 COMPLETED; g1 PSPG c15 and the entire h1/go2/dr_legs
# c15 matrix are orphaned.
#
# The existing `resume_dr_legs_contact5.sh` wrapper (PID 175497) already detected
# the dead master and is re-running only the three dr_legs contact_5 cells,
# GPU-exclusive. That wrapper does NOT cover the orphaned contact_15 cells.
#
# This second wrapper waits (non-busy) for BOTH the dr_legs c5 wrapper AND its
# live training child to exit so nothing runs concurrently on the single GPU
# (environments/solvers stay strictly sequential), then re-invokes the existing
# restart-safe master script for the FULL budget set. Every already-COMPLETED
# cell is skipped by its marker (no FORCE), so only the orphaned contact_15 cells
# actually run, in order. Reversible; touches no solver/env logic. No commit.
set -uo pipefail
cd ~/repos/isaaclab-dvi

SWEEP_DIR="results/rl_solver_sweep"
RESUME_LOG="$SWEEP_DIR/resume_master_after_wrapper.log"

# The dr_legs c5 resume wrapper and (optionally) its current train child.
WAIT_WRAPPER_PID=${WAIT_WRAPPER_PID:-175497}

{
  echo "===== MASTER-CONTINUATION WRAPPER START UTC $(date -u +%FT%TZ) ====="
  echo "waiting for dr_legs c5 resume wrapper PID $WAIT_WRAPPER_PID (and any live train child) to exit..."
} | tee -a "$RESUME_LOG"

# Wait for the dr_legs c5 resume wrapper to finish.
while kill -0 "$WAIT_WRAPPER_PID" 2>/dev/null; do
  sleep 60
done

# Belt-and-suspenders: also wait until no rsl_rl train.py is running, so we never
# overlap on the GPU even if the wrapper's child briefly outlives it.
while pgrep -f 'scripts/reinforcement_learning/rsl_rl/train.py' >/dev/null 2>&1; do
  sleep 30
done

echo "dr_legs c5 wrapper PID $WAIT_WRAPPER_PID exited and GPU idle of train.py; starting master continuation UTC $(date -u +%FT%TZ)" | tee -a "$RESUME_LOG"

# Re-run the FULL budget matrix. Completed cells are skipped by COMPLETED marker
# (no FORCE), so only the orphaned contact_15 cells execute, sequentially.
BUDGETS="5 10 15" bash "$SWEEP_DIR/run_locomotion_sweep.sh" 2>&1 | tee -a "$RESUME_LOG"

echo "===== MASTER-CONTINUATION WRAPPER END UTC $(date -u +%FT%TZ) =====" | tee -a "$RESUME_LOG"

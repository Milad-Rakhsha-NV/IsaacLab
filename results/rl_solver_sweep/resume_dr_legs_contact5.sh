#!/usr/bin/env bash
# Deferred, sequential, restart-safe resume for the 3 dr_legs contact_5 runs
# that failed at 2026-07-23T10:22Z with the missing dr_legs.usda symlink
# (asset symlink created 13:46; dr_legs contact_10 subsequently ran clean).
#
# This waits for the currently-running master sweep to exit so nothing runs
# concurrently on the GPU (environments/solvers stay sequential), then invokes
# the existing restart-safe script for the contact_5 budget only. All other
# contact_5 runs already have COMPLETED markers and will be skipped, leaving
# just the 3 dr_legs re-runs. No FORCE, so nothing completed is clobbered.
set -uo pipefail
cd ~/repos/isaaclab-dvi

WAIT_PID=${WAIT_PID:-144300}
SWEEP_DIR="results/rl_solver_sweep"
RESUME_LOG="$SWEEP_DIR/resume_dr_legs_contact5.log"

{
  echo "===== RESUME WRAPPER START UTC $(date -u +%FT%TZ) ====="
  echo "waiting for master sweep PID $WAIT_PID to exit before resuming..."
} | tee -a "$RESUME_LOG"

# Wait (non-busy) for the master sweep to finish.
while kill -0 "$WAIT_PID" 2>/dev/null; do
  sleep 60
done

echo "master sweep PID $WAIT_PID exited; starting dr_legs contact_5 resume UTC $(date -u +%FT%TZ)" | tee -a "$RESUME_LOG"

# Re-run ONLY the contact_5 budget. Completed runs are skipped by the script.
BUDGETS=5 bash "$SWEEP_DIR/run_locomotion_sweep.sh" 2>&1 | tee -a "$RESUME_LOG"

echo "===== RESUME WRAPPER END UTC $(date -u +%FT%TZ) =====" | tee -a "$RESUME_LOG"

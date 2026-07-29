#!/usr/bin/env bash
set -uo pipefail
source ~/miniforge3/etc/profile.d/conda.sh
conda activate dvi
cd ~/repos/isaaclab-dvi

for CONTACT_ITERS in 5 10 15; do
  OUT="$PWD/results/rl_solver_sweep/raw/g1/newton_dvi_coupling2/contact_${CONTACT_ITERS}"
  mkdir -p "$OUT"
  LOG="$OUT/run.log"
  if [[ -f "$OUT/COMPLETED" ]]; then
    echo "Already completed: $OUT"
    continue
  fi
  {
    echo "===== START g1 newton_dvi_coupling2 UTC $(date -u +%FT%TZ) ====="
    echo "task=Isaac-Velocity-Flat-G1-v0 preset=newton_dvi max_iterations=1500 contact_iterations=$CONTACT_ITERS coupling_iterations=2"
    git rev-parse HEAD > "$OUT/isaaclab_git_revision.txt"
    git -C ~/repos/newton-dvi rev-parse HEAD > "$OUT/newton_git_revision.txt" 2>/dev/null || true
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader > "$OUT/gpu.txt" 2>/dev/null || true
  } | tee "$LOG"
  ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Velocity-Flat-G1-v0 --headless --max_iterations 1500 \
    presets=newton_dvi \
    env.sim.physics=newton_dvi \
    env.sim.physics.solver_cfg.contact_max_iterations="$CONTACT_ITERS" \
    env.sim.physics.solver_cfg.coupling_iterations=2 \
    2>&1 | tee -a "$LOG"
  status=${PIPESTATUS[0]}
  echo "===== END g1 newton_dvi_coupling2 contact_${CONTACT_ITERS} UTC $(date -u +%FT%TZ) exit=$status =====" | tee -a "$LOG"
  if [[ $status -eq 0 ]] && grep -q '^Training time:' "$LOG"; then
    date -u +%FT%TZ > "$OUT/COMPLETED"
  else
    echo "FAILED: inspect $LOG" | tee -a "$LOG"
    exit "$status"
  fi
done

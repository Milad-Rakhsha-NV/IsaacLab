#!/usr/bin/env bash
# Sequential Anymal-C APGD sweep over 10 and 5 contact iterations at zero compliance.
set -uo pipefail
source ~/miniforge3/etc/profile.d/conda.sh
conda activate dvi
cd ~/repos/isaaclab-dvi
BASE="$(pwd)/results/rl_solver_sweep/raw/anymal_c"
MASTER="$BASE/apgd_contact_iterations_10_5_master.log"
for contact in 10 5; do
  tag="apgd_coupling2_cache_on_contact${contact}_compliance_0"
  ROOT="$BASE/$tag"
  LOG="$ROOT/run.log"
  mkdir -p "$ROOT"
  if [[ -f "$ROOT/COMPLETED" ]]; then
    echo "Already completed: $ROOT" | tee -a "$MASTER"
    continue
  fi
  {
    echo "===== START anymal_c APGD contact_iterations=$contact compliance=0.0 UTC $(date -u +%FT%TZ) ====="
    echo "task=Isaac-Velocity-Flat-Anymal-C-v0 preset=newton_dvi_apgd max_iterations=500 joint_solver=sparse_ldl contact_solver=sparse_apgd contact_iterations=$contact coupling_iterations=2 post_stabilize_joints=false cache_factorization=true contact_compliance=0.0"
    git rev-parse HEAD > "$ROOT/isaaclab_git_revision.txt"
    git -C ~/repos/newton-dvi rev-parse HEAD > "$ROOT/newton_git_revision.txt" 2>/dev/null || true
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader > "$ROOT/gpu.txt" 2>/dev/null || true
  } | tee "$LOG" | tee -a "$MASTER"
  ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Velocity-Flat-Anymal-C-v0 --headless --max_iterations 500 \
    presets=newton_dvi_apgd env.sim.physics=newton_dvi_apgd \
    env.sim.physics.solver_cfg.contact_max_iterations="$contact" \
    env.sim.physics.solver_cfg.coupling_iterations=2 \
    env.sim.physics.solver_cfg.post_stabilize_joints=false \
    env.sim.physics.solver_cfg.cache_factorization=true \
    env.sim.physics.solver_cfg.contact_compliance=0.0 \
    2>&1 | tee -a "$LOG" | tee -a "$MASTER"
  status=${PIPESTATUS[0]}
  echo "===== END anymal_c APGD contact_iterations=$contact compliance=0.0 UTC $(date -u +%FT%TZ) exit=$status =====" | tee -a "$LOG" | tee -a "$MASTER"
  if [[ "$status" -eq 0 ]] && grep -q '^Training time:' "$LOG"; then
    date -u +%FT%TZ > "$ROOT/COMPLETED"
  else
    echo "FAILED: inspect $LOG" | tee -a "$LOG" | tee -a "$MASTER"
    exit "$status"
  fi
done
echo "===== SWEEP COMPLETE UTC $(date -u +%FT%TZ) =====" | tee -a "$MASTER"

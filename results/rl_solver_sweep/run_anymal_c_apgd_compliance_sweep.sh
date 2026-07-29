#!/usr/bin/env bash
# Anymal-C APGD compliance sweep. All settings match the validated APGD run;
# only contact_compliance changes.
set -uo pipefail
source ~/miniforge3/etc/profile.d/conda.sh
conda activate dvi
cd ~/repos/isaaclab-dvi
BASE="$(pwd)/results/rl_solver_sweep/raw/anymal_c"
for compliance in 1e-7 1e-6 1e-5; do
  tag="apgd_coupling2_cache_on_contact20_compliance_${compliance}"
  ROOT="$BASE/$tag"
  LOG="$ROOT/run.log"
  mkdir -p "$ROOT"
  if [[ -f "$ROOT/COMPLETED" ]]; then
    echo "Already completed: $ROOT"
    continue
  fi
  {
    echo "===== START anymal_c APGD compliance=$compliance UTC $(date -u +%FT%TZ) ====="
    echo "task=Isaac-Velocity-Flat-Anymal-C-v0 preset=newton_dvi_apgd max_iterations=500 joint_solver=sparse_ldl contact_solver=sparse_apgd contact_iterations=20 coupling_iterations=2 post_stabilize_joints=false cache_factorization=true contact_compliance=$compliance"
    git rev-parse HEAD > "$ROOT/isaaclab_git_revision.txt"
    git -C ~/repos/newton-dvi rev-parse HEAD > "$ROOT/newton_git_revision.txt" 2>/dev/null || true
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader > "$ROOT/gpu.txt" 2>/dev/null || true
  } | tee "$LOG"
  ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Velocity-Flat-Anymal-C-v0 --headless --max_iterations 500 \
    presets=newton_dvi_apgd env.sim.physics=newton_dvi_apgd \
    env.sim.physics.solver_cfg.contact_max_iterations=20 \
    env.sim.physics.solver_cfg.coupling_iterations=2 \
    env.sim.physics.solver_cfg.post_stabilize_joints=false \
    env.sim.physics.solver_cfg.cache_factorization=true \
    env.sim.physics.solver_cfg.contact_compliance="$compliance" \
    2>&1 | tee -a "$LOG"
  status=${PIPESTATUS[0]}
  echo "===== END anymal_c APGD compliance=$compliance UTC $(date -u +%FT%TZ) exit=$status =====" | tee -a "$LOG"
  if [[ "$status" -eq 0 ]] && grep -q '^Training time:' "$LOG"; then
    date -u +%FT%TZ > "$ROOT/COMPLETED"
  else
    echo "FAILED: inspect $LOG" | tee -a "$LOG"
    exit "$status"
  fi
done

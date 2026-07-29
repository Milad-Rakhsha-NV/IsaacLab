#!/usr/bin/env bash
# Sequential 500-PPO comparison for manager-based Anymal-C flat locomotion.
# DVI Jacobi and APGD use contact_max_iterations=40 from PhysicsCfg; only the
# contact solver changes. Both DVI runs use coupling_iterations=1 and
# post_stabilize_joints=true. MJWarp is the native baseline.
set -uo pipefail
source ~/miniforge3/etc/profile.d/conda.sh
conda activate dvi
cd ~/repos/isaaclab-dvi

ROOT="$(pwd)/results/rl_solver_sweep/raw/anymal_c"
TASK="Isaac-Velocity-Flat-Anymal-C-v0"
ITER=500

run_one() {
  local label="$1" preset="$2" solver="$3"
  local out="$ROOT/$label"
  local log="$out/run.log"
  mkdir -p "$out"
  if [[ -f "$out/COMPLETED" ]]; then
    echo "SKIP completed $label"
    return 0
  fi
  : > "$log"
  {
    echo "===== START anymal_c $label UTC $(date -u +%FT%TZ) ====="
    echo "task=$TASK preset=$preset solver=$solver max_iterations=$ITER contact_iterations=40 coupling_iterations=1 post_stabilize_joints=true"
    git rev-parse HEAD > "$out/isaaclab_git_revision.txt"
    git -C ~/repos/newton-dvi rev-parse HEAD > "$out/newton_git_revision.txt" 2>/dev/null || true
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader > "$out/gpu.txt" 2>/dev/null || true
  } | tee -a "$log"

  if [[ "$solver" == "dvi" ]]; then
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
      --task "$TASK" --headless --max_iterations "$ITER" \
      presets="$preset" env.sim.physics="$preset" \
      env.sim.physics.solver_cfg.coupling_iterations=1 \
      env.sim.physics.solver_cfg.post_stabilize_joints=true \
      env.sim.physics.solver_cfg.contact_max_iterations=40 \
      2>&1 | tee -a "$log"
  else
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
      --task "$TASK" --headless --max_iterations "$ITER" \
      presets="$preset" env.sim.physics="$preset" \
      2>&1 | tee -a "$log"
  fi
  local status=${PIPESTATUS[0]}
  echo "===== END anymal_c $label UTC $(date -u +%FT%TZ) exit=$status =====" | tee -a "$log"
  if [[ "$status" -eq 0 ]] && grep -q '^Training time:' "$log"; then
    date -u +%FT%TZ > "$out/COMPLETED"
  else
    echo "FAILED: inspect $log" | tee -a "$log"
  fi
}

run_one mjwarp newton_mjwarp mjwarp
run_one jacobi newton_dvi dvi
run_one apgd newton_dvi_apgd dvi

echo "===== ANYMAL-C MANAGER SWEEP COMPLETE $(date -u +%FT%TZ) ====="

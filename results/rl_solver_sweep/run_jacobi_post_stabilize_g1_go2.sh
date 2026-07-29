#!/usr/bin/env bash
set -uo pipefail
source ~/miniforge3/etc/profile.d/conda.sh
conda activate dvi
cd ~/repos/isaaclab-dvi
CONTACT_ITERS=${CONTACT_ITERS:-10}
ROOT="$(pwd)/results/rl_solver_sweep/raw"
run_one() {
  local label="$1" task="$2" iterations="$3"
  local out="$ROOT/$label/newton_dvi_post_stabilize/contact_${CONTACT_ITERS}"
  local log="$out/run.log"
  mkdir -p "$out"
  if [[ -f "$out/COMPLETED" ]]; then echo "SKIP completed $label"; return 0; fi
  : > "$log"
  {
    echo "===== START $label newton_dvi post_stabilize UTC $(date -u +%FT%TZ) ====="
    echo "task=$task preset=newton_dvi max_iterations=$iterations contact_iterations=$CONTACT_ITERS coupling_iterations=1 post_stabilize_joints=true"
    git rev-parse HEAD > "$out/isaaclab_git_revision.txt"
    git -C ~/repos/newton-dvi rev-parse HEAD > "$out/newton_git_revision.txt" 2>/dev/null || true
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader > "$out/gpu.txt" 2>/dev/null || true
  } | tee -a "$log"
  ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task "$task" --headless --max_iterations "$iterations" \
    presets=newton_dvi env.sim.physics=newton_dvi \
    env.sim.physics.solver_cfg.contact_max_iterations="$CONTACT_ITERS" \
    env.sim.physics.solver_cfg.coupling_iterations=1 \
    env.sim.physics.solver_cfg.post_stabilize_joints=true \
    2>&1 | tee -a "$log"
  local status=${PIPESTATUS[0]}
  echo "===== END $label newton_dvi post_stabilize UTC $(date -u +%FT%TZ) exit=$status =====" | tee -a "$log"
  if [[ $status -eq 0 ]] && grep -q '^Training time:' "$log"; then
    date -u +%FT%TZ > "$out/COMPLETED"
  else
    echo "FAILED: inspect $log" | tee -a "$log"
    return 1
  fi
}
run_one g1 Isaac-Velocity-Flat-G1-v0 1500
run_one go2 Isaac-Velocity-Flat-Unitree-Go2-v0 1000
echo "===== SWEEP COMPLETE $(date -u +%FT%TZ) ====="

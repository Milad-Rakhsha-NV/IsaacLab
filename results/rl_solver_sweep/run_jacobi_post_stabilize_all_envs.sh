#!/usr/bin/env bash
# Sequential Jacobi/post-stabilization benchmark across the requested environments.
set -uo pipefail
source ~/miniforge3/etc/profile.d/conda.sh
conda activate dvi
cd ~/repos/isaaclab-dvi

MAX_ITERATIONS=${MAX_ITERATIONS:-1000}
CONTACT_ITERS=${CONTACT_ITERS:-10}
ROOT="$(pwd)/results/rl_solver_sweep/raw"
mkdir -p "$ROOT"

# label, task, output preset
RUNS=(
  "ant Isaac-Ant-Direct-v0 newton_dvi"
  "humanoid Isaac-Humanoid-Direct-v0 newton_dvi"
  "h1 Isaac-Velocity-Flat-H1-v0 newton_dvi"
  "g1 Isaac-Velocity-Flat-G1-v0 newton_dvi"
  "go2 Isaac-Velocity-Flat-Unitree-Go2-v0 newton_dvi"
)

for row in "${RUNS[@]}"; do
  read -r label task preset <<< "$row"
  out="$ROOT/$label/${preset}_post_stabilize/contact_${CONTACT_ITERS}"
  mkdir -p "$out"
  log="$out/run.log"
  if [[ -f "$out/COMPLETED" ]]; then
    echo "SKIP completed $label"
    continue
  fi
  : > "$log"
  {
    echo "===== START $label $preset post_stabilize UTC $(date -u +%FT%TZ) ====="
    echo "task=$task preset=$preset max_iterations=$MAX_ITERATIONS contact_iterations=$CONTACT_ITERS coupling_iterations=1 post_stabilize_joints=true"
    git rev-parse HEAD > "$out/isaaclab_git_revision.txt"
    git -C ~/repos/newton-dvi rev-parse HEAD > "$out/newton_git_revision.txt" 2>/dev/null || true
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader > "$out/gpu.txt" 2>/dev/null || true
    python -m pip show mujoco-warp > "$out/mujoco_warp.txt" 2>&1 || true
  } | tee -a "$log"

  ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task "$task" --headless --max_iterations "$MAX_ITERATIONS" \
    presets="$preset" env.sim.physics="$preset" \
    env.sim.physics.solver_cfg.contact_max_iterations="$CONTACT_ITERS" \
    env.sim.physics.solver_cfg.coupling_iterations=1 \
    env.sim.physics.solver_cfg.post_stabilize_joints=true \
    2>&1 | tee -a "$log"
  status=${PIPESTATUS[0]}
  echo "===== END $label $preset post_stabilize UTC $(date -u +%FT%TZ) exit=$status =====" | tee -a "$log"
  if [[ $status -eq 0 ]] && grep -q '^Training time:' "$log"; then
    date -u +%FT%TZ > "$out/COMPLETED"
  else
    echo "FAILED: inspect $log" | tee -a "$log"
    # Continue to the next environment so one failure does not discard the sweep.
  fi
done

echo "===== SWEEP COMPLETE $(date -u +%FT%TZ) ====="

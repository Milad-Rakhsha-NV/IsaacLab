#!/usr/bin/env bash
# APGD sweep matching the selected Jacobi table configurations exactly;
# only the contact solver is changed from Jacobi to APGD.
set -uo pipefail
source ~/miniforge3/etc/profile.d/conda.sh
conda activate dvi
cd ~/repos/isaaclab-dvi

ROOT="$(pwd)/results/rl_solver_sweep/raw"

run_one() {
  local label="$1" task="$2" iterations="$3" contact="$4" coupling="$5" post="$6"
  local out="$ROOT/$label/newton_dvi_apgd/contact_${contact}"
  local log="$out/run.log"
  mkdir -p "$out"
  if [[ -f "$out/COMPLETED" ]]; then
    echo "SKIP completed $label"
    return 0
  fi
  : > "$log"
  {
    echo "===== START $label newton_dvi_apgd UTC $(date -u +%FT%TZ) ====="
    echo "task=$task preset=newton_dvi_apgd max_iterations=$iterations contact_iterations=$contact coupling_iterations=$coupling post_stabilize_joints=$post solver_comparison=Jacobi_to_APGD"
    git rev-parse HEAD > "$out/isaaclab_git_revision.txt"
    git -C ~/repos/newton-dvi rev-parse HEAD > "$out/newton_git_revision.txt" 2>/dev/null || true
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader > "$out/gpu.txt" 2>/dev/null || true
  } | tee -a "$log"
  ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task "$task" --headless --max_iterations "$iterations" \
    presets=newton_dvi_apgd env.sim.physics=newton_dvi_apgd \
    env.sim.physics.solver_cfg.contact_max_iterations="$contact" \
    env.sim.physics.solver_cfg.coupling_iterations="$coupling" \
    env.sim.physics.solver_cfg.post_stabilize_joints="$post" \
    2>&1 | tee -a "$log"
  local status=${PIPESTATUS[0]}
  echo "===== END $label newton_dvi_apgd UTC $(date -u +%FT%TZ) exit=$status =====" | tee -a "$log"
  if [[ "$status" -eq 0 ]] && grep -q '^Training time:' "$log"; then
    date -u +%FT%TZ > "$out/COMPLETED"
  else
    echo "FAILED: inspect $log" | tee -a "$log"
  fi
}

# Exact selected Jacobi table settings, with only Jacobi -> APGD changed.
run_one ant      Isaac-Ant-Direct-v0                 1000 10 1 true
run_one humanoid Isaac-Humanoid-Direct-v0            1000 10 1 true
run_one h1       Isaac-Velocity-Flat-H1-v0            1000 10 1 false
run_one g1       Isaac-Velocity-Flat-G1-v0            1500 15 2 false
run_one go2      Isaac-Velocity-Flat-Unitree-Go2-v0    500 15 1 true

echo "===== APGD TABLE SWEEP COMPLETE $(date -u +%FT%TZ) ====="

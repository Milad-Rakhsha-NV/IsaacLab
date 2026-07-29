#!/usr/bin/env bash
# Sequential G1 comparison: full coupling-2/cache-on, post-stabilized coupling-1/cache-on,
# and full coupling-2/cache-off. 1500 PPO iterations each.
set -uo pipefail
source ~/miniforge3/etc/profile.d/conda.sh
conda activate dvi
cd ~/repos/isaaclab-dvi

CONTACT_ITERS=${CONTACT_ITERS:-15}
MAX_ITERATIONS=${MAX_ITERATIONS:-1500}
ROOT="$(pwd)/results/rl_solver_sweep/raw/g1"

run_one() {
  local label="$1" coupling="$2" post="$3" cache="$4"
  local out="$ROOT/$label/contact_${CONTACT_ITERS}"
  local log="$out/run.log"
  mkdir -p "$out"
  if [[ -f "$out/COMPLETED" ]]; then
    echo "SKIP completed $label"
    return 0
  fi
  : > "$log"
  {
    echo "===== START g1 $label UTC $(date -u +%FT%TZ) ====="
    echo "task=Isaac-Velocity-Flat-G1-v0 preset=newton_dvi max_iterations=$MAX_ITERATIONS contact_iterations=$CONTACT_ITERS block_iteration=$coupling coupling_iterations=$coupling post_stabilize_joints=$post ldl_caching=$cache cache_factorization=$cache"
    git rev-parse HEAD > "$out/isaaclab_git_revision.txt"
    git -C ~/repos/newton-dvi rev-parse HEAD > "$out/newton_git_revision.txt" 2>/dev/null || true
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader > "$out/gpu.txt" 2>/dev/null || true
  } | tee -a "$log"
  ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Velocity-Flat-G1-v0 --headless --max_iterations "$MAX_ITERATIONS" \
    presets=newton_dvi env.sim.physics=newton_dvi \
    env.sim.physics.solver_cfg.contact_max_iterations="$CONTACT_ITERS" \
    env.sim.physics.solver_cfg.coupling_iterations="$coupling" \
    env.sim.physics.solver_cfg.post_stabilize_joints="$post" \
    env.sim.physics.solver_cfg.cache_factorization="$cache" \
    2>&1 | tee -a "$log"
  local status=${PIPESTATUS[0]}
  echo "===== END g1 $label UTC $(date -u +%FT%TZ) exit=$status =====" | tee -a "$log"
  if [[ $status -eq 0 ]] && grep -q '^Training time:' "$log"; then
    date -u +%FT%TZ > "$out/COMPLETED"
  else
    echo "FAILED: inspect $log" | tee -a "$log"
    return 1
  fi
}

run_one newton_dvi_coupling2_cache_on 2 false true
run_one newton_dvi_post_stabilize_cache_on 1 true true
run_one newton_dvi_coupling2_cache_off 2 false false

echo "===== SWEEP COMPLETE $(date -u +%FT%TZ) ====="

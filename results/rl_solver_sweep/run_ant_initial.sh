#!/usr/bin/env bash
set -euo pipefail
source ~/miniforge3/etc/profile.d/conda.sh
conda activate dvi
cd ~/repos/isaaclab-dvi

STAMP=${STAMP:-$(date -u +%Y%m%d_%H%M%S)}
ENV_NAME=${ENV_NAME:-ant}
MAX_ITERATIONS=${MAX_ITERATIONS:-1000}
CONTACT_ITERS=${CONTACT_ITERS:-10}
OUT="$(pwd)/results/rl_solver_sweep/raw/${ENV_NAME}/initial_contact_${CONTACT_ITERS}"
mkdir -p "$OUT"

git rev-parse HEAD > "$OUT/isaaclab_git_revision.txt"
(git -C ~/repos/newton-dvi rev-parse HEAD || true) > "$OUT/newton_git_revision.txt"
(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true) > "$OUT/gpu.txt"

run_one() {
  local name="$1" preset="$2"; shift 2
  local log="$OUT/${name}.log"
  printf '\n===== START %s UTC %s =====\n' "$name" "$(date -u +%FT%TZ)" | tee "$log"
  printf 'preset=%s max_iterations=%s contact_iterations=%s\n' "$preset" "$MAX_ITERATIONS" "$CONTACT_ITERS" | tee -a "$log"
  ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Ant-Direct-v0 --headless \
    --max_iterations "$MAX_ITERATIONS" \
    env.sim.physics="$preset" "$@" 2>&1 | tee -a "$log"
  local status=${PIPESTATUS[0]}
  printf '===== END %s UTC %s exit=%s =====\n' "$name" "$(date -u +%FT%TZ)" "$status" | tee -a "$log"
  return "$status"
}

# All DVI variants use sparse LDL for bilateral joints and exactly 10 contact
# iterations. MJWarp is the native baseline; no artificial iteration override.
run_one jacobi newton_dvi \
  env.sim.physics.solver_cfg.contact_max_iterations="$CONTACT_ITERS"
run_one apgd newton_dvi_apgd \
  env.sim.physics.solver_cfg.contact_max_iterations="$CONTACT_ITERS"
run_one pspg newton_dvi_pspg \
  env.sim.physics.solver_cfg.contact_max_iterations="$CONTACT_ITERS"
run_one mjwarp newton_mjwarp

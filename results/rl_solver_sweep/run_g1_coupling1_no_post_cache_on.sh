#!/usr/bin/env bash
set -uo pipefail
source ~/miniforge3/etc/profile.d/conda.sh
conda activate dvi
cd ~/repos/isaaclab-dvi
CONTACT_ITERS=15
MAX_ITERATIONS=1500
out="$(pwd)/results/rl_solver_sweep/raw/g1/newton_dvi_coupling1_no_post_cache_on/contact_${CONTACT_ITERS}"
log="$out/run.log"
mkdir -p "$out"
if [[ -f "$out/COMPLETED" ]]; then echo "SKIP completed"; exit 0; fi
: > "$log"
{
 echo "===== START g1 newton_dvi_coupling1_no_post_cache_on UTC $(date -u +%FT%TZ) ====="
 echo "task=Isaac-Velocity-Flat-G1-v0 preset=newton_dvi max_iterations=$MAX_ITERATIONS contact_iterations=$CONTACT_ITERS block_iteration=1 coupling_iterations=1 post_stabilize_joints=false ldl_caching=true cache_factorization=true"
 git rev-parse HEAD > "$out/isaaclab_git_revision.txt"
 git -C ~/repos/newton-dvi rev-parse HEAD > "$out/newton_git_revision.txt" 2>/dev/null || true
 nvidia-smi --query-gpu=name,driver_version --format=csv,noheader > "$out/gpu.txt" 2>/dev/null || true
} | tee -a "$log"
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
 --task Isaac-Velocity-Flat-G1-v0 --headless --max_iterations "$MAX_ITERATIONS" \
 presets=newton_dvi env.sim.physics=newton_dvi \
 env.sim.physics.solver_cfg.contact_max_iterations="$CONTACT_ITERS" \
 env.sim.physics.solver_cfg.coupling_iterations=1 \
 env.sim.physics.solver_cfg.post_stabilize_joints=false \
 env.sim.physics.solver_cfg.cache_factorization=true \
 2>&1 | tee -a "$log"
status=${PIPESTATUS[0]}
echo "===== END g1 newton_dvi_coupling1_no_post_cache_on UTC $(date -u +%FT%TZ) exit=$status =====" | tee -a "$log"
if [[ $status -eq 0 ]] && grep -q '^Training time:' "$log"; then date -u +%FT%TZ > "$out/COMPLETED"; else exit 1; fi

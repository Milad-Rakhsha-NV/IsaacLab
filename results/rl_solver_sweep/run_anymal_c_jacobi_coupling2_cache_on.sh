#!/usr/bin/env bash
# Anymal-C manager-based 500-PPO Jacobi benchmark:
# coupling_iterations=2, post_stabilize_joints=false, cache_factorization=true,
# contact_max_iterations=20. The prior coupling=1/post=true run is preserved.
set -uo pipefail
source ~/miniforge3/etc/profile.d/conda.sh
conda activate dvi
cd ~/repos/isaaclab-dvi
ROOT="$(pwd)/results/rl_solver_sweep/raw/anymal_c/jacobi_coupling2_cache_on_contact20"
LOG="$ROOT/run.log"
mkdir -p "$ROOT"
if [[ -f "$ROOT/COMPLETED" ]]; then echo "Already completed: $ROOT"; exit 0; fi
{
 echo "===== START anymal_c jacobi_coupling2_cache_on_contact20 UTC $(date -u +%FT%TZ) ====="
 echo "task=Isaac-Velocity-Flat-Anymal-C-v0 preset=newton_dvi max_iterations=500 contact_iterations=20 coupling_iterations=2 post_stabilize_joints=false cache_factorization=true solver=sparse_jacobi"
 git rev-parse HEAD > "$ROOT/isaaclab_git_revision.txt"
 git -C ~/repos/newton-dvi rev-parse HEAD > "$ROOT/newton_git_revision.txt" 2>/dev/null || true
 nvidia-smi --query-gpu=name,driver_version --format=csv,noheader > "$ROOT/gpu.txt" 2>/dev/null || true
} | tee "$LOG"
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
 --task Isaac-Velocity-Flat-Anymal-C-v0 --headless --max_iterations 500 \
 presets=newton_dvi env.sim.physics=newton_dvi \
 env.sim.physics.solver_cfg.contact_max_iterations=20 \
 env.sim.physics.solver_cfg.coupling_iterations=2 \
 env.sim.physics.solver_cfg.post_stabilize_joints=false \
 env.sim.physics.solver_cfg.cache_factorization=true \
 2>&1 | tee -a "$LOG"
status=${PIPESTATUS[0]}
echo "===== END anymal_c jacobi_coupling2_cache_on_contact20 UTC $(date -u +%FT%TZ) exit=$status =====" | tee -a "$LOG"
if [[ "$status" -eq 0 ]] && grep -q '^Training time:' "$LOG"; then date -u +%FT%TZ > "$ROOT/COMPLETED"; else echo "FAILED: inspect $LOG" | tee -a "$LOG"; fi

#!/usr/bin/env bash
set -uo pipefail
source ~/miniforge3/etc/profile.d/conda.sh
conda activate dvi
cd ~/repos/isaaclab-dvi
CONTACT_ITERS=10
MAX_ITERATIONS=1000
out="$(pwd)/results/rl_solver_sweep/raw/humanoid/newton_mjwarp/contact_${CONTACT_ITERS}"
log="$out/run.log"
mkdir -p "$out"
if [[ -f "$out/COMPLETED" ]]; then echo "SKIP completed"; exit 0; fi
: > "$log"
{
 echo "===== START humanoid newton_mjwarp UTC $(date -u +%FT%TZ) ====="
 echo "task=Isaac-Humanoid-Direct-v0 preset=newton_mjwarp max_iterations=$MAX_ITERATIONS contact_iterations=$CONTACT_ITERS native_mjwarp_contact_budget=default"
 git rev-parse HEAD > "$out/isaaclab_git_revision.txt"
 git -C ~/repos/newton-dvi rev-parse HEAD > "$out/newton_git_revision.txt" 2>/dev/null || true
 nvidia-smi --query-gpu=name,driver_version --format=csv,noheader > "$out/gpu.txt" 2>/dev/null || true
} | tee -a "$log"
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
 --task Isaac-Humanoid-Direct-v0 --headless --max_iterations "$MAX_ITERATIONS" \
 presets=newton_mjwarp env.sim.physics=newton_mjwarp \
 2>&1 | tee -a "$log"
status=${PIPESTATUS[0]}
echo "===== END humanoid newton_mjwarp UTC $(date -u +%FT%TZ) exit=$status =====" | tee -a "$log"
if [[ $status -eq 0 ]] && grep -q '^Training time:' "$log"; then date -u +%FT%TZ > "$out/COMPLETED"; else exit 1; fi

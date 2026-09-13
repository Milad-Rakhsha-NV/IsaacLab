#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/home/horde/repos/isaaclab-dvi-prev-dvi-paper-lfsless-test
NEWTON=/home/horde/repos/newton-dvi-prev-dvi-paper-lfsless-test
PY=/home/horde/miniforge3/envs/dvi/bin/python
SOURCE_PATH=$(find "$ROOT/source" -mindepth 1 -maxdepth 1 -type d -printf '%p:')
export PYTHONPATH="${SOURCE_PATH}${NEWTON}"
export CUDA_VISIBLE_DEVICES=0
cd "$ROOT"
RUN_NAME="go2_dvi_jacobi_c2_nopost_$(date -u +%Y%m%dT%H%M%SZ)"
CONSOLE="$ROOT/logs/go2_dvi_jacobi_c2_nopost_console.log"
printf 'RUN_NAME=%s\n' "$RUN_NAME" | tee "$CONSOLE"
"$PY" scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Flat-Unitree-Go2-v0 \
  --num_envs 4096 \
  --seed 42 \
  --max_iterations 500 \
  --run_name "$RUN_NAME" \
  presets=newton_dvi env.sim.physics=newton_dvi 2>&1 | tee -a "$CONSOLE"
rc=${PIPESTATUS[0]}
echo "TRAIN_EXIT=$rc" | tee -a "$CONSOLE"
exit "$rc"

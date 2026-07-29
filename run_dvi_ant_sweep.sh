#!/usr/bin/env bash
set -u
# Activate the dvi conda env so isaaclab.sh resolves the correct interpreter
# (picks $CONDA_PREFIX/bin/python). Without this it falls through to system
# python3 which lacks gymnasium/isaacsim.
source ~/miniforge3/etc/profile.d/conda.sh
conda activate dvi
cd ~/repos/isaaclab-dvi
STAMP=$(date +%Y%m%d_%H%M%S)
for PRESET in ${SWEEP_PRESETS:-newton_dvi newton_dvi_apgd newton_dvi_aspg}; do
  echo "===== $(date -u +%H:%M:%S) START $PRESET ====="
  ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Ant-Direct-v0 \
    --headless \
    --max_iterations 1000 \
    env.sim.physics=$PRESET \
    2>&1 | tee sweep_logs/${PRESET}_${STAMP}.log
  echo "===== $(date -u +%H:%M:%S) DONE  $PRESET (exit ${PIPESTATUS[0]}) ====="
done
echo "===== SWEEP COMPLETE ====="

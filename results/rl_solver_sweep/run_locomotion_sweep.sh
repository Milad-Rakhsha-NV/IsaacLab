#!/usr/bin/env bash
set -uo pipefail
source ~/miniforge3/etc/profile.d/conda.sh
conda activate dvi
cd ~/repos/isaaclab-dvi

MAX_ITERATIONS=${MAX_ITERATIONS:-1000}
CONTACT_ITERS=${CONTACT_ITERS:-10}
BUDGETS=${BUDGETS:-"5 10 15"}
# Keep the matrix sequential and restart-safe: completed logs are skipped.
# Override with FORCE=1 to rerun an existing result.
FORCE=${FORCE:-0}
ROOT="$(pwd)/results/rl_solver_sweep/raw"
mkdir -p "$ROOT"

# task label, Isaac Lab task, solver preset, native-mjwarp flag
RUNS=(
  "g1 Isaac-Velocity-Flat-G1-v0 newton_dvi"
  "g1 Isaac-Velocity-Flat-G1-v0 newton_dvi_apgd"
  "g1 Isaac-Velocity-Flat-G1-v0 newton_dvi_pspg"
  "g1 Isaac-Velocity-Flat-G1-v0 newton_mjwarp mjwarp"
  "h1 Isaac-Velocity-Flat-H1-v0 newton_dvi"
  "h1 Isaac-Velocity-Flat-H1-v0 newton_dvi_apgd"
  "h1 Isaac-Velocity-Flat-H1-v0 newton_dvi_pspg"
  "h1 Isaac-Velocity-Flat-H1-v0 newton_mjwarp mjwarp"
  "go2 Isaac-Velocity-Flat-Unitree-Go2-v0 newton_dvi"
  "go2 Isaac-Velocity-Flat-Unitree-Go2-v0 newton_dvi_apgd"
  "go2 Isaac-Velocity-Flat-Unitree-Go2-v0 newton_dvi_pspg"
  "go2 Isaac-Velocity-Flat-Unitree-Go2-v0 newton_mjwarp mjwarp"
  "dr_legs Isaac-DrLegs-Walk-v0 newton_dvi"
  "dr_legs Isaac-DrLegs-Walk-v0 newton_dvi_apgd"
  "dr_legs Isaac-DrLegs-Walk-v0 newton_dvi_pspg"
)


for CONTACT_ITERS in $BUDGETS; do
for row in "${RUNS[@]}"; do
  read -r label task preset kind <<< "$row"
  if [[ "$kind" == mjwarp && "$CONTACT_ITERS" != 10 ]]; then continue; fi
  budget="contact_${CONTACT_ITERS}"
  out="$ROOT/$label/$preset/$budget"
  mkdir -p "$out"
  log="$out/run.log"
  if [[ "$FORCE" != 1 && -f "$out/COMPLETED" ]]; then
    echo "SKIP completed $label $preset"
    continue
  fi
  {
    echo "===== START $label $preset UTC $(date -u +%FT%TZ) ====="
    echo "task=$task preset=$preset max_iterations=$MAX_ITERATIONS contact_iterations=$CONTACT_ITERS"
    git rev-parse HEAD > "$out/isaaclab_git_revision.txt"
    git -C ~/repos/newton-dvi rev-parse HEAD > "$out/newton_git_revision.txt" 2>/dev/null || true
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader > "$out/gpu.txt" 2>/dev/null || true
    python -m pip show mujoco-warp > "$out/mujoco_warp.txt" 2>&1 || true
  } | tee "$log"
  # Manager-based locomotion selects backend-specific sensors/actuators through
  # the global preset as well as the physics path.  Set both explicitly.
  args=(presets="$preset" env.sim.physics="$preset")
  if [[ "$kind" != mjwarp ]]; then
    args+=(env.sim.physics.solver_cfg.contact_max_iterations="$CONTACT_ITERS")
  fi
  ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task "$task" --headless --max_iterations "$MAX_ITERATIONS" \
    "${args[@]}" 2>&1 | tee -a "$log"
  status=${PIPESTATUS[0]}
  echo "===== END $label $preset UTC $(date -u +%FT%TZ) exit=$status =====" | tee -a "$log"
  if [[ $status -eq 0 ]] && grep -q '^Training time:' "$log"; then
    date -u +%FT%TZ > "$out/COMPLETED"
  else
    echo "FAILED: inspect $log" | tee -a "$log"
  fi
done
done

# RFT training and MPM transfer study

This study implements inexpensive granular interaction through resistive force theory (RFT), trains Go2 with that model, and tests the frozen policy in particle-based MPM. A rigid-ground policy provides the reference. **MPM is excluded from training.**

Both repositories use branch **`milad/rft-mpm`**:

| Repository | Remote alias | Working directory |
|---|---|---|
| [Milad-Rakhsha-NV/IsaacLab](https://github.com/Milad-Rakhsha-NV/IsaacLab/tree/milad/rft-mpm) | `milad-rakhsha-nv` | `/home/horde/Documents/Repos/IsaacLab` |
| [Milad-Rakhsha/newton](https://github.com/Milad-Rakhsha/newton/tree/milad/rft-mpm) | `milad-rakhsha` | `/home/horde/Documents/Repos/newton-go2-rft` |

The computational Newton revision is pinned to `32f455bc4430f0954ae564fb40681fbaca95e7f3`. The original experiments installed this revision from the NV fork; the dependency now resolves the same immutable revision from the personal fork. Existing numerical results have not been rerun or relabeled as coming from different solver code.

## Implementation

Newton provides `SolverRFT`: a triangular-surface force integrator based on the published 3D-RFT angular fit. It clips surfaces to the occupied layer, integrates depth-dependent traction and torque, exposes per-world GPU material/height arrays, composes with rigid dynamics, and records granular impulses. Finite-layer clipping, rigid-contact export and restoration of MPM history between proxy passes were added during this study. See the [Newton summary](https://github.com/Milad-Rakhsha/newton/blob/milad/rft-mpm/RFT_MPM_STUDY.md).

Isaac Lab provides:

- Direct Go2 environments for rigid ground and reset-randomized RFT sand, with a shared observation/action contract, actuator configuration, self-collision, posture and gait criteria.
- RFT collision-surface binding for the whole robot, including feet, calves and thighs. Primitive colliders are triangulated and transformed into body-local surfaces. Granular forces enter articulated dynamics through the composed solver.
- Contact-dependent rewards and diagnostics that combine rigid contact cues with granular body impulses. Granular loads remain separate from discrete rigid contacts.
- A fixed rigid substrate and a reset-sampled granular free surface. Spawning, body-height rewards and failure thresholds reference the surface; the zero-layer case reduces to the corresponding rigid-floor configuration.
- Reset-based material randomization with a performance-gated material curriculum. Commands have no curriculum.
- Frozen-policy MPM evaluation with prepared beds, force arrows, capacity/boundary checks, saved state traces and explicit policy/terrain labels.
- Training, acceptance-check, material-probe, video and report workflows under `scripts/benchmarks/`.

Key source locations are [the direct task configuration](source/isaaclab_tasks/isaaclab_tasks/core/velocity/config/go2/rft_env_cfg.py), [the RFT adapter](source/isaaclab_newton/isaaclab_newton/physics/rft_solver.py), [the evaluator](scripts/benchmarks/evaluate_go2_granular.py), and [the report generator](scripts/benchmarks/report_go2_granular.py).

## Go2 training configuration

Both fresh runs used **4,096 environments**, **1,500 PPO iterations**, **24 steps per environment per iteration**, and **seed 42**: 147,456,000 training transitions per policy. The final checkpoint is `model_1499.pt`. The policy acts at 50 Hz.

| Quantity | Distribution or setting |
|---|---|
| Forward command | Uniform 0.1–1.0 m/s |
| Lateral command | Uniform −0.1–0.1 m/s |
| Yaw-rate command | Uniform −0.3–0.3 rad/s |
| Standing trials | 10% of resets |
| RFT layer thickness | Uniform 0–118 mm from the first episode |
| Bulk density at full material difficulty | Uniform 1,400–1,800 kg/m³ |
| Internal friction | Uniform 0.3–0.5 |
| Surface/internal friction ratio | Uniform 0.6–1.0 |
| Strength multiplier | Log-uniform 0.8–1.2 |
| Experimental bearing multiplier | 0.87574 |
| Velocity regularization | 0.04 m/s |
| Inertial correction | Disabled |

The 118 mm depth cap is half the approximately 236 mm **knee-to-sole length**. It is unrelated to the 22 mm collision-sphere radius. Each resetting environment receives its own material and command sample; other environments retain their state. Material distributions widen with sustained completion and tracking performance; depth and command ranges are available from the start. RFT reached material difficulty 1.0.

```bash
uv run isaaclab train --rl_library rsl_rl \
  --task Isaac-Velocity-Flat-UnitreeGo2-Rigid-Direct \
  --num_envs 4096 --max_iterations 1500 --seed 42 \
  --experiment_name go2_rigid_max1 'env.forward_command_range=[0.1,1.0]'

uv run isaaclab train --rl_library rsl_rl \
  --task Isaac-Velocity-Flat-UnitreeGo2-RFT-Curriculum-Direct \
  --num_envs 4096 --max_iterations 1500 --seed 42 \
  --experiment_name go2_rft_max1 'env.forward_command_range=[0.1,1.0]'
```

The saved configurations, TensorBoard events, curriculum records and final checkpoints are archived under `results/go2-rft-mpm/rsl_rl/`.

## Go2 frozen-policy results

Each final recording lasts 10 s, contains 500 frames at 50 FPS, uses seed 123 and requests zero lateral velocity and yaw. Both commands evaluate the **same policies trained up to 1.0 m/s**.

| Training / evaluation | Forward command | Mean planar error | First failure threshold |
|---|---:|---:|---|
| Rigid / rigid | 0.50 m/s | 0.018 m/s | None in 10 s |
| RFT / RFT | 0.50 m/s | 0.051 m/s | None in 10 s |
| RFT / MPM | 0.50 m/s | 0.166 m/s | None in 10 s |
| Rigid / MPM | 0.50 m/s | 0.478 m/s | 0.60 s |
| Rigid / rigid | 1.00 m/s | 0.034 m/s | None in 10 s |
| RFT / RFT | 1.00 m/s | 0.068 m/s | None in 10 s |
| RFT / MPM | 1.00 m/s | 0.969 m/s | 0.62 s |
| Rigid / MPM | 1.00 m/s | 0.972 m/s | 0.52 s |

The RFT-trained controller transfers better than the rigid reference in the nominal 0.5 m/s MPM recording. This is a single-seed observation, not a general robustness guarantee. Metrics include initial acceleration and time after failure. The 1.0 m/s videos were retitled from saved trajectories; the 0.5 m/s videos are new rollouts of the same checkpoints. Numerical trajectories were not edited.

MPM uses nominal depth 118 mm, density 1,600 kg/m³, internal friction 0.4 and a fully rough interface; PIC transfer, 20 mm grid, four substeps per 5 ms physics frame, 50 stress iterations, and one coupling pass with feedback factor 0.1. The bed is prepared for 0.5 s with the robot held fixed, preserving the resulting material history. Main video beds are **12 × 1.6 m at 1.0 m/s** and **12 × 3.2 m at 0.5 m/s**. Policies share the same bed at each speed; the width difference limits a strict comparison across speeds. A narrow-bed trial that left the bed is excluded.

The published report also includes short three-material comparisons at 0.5 m/s and the controlled MPM startup investigation at 1.0 m/s. The 1.0 m/s material-comparison section was removed from the report during cleanup. Rigid and RFT learning curves now share each metric axis; the analysis has 0.50 and 1.00 m/s subsections.

## Validation and outstanding limits

- The rigid policy passed the specified held-out acceptance checks. RFT standing drift and loaded-foot slip still fail; finishing the training budget does not establish convergence.
- Newton analytic and convergence investigations verify force integration, moments, rate/scaling behavior and composition contracts. They do not independently validate physical sand accuracy.
- MPM has not passed the force-convergence gate. Matching density, friction and depth does not establish an equivalent RFT/MPM response.
- The 1.0 m/s MPM collapse is coupling-sensitive. Smaller time steps alone did not remove it; four coupling passes or full feedback avoided the failure threshold in controlled **2 s** tests. Those checks do not establish accurate tracking or 10 s stability. The main videos retain the original numerical configuration.
- RFT omits evolving free surfaces, compaction and plastic history. Its near-rest bearing term is experimental, and exact tangential sticking is absent.
- The G1 extension has completed rigid training and passed all seven held-out acceptance gates. RFT training is in progress with a thin-to-deep material curriculum; the full G1 transfer study is not yet complete. See [G1 status and configuration](G1_RFT_STATUS.md) and the [live G1 report](http://localhost:18010/g1/report/).

The G1 drive regression failed before the integrator fix and passed afterward; the final focused file passed two tests. The regression protects bounded implicit-drive motion and unchanged free-space control through each coupling path, distinct from the existing leg-surface/reset test. No test-only production API was introduced.

## Published report and data

The [results README](results/go2-rft-mpm/README.md) explains the archive, Git LFS retrieval, serving the report, checking file hashes, and restoring inputs for plot regeneration. The [report HTML](results/go2-rft-mpm/report/index.html), eight videos, graphs, raw trajectories, material probes, final checkpoints and training configurations are committed with this branch. Large binary artifacts are stored through Git LFS.

The existing machine-local preview remains at `http://localhost:18010/report/` through the established SSH tunnel. The committed report can be served independently from any machine with a clone; it does not require the original workspace or running training services.

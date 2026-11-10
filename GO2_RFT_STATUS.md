# Go2 RFT training and MPM inference

Updated November 10, 2026 (Pacific). Fresh training is running. **Convergence and RFT-to-MPM transfer are not yet established.**

## Current experiment

- One shared Go2 policy trains with RFT only, across **4,096 environments**. MPM is reserved for frozen-policy inference.
- Commands use their full distribution from the first reset: forward 0.1–0.5 m/s, lateral ±0.1 m/s, yaw ±0.3 rad/s, and 10% standing trials. There is **no command curriculum**.
- Only material randomization has a performance-driven curriculum. Each resetting environment receives a new field; other environments retain their samples.
- A separate 4,096-environment rigid-flat control uses the same observation, action, reward, command, and PPO settings.
- Both fresh runs use seed 42 and initial action standard deviation 1.0. They start from scratch.

| Run | Task | Planned PPO iterations | Log directory under `logs/rsl_rl/` |
|---|---|---:|---|
| RFT material curriculum | `Isaac-Velocity-Flat-UnitreeGo2-RFT-Curriculum-Direct` | 1,500 | `go2_rft_material_curriculum/2026-11-10_19-05-34/` |
| Rigid control | `Isaac-Velocity-Flat-UnitreeGo2-Rigid-Direct` | 600 | `go2_rigid_infinite/2026-11-10_19-05-34/` |

Each PPO iteration collects 24 steps per environment, or 98,304 transitions. These budgets are starting budgets, not evidence of convergence. Checkpoints are saved every 50 iterations. Source at launch: Isaac Lab `1209b5518f`; Newton `cb4f57292ff8da2c7d422817ca9cdc539734b1f6`.

The earlier curriculum run with reduced initial commands was stopped and has been superseded. Its checkpoint is not resumed for this protocol.

## Live status and videos

On another machine, keep this tunnel running:

```bash
ssh -o ExitOnForwardFailure=yes -N -L 18010:127.0.0.1:8010 horde@mrakhsha-41eslp
```

Open [live status and curves](http://localhost:18010/live/), which refreshes every 30 seconds. It shows actual service state, iteration counts, checkpoint names, tracking error, material difficulty, episode completion, and throughput. [Raw live metrics](http://localhost:18010/live/status.json) include snapshot time and event age so stale logging is visible.

The [video gallery](http://localhost:18010/report/) labels historical recordings affected by the fluid-density bug. New-policy recordings must be identified by checkpoint and evaluated field. On the server itself, use port 8010 instead of the tunnel port 18010.

The gallery, monitor, and training jobs run as independent user services. A chat interruption does not terminate them:

```bash
systemctl --user status go2-rft-material-training go2-rigid-plane-training go2-granular-monitor go2-granular-gallery
journalctl --user -u go2-granular-monitor -n 20
```

Training stdout is served at `/training/rft.log` and `/training/rigid.log`. Actual artifacts live under `logs/`; checkpoints and videos are excluded from Git.

The `go2-granular-validation` service records an early RFT checkpoint preview, then waits for both training jobs to finish successfully. It automatically evaluates frozen policies on held-out achieved/full material banks, the rigid and finite-layer controls, and nominal MPM, and rebuilds the figures/gallery. Its current stage and completed recordings appear on the live page. MPM evaluation starts only after training ends. These scheduled evaluations remain pending until their artifacts exist.

## Corrections and verification

1. **Ambient fluid density:** the previous RFT adapter field named `density` was also accepted by the rigid solver as fluid density, adding unintended drag. The adapter now uses `bulk_density`, leaving MuJoCo ambient fluid density at zero. Historical RFT training and transfer results are confounded and cannot validate RFT performance. The frozen-policy evaluator asserts that ambient density is zero.
2. **Floor coverage:** a finite 100 m box left many of the 4,096 clones unsupported. The rigid control and finite-layer substrate now use an infinite collision plane. The rendered mesh remains finite. Earlier rigid runs using the box are invalid controls.
3. **Zero-layer limit:** RFT now clips surfaces between an optional bottom and free surface. At zero thickness it contributes exactly zero granular impulse. With identical prescribed actions, zero-layer RFT plus substrate agrees with the plain rigid task within 0.0005 over 100 control steps. The test includes clones 120 m from the origin.
4. **Command independence:** the same sampled commands are reproduced at material levels zero and 0.8; lateral/yaw commands are present at level zero. This regression failed before removing the command curriculum and passed afterward.
5. **Articulated coupling:** a nonzero-load momentum balance verifies that granular body impulses reach articulated dynamics exactly once, including fixed foot bodies.

Latest native Go2 integration check: **2 tests passed**, including the new command and distant-floor coverage. Full repository formatting/lint checks passed after the report formatting correction. Previously completed Newton checks remain applicable: **19 RFT tests**, and all eight standalone numerical gates on both CPU and CUDA. A warning-as-error Sphinx build passed before the latest task-only command/floor changes.

Both dependency declarations and `uv.lock` pin Newton `cb4f57292ff8da2c7d422817ca9cdc539734b1f6`. The active editable Newton checkout is at that clean revision; dependencies are not synchronized while training is active.

## Material curriculum

A single global difficulty `a` starts at zero and interpolates reset distributions continuously. Ten percent of episodes remain nominal-material anchors until `a=1`; anchors use the same full command distribution and do not qualify for progression.

| Property | Nominal, `a=0` | Full target, `a=1` |
|---|---:|---:|
| Bulk density | 1,600 kg/m³ | Uniform 1,400–1,800 kg/m³ |
| Internal friction coefficient | 0.4 | Uniform 0.3–0.5 |
| Surface/internal friction ratio | 1.0 | Uniform 0.6–1.0 |
| Strength multiplier | 1.0 | Log-uniform 0.8–1.2 |
| Free-surface elevation | 0 m | Uniform −0.02–0.02 m |

For example, density is `1600 + a*(sample-1600)`, and the strength multiplier is `exp(a*log_sample)`. Field bulk velocity is fixed at zero. Velocity randomization refers to robot commands.

Progression requires two consecutive windows of at least 4,096 qualifying completed episodes, each with at least 95% full 10 s episode completion and mean planar tracking error at most 0.10 m/s. Failures count; nominal anchors and episodes from an earlier level are excluded. A pass increases `a` by 0.1. Completion below 70% retreats by 0.1. Once `a=1` is reached, the full distribution freezes and cannot shrink to hide poor performance.

A stall before `a=1` means full-range training has not been achieved. Training return alone does not establish convergence. Difficulty snapshots are logged alongside checkpoints, but the full scheduler state is not embedded in RSL-RL checkpoints; a resume requires explicitly setting the matching initial difficulty and collecting fresh progression windows.

## Physics and task contract

The direct environment binds all 27 robot colliders, including hips, thighs, calves, feet, torso and head. Collision surfaces supply 1,536 nondegenerate RFT triangles at the training resolution. Triangles preserve original rigid collision geometry, mass and inertia; they approximate exterior surfaces rather than an exact Boolean union.

The Newton adapter evaluates RFT body wrenches, advances MJWarp articulation dynamics once per substep, and preserves external-force inputs. Integrated body impulses provide control-interval force visualizations. MPM uses the existing two-way proxy coupling only during evaluation.

The actor observes 45 proprioceptive/command values; the critic adds three true base linear-velocity values. Material identity and base linear velocity are absent from the actor. Actions are 12 joint-position offsets scaled by 0.25 rad. Control runs at 50 Hz; physics at 200 Hz with four Newton substeps. Episodes last 10 s. New runs spawn and measure base height relative to the sampled surface.

Default RFT training uses a half-space with no support plane and no particles. `bed_depth` enables a finite clipped layer with an explicit rigid substrate. Lowering the free surface of an unsupported half-space does not produce rigid terrain. Only the zero-thickness layer over a substrate has the verified rigid limit; positive-thickness confinement accuracy remains unvalidated.

The task enables an **experimental near-rest bearing extension** (`bearing=0.87574`, velocity regularization 0.04 m/s). It blends the published flowing RFT law with normal bearing and regularized sliding resistance. It supplies no exact tangential sticking or stress history. Its support/calibration is not independently validated; zero-action standing tests can sink into a low posture. These limitations remain visible in validation and are not fixed by a curriculum.

## Validation after training

Use the same frozen weights and compatible observations throughout:

1. Evaluate the achieved curriculum level and the full target distribution on held-out reset/command banks. Report completion, tracking, posture, motor use, and material coverage, alongside learning curves.
2. Evaluate the rigid policy on plain rigid ground and zero-thickness RFT over the same substrate. Then examine small positive layer depths separately.
3. Evaluate the RFT-trained policy in nominal RFT and in an explicitly matched MPM material setup. Record force-arrow videos and trajectories, retaining failures.
4. Establish MPM resolution, timestep, preparation and coupling convergence before interpreting differences as transfer error. The existing coarse finite-bed MPM setup is an exploratory reference, not an accepted material ground truth.

A completed training budget, a favorable reward plateau, or one video does not establish robust transfer. Multiple training seeds and a converged MPM reference are still required for a final study.

## Reproduction

From `/home/horde/Documents/Repos/IsaacLab`, in the existing uv environment:

```bash
uv run --no-sync isaaclab train --rl_library rsl_rl \
  --task Isaac-Velocity-Flat-UnitreeGo2-RFT-Curriculum-Direct \
  --num_envs 4096 --max_iterations 1500 --seed 42 \
  --experiment_name go2_rft_material_curriculum

uv run --no-sync isaaclab train --rl_library rsl_rl \
  --task Isaac-Velocity-Flat-UnitreeGo2-Rigid-Direct \
  --num_envs 4096 --max_iterations 600 --seed 42 \
  --experiment_name go2_rigid_infinite
```

Isaac Lab: [milad/go2-rft-mpm](https://github.com/Milad-Rakhsha-NV/IsaacLab/tree/milad/go2-rft-mpm). Newton: [milad/go2-rft](https://github.com/Milad-Rakhsha-NV/newton/tree/milad/go2-rft).

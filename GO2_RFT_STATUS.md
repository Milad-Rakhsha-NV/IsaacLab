# Go2: rigid, RFT and MPM policy comparisons

## Current study: forward commands up to 1.0 m/s

Both policies start fresh with **4,096 environments, 1,500 PPO iterations and seed 42**. Forward commands are reset-uniform over **0.1–1.0 m/s** from the first episode, with lateral ±0.1 m/s, yaw ±0.3 rad/s and 10% standing. There is no command curriculum. RFT retains its material curriculum and reset-randomized 0–118 mm layers. The previous policies used a maximum forward command of 0.5 m/s.

- [Live training and evaluation status](http://localhost:18010/live/).
- [Current report](http://localhost:18010/report/).
- [Automatically updated Markdown status](http://localhost:18010/max1/STATUS.md).
- [Previous 0.5 m/s report and four recordings](http://localhost:18010/previous_0p5_report/).

New training runs are `go2_rigid_max1/2026-11-11_08-33-41` and `go2_rft_max1/2026-11-11_08-33-41` under `logs/rsl_rl/`. Core training source is `66afe5d303`. The two native task tests pass; the command-range regression fails with the previous hard-coded sampler. A 16-environment, three-iteration PPO smoke confirms the new range in the saved configuration. The extended MPM bed also has a dedicated inference smoke test.

After both training services finish, `run_go2_speed_comparison.py` checks stand, 0.5 m/s and 1.0 m/s commands, plus three held-out reset-randomization banks per policy. It generates exactly four 10 s videos at **1.0 m/s forward, zero lateral and zero yaw**: rigid policy / rigid floor, RFT policy / RFT field, RFT policy / MPM sand, and rigid policy / MPM sand. The MPM videos use a prepared **12 × 1.6 × 0.118 m** bed, PIC transfer, 20 mm grid, four substeps, 50 stress iterations and one coupling pass with relaxation 0.1. MPM remains inference-only. Both MPM policies use identical material and numerical settings.

The pipeline also repeats the three-material tracking and support-force comparison at 1.0 m/s using the new frozen checkpoints. Short material trials use a 5 × 1.6 m bed. The earlier standalone force-convergence checks are retained; **MPM material equivalence remains unvalidated**. Acceptance failures are reported independently of training-budget completion and video generation. Each new video is published after its file, command and decoded-frame checks pass; remaining recordings stay clearly marked as pending. Physical failures remain visible.

The training and evaluation jobs run as durable services: `go2-max1-rigid-training`, `go2-max1-rft-training` and `go2-max1-evaluation`. Logs are `/training/rigid-max1.log`, `/training/rft-max1.log` and `/training/max1-evaluation.log`. Use the existing SSH tunnel described below. Generated artifacts stay under ignored `logs/go2_granular/max1/` paths.

```bash
uv run --no-sync isaaclab train --rl_library rsl_rl \
  --task Isaac-Velocity-Flat-UnitreeGo2-Rigid-Direct \
  --num_envs 4096 --max_iterations 1500 --seed 42 \
  --experiment_name go2_rigid_max1 'env.forward_command_range=[0.1,1.0]'

uv run --no-sync isaaclab train --rl_library rsl_rl \
  --task Isaac-Velocity-Flat-UnitreeGo2-RFT-Curriculum-Direct \
  --num_envs 4096 --max_iterations 1500 --seed 42 \
  --experiment_name go2_rft_max1 'env.forward_command_range=[0.1,1.0]'
```

## Previous study: forward commands up to 0.5 m/s

Updated November 11, 2026 (Pacific). **RFT training finished 1,500 iterations with 4,096 environments and the full material curriculum.** The final checkpoint is not accepted as converged: standing drift, loaded-foot slip, foot placement and stepping criteria failed. The report now compares the current policies at the maximum forward command; previous RFT/MPM experiments remain excluded.

The material investigation is documented in [GO2_MATERIAL_VALIDATION.md](GO2_MATERIAL_VALIDATION.md) and the [graphical report](http://localhost:18010/investigation/). It checks native MPM force convergence, RFT force matching, preparation and coupling history, and frozen-policy material sensitivity. The current MPM configuration has not passed the force-convergence gate. Matching density, friction and initial depth alone does not establish equivalent material response. No policy was retrained during this investigation.

## Current RFT run

- Task: `Isaac-Velocity-Flat-UnitreeGo2-RFT-Curriculum-Direct`.
- Run: `logs/rsl_rl/go2_rft_knee_depth_curriculum/2026-11-11_02-42-39/`.
- The previous 0–22 mm start / 0–100 mm target run is superseded and excluded from the report. Fresh initialization, seed 42, 1,500 PPO iterations completed; 24 steps per environment per iteration.
- Isaac Lab source at launch: `0e27c65786`; Newton pin: `6528eb5e7bb823fe35f0d238b2571431bb55a811`.
- Native task checks passed from the installed pinned Newton package. No dependency synchronization runs while training is active.
- MPM was excluded from training. The final frozen checkpoint is `model_1499.pt`. Training took 2,914.81 s (48.6 min) and collected 147,456,000 transitions.
- The last training tracking error was 0.0356 m/s with material difficulty 1.0; these training metrics do not supersede the failed held-out acceptance checks.

The fixed nominal-material 20 s tests had tracking errors of 0.0761 m/s while standing, 0.0374 m/s at 0.30 m/s, and 0.0639 m/s at 0.50 m/s. Loaded-foot slip was 0.176–0.290 m/s, above the 0.10 m/s gate. At 0.30 m/s at least one foot did not register repeated swing/landing events. One of 384 held-out randomized trials violated foot placement; none fell. The policy therefore remains a diagnostic checkpoint.

## Material height and reset randomization

The relevant length is **knee to sole**, measured as **235.915 mm** in the Go2 USD visual geometry and rounded to **236 mm** in the task configuration. Half is **118 mm**. Layer thickness is sampled independently at reset over **0–118 mm**, from the first episode onward. The initial range is checked against half the configured knee-to-sole length. The separate collision sphere has a 22 mm radius; it does not define this depth limit.

[Measured geometry and dimension PNG](http://localhost:18010/measurements/foot-lengths.png).

The bed has a fixed rigid substrate at `z=0`. Its RFT bottom remains fixed; the sampled free surface determines layer thickness. This is a finite sand layer over a floor, rather than a shifted unsupported half-space. Robot spawning, height reward and height termination use the sampled free surface.

| Quantity | Initial curriculum distribution | Full target distribution |
|---|---|---|
| Layer thickness | Uniform 0–0.118 m | Uniform 0–0.118 m |
| Bulk density | 1,600 kg/m³ | Uniform 1,400–1,800 kg/m³ |
| Internal friction | 0.4 | Uniform 0.3–0.5 |
| Surface/internal friction ratio | 1.0 | Uniform 0.6–1.0 |
| Strength multiplier | 1.0 | Log-uniform 0.8–1.2 |
| Forward command | Uniform 0.1–0.5 m/s | Same |
| Lateral command | Uniform −0.1–0.1 m/s | Same |
| Yaw command | Uniform −0.3–0.3 rad/s | Same |
| Standing trials | 10% | Same |

The depth range remains 0–118 mm throughout training. Density, friction and roughness interpolate linearly from nominal values, and strength interpolates in log space. The granular field is stationary; velocity randomization refers to commands.

All material and command samples change only for resetting environments. Other environments retain their material, height and state. There is **no command curriculum**.

The material curriculum advances by 0.1 after two qualifying windows of at least 4,096 episodes, each with ≥95% full 10 s completion and ≤0.10 m/s mean planar tracking error. Ten percent of episodes remain nominal-material anchors with the same depth distribution before full difficulty; anchors do not qualify for progression. Success below 70% causes retreat. At full difficulty the target distribution freezes.

Each run saves `material_curriculum.json` with the knee-to-sole reference length, initial depth cap, separate collision diameter, initial/target depth ranges and substrate height. Difficulty changes are saved alongside checkpoints. A stopped run requires explicit difficulty restoration and fresh progression windows when resumed.

## Shared locomotion fixes

The RFT curriculum inherits the validated rigid task's robot configuration, native DC motor model, self-collision, gait configuration, observations and action scaling. Both use 45 actor inputs, 48 critic inputs, 12 joint-position offsets scaled by 0.25 rad, and 50 Hz control.

The same reward settings cover hip/joint posture, standing posture, vertical/angular motion, joint acceleration and limits, foot slip, swing time, undesired rigid non-foot collisions and stance width. Feet crossing the body centerline terminate an episode independently of fall detection. Nonfinite joint states also terminate.

Native contact sensors report rigid collisions, including the substrate and self-collision. **Granular RFT loads are separate body wrenches.** The task combines rigid-contact cues with control-interval RFT body impulses for loaded/unloaded timing and contact-dependent rewards. Granular-load timers run at the 50 Hz control rate; the rigid-only control retains native sensor timers. A load above 1 N counts as loaded. Granular loading of calves and thighs is allowed; only rigid non-foot collisions receive the undesired-contact penalty. These cues affect rewards only; granular forces still enter articulated dynamics exactly once.

All robot collision surfaces remain coupled, including hips, thighs, calves, feet and torso. The RFT adapter exports rigid collision contacts through its composed MJWarp solver and retains separate granular impulses.

## Validated rigid reference

The rigid policy was trained with 4,096 environments for 600 iterations in 317.64 s. The fixed-command tests ran for 20 s; three held-out banks contained 128 randomized-command trials each, lasting 10 s. One 34 s recording tests standing, walking, the 0.50 m/s training maximum, lateral motion, turning and stopping.

| Rigid test | Mean planar velocity error | Mean contact slip |
|---|---:|---:|
| Stand | 0.0063 m/s | 0.0049 m/s |
| Walk at 0.30 m/s | 0.0140 m/s | 0.0476 m/s |
| Walk at 0.50 m/s | 0.0178 m/s | 0.0684 m/s |
| Held-out bank 1 | 0.0200 m/s | 0.0419 m/s |
| Held-out bank 2 | 0.0207 m/s | 0.0451 m/s |
| Held-out bank 3 | 0.0222 m/s | 0.0470 m/s |
| Command-sequence video | 0.0206 m/s | 0.0455 m/s |

All seven specified acceptance gates passed. There were no falls or foot-placement violations. Each foot stepped repeatedly in both walking tests. States remained finite, maximum torso tilt was 0.097 rad, maximum joint-limit overshoot was 0.0018 rad, and no motor saturation or contact-buffer overflow was observed. This is a checked one-seed simulation baseline.

## Verification and post-training evaluation

- Both native Go2 integration tests pass in 13.80 s with the corrected 118 mm cap, masked depth/material resets, articulated momentum balance and the zero-layer rigid limit. Restoring the previous collider-radius guard makes the 118 mm configuration fail as expected.
- A controlled granular-load test detects foot contact above the rigid floor while native collision forces are zero. Removing the granular-load cue makes this regression fail.
- Removing the composed solver's rigid-contact export makes the contact-enabled zero-layer test fail with `NotImplementedError`; forwarding it makes the test pass.
- All 19 Newton RFT tests pass. Newton and Isaac Lab formatting/lint checks pass.
- The 16-environment, three-iteration PPO smoke passes with the shared gait criteria and corrected 0–118 mm layer range; its saved manifest records the 236 mm reference length and 118 mm cap.

The final RFT policy was tested on fixed stand/walk/maximum commands and three held-out full-distribution banks. Full material-curriculum completion passed, but the gait and standing criteria above did not. The new recordings hold a single maximum forward command for their full duration.

An exploratory MPM inference test completed with the same frozen RFT weights despite those failed gait gates. The robot stayed upright for 10 s, remained inside the bed margin, and had mean planar tracking error 0.2318 m/s. This is poorer command tracking than the RFT clip, not a passed transfer-validation result. The MPM rigid entry includes the backing floor and uses the validated rigid solver settings. MPM resolution, preparation and coupling convergence remain necessary before treating transfer differences as material-model error.

The RFT configuration retains an experimental near-rest bearing term. Finite-layer clipping does not model confinement, compaction, evolving free surfaces or exact tangential sticking. The local numerical checks do not independently validate that material closure.

## Current maximum-forward recordings

All four recordings request **0.50 m/s forward**, zero lateral velocity and zero yaw, for **10 s**. Two frozen policies are compared: the validated rigid-ground checkpoint (`model_599.pt`) is tested on a rigid floor and in MPM sand; the final RFT checkpoint (`model_1499.pt`) is tested in RFT sand and MPM sand. Neither policy was trained in MPM, and neither receives fine-tuning for these recordings. Material parameters are fixed during recording. Training budgets were 600 iterations for the rigid policy and 1,500 for the RFT policy.

All sand recordings use a 118 mm initial depth, density 1,600 kg/m³, internal friction 0.4 and surface/internal friction ratio 1.0. Matching these inputs does not establish equivalence of the RFT and MPM constitutive laws. The MPM bed is 7 m long, with a 20 mm grid and 1,016,400 particles; its backing floor and bed-exit check use the same extent. The longer bed avoids reaching the old short bed's edge during a 10 s walk.

| Policy trained on | Terrain simulated in video | Mean planar velocity error | First failure check |
|---|---|---:|---|
| Rigid ground | Rigid floor | 0.0209 m/s | None in 10 s |
| RFT sand with material randomization | RFT sand | 0.0682 m/s | None in 10 s |
| RFT sand with material randomization | MPM sand | 0.2318 m/s | None in 10 s |
| Rigid ground | MPM sand | 0.4862 m/s | Low body height at 0.66 s |

All four videos contain 500 frames at 50 FPS, with the commanded vector `[0.5, 0, 0]` verified throughout their saved traces. Both MPM runs completed without execution errors, nonfinite particle positions, rigid-contact overflow or entering the bed-edge margin. All four evaluations use seed 123. This is a single-seed inference comparison.

The rigid-trained policy in MPM crossed the 0.16 m height threshold at 0.66 s, when its height was 0.1515 m above the initial sand surface. Its maximum torso tilt was 0.220 rad. The recorded failure is loss of body clearance. It averaged 0.0096 m/s forward during the last 5 s and finished at 0.1482 m body height. The full rollout remains visible without resets. The failure check also covers projected-gravity z greater than −0.3 and nonfinite body positions.

Each video has an embedded banner naming its training terrain, evaluation terrain and command. The report repeats these labels above each player and shows all four clips with explicit command/depth captions, green granular-force arrows, and speed/error/height curves relative to the initial free surface. Numerical summaries and state traces remain linked from each recording.

## Report and live access

- [Current report](http://localhost:18010/report/).
- [Live status and learning curves](http://localhost:18010/live/), refreshed every 30 s.
- [Rigid-trained policy on a rigid floor](http://localhost:18010/rigid_max_forward/rollout.mp4).
- [RFT-trained policy on RFT sand](http://localhost:18010/rft_max_forward/rollout.mp4).
- [RFT-trained policy on MPM sand](http://localhost:18010/mpm_max_forward/rollout.mp4).
- [Rigid-trained policy on MPM sand](http://localhost:18010/rigid_mpm_max_forward/rollout.mp4).

From a different machine, keep this tunnel running:

```bash
ssh -o ExitOnForwardFailure=yes -N -L 18010:127.0.0.1:8010 horde@mrakhsha-41eslp
```

On the server itself, use `http://127.0.0.1:8010/`. The HTTP server and monitor run independently of this chat. Training stdout is at `/training/rft-knee-depth.log`; final evaluation status is at `/training/rft-evaluation-status.json`. Generated figures, videos and checkpoints remain under ignored `logs/` paths.

```bash
systemctl --user status go2-rft-knee-depth-training go2-rft-knee-depth-validation go2-granular-monitor

uv run --no-sync isaaclab train --rl_library rsl_rl \
  --task Isaac-Velocity-Flat-UnitreeGo2-RFT-Curriculum-Direct \
  --num_envs 4096 --max_iterations 1500 --seed 42 \
  --experiment_name go2_rft_knee_depth_curriculum
```

Isaac Lab branch: [milad/go2-rft-mpm](https://github.com/Milad-Rakhsha-NV/IsaacLab/tree/milad/go2-rft-mpm).
Newton branch: [milad/go2-rft](https://github.com/Milad-Rakhsha-NV/newton/tree/milad/go2-rft).

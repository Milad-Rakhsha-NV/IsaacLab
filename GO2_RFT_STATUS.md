# Go2: rigid, RFT and MPM policy comparisons

Go2 complete: eight videos, both command plots, material comparisons and coupling checks

[Report](http://localhost:18010/report/)

Both policies trained with 4,096 environments for 1,500 iterations, seed 42, and forward commands 0.1–1.0 m/s.
The same frozen checkpoints are evaluated at 1.0 and 0.5 m/s. Every video is 10 s and has 500 frames at 50 FPS.
All eight recordings use evaluation seed 123 and fixed material parameters; these are single-seed comparisons.
The 1.0 m/s videos were regenerated from saved trajectories with new banners; 0.5 m/s uses new rollouts.

| Training / evaluation terrain | Forward command | Tracking error | First failure threshold |
|---|---:|---:|---|
| Rigid-trained policy on a rigid floor | 1.00 m/s | 0.034 m/s | None in 10 s |
| Rigid-trained policy on a rigid floor | 0.50 m/s | 0.018 m/s | None in 10 s |
| RFT-trained policy on RFT sand | 1.00 m/s | 0.068 m/s | None in 10 s |
| RFT-trained policy on RFT sand | 0.50 m/s | 0.051 m/s | None in 10 s |
| RFT-trained policy on MPM sand | 1.00 m/s | 0.969 m/s | 0.62 s |
| RFT-trained policy on MPM sand | 0.50 m/s | 0.166 m/s | None in 10 s |
| Rigid-trained policy on MPM sand | 1.00 m/s | 0.972 m/s | 0.52 s |
| Rigid-trained policy on MPM sand | 0.50 m/s | 0.478 m/s | 0.60 s |

MPM uses one coupling pass and feedback relaxation 0.1 in all main videos. The 1.0 m/s bed is 12 × 1.6 m;
the 0.5 m/s bed is 12 × 3.2 m to avoid a boundary exit. Both policies share a bed at each command.
The width difference limits a strict comparison across speeds. Sand depth, density and friction are unchanged.

An excluded narrow-bed 0.5 m/s trial reached the bed margin at 7.2 s before its 9.38 s clearance failure.
That fall is not counted as a material-induced failure.

The immediate 1.0 m/s collapse is sensitive to coupling numerics. Finer time steps did not remove it,
but four coupling passes or full feedback avoided the failure threshold in 2 s tests. This is not a
10 s convergence result. The video falls cannot be attributed solely to the RFT policy or material law.

MPM force convergence and material equivalence remain unvalidated. RFT standing and slip acceptance checks remain failed.
G1 training is stopped while this Go2 report is reviewed.

## Training and material settings

RFT reached material difficulty 1.0. Reset-based randomization covers density 1,400–1,800 kg/m³, internal friction 0.3–0.5, surface/internal friction ratio 0.6–1.0 and log-uniform strength 0.8–1.2. Layers span 0–118 mm from the first episode over a fixed substrate, half Go2's approximately 236 mm knee-to-sole length. Material distributions widen with performance; commands have no curriculum. Lateral commands span ±0.1 m/s, yaw rates ±0.3 rad/s, and 10% of resets request standing.

Both checkpoints are `model_1499.pt` in `logs/rsl_rl/go2_rigid_max1/2026-11-11_08-33-41/` and `logs/rsl_rl/go2_rft_max1/2026-11-11_08-33-41/`. MPM is excluded from training. The report combines rigid and RFT learning curves on shared axes, with comparison subsections at 0.5 and 1.0 m/s, a three-material comparison at 0.5 m/s, and the controlled MPM startup comparison. Material trials last 3 s and use 5 × 1.6 m beds; support includes contacts after failure. The older force-convergence investigation remains applicable and has not passed its acceptance gate.

The rigid policy passed the held-out checks. RFT standing drift and loaded-foot slip still fail. A completed training budget does not establish policy convergence or material equivalence.

## Access and artifacts

[Live report](http://localhost:18010/report/) · [Generated status](http://localhost:18010/max1/STATUS.md)

Keep the existing SSH tunnel running: `ssh -o ExitOnForwardFailure=yes -N -L 18010:127.0.0.1:8010 horde@mrakhsha-41eslp`. Generated videos, state traces, summaries and figures remain under ignored `logs/go2_granular/` paths. The 1.0 m/s original movies are preserved beside their retitled copies; their numerical trajectories are unchanged.

## Controlled MPM startup check at 1.0 m/s

These 2 s trials use the same checkpoint, seed 123 and prepared 5 × 1.6 × 0.118 m bed. The physics frame is 5 ms. Only the listed numerical control changes.

| Substeps | Coupling passes | Feedback factor | Tracking error | First failure threshold |
|---:|---:|---:|---:|---|
| 4 | 1 | 0.1 | 0.853 m/s | 0.58 s |
| 8 | 1 | 0.1 | 0.845 m/s | 0.56 s |
| 16 | 1 | 0.1 | 0.825 m/s | 0.56 s |
| 4 | 4 | 0.1 | 0.467 m/s | None in 2 s |
| 4 | 1 | 1 | 0.421 m/s | None in 2 s |

These results demonstrate coupling sensitivity. They do not establish a converged MPM reference, accurate 1.0 m/s tracking, or 10 s stability for the alternative settings.

## Previous study: forward commands up to 0.5 m/s

Updated November 11, 2026 (Pacific). **RFT training finished 1,500 iterations with 4,096 environments and the full material curriculum.** The final checkpoint is not accepted as converged: standing drift, loaded-foot slip, foot placement and stepping criteria failed. The report now compares the current policies at the maximum forward command; previous RFT/MPM experiments remain excluded.

The material investigation is documented in [GO2_MATERIAL_VALIDATION.md](GO2_MATERIAL_VALIDATION.md) and the [graphical report](http://localhost:18010/investigation/). It checks native MPM force convergence, RFT force matching, preparation and coupling history, and frozen-policy material sensitivity. The current MPM configuration has not passed the force-convergence gate. Matching density, friction and initial depth alone does not establish equivalent material response. No policy was retrained during this investigation.

### Current RFT run

- Task: `Isaac-Velocity-Flat-UnitreeGo2-RFT-Curriculum-Direct`.
- Run: `logs/rsl_rl/go2_rft_knee_depth_curriculum/2026-11-11_02-42-39/`.
- The previous 0–22 mm start / 0–100 mm target run is superseded and excluded from the report. Fresh initialization, seed 42, 1,500 PPO iterations completed; 24 steps per environment per iteration.
- Isaac Lab source at launch: `0e27c65786`; Newton pin: `6528eb5e7bb823fe35f0d238b2571431bb55a811`.
- Native task checks passed from the installed pinned Newton package. No dependency synchronization runs while training is active.
- MPM was excluded from training. The final frozen checkpoint is `model_1499.pt`. Training took 2,914.81 s (48.6 min) and collected 147,456,000 transitions.
- The last training tracking error was 0.0356 m/s with material difficulty 1.0; these training metrics do not supersede the failed held-out acceptance checks.

The fixed nominal-material 20 s tests had tracking errors of 0.0761 m/s while standing, 0.0374 m/s at 0.30 m/s, and 0.0639 m/s at 0.50 m/s. Loaded-foot slip was 0.176–0.290 m/s, above the 0.10 m/s gate. At 0.30 m/s at least one foot did not register repeated swing/landing events. One of 384 held-out randomized trials violated foot placement; none fell. The policy therefore remains a diagnostic checkpoint.

### Material height and reset randomization

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

### Shared locomotion fixes

The RFT curriculum inherits the validated rigid task's robot configuration, native DC motor model, self-collision, gait configuration, observations and action scaling. Both use 45 actor inputs, 48 critic inputs, 12 joint-position offsets scaled by 0.25 rad, and 50 Hz control.

The same reward settings cover hip/joint posture, standing posture, vertical/angular motion, joint acceleration and limits, foot slip, swing time, undesired rigid non-foot collisions and stance width. Feet crossing the body centerline terminate an episode independently of fall detection. Nonfinite joint states also terminate.

Native contact sensors report rigid collisions, including the substrate and self-collision. **Granular RFT loads are separate body wrenches.** The task combines rigid-contact cues with control-interval RFT body impulses for loaded/unloaded timing and contact-dependent rewards. Granular-load timers run at the 50 Hz control rate; the rigid-only control retains native sensor timers. A load above 1 N counts as loaded. Granular loading of calves and thighs is allowed; only rigid non-foot collisions receive the undesired-contact penalty. These cues affect rewards only; granular forces still enter articulated dynamics exactly once.

All robot collision surfaces remain coupled, including hips, thighs, calves, feet and torso. The RFT adapter exports rigid collision contacts through its composed MJWarp solver and retains separate granular impulses.

### Validated rigid reference

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

### Verification and post-training evaluation

- Both native Go2 integration tests pass in 13.80 s with the corrected 118 mm cap, masked depth/material resets, articulated momentum balance and the zero-layer rigid limit. Restoring the previous collider-radius guard makes the 118 mm configuration fail as expected.
- A controlled granular-load test detects foot contact above the rigid floor while native collision forces are zero. Removing the granular-load cue makes this regression fail.
- Removing the composed solver's rigid-contact export makes the contact-enabled zero-layer test fail with `NotImplementedError`; forwarding it makes the test pass.
- All 19 Newton RFT tests pass. Newton and Isaac Lab formatting/lint checks pass.
- The 16-environment, three-iteration PPO smoke passes with the shared gait criteria and corrected 0–118 mm layer range; its saved manifest records the 236 mm reference length and 118 mm cap.

The final RFT policy was tested on fixed stand/walk/maximum commands and three held-out full-distribution banks. Full material-curriculum completion passed, but the gait and standing criteria above did not. The new recordings hold a single maximum forward command for their full duration.

An exploratory MPM inference test completed with the same frozen RFT weights despite those failed gait gates. The robot stayed upright for 10 s, remained inside the bed margin, and had mean planar tracking error 0.2318 m/s. This is poorer command tracking than the RFT clip, not a passed transfer-validation result. The MPM rigid entry includes the backing floor and uses the validated rigid solver settings. MPM resolution, preparation and coupling convergence remain necessary before treating transfer differences as material-model error.

The RFT configuration retains an experimental near-rest bearing term. Finite-layer clipping does not model confinement, compaction, evolving free surfaces or exact tangential sticking. The local numerical checks do not independently validate that material closure.

### Current maximum-forward recordings

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

### Report and live access

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

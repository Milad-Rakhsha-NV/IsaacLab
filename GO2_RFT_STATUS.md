# Go2: validated rigid control and 0–118 mm RFT training

Updated November 10, 2026 (Pacific). **The new RFT domain-randomization run is active with 4,096 environments.** The report contains the validated rigid control and this run only. Previous RFT/MPM results have been removed from the report and dashboard.

## Current RFT run

- Task: `Isaac-Velocity-Flat-UnitreeGo2-RFT-Curriculum-Direct`.
- Run: `logs/rsl_rl/go2_rft_knee_depth_curriculum/2026-11-11_02-42-39/`.
- The previous 0–22 mm start / 0–100 mm target run is superseded and excluded from the report. Fresh initialization, seed 42, 1,500 PPO iterations planned; 24 steps per environment per iteration.
- Isaac Lab source at launch: `0e27c65786`; Newton pin: `6528eb5e7bb823fe35f0d238b2571431bb55a811`.
- Native task checks passed from the installed pinned Newton package. No dependency synchronization runs while training is active.
- MPM is excluded from training. Final frozen-policy evaluation runs automatically after training finishes.

Training is not accepted as converged until the final material distribution and held-out gait/tracking checks pass. Current iteration counts, material difficulty, episode completion and tracking error are available on the live dashboard.

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

After training, the final RFT policy is tested on fixed stand/walk/maximum commands and three held-out full-distribution banks. Full material-curriculum completion is an acceptance gate. The final report will distinguish a stalled curriculum from full-distribution convergence. A new RFT video is recorded with explicit commands and layer depth.

Only after those RFT checks pass is the same frozen policy evaluated in a depth-matched MPM bed. The MPM rigid entry includes the backing floor and uses the validated rigid solver settings. MPM resolution, preparation and coupling convergence remain necessary before treating transfer differences as material-model error.

The RFT configuration retains an experimental near-rest bearing term. Finite-layer clipping does not model confinement, compaction, evolving free surfaces or exact tangential sticking. The local numerical checks do not independently validate that material closure.

## Report and live access

- [Current report](http://localhost:18010/report/).
- [Live status and learning curves](http://localhost:18010/live/), refreshed every 30 s.
- [Validated rigid-control video](http://localhost:18010/rigid_gait_control/rollout.mp4).

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

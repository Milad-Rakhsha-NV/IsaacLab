# G1: rigid, RFT and MPM policy study

Both G1 training stages and native-simulator acceptance checks are complete on `milad/rft-mpm`. The rigid baseline passes all seven gates. RFT reaches the full material curriculum and passes five of eight gates; foot placement, tracking and loaded-foot velocity remain failed. Final paired-speed recordings, MPM inference and the results archive are in progress.

[Live G1 report](http://localhost:18010/g1/report/) · [Generated live status](http://localhost:18010/g1/STATUS.md) · [Published Go2 results](results/go2-rft-mpm/README.md)

## Training and policy initialization

| Setting | Rigid baseline | RFT adaptation |
|---|---|---|
| Initialization | Fresh weights | Accepted final rigid weights; optimizer reset |
| Environments and seed | 4,096; seed 42 | 4,096; seed 42 |
| PPO iterations | 1,500 | 1,500 additional |
| Transitions in this stage | 147,456,000 | 147,456,000 additional |
| End-to-end training time | 27.4 min | 51.8 min additional |
| Final checkpoint | `model_1499.pt` | `model_2998.pt` |
| Material curriculum | None | Full difficulty reached around iteration 1,270 |

This is a two-stage adaptation study, with more total training exposure for RFT. Learning plots use iteration within each stage. Short independent diagnostics shared the GPU during portions of training, so throughput includes that contention. No MPM simulation supplies training transitions.

Commands cover forward 0.1–1.0 m/s, lateral ±0.1 m/s and yaw ±0.3 rad/s from the first reset, with 10% standing. **There is no command curriculum.** Each resetting RFT environment samples its own material; other environments retain their state.

| RFT parameter | Initial range | Target at full difficulty |
|---|---|---|
| Layer depth over fixed substrate | Uniform 0–10 mm | Uniform 0–118 mm |
| Bulk density | 1,600 kg/m³ | Uniform 1,400–1,800 kg/m³ |
| Internal friction | 0.4 | Uniform 0.3–0.5 |
| Surface/internal friction ratio | 1.0 | Uniform 0.6–1.0 |
| Strength multiplier | 1.0 | Log-uniform 0.8–1.2 |

The performance-gated material curriculum widens depth and material properties. RFT retains bearing coefficient 0.87574, velocity regularization 0.04 m/s and zero inertial correction. The 118 mm target is below half the approximately 350 mm G1 knee-to-sole length. The final policy receives about 230 iterations at full material difficulty.

## Held-out native-simulator validation

Fixed commands use 16 environments for 20 s. Three randomized banks each use 128 environments for 10 s; both policies receive randomized commands and RFT also receives full-range materials. Observation noise and action exploration are disabled. All measurements include initial transients.

Criteria require zero falls, valid foot placement in every environment, mean tracking error below 0.10 m/s, standing drift below 0.03 m/s and loaded-foot velocity below 0.10 m/s. Both feet must register at least three step events in each fixed walking trial. The loaded-foot metric uses ankle/body velocity and includes rolling; it is a **proxy for slip**, not a contact-point slip measurement.

| Policy | Trial | Tracking error [m/s] | Falls | Valid foot placement | Loaded-foot velocity [m/s] |
|---|---|---:|---:|---:|---:|
| RIGID | Standing, 20 s | 0.003 | 0.0% | 100.0% | 0.003 |
| RIGID | 0.50 m/s, 20 s | 0.026 | 0.0% | 100.0% | 0.019 |
| RIGID | 1.00 m/s, 20 s | 0.043 | 0.0% | 100.0% | 0.031 |
| RIGID | Randomized bank, seed 123, 10 s | 0.036 | 0.0% | 100.0% | 0.019 |
| RIGID | Randomized bank, seed 456, 10 s | 0.038 | 0.0% | 100.0% | 0.020 |
| RIGID | Randomized bank, seed 789, 10 s | 0.040 | 0.0% | 100.0% | 0.021 |
| RFT | Standing, 20 s | 0.010 | 0.0% | 100.0% | 0.032 |
| RFT | 0.50 m/s, 20 s | 0.093 | 0.0% | 100.0% | 0.134 |
| RFT | 1.00 m/s, 20 s | 0.141 | 0.0% | 100.0% | 0.185 |
| RFT | Randomized bank, seed 123, 10 s | 0.067 | 0.0% | 100.0% | 0.117 |
| RFT | Randomized bank, seed 456, 10 s | 0.069 | 0.0% | 99.2% | 0.119 |
| RFT | Randomized bank, seed 789, 10 s | 0.069 | 0.0% | 100.0% | 0.114 |

Rigid passes every gate. RFT passes no-falls, standing, capacity, stepping and full-curriculum gates. It fails the aggregate foot-placement gate because one of 384 randomized environments violates the lateral clearance threshold. Its 1.0 m/s nominal tracking error and loaded-foot velocities exceed the acceptance limits. Finishing the training budget therefore does not establish complete locomotion validation.

## Inference and recording protocol

The final comparison contains four panels with **two 10 s videos each**, commanding 1.0 and 0.5 m/s with zero lateral velocity and yaw. Each pair uses the same frozen checkpoint and evaluation seed 123:

- Rigid-trained policy on a rigid floor.
- RFT-adapted policy in RFT sand.
- RFT-adapted policy in MPM sand.
- Rigid-trained policy in MPM sand.

Sand starts 118 mm deep at density 1,600 kg/m³, internal friction 0.4 and surface/internal ratio 1.0. The MPM videos share a prepared 12 × 3.2 m bed, PIC transfer, a 20 mm grid, four substeps per 5 ms physics frame, 50 stress iterations and one coupling pass with feedback factor 1.0. Preparation advances actual MPM material history for 0.5 s with the robot fixed. The earlier Go2 main videos used factor 0.1, so cross-robot outcomes are not a controlled numerical comparison.

Controlled 2 s G1 checks at both commands compare one-pass feedback factors 0.1 and 1.0 and four-pass refinement at 1.0, holding checkpoint, prepared 5 × 3.2 × 0.118 m bed and seed fixed. The feedback factor blends reaction wrenches as `w_next = (1 - alpha) * w_previous + alpha * w_MPM`; previous feedback persists across steps, so smaller factors smooth successive updates. Extra passes repeat the same physical interval with material history restored. These checks precede the long MPM videos. A three-material bank at 0.5 m/s compares tracking and support. Numerical traces, capacity checks, failure times and bed exits are retained; episodes continue after failure without resets.

## Shared mechanics and numerical checks

The direct tasks use the maintained minimal G1 asset with 37 actuated joints and 123 actor/critic inputs, including base linear velocity. Both policies use identical mechanics, actuators, observations, rewards, self-collision and implicit-fast integration. The shared foot-slip penalty is 1.0. Native tests verify full-leg coupling, clone-local reset and rigid/RFT/MPM drive parity before material contact; the focused file passes two tests.

Four invisible capsules cover thighs and calves, spanning parent frames to knee/ankle anchors with radii 45 and 35 mm. These approximations augment the authored foot and torso colliders without changing mass, inertia or visible meshes. RFT triangulates the same collision surfaces. Body-clearance failure is 0.4 m above the initial surface; the foot-placement threshold is 10 mm on each assigned side of the base.

Development trials with disabled self-collision, weaker slip cost or failed RFT initialization are excluded from the final comparison. Full-depth RFT training from scratch collapsed into short episodes; thin-layer training from scratch learned standing without command following. The accepted rigid policy completes separate 10 s checks in zero-depth and 10 mm RFT layers without failure, with 0.029 and 0.030 m/s tracking errors at a 0.5 m/s command. Four- versus sixteen-substep probes produced nearly identical failure times, giving no evidence that timestep reduction resolves the initialization problem. RFT therefore uses the explicitly reported rigid-policy initialization.

A G1-specific MPM scene inheritance bug retained the rigid plane at the free surface. Removing it leaves the backing floor 118 mm below the surface. A force-audit regression rejects the old initial rigid support of 252.5 N and accepts the corrected 0.10 N over the first 0.1 s, using a 5%-of-weight tolerance. The corrected 2 s rigid-policy contact probe falls at 1.16 s with finite particles and no capacity overflow. The invalid case is excluded from policy comparisons. Published Go2 MPM scenes did not inherit that plane.

## Interpretation limits

RFT omits evolving surfaces, compaction and plastic history; the bearing extension remains experimental. Matching density, friction and depth does not establish RFT/MPM material equivalence. The earlier standalone MPM force-convergence gate remains failed, and G1-specific force calibration is unvalidated. Support includes backing-floor and non-foot contacts after collapse, so total support alone does not establish successful walking. Native contact sensors are unavailable through the MPM proxy coupler; MPM evaluation disables gait rewards/sensors while preserving actor inputs.

Registered tasks are `Isaac-Velocity-Flat-G1-Granular-Rigid-Direct` and `Isaac-Velocity-Flat-G1-Granular-RFT-Direct`. The durable controller is [run_g1_granular_study.py](scripts/benchmarks/run_g1_granular_study.py), and the report generator is [report_g1_granular.py](scripts/benchmarks/report_g1_granular.py). Generated artifacts currently remain under ignored `logs/go2_granular/g1/` and `logs/rsl_rl/`; the final archive is pending.

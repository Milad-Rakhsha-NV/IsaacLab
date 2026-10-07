# Release-3 Newton DVI integration and locomotion validation

<!-- current-campaign-note -->
Current status (October 7): the release-3 integration and seven-environment
solver campaign are complete, with the behavior qualifications below. Subsequent
Go2 experiments added **opt-in experimental revolute armature rows**, completed
two-coupling Jacobi/APGD/P-SPG-FB training, and optimized Jacobi throughput.
One-coupling trials were rejected for excessive foot overlap. After the initial
stop for background work, the fresh Jacobi run completed all 500 updates at
**164,852 samples/s** with two coupling iterations, no pauses and no detected
competing compute worker. Its new reward curve and 4K video replace the historical
Jacobi entries. See the completed-run section at the end of this log.

Current experimental implementation: Isaac Lab `77b15f923f`, Newton `8085486f`, both on
`milad/develop`. The [Newton upgrade log](../newton-dvi/release16_dvi_integration_status.md)
records solver changes, experimental limitations and regression coverage.

Historical October 5 campaign: all 21 DVI solver/task policies passed finite-state
playback checks with 64 and 4096 worlds, and all 28 stationary 4K videos passed the
artifact audit. That Go2 campaign used coupling=2 and post-stabilization=false,
and tracked about 0.53 m/s for a 0.5 m/s command. Later rotor-row experiments use
different actuator/armature settings and final joint stabilization. Finite-state
validation does not imply perfect locomotion: premature episode ends, weak Dr Legs
tracking, joint-anchor drift and unavailable wrench channels remain documented.

Original Jacobi validation date: 2026-10-02. The following starting revisions and
October 2 results are retained as history; later commits incorporated those repairs.
Isaac Lab: `milad/develop`, starting HEAD `58576b219451d5bf466956e2eeff2795d32fe9a5`.
Newton DVI: `milad/develop`, starting HEAD `d37ce6b0`.

## Acceptance criterion

Parity means finite, stable training and useful locomotion behavior. Isaac Lab 3.0
MDP changes and reward rescaling are permitted. Matching the old numerical rewards
is not required and a reward-scale gap alone does not establish a regression.
The earlier version of this document is preserved in the starting commit. Its
claim that Ant's reward gap proved failed reproduction is superseded by actual
checkpoint rollouts. Its Humanoid mass/inertia hypothesis was not substantiated:
the compared imported model buffers matched the working paper baseline.

The budgets come from `Newton-DVI-tech-report/results/reinforcement_learning/parameters.csv`,
the ICRA experiment table, and Dr Legs' archived coupling-2 launch script. Later
plot refreshes and the armature derivation are recorded separately below.

## Repairs

1. **Newton position-target indexing.** Newton 1.6 defaults to coordinate-shaped
   targets, with seven coordinates for a floating root and four for a ball joint.
   DVI read those targets using velocity indices. It now uses
   `model.joint_target_q_start` and decodes ball quaternion targets, including the
   equivalent negative hemisphere. Legacy velocity-shaped targets still work.
2. **Reset and observation FK isolation.** Combined joint-state reset writes
   marked every articulation for FK. `forward()` also overwrote every world's
   maximal body state. Both now update only pending articulations. Peers retain
   the poses and velocities produced by DVI. This resolved the reproduced
   full-scale Humanoid NaNs.
3. **Release asset-write contracts.** Warp launches receive native arrays rather
   than ProxyArray wrappers; indexed velocity writes use the specialized dispatcher.
   COM randomization accepts the release's seven-component poses. Material
   randomization can query shape counts in backend order.
4. **Contact sensing.** The retained sensor implementation used obsolete kernel
   signatures and buffer handling. It now follows release-3 buffer, history,
   friction/normal force, and reset handling. Environment-path regular expressions
   are resolved to native body/shape indices; replacing only `.*` with `*` could
   not match the release's `env_[^/]+` patterns.
5. **Collision imports.** Fixed-joint collapsing, configured self-collision
   filtering, and shape contraction reach the clone prototypes. DVI retains the
   paper's convex-hull geometry; other solvers retain authored approximations.
6. **Historical actuator presets.** ANYmal C and Go2 lost the paper's DVI implicit
   PD overrides. Restored ANYmal's stiffness 80, damping 5, armature 0.06 and Go2's
   stiffness 25/damping 0.5. Collapsed feet use SHANK/calf contact selections.
   Restored zero shape margin and task-specific recovery/damping settings.
   Go2 uses zero joint_alpha for the selected standing-gait checkpoint; the
   July 0.005 and latest-paper 0.002 variants were also trained and recorded.
   The release standing-height objective remains enabled after a controlled
   paper-reward comparison yielded crouching.
   Stock actuator presets remain available for other backends.
7. **Closed loops and DR Legs assets.** Closed-loop USD graphs use ClosedLoopView,
   which now reports per-joint coordinate/velocity widths required by release data
   bindings. The missing ignored DR Legs USD and machine-specific Geometry symlink
   are bypassed by resolving the same pinned Disney asset revision
   `8e8df07d2e4829442d3d3d3aeecee1857f9951d7` through Newton's asset cache.
8. **Native actuator bridge.** Imported explicit actuators bind to the retained
   manager's model/control buffers and advance once per physics tick outside the
   physics graph, with the release post-actuator callback interface.

## Reproduce the runs

Activate `dvi` in this repository. Editable imports resolve to both DVI repositories.
A DR Legs smoke run also passed after plain `conda activate dvi`, without the
validation wrapper or PYTHONPATH overrides. Use the release unified training
entrypoint and `--visualizer none`; the old `--headless` flag is no longer
accepted by this entrypoint.

```bash
conda activate dvi
uv run --no-project --python "$CONDA_PREFIX/bin/python" python scripts/reinforcement_learning/validate_dvi_release.py --task all --phase smoke
uv run --no-project --python "$CONDA_PREFIX/bin/python" python scripts/reinforcement_learning/validate_dvi_release.py --task all --phase train
```

The runner saves exact commands, recipes, dependency versions, exit codes and
console logs. `--task humanoid` (or another alias) selects one environment;
`--dry-run` prints commands. `--solver apgd` and `--solver pspg` select alternate
DVI contact solvers. Their smoke tests use the Jacobi table's iteration budgets;
they are API/integration checks, not complete reproductions of every archived
APGD/PSPG experiment.

| Alias | Release task | Substeps | Contact iterations | Coupling | Post-stabilize | PPO iterations |
|---|---|---:|---:|---:|---|---:|
| ant | Isaac-Ant-Direct | 1 | 10 | 1 | true | 1000 |
| humanoid | Isaac-Humanoid-Direct | 2 | 10 | 1 | true | 1000 |
| anymal_c | IsaacContrib-Velocity-Flat-AnymalC | 1 | 20 | 2 | false | 500 |
| go2 | Isaac-Velocity-Flat-UnitreeGo2 | 1 | 15 | 1 | true | 500 |
| h1 | Isaac-Velocity-Flat-H1 | 1 | 10 | 1 | false | 1000 |
| g1 | Isaac-Velocity-Flat-G1 | 1 | 15 | 2 | false | 1500 |
| dr_legs | Isaac-DrLegs-Walk-v0 | 1 | 10 | 2 | false | 1000 |

All full runs use 4096 worlds, seed 42 and cached factorization. Physics remains
semi-implicit DVI with the historical tangential friction projection.

## Training and behavior validation

All seven environments completed their original Jacobi DVI PPO budgets at 4096
worlds with seed 42: Ant 1000, Humanoid 1000, ANYmal C 500, Go2 500, H1 1000,
G1 1500, and DR Legs 1000. Checkpoint selection is stated below; a completed
training budget does not imply that its last checkpoint is always the best policy.
APGD and PSPG each passed three PPO iterations on all seven robots with 64 worlds.
The Jacobi smoke sequence also passed on all seven environments.

Every selected policy was replayed for 1200 steps at 16 worlds with video and at
4096 worlds with native-state checks. Observations and all native body poses and
velocities stayed finite on every sampled step. Velocity tasks use a fixed
0.5 m/s forward command, except DR Legs at 0.2 m/s within its [-0.3, 0.3] training
range. Observation noise, random pushes, and external forces are disabled.
Ant and Humanoid retain their direct forward locomotion objectives. Rollouts span
20 seconds for direct tasks and 24 seconds for manager tasks. Early terminations
exclude normal time limits; worlds can terminate repeatedly within a rollout.
These are finite samples, not a statistical robustness guarantee.

| Policy | 16-world forward speed (m/s) | Upright samples | Early ends (16 worlds) | 4096-world forward speed (m/s) | Early / total ends (4096 worlds) |
|---|---:|---:|---:|---:|---:|
| [Ant rollout](logs/dvi_release_validation_behavior/ant.mp4) | 4.594 | 100.000% | 0 | 4.610 | 0 / 4096 |
| [Humanoid rollout](logs/dvi_release_validation_behavior/humanoid.mp4) | 3.804 | 99.964% | 4 | 3.746 | 578 / 4672 |
| [ANYmal C rollout](logs/dvi_release_validation_behavior/anymal_c.mp4) | 0.486 | 100.000% | 0 | 0.491 | 2 / 4098 |
| [Go2 rollout](logs/dvi_release_validation_behavior/go2.mp4) | 0.435 | 100.000% | 0 | 0.422 | 524 / 4443 |
| [H1 rollout](logs/dvi_release_validation_behavior/h1.mp4) | 0.465 | 100.000% | 0 | 0.463 | 0 / 4096 |
| [G1 rollout](logs/dvi_release_validation_behavior/g1.mp4) | 0.476 | 100.000% | 0 | 0.474 | 5 / 4101 |
| [DR Legs rollout](logs/dvi_release_validation_behavior/dr_legs.mp4) | 0.055 | 100.000% | 0 | 0.053 | 4 / 8194 |

The recordings show assembled robots with sustained stepping/locomotion.
Humanoid still has occasional falls, consistent with its archived episode-survival
curve. Go2 and DR Legs require the qualifications below; equal reward totals were
not used to declare parity.

Videos, JSON measurements, and NPZ state traces are under
`logs/dvi_release_validation_behavior/`. Checkpoints are under
`logs/rsl_rl/release3_dvi_validation/`.
To replay all selected checkpoints and record fresh videos:

```bash
bash logs/dvi_release_validation_behavior/replay_selected_checkpoints.sh
```

Selected checkpoints:

- ant: `2026-10-02_17-41-17_ant_jacobi10/model_999.pt`
- humanoid: `2026-10-02_17-44-34_humanoid_jacobi10/model_999.pt`
- anymal_c: `2026-10-02_18-18-01_anymal_c_jacobi20/model_499.pt`
- go2: `2026-10-02_18-21-48_go2_jacobi15/model_450.pt`
- h1: `2026-10-02_18-22-30_h1_jacobi10/model_999.pt`
- g1: `2026-10-02_18-22-30_g1_jacobi15/model_1499.pt`
- dr_legs: `2026-10-02_18-25-04_dr_legs_jacobi10/model_999.pt`

The evaluator accepts the normal playback arguments plus `--metrics PATH`,
`--steps N`, `--forward-speed V`, and optionally `--video-file PATH`. Supply the
same physics overrides printed by the recipe runner. For example:

```bash
uv run --no-project --python "$CONDA_PREFIX/bin/python" python scripts/reinforcement_learning/evaluate_dvi_release.py --task Isaac-Humanoid-Direct --checkpoint logs/rsl_rl/release3_dvi_validation/2026-10-02_17-44-34_humanoid_jacobi10/model_999.pt --num_envs 16 --visualizer none --metrics logs/dvi_release_validation_behavior/humanoid.json --video-file logs/dvi_release_validation_behavior/humanoid.mp4 presets=newton_dvi env.sim.physics=newton_dvi env.sim.physics.num_substeps=2 env.sim.physics.solver_cfg.contact_max_iterations=10 env.sim.physics.solver_cfg.coupling_iterations=1 env.sim.physics.solver_cfg.post_stabilize_joints=true
```

## Behavior qualifications and paper-branch checks

Paper-branch replays use Isaac Lab `e72584b71bf5d9394e81026e32883ab6af04b548`
and Newton `6e9b05cf398e67eb9b54ad6a775121dfe394d8f5`, with the same current
conda runtime and the same policy weights to isolate repository physics changes.
They are controlled policy replays, not newly completed paper-branch training runs.


Go2 uses the release standing-height objective with DVI joint_alpha=0.0.
Its 500-iteration training run completed; the selected playback checkpoint is
iteration 450, which tracks 0.5 m/s at 0.435 m/s in 16 worlds with a standing gait,
0.371 m median root height, and no early terminations. At 4096 worlds it averages
0.422 m/s, stays upright in 99.999% of samples, and has 524 early terminations
among 4443 episode ends. Some large-batch failures remain. The last checkpoint
of this run is less stable and is not recommended for playback.

The July joint_alpha=0.005 and latest-paper 0.002 settings also completed 500
iterations, but yielded slower policies (about 0.25 m/s for the same command).
Removing the added height objective improved numerical tracking while producing
a low crouched gait (0.166 m median root height). That change was rejected after
visual inspection; the standing-height objective is retained. Variant recordings
and measurements remain under `go2_alpha0*`, `go2_alpha002*`, `go2_paper_mdp*`,
and `*go2*july_alpha005*` for inspection.

The selected Go2 policy also behaves similarly on the paper physics and upgraded
physics at 4096 worlds: 0.419 versus 0.422 m/s, with 494 versus 524 early
terminations over 24 seconds. This supports behavior parity for this policy,
including its remaining failures. It is finite, single-seed validation.

DR Legs maintains an assembled closed-loop linkage and sustained stepping, but
forward-command tracking remains modest. Its archived full run also had modest
XY tracking: the final-50 training mean was 0.208 m/s error. Replaying the same
final policy on the paper branches produced essentially the same
forward/lateral speeds (0.0548/0.1276 m/s versus 0.0552/0.1266 on the upgraded
branches), with no early terminations in either 16-world sample. That supports
dynamics and gait parity for this sample; it does not prove strong command
tracking. First-world maximum joint-anchor drift was 6.4 mm on the paper branches
versus 7.2 mm after the upgrade.

Replaying H1's final policy on the paper branches gave 0.477 m/s versus 0.465 m/s
after the upgrade, with no early terminations in either 16-world sample. Its
first-world maximum anchor drift was 6.5 cm versus 6.7 cm. This is comparable to
existing DVI constraint behavior, rather than evidence of a newly exact joint
constraint. Joint-anchor measurements are diagnostic and cover one sampled world.

`training_progress.csv/.png` records episode length and velocity-tracking error.
`paper_release_training_comparison.csv/.json/.png` compares episode survival with
the archived source logs. Release MDPs differ, so those curves support the behavior
checks rather than impose numerical equality. Rewards remain recorded in console
logs for traceability.

## Environment and checks

The conda `dvi` environment uses Python 3.12.14, Warp 1.17.0, RSL-RL 5.5.1,
Torch 2.10.0+cu128, newton-usd-schemas 0.5.0 and usd-exchange 3.0.0. Conflicting
usd-core was removed so there is one pxr installation. Added release dependencies
omniverseclient, rich and fast-simplification. Isaac Lab's package metadata still
reports 25.1.0 in this release-3 checkout; this is the repository version field.

Isaac Lab's `[tool.uv]` override can replace editable Newton with stock release/1.6.
If reinstalling Newton, run from outside the Lab project with configuration disabled:

```bash
cd /tmp
uv --no-config pip install --python "$CONDA_PREFIX/bin/python" --no-deps -e /home/mrakhsha/Documents/DVI/newton-dvi
```

Regression tests operate on real replicated Newton models and native buffers.
Target-layout, peer-reset preservation, scalar drive writes, COM pose writes,
forward-without-pending-writes, fixed-link collapsing, sensor regex selection, and
closed-loop coordinate tests were observed failing with their fixes removed.
CPU and CUDA variants are covered where applicable. The focused Lab suite passes
53 tests. Newton target-layout checks pass 6 device tests (with revolute/prismatic,
ball and D6 layout subcases); the existing DVI selection suite passes 42 tests.
Re-run the focused checks from the respective repository roots:

```bash
# newton-dvi
uv run --no-project --python "$CONDA_PREFIX/bin/python" python -m unittest newton.tests.test_dvi_target_layout newton.tests.test_dvi_selection_api

# isaaclab-dvi
uv run --no-project --python "$CONDA_PREFIX/bin/python" python -m pytest -q source/isaaclab_newton/test/assets/test_dvi_articulation_writes.py source/isaaclab_newton/test/assets/test_closed_loop_coordinates.py source/isaaclab_newton/test/assets/test_joint_coordinates.py source/isaaclab_newton/test/sensors/test_contact_sensor_regex.py source/isaaclab_newton/test/sensors/test_contact_sensor_history.py source/isaaclab_newton/test/cloner/test_dvi_collision_configuration.py source/isaaclab_newton/test/cloner/test_collision_approximation.py source/isaaclab_newton/test/cloner/test_visual_shape_import.py
```

Ruff checks/formatting and `git diff --check` pass for the changed files.
Test and pre-fix regression logs are retained in
`logs/dvi_release_validation_behavior/checks/`.
The source patches, new files, starting HEADs, and hashes are retained under
`logs/dvi_release_validation_behavior/source_snapshot/` for reproduction without
a commit.

Runs share the GPU with other workloads. Their elapsed times/throughput must not
be presented as controlled solver benchmarks. The full stock PhysX/MJWarp/Kamino
matrix and every archived comparator training run have not been validated here.

<!-- release3-oct4-campaign -->
## October 4–5 full solver campaign

All new training uses seed 42 and 4096 environments with the archived PPO budgets.
Go2 uses coupling=2 and post-stabilization=false, joint_alpha=0, and the release standing-height objective.
This Go2 comparison changes two solver settings together as requested; it is not a single-variable ablation.
APGD uses contact tolerance 1e-4; P-SPG-FB uses 1e-4 for Ant/Humanoid and the task default elsewhere.
Contact iteration budgets remain 10/10/20/15/10/15/10 in the environment order below.
The other six Jacobi policies reuse the completed October 2 training; Go2 is retrained for 500 iterations.
Playback uses 64 worlds, 3840×2160 H.264, fixed forward commands (0.5 m/s; DR Legs 0.2), and 1200 steps.
Videos retain physical world positions and use a stationary lower oblique camera, tightly framing the initial formation. Robots move relative to the fixed ground; no per-robot tracking offsets are applied.
Camera position, angle and zoom remain fixed throughout each recording. Fast robots can leave the close view; all 64 worlds continue to simulate.
The earlier recentered movies were rejected and replaced; [current rendering and reproduction settings](../Newton-DVI-tech-report/results/reinforcement_learning/README.md#reproduction).
Direct Ant/Humanoid keep their native forward goal. Commands, checkpoints and raw metrics are in campaign.json.
The table uses world-X speed for direct Ant/Humanoid and body-forward speed for velocity-command tasks.
A separate 4096-world playback measures finite native state and behavior. Reward scale is not a parity criterion.
Joint-anchor separation is measured over the first world's trajectory; peaks above 0.02 m are flagged for inspection. This does not establish a merge regression or sample every world's joint drift.
H1/G1 drift is also observed on the retained paper branches; [measurements and limits](#behavior-qualifications-and-paper-branch-checks).
Each new solver/task result uses the final checkpoint of one seed-42 run. Earlier training jobs overlapped; following the October 5 correction, training runs strictly one at a time. Wall times are not solver throughput comparisons.
Release-3 direct Ant/Humanoid add foot-wrench observations; DVI currently leaves these channels zero. See [direct-task compatibility and retained-branch checks](logs/dvi_release3_matrix/2026-10-04/paper_branch_comparisons/).
Videos only: `videos/dvi_release3/2026-10-04_initial_64env/` and `videos/dvi_release3/2026-10-04_all_solvers_64env/`.
The initial-policy directory now uses the retrained Go2 Jacobi checkpoint. Its earlier crouching diagnostic clip is retained with the rejected recordings.

| Run | Status | Forward m/s (4096) | Height m | Premature ends / all ends | Observations | Video |
|---|---|---:|---:|---:|---|---|
| jacobi_ant | complete | 4.694 | 0.591 | 0 / 4096 |  | [4K](/home/mrakhsha/Documents/DVI/isaaclab-dvi/videos/dvi_release3/2026-10-04_all_solvers_64env/jacobi_ant.mp4) |
| jacobi_humanoid | complete | 3.840 | 1.262 | 564 / 4656 | premature terminations observed | [4K](/home/mrakhsha/Documents/DVI/isaaclab-dvi/videos/dvi_release3/2026-10-04_all_solvers_64env/jacobi_humanoid.mp4) |
| jacobi_anymal_c | complete | 0.491 | 0.647 | 2 / 4098 | premature terminations observed | [4K](/home/mrakhsha/Documents/DVI/isaaclab-dvi/videos/dvi_release3/2026-10-04_all_solvers_64env/jacobi_anymal_c.mp4) |
| jacobi_go2 | complete | 0.525 | 0.388 | 12 / 4107 | premature terminations observed | [4K](/home/mrakhsha/Documents/DVI/isaaclab-dvi/videos/dvi_release3/2026-10-04_all_solvers_64env/jacobi_go2.mp4) |
| jacobi_h1 | complete | 0.463 | 0.974 | 0 / 4096 | first-world joint-anchor separation peaks at 0.067 m | [4K](/home/mrakhsha/Documents/DVI/isaaclab-dvi/videos/dvi_release3/2026-10-04_all_solvers_64env/jacobi_h1.mp4) |
| jacobi_g1 | complete | 0.474 | 0.743 | 4 / 4100 | premature terminations observed; first-world joint-anchor separation peaks at 0.029 m | [4K](/home/mrakhsha/Documents/DVI/isaaclab-dvi/videos/dvi_release3/2026-10-04_all_solvers_64env/jacobi_g1.mp4) |
| jacobi_dr_legs | complete | 0.053 | 0.272 | 3 / 8193 | premature terminations observed; mean forward tracking error exceeds 25% | [4K](/home/mrakhsha/Documents/DVI/isaaclab-dvi/videos/dvi_release3/2026-10-04_all_solvers_64env/jacobi_dr_legs.mp4) |
| apgd_ant | complete | 4.478 | 0.624 | 0 / 4096 |  | [4K](/home/mrakhsha/Documents/DVI/isaaclab-dvi/videos/dvi_release3/2026-10-04_all_solvers_64env/apgd_ant.mp4) |
| apgd_humanoid | complete | 3.071 | 1.234 | 1405 / 5496 | premature terminations observed | [4K](/home/mrakhsha/Documents/DVI/isaaclab-dvi/videos/dvi_release3/2026-10-04_all_solvers_64env/apgd_humanoid.mp4) |
| apgd_anymal_c | complete | 0.484 | 0.648 | 2 / 4098 | premature terminations observed | [4K](/home/mrakhsha/Documents/DVI/isaaclab-dvi/videos/dvi_release3/2026-10-04_all_solvers_64env/apgd_anymal_c.mp4) |
| apgd_go2 | complete | 0.450 | 0.389 | 9 / 4103 | premature terminations observed | [4K](/home/mrakhsha/Documents/DVI/isaaclab-dvi/videos/dvi_release3/2026-10-04_all_solvers_64env/apgd_go2.mp4) |
| apgd_h1 | complete | 0.470 | 0.961 | 0 / 4096 | first-world joint-anchor separation peaks at 0.085 m | [4K](/home/mrakhsha/Documents/DVI/isaaclab-dvi/videos/dvi_release3/2026-10-04_all_solvers_64env/apgd_h1.mp4) |
| apgd_g1 | complete | 0.491 | 0.738 | 0 / 4096 | first-world joint-anchor separation peaks at 0.032 m | [4K](/home/mrakhsha/Documents/DVI/isaaclab-dvi/videos/dvi_release3/2026-10-04_all_solvers_64env/apgd_g1.mp4) |
| apgd_dr_legs | complete | 0.039 | 0.274 | 3 / 8193 | premature terminations observed; mean forward tracking error exceeds 25% | [4K](/home/mrakhsha/Documents/DVI/isaaclab-dvi/videos/dvi_release3/2026-10-04_all_solvers_64env/apgd_dr_legs.mp4) |
| pspg_ant | complete | 4.647 | 0.583 | 459 / 4155 | premature terminations observed; upright fraction 0.9930 | [4K](/home/mrakhsha/Documents/DVI/isaaclab-dvi/videos/dvi_release3/2026-10-04_all_solvers_64env/pspg_ant.mp4) |
| pspg_humanoid | complete | 5.251 | 1.286 | 354 / 4449 | premature terminations observed | [4K](/home/mrakhsha/Documents/DVI/isaaclab-dvi/videos/dvi_release3/2026-10-04_all_solvers_64env/pspg_humanoid.mp4) |
| pspg_anymal_c | complete | 0.498 | 0.643 | 1 / 4097 | premature terminations observed | [4K](/home/mrakhsha/Documents/DVI/isaaclab-dvi/videos/dvi_release3/2026-10-04_all_solvers_64env/pspg_anymal_c.mp4) |
| pspg_go2 | complete | 0.442 | 0.376 | 6 / 4099 | premature terminations observed | [4K](/home/mrakhsha/Documents/DVI/isaaclab-dvi/videos/dvi_release3/2026-10-04_all_solvers_64env/pspg_go2.mp4) |
| pspg_h1 | complete | 0.497 | 0.957 | 0 / 4096 | first-world joint-anchor separation peaks at 0.072 m | [4K](/home/mrakhsha/Documents/DVI/isaaclab-dvi/videos/dvi_release3/2026-10-04_all_solvers_64env/pspg_h1.mp4) |
| pspg_g1 | complete | 0.507 | 0.746 | 0 / 4096 | first-world joint-anchor separation peaks at 0.033 m | [4K](/home/mrakhsha/Documents/DVI/isaaclab-dvi/videos/dvi_release3/2026-10-04_all_solvers_64env/pspg_g1.mp4) |
| pspg_dr_legs | complete | 0.047 | 0.273 | 5 / 8196 | premature terminations observed; mean forward tracking error exceeds 25% | [4K](/home/mrakhsha/Documents/DVI/isaaclab-dvi/videos/dvi_release3/2026-10-04_all_solvers_64env/pspg_dr_legs.mp4) |


Training curves and archived parameters, logs, checkpoint checksums, and metrics: [release-3 results](../Newton-DVI-tech-report/results/reinforcement_learning/README.md).
Detailed Go2 comparisons: [fixed-policy solver settings](logs/dvi_release3_matrix/2026-10-04/go2_fix/README.md).

<!-- RELEASE3_ALL_REPRODUCTION_20261005 -->

## ALL RL reproduction

DVI: 21 complete policies; MJWarp: 6/6 complete full-budget runs and native-state playbacks. Training runs one at a time.

[Measured results, issues and figure links](../Newton-DVI-tech-report/results/reinforcement_learning/README.md). [Dataset and exact reproduction commands](../Newton-DVI-tech-report/results/reinforcement_learning/README.md#reproduction).

## Dr Legs Jacobi parameter audit and stability checks — October 6

The retained release-3 Jacobi run uses the same substantive settings as the paper's
August 18 two-coupling run. This comparison uses the resolved `params/env.yaml` and
`params/agent.yaml` from both runs, rather than the task preset defaults.

| Parameter | Archived two-coupling run | Release-3 Jacobi run |
|---|---|---|
| Physics interval / control interval | 0.004 s / 0.020 s | 0.004 s / 0.020 s |
| Physics substeps | 1 | 1 |
| Jacobi contact iteration cap | 10 | 10 |
| Bilateral/contact coupling sweeps | 2 | 2 |
| Bilateral post-stabilization | disabled | disabled |
| Cached factorization | enabled | enabled |
| Joint / preconditioner regularization | 1e-4 / 1e-4 | 1e-4 / 1e-4 |
| Joint-limit iteration cap | 20 | 20 |
| Driven-joint stiffness / damping / effort limit | 5 / 0.2 / 3.1 Nm | 5 / 0.2 / 3.1 Nm |
| PPO budget / environments / seed | 1000 / 4096 / 42 | 1000 / 4096 / 42 |

Reward weights, command ranges, action scale and the main PPO hyperparameters match.
The new explicit `init_at_random_ep_len=true` also matches the old training entry
point's hard-coded `runner.learn(..., init_at_random_ep_len=True)`. Both launches
override the task preset's 20-contact-iteration, four-substep defaults.

At PPO iteration 662 the release-3 Jacobi log contains mean reward -4.530077184e9,
angular-velocity reward -2.35938112e8 and value loss 1.6323173704102052e19. Adjacent
iterations return to the usual reward and loss range. This is a real logged outlier;
it remains in the numerical inputs. Median reward over the final 100 iterations is
379.86, versus 381.06 in the archived two-coupling run. The archived final planar
velocity-command error was 0.2127 m/s; the current final value is 0.2153 m/s. These
training metrics suggest weak tracking was already present in the archived run;
they do not replace a comparison of the original policies' behavior.

Frozen-policy diagnostics use `model_600.pt`, 4096 environments, seed 42, sampled
policy actions, training command ranges and randomized initial episode phases.
State is inspected during reward computation, before automatic reset can erase a
failure. The detector stops if any body's angular-speed norm exceeds 1000 rad/s,
the root's angular-speed norm exceeds 1000 rad/s, or a reward is non-finite.
Each case runs separately; no policy optimization occurs.

| Diagnostic | Completed control steps | Peak body angular speed (rad/s) | Detector result |
|---|---:|---:|---|
| Current solver, original settings | 56 | 179,973.69 | triggered |
| Current solver, factorization caching disabled | 43 | 1,408.75 | triggered |
| Current solver, 20 contact iterations | 42 | 1,680.72 | triggered |
| Current solver, two physics substeps | 1200 | 593.69 | not triggered |
| Original DVI code, original settings on the current Lab/Newton stack | 1200 | 761.73 | not triggered |

The original-code check loads the DVI package from Newton revision
`1a7473f12b5cd91164f70d8078c77b57efd73fd7` under a separate module name and verifies
the instantiated solver's source path. It retains the current Newton model,
collision pipeline, Isaac Lab and release-3 policy; it does not recreate the whole
original software stack. The instability is present in native body velocities,
with normalized quaternions, so it is not solely a reward or plotting problem.
Disabling caching or doubling contact iterations did not remove it in these checks.
Two physics substeps are a candidate for improved robustness, but this finite
rollout is not full training validation. A specific merge change has not yet been
isolated as the cause; the original and current kernels can follow different
trajectories through floating-point accumulation.

Diagnostic logs, compact failing-state snapshots and the reproduction script are
stored [outside the report repository](/home/mrakhsha/Documents/DVI/diagnostics/drlegs_jacobi_20261006).
After `conda activate dvi`, run `bash reproduce.sh current600` from that directory;
the other case names are `no_cache600`, `contact20_600`, `substeps2_600` and
`paper_dvi600_verified`. Historical checkpoints, curves and training recipes have
not been replaced by these diagnostics.

## Go2 Jacobi: fixed MJWarp-controller transfer — October 6

The DVI-trained controller cannot by itself establish physics correctness after a
parameter change. A new audit therefore fixes the MJWarp Go2 checkpoint weights
and plays that controller on MJWarp and Jacobi DVI physics. All 22 completed
rollouts ran sequentially: 64 environments, 12 seconds, seed 42, fixed 0.5 m/s
forward command. No training occurred. The MJWarp DC-motor configuration,
armature, observation terms, action coordinates and nominal joint positions are
preserved when replacing the physics backend. This is closed-loop controller
transfer; different observations produce different actions.

With matched drives, default Jacobi physics tracks 0.527 m/s without premature
resets in this finite test. MJWarp tracks 0.523 m/s without resets. Both swing all
four feet, but DVI's front-foot excursion remains smaller. Foot excursions are
measured relative to each solver's loaded stance height, avoiding a misleading
comparison caused by collision margins.

The audit exposed a contact geometry defect: DVI omitted thickness offsets for
world-attached shapes. Both shape orders fail an analytic sphere-plane regression
before the correction; CPU/CUDA and graph replay pass after it. Matching MJWarp's
two 1 cm shape margins now raises DVI stance by 20 mm instead of 10 mm, bringing
root height within about 2 mm of MJWarp. Zero-margin behavior is unchanged. The
fix passes 20 focused geometry, contact-slip and shared solver regression checks.

Increasing Jacobi contact iterations from 15 to 60 greatly reduces residuals but
barely changes gait. More coupling sweeps help: matched margins with four sweeps
and 15 contact iterations has no premature resets, 0.516 m/s tracking and roughly
19.5 ms environment stepping time. Eight sweeps with final bilateral stabilization
has no resets, 0.516 m/s tracking and a maximum anchor separation of 0.56 mm, at
roughly 37.2 ms per step. Its front-foot excursions are still 12.0/9.8 mm, versus
MJWarp's 20.9/20.1 mm. Thus the behavior gap is reduced, not fully closed. Contact
compliance 1e-4 and removing armature both cause instability and are rejected.

A native Jacobian audit also confirms a structural armature difference. DVI's
isotropic child-body augmentation, pulled back to generalized coordinates, adds
0.06 at each hip and 0.04 at each thigh rather than the reference 0.02. It also
adds free-root rotational inertia and off-diagonal terms. Source comments now
state that this is a body-space regularization rather than exact joint-space
rotor inertia. The armature implementation is unchanged; its causal contribution
to the gait gap needs an isolated dynamics comparison before a replacement.

[Measurements, limitations and reproduction commands](logs/dvi_go2_mjwarp_policy_transfer/2026-10-06/README.md),
[comparison figure](logs/dvi_go2_mjwarp_policy_transfer/2026-10-06/physics_gap.png),
and [all-case summary](logs/dvi_go2_mjwarp_policy_transfer/2026-10-06/summary.csv)
are kept in ignored Lab logs, outside the paper repository. No task defaults,
training checkpoints or manuscript results were replaced. No commits or pushes.

## Go2 Jacobi: experimental revolute armature rows — October 6

Added opt-in `use_armature_rows` in Newton DVI and its Lab configuration. One
unrestricted rotor impulse row per revolute joint represents relative joint
inertia while retaining body-local inverse mass and the existing sparse 6×6
joint tiles. The default remains false. Nonzero armature on other joint types
is unsupported, and the joint solver must be sparse LDL.

Independent generalized-mass tests verify the corrected inertia response,
including multiple worlds, free spin, refinement and CUDA graph replay. The
legacy isotropic approximation fails the nonzero-armature response comparisons.
However, this prototype fails the fixed-MJWarp-policy Go2 test at the requested
one/two coupling sweeps. Across 64 environments and 12 s, rotor rows give 552
premature resets with one sweep; two sweeps reach non-finite state at step 448.
A final bilateral solve still gives 877/283 resets for one/two sweeps. Legacy
baselines give 4/1 resets. No training was run.

Additional frozen-matrix audits find accurate sparse joint solves: sampled
relative residuals remain below 2e-6. Final stabilization makes anchors tight
but contact residuals remain large. A frozen sticking-contact calculation also
indicates slower joint/contact splitting with the corrected inertia. This is
evidence to investigate contact response and coupling next, not proof of a
complete failure mechanism or a successful Go2 fix. Eight sweeps were not used
as a workaround. Runtime measurements from failing trajectories are not clean
implementation-overhead comparisons.

[Trial report and reproduction details](logs/dvi_go2_mjwarp_policy_transfer/2026-10-06/rotor_rows/REPORT.md)
include all completed and failed cases, matrix snapshots, validation logs and
source provenance. The paper repository and training results remain unchanged.

## Go2 Jacobi: fresh rotor-row training — October 6

Prepared a fresh seed-42 PPO run with 4096 environments and 500 iterations,
using the rotor-row experiment's explicit DC-motor drives and armature 0.02.
Jacobi uses 15 contact iterations, two coupling sweeps and final bilateral
stabilization. The shape margins are 1 cm, physics dt is 5 ms, and there is one
physics substep. Normal training commands, observation noise and disturbances
remain enabled. This differs from the legacy implicit-drive training recipe;
it follows the corrected-armature inference experiment.

A 64-environment, 48-step random-action preflight stayed finite, with 50
premature resets. That checks execution, not useful behavior. The run was
initially queued behind an unrelated Ant trainer. The user clarified that
unrelated projects may run concurrently and only this task's own training jobs
must be serialized. The Go2 supervisor now proceeds independently of that Ant
job, which was left untouched. It will export scalar curves to NPZ and a plot,
then evaluate and record the final policy in 64 environments with the approved
stationary camera. Any failure is recorded without automatically changing
solver settings. The linked live report gives the current training progress.

The run uses frozen Newton/Lab sources so subsequent workspace edits do not
change the queued experiment. [Live status, artifacts and reproduction details](logs/dvi_go2_rotor_training/2026-10-06/REPORT.md)
are kept in Lab logs; previous checkpoints and paper results are preserved.

The run completed all 500 iterations with no reported non-finite training
failure. Final-50 averages were reward 33.72, episode length 982.4 policy steps
(19.65 s), and XY command-tracking error 0.109 m/s. The final checkpoint
`model_499.pt` then completed 12 s of playback across 64 environments with
zero premature resets, 100% upright samples, mean body-forward speed 0.552 m/s
for a 0.5 m/s command, and maximum sampled joint-anchor error 0.387 mm in world 0.
Every robot translated 6.31–6.70 m horizontally. Initial headings vary; the
mean world-X velocity is consequently not the forward-command tracking metric.

Contact accuracy remains a qualification. Offline analysis of world 0's
25 Hz body poses, using foot sphere geometry read from the exact frozen model,
shows all four feet lifting but physical ground penetration during stance.
After discarding the first 2 s, maximum penetrations for FL/FR/RL/RR were
15.28/11.24/0/4.57 mm. These are geometric overlaps relative to the flat plane,
not contact residuals or all-world penetration bounds. This supports successful
training and forward locomotion with the new armature treatment, but does not
establish that the contact problem is fully solved.

The [stationary 4K video](logs/dvi_go2_rotor_training/2026-10-06/videos/go2_jacobi_rotor_64env.mp4)
decodes correctly at 3840×2160, 25 fps, 300 frames / 12 s, with no camera motion.
[Training curves](logs/dvi_go2_rotor_training/2026-10-06/training_progress.png),
[training NPZ](logs/dvi_go2_rotor_training/2026-10-06/training_metrics.npz),
and [motion/contact diagnostic](logs/dvi_go2_rotor_training/2026-10-06/evaluation/motion_check.png)
are retained with the final checkpoint, exact commands, model/config snapshots
and logs. No additional training run was launched.

## Go2 rotor rows: APGD and P-SPG-FB comparison — October 6

At the user's request, launched fresh APGD and P-SPG-FB runs sequentially with
the successful Jacobi rotor-row setup: seed 42, 4096 environments, 500 PPO
updates, 15 contact iterations, two coupling sweeps and final joint stabilization.
APGD uses tolerance 1e-4; P-SPG-FB uses no tolerance-based early exit, matching
the archived solver recipes. Resolved APGD environment configuration differs
from Jacobi only in contact solver type and tolerance; PPO differs only in
experiment name. The frozen Newton/Lab sources and recorder are shared with
the completed Jacobi run, avoiding another source copy in the report repository.

The supervisor performs a finite-state preflight, training, curve export,
64-environment stationary 4K playback and geometric foot-height analysis for
each solver before starting the next. A failure is retained with its logs;
parameters are not changed automatically. Unrelated user jobs are left alone.
[Campaign report and current status](logs/dvi_go2_rotor_solvers/2026-10-06/REPORT.md)
and [videos only](logs/dvi_go2_rotor_solvers/2026-10-06/videos/) are kept in Lab logs.

APGD completed all 500 updates. Final-50 averages were reward 32.14, episode
length 19.51 s and XY command error 0.112 m/s. Its 64-environment, 12 s playback
had zero premature resets, 100% upright samples and mean body-forward speed
0.548 m/s for the 0.5 m/s command. Every robot translated 6.27–6.63 m.
World 0's maximum sampled anchor error was 0.427 mm; maximum physical foot
overlap after 2 s was 2.49 mm, compared with Jacobi's 15.28 mm. Foot vertical
excursions (p95−p05) were 54.1/26.1/70.4/59.4 mm for FL/FR/RL/RR.
These single-robot geometric diagnostics show smaller sampled overlap in this
rollout, not a general contact-accuracy guarantee. P-SPG-FB passed preflight
and trained sequentially after APGD's evaluation completed.

P-SPG-FB also completed all 500 updates. Final-50 averages were reward 32.16,
episode length 19.50 s and XY command error 0.116 m/s. Its final 64-environment,
12 s playback had zero premature resets and 100% upright samples. Mean
body-forward speed was 0.549 m/s, and all robots traveled 6.24–6.71 m.
World 0's maximum sampled anchor error was 0.491 mm; physical foot overlap
after 2 s peaked at 1.33 mm. Foot excursions were 54.4/31.0/55.4/47.4 mm.

Both new videos decode at 3840×2160, 25 fps and 300 frames, with zero camera
displacement and the same approved stationary grid view as Jacobi. Actual
saved training YAMLs match Jacobi except for contact algorithm, APGD tolerance
and output names/paths. Smaller sampled overlap is encouraging, but these
are different learned policies and a single seed; contact accuracy is not
isolated. All three policies overshoot the 0.5 m/s playback command by about
10%. P-SPG-FB was also more expensive: reported collection plus learning time
was 36.4 min versus APGD's 13.9 min. This is an observed run cost, not an
isolated throughput benchmark.

[Final comparison and reproduction details](logs/dvi_go2_rotor_solvers/2026-10-06/COMPARISON.md),
[overlaid training curves](logs/dvi_go2_rotor_solvers/2026-10-06/solver_comparison.png),
[APGD video](logs/dvi_go2_rotor_solvers/2026-10-06/videos/go2_apgd_rotor_64env.mp4)
and [P-SPG-FB video](logs/dvi_go2_rotor_solvers/2026-10-06/videos/go2_pspg_rotor_64env.mp4)
are retained with scalar NPZ, final checkpoints, commands and diagnostics.
The video directory contains only the two new videos. No paper datasets,
previous checkpoints or unrelated jobs were changed. No commits or pushes.

## H1 APGD constraint investigation — October 6 follow-up

This was inference with the existing policy, **not new H1 training**. The APGD
numerical safeguards did not remove the original joint-anchor problem: maximum
sampled anchor separation was about 93.35 mm both before and after the APGD
patch. Increasing contact iterations alone to 40 also left about 93 mm drift.
The full measurements are in
[`dvi_h1_apgd_investigation`](logs/dvi_h1_apgd_investigation/2026-10-06/comparison.csv).

With final joint stabilization and joint-limit recovery speed 10 rad/s, the
64-world, 24-second seed-42 comparison gave:

| Coupling sweeps | Maximum anchor separation | Maximum joint-limit excess | Premature terminations | Upright fraction |
|---|---:|---:|---:|---:|
| 1 | 3.87 mm | 17.88 degrees | 259 | 0.9482 |
| 2 | 1.87 mm | 7.17 degrees | 8 | 0.9977 |

Four sweeps with stabilization and recovery 10 achieved 0.37 mm anchor separation,
1.52 degrees limit excess and no early terminations in a separate 24-second,
64-world seed-42 rollout. The larger 4096-world checks still recorded failures.
These are measured joint-anchor and joint-limit errors, not mesh self-penetration
measurements. They do not establish that APGD alone caused the original behavior.
[Coupling comparison](logs/dvi_h1_apgd_coupling_comparison/2026-10-06/comparison.csv)
and [single-robot follow-camera recordings](logs/dvi_h1_apgd_coupling_videos/2026-10-06/)
retain the commands, diagnostics and recordings. No global H1 preset was changed
to a more expensive coupling count based on these trials.

## Go2 coupling and completed reruns — October 7

The rotor-row trials explicitly override the task's plain DVI preset. They use
seed 42, 4096 environments, 500 PPO updates, physics dt 0.005 s, policy dt 0.02 s,
15 contact iterations, armature 0.02 kg m², explicit DC-motor torques and final
joint stabilization. Native drive gains are zero. APGD tolerance is 1e-4;
Jacobi/P-SPG-FB use their archived settings with no tolerance override.

**One coupling sweep was rejected.** Jacobi and APGD completed 500 updates but
their final-50 mean rewards were 17.410 and 19.721, with maximum sampled physical
foot overlap of 87.07 and 74.70 mm. Finite states and upright robots alone missed
this abnormality. The P-SPG-FB run was stopped before completion when the user
requested returning to two sweeps. Artifacts remain in the
[one-coupling campaign](logs/dvi_go2_rotor_c1/2026-10-07/REPORT.md).

**Two coupling sweeps are retained.** The October 6 Jacobi policy remains the
completed reference. APGD and P-SPG-FB were rerun sequentially on October 7:

| Solver | Completed run | Reward, final 50 updates | Logged median samples/s | Body-forward speed | Maximum foot overlap |
|---|---|---:|---:|---:|---:|
| Jacobi | October 6, retained | 33.724 | 68,956.5, historical shared-GPU run | 0.552 m/s | 15.28 mm |
| APGD | October 7, fresh 500 updates | 31.010 | 83,136 | 0.517 m/s | 0.00 mm |
| P-SPG-FB | October 7, fresh 500 updates | 33.055 | 27,113 | 0.521 m/s | 2.17 mm |
| MJWarp | Retained baseline | 34.053 | 273,101 | See baseline evaluation | See baseline evaluation |

All three DVI policies had finite playback, no premature terminations and upright
fraction 1.0 over 64 worlds for 12 seconds with a 0.5 m/s command. Foot overlap
uses physical sphere geometry in world 0, sampled at 25 Hz after the first two
seconds; these maxima are not bounds over all worlds or complete contact
residuals. Different learned policies prevent treating this as an identical-control
comparison. All rewards and timing rates come from each run's own scalar data.

The APGD/P-SPG-FB campaign used five in-memory process suspensions when other GPU
workers appeared: four during APGD, one during P-SPG-FB. `SIGSTOP`/`SIGCONT`
preserved policy, optimizer, rollout and simulation state; there was no restart
or checkpoint reload. Each completed run logged 500 consecutive updates. Paused
iterations remain in the raw timings, and sampling allowed brief overlaps before
detection. These are observed training rates, not isolated solver benchmarks.

[Comparison and provenance](logs/dvi_go2_rotor_c2/2026-10-07/COMPARISON.md),
[saved scalar/configuration checks](logs/dvi_go2_rotor_c2/2026-10-07/comparison.json)
and the [videos-only directory](logs/dvi_go2_rotor_c2/2026-10-07/videos/)
retain the completed results. Recordings are 3840×2160 at 25 fps, with a stationary
camera 3 m high and 10.61 m from the initial grid center. The Jacobi entry links
to its retained recording. The existing report figure and compact NPZ contain
the retained Jacobi curve and latest completed APGD/P-SPG-FB curves.

The manager-based and core Go2 task presets both retain `coupling_iterations=2`
and `post_stabilize_joints=False`. The default `use_armature_rows` remains False,
and the plain Go2 DVI preset has zero armature. Reproducing the rotor experiment
requires its explicit armature, actuator and stabilization overrides; setting
the experimental flag alone does not reproduce it.

## Go2 Jacobi timing and stopped rerun — October 7

The historical 69k samples/s is from the October 6 full training run, before the
Jacobi optimization and while another job used the GPU. It is not a measurement
of the isolated current code. Sequential short PPO timing runs measured 126,769
before optimization and 165,065 afterward, using the last 25 of 35 updates, the
same checkpoint and the same two-coupling rotor configuration. These runs include
PPO optimization and saved no new policy checkpoints.

The [performance investigation](logs/dvi_go2_rotor_training/2026-10-06/performance_2026-10-07/README.md)
separates the effects: on one frozen Go2 state, before optimization, the rotor
rows added about 3.7% to solver-step time, the second coupling sweep added about
66% relative to one sweep, and the final joint solve cost about 0.752 ms. The
Jacobi kernel changes reduced the frozen graph time from 6.302 to 4.418 ms.
These scopes differ from complete RL throughput and do not replace its timing.

A fresh full 500-update Jacobi run was attempted to replace the old curve and
measure its own FPS. Competing Isaac Sim workers invalidated two partial attempts;
the next attempt was stopped on October 7 at the user's request. Only this
campaign's processes were terminated. No clean complete result was accepted,
no new video was completed, and no partial timing was substituted into the figure.
The [stopped campaign](logs/dvi_go2_jacobi_c2_clean/2026-10-07/REPORT.md) retains
commands, source hashes, telemetry and partial logs. Completing this measurement
requires a later GPU window and a new uninterrupted run.

## Armature coverage and unresolved behavior

The initialized-model audit found these physical armatures; `null` in a YAML
means the imported asset value survives and does not establish zero armature.

| Task | Loaded armature (kg m²) | Consequence for the experimental option |
|---|---|---|
| Ant | 0.05 | Eight revolute joints; not retrained with rows. |
| Humanoid | 0.01 | Includes nonzero D6 joints; unsupported by the current implementation. |
| ANYmal C | 0.06 | Twelve revolute joints; not retrained with rows. The retained MJWarp case uses 0.01. |
| H1 | 0.1 | Nineteen revolute joints; not retrained with rows. |
| G1 | 0.01 / 0.05 / 0.001 | Thirty-seven revolute joints; DVI has upper-arm overrides that differ from MJWarp. Not retrained with rows. |
| Go2 | 0 in plain preset; 0.02 in rotor trials | Only the explicit rotor trials above exercise the new treatment. |
| Dr Legs | 0 in the pinned asset | Enabling rows alone introduces no physical rotor inertia. A different cached asset revision has nonzero values. |

The full audit is retained in the
[RL results README](../Newton-DVI-tech-report/results/reinforcement_learning/README.md#armature-coverage-audit--october-7-2026).
Dr Legs' large Jacobi reward outlier and modest tracking remain documented in its
October 6 audit above. H1 joint-limit violations and low-coupling Go2 policy
transfer are unresolved beyond the measured improvements; neither reward curves
nor finite-state checks establish general contact accuracy.

## Cleanup and validation for review — October 7

- `DVISolverCfg.use_armature_rows` is explicitly experimental, defaults to False,
  and is forwarded once by `NewtonDVIManager` to `SolverDVI`. The docs name the
  sparse-LDL/revolute restriction, finite nonnegative values, construction-time
  selection, iterative contact/limit coupling and separate implicit-drive
  approximation. Existing task defaults are preserved.
- The Lab changelog fragment uses `.minor.rst`, as required for the new public
  configuration field. Unused imports and formatting in the two touched manager
  files were cleaned up. Newton's solver changes and regression coverage are
  recorded in its linked upgrade log.
- Historical release integration coverage is the 53-test Lab suite and Newton's
  target/selection suites recorded above. This cleanup reran 38 focused Newton
  tests on CPU only. It does not claim a new full Lab/GPU matrix run. Validation
  output is kept under `logs/dvi_upgrade_review/2026-10-07/`.
- The live Lab configuration/import check passed with CUDA disabled: the default
  serializes as False and the opt-in value as True; module paths resolve to these
  two working trees. Both touched Lab files pass Ruff lint and formatting, and
  both repositories pass `git diff --check`. Newton's focused clean files pass;
  pre-existing findings in older DVI modules remain explicitly recorded in the
  [cleanup check record](logs/dvi_upgrade_review/2026-10-07/README.md).
- The Newton API review treats rotor rows as a bounded opt-in experiment. Default
  dynamics remain compatible; broader joint support and contact/drive validation
  are required before considering a default-on change.
- Retain raw data and snapshots under ignored Lab logs. Do not include generated
  experiment artifacts, the independent Kamino plan, or unrelated Isaac Sim
  changes in this change set. Training remains stopped; no push is requested.
- Install/update the paired Newton change before this manager, which forwards
  `use_armature_rows` even when False. The cleanup was subsequently committed in
  Newton `8085486f` and Lab `77b15f923f`, before the completed run below.

## Completed clean Go2 Jacobi run — October 7

At the user's request, restarted from scratch after the earlier interrupted
attempts. This run used the committed Newton/Lab revisions above, seed 42, 4096
environments and 500 PPO updates. Resolved physics/PPO settings match the previous
two-coupling Jacobi trial except output names: rotor rows at armature 0.02, explicit
DC-motor torques, 15 contact iterations, two coupling sweeps, final joint
stabilization, physics dt 0.005 s and policy dt 0.02 s.

- All 500 consecutive updates completed with one initialization, no checkpoint
  resume and no process suspensions. Median `Perf/total_fps` across all updates
  is **164,852 samples/s**, including PPO optimization; no warmup samples were
  excluded. Final-50 mean episode reward is **31.892**.
- The GPU monitor sampled every two seconds. All 151 samples containing the
  training CUDA process showed no competing compute worker. Desktop graphics
  remained active. No simultaneous training or playback job was launched.
- Model 499 played for 12 seconds across 64 environments with finite state,
  upright fraction 1.0, no premature terminations and mean body-forward speed
  **0.540 m/s** for a 0.5 m/s command. Every robot traveled at least 6.13 m in XY.
- Maximum sampled first-world joint-anchor error was **0.584 mm**. Maximum
  sampled physical foot overlap after the first two seconds was **9.23 mm**;
  contact accuracy remains imperfect. These are sampled world-0 diagnostics,
  not bounds over all worlds. The new policy is separately trained.
- The stationary 64-world video is 3840×2160, 25 fps and 300 frames, using the
  approved 3 m camera height and 10.61 m distance. Camera drift is zero; robot
  world positions are preserved. All three comparison videos decoded successfully.

[Campaign, checkpoint and source provenance](logs/dvi_go2_jacobi_c2_clean/2026-10-07_r4/REPORT.md),
[new video](logs/dvi_go2_jacobi_c2_clean/2026-10-07_r4/videos/go2_jacobi_armature_c2_64env.mp4),
[updated comparison](logs/dvi_go2_rotor_c2/2026-10-07/COMPARISON.md), and
[PNG figure](../Newton-DVI-tech-report/results/reinforcement_learning/figures/go2_armature_mjwarp_progress.png)
contain the completed results. The existing PDF/SVG figure and compact NPZ were
updated too. APGD, P-SPG-FB and MJWarp curve arrays were retained exactly. The
figure uses the new run's own measured FPS, replacing the previous exclusion
label; it does not substitute the separate 35-update timing benchmark. No further
commits or pushes were made during this run.

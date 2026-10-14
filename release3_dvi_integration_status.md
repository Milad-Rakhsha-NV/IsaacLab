<!-- jacobi-performance-audit-20261012 -->
## Jacobi slowdown investigated; Dr Legs held — October 12, 2026

User requested holding Dr Legs and investigating the Jacobi throughput drop.
No training/controller is running and no production solver changes or commits
were made. Five robots were timed serially on two frozen scenes each at 4,096
worlds, alternating current legacy and armature-enabled CUDA graphs. Full
evidence: `isaaclab-dvi/logs/dvi_performance_audit/2026-10-12/REPORT.html`.

| Robot | Historical Jacobi FPS | Armature Jacobi FPS | Matched walking-scene physics slowdown |
| --- | ---: | ---: | ---: |
| Ant | 346,558 | 293,726 | 1.55x |
| H1 | 120,875 | 69,518 | 2.59x |
| G1 | 66,104 | 33,367 | 2.02x |
| Go2 | 136,838 | 78,810 | 1.35x |
| ANYmal C | 121,023 | 70,431 | 1.33x |

Most added cost is response preparation and its use inside unilateral products,
not a new factorization. G1 is 13.658 versus 27.576 ms per captured physics step;
response preparation alone is 7.413 ms, application 0.2845 ms each, and diagonal
1.0246 ms. H1 additionally performs two bilateral solves instead of one because
armature mode forces the final correction despite post_stabilize_joints=false.
The sparse LDL backend and cached factors remain in use. Padded dense basis
work and fixed inactive limit slots are concrete optimization targets.

Training timing supports a rollout cost: G1 collection rises 1.403 to 2.842 s
per update, while PPO learning rises 0.084 to 0.104 s. Historical throughput is
not a fully matched current-version control; Ant policy inputs changed 36 to
60. Refinement diagnostic timings are archived but no setting was adopted;
G1/Go2/H1's one refinement predates the historical runs. Humanoid has no new
matched run. Physics-step timings exclude collision/PPO and are not training
FPS. Dr Legs remains held pending user direction; no solver fix is claimed.

## APGD/PSPG campaign started — October 11, 2026 (Pacific)

User authorized a ten-hour monitored serial campaign: Ant, ANYmal C, H1,
Go2, G1, and Allegro cube reorientation with APGD first, then PSPG; then
Dr Legs with Jacobi, APGD, and PSPG. The Franka/manipulation queue stays
stopped. No commits or pushes. The campaign starts from the current
post-pull armature implementation, whose unchanged physics fingerprint
passed 67 CPU/CUDA tests. Existing full training budgets and physical/PPO
settings are retained; solver-specific recipe tolerances are recorded.
Monitor window ends at 7:19 p.m. Pacific, October 11. This is a monitoring
window, not a claim that all training will fit within ten hours.

Campaign: `isaaclab-dvi/logs/dvi_solver_matrix/2026-10-11/`. See `plan.json`,
`source_manifest.json`, `status.json`, and `events.jsonl`. Each completed
run receives videos, population evaluation and a reward/FPS plot update.
Existing locomotion plot titles/style and unrelated solver arrays are
preserved; new all-solver panels retain the earlier Allegro/Dr Legs
comparison figures as separate artifacts.

<!-- solver-matrix-20261011-results:start -->

Completed runs (one seed; these are measured results, not statistical parity):

| Run | Updates | Final-50 reward | Median training FPS | Population early terminations | Review |
| --- | ---: | ---: | ---: | ---: | --- |
| ant_apgd | 1000 | 87.55 | 174,777 | 8 | completed_reward_and_behavior_comparison_with_rare_falls |
| anymal_c_apgd | 500 | 23.66 | 44,286 | 6 | completed_reward_and_behavior_comparison_with_rare_falls |
| h1_apgd | 1000 | 32.77 | 44,030 | 0 | reviewed_single_seed_comparison |
| go2_apgd | 500 | 33.50 | 51,092 | 24 | reviewed_single_seed_comparison |
| g1_apgd | 1500 | 31.83 | 17,132 | 0 | reviewed_single_seed_comparison |
| allegro_apgd | 5000 | 24.75 | 13,483 | n/a | manual_review_pending |
| ant_pspg | 1000 | 86.64 | 67,757 | 5 | manual_review_pending |
| anymal_c_pspg | 500 | 24.12 | 14,130 | 24 | manual_review_pending |
| h1_pspg | 1000 | 32.17 | 16,940 | 0 | manual_review_pending |
| go2_pspg | 500 | 32.06 | 18,006 | 0 | manual_review_pending |

Videos, full evaluation metrics, source hashes, and review caveats are linked in
`isaaclab-dvi/logs/dvi_solver_matrix/2026-10-11/index.html`. The current queue
and live update count are recorded in `status.json` and
`monitor_observations.jsonl`; the table includes completed artifacts only.

<!-- solver-matrix-20261011-results:end -->

Known Dr Legs compatibility issue: its model contains closed loops, while
the current compact armature response explicitly requires a tree. This
must be investigated before those three training runs; disabling armature
would not validate the requested modified implementation.

## Training paused at user request — October 11, 2026 (Pacific)

Franka Reach baseline training and the automatic remaining-task queue were
stopped while the armature results are reviewed. The stop observation was
643/1000 updates; checkpoint `model_600.pt` and all earlier checkpoints are
preserved. No Franka Reach candidate full training was launched. Completed
Allegro and Cabinet artifacts remain unchanged. See the campaign file
`user_stop_20261011.json` for the verified stopped process IDs. No commits.

## Jacobi manipulation validation in progress — October 10, 2026 (Pacific)

Both `milad/develop` branches were fast-forwarded without creating commits:
Newton `800aa746bd0ef7a7de4834e20fbbebf4ef0f95d2`, IsaacLab
`d59886a7ed16469500b0da6b0784a74e702c2686`. Local armature changes were backed
up and reapplied, preserving the incoming manipulation/reset/contact fixes.

The campaign is recorded in
`isaaclab-dvi/logs/dvi_manipulation_validation/2026-10-10/investigation.md`.
Allegro cube reorientation is first, Franka cabinet second, followed by other
applicable manipulation tasks. Shadow Hand is excluded. APGD/PSPG environment
runs are deferred. Each selected environment needs matched reward evidence,
physical-state checks, plots and playback videos before acceptance.

- Post-merge existing mechanics suite: **47 CPU/CUDA tests passed**.
- Allegro baseline and corrected candidate completed all **5000 updates** at
  4096 worlds, seed 42. Final-50 training reward is **57.5273 baseline versus
  54.7671 candidate** (candidate 4.8% lower); median training throughput is
  **28,804 versus 34,049 steps/s** (candidate 18.2% higher). Both final videos,
  the full reward/FPS plot, and 4096-world evaluations are available in the
  campaign directory. Evaluation episode return is **69.1081 versus 64.7150**
  (candidate 6.4% lower); reported success metric is 0.72993 versus 0.73321.
  Those historical success values average persistent logs over time; they
  are not episode-weighted rates. Future evaluations record corrected rates.
  Early endings are 385/12422 versus 373/12439. All 8000 candidate internal
  substeps are finite before sanitization; peak speeds are 2.823 m/s and
  63.606 rad/s, passing the recorded cube-spin regression screen.
  One seed does not establish statistical reward equivalence. Allegro authors
  zero armature, so this establishes compatibility evidence, not nonzero
  inertia in isolation. See `allegro_review.json` for scope and limitations.
- The original omega=0.15 candidate is **rejected and archived** because final
  playback exposed rare free-cube spin outliers (peak 465 rad/s). Independent
  replay of the captured onset reproduced the Jacobi instability. The current
  candidate was retrained from scratch with omega=0.10 at the same 20-contact-
  iteration budget; the baseline retains 0.15. Physical and PPO settings are
  unchanged. Earlier policy-transfer returns of 83.29 are diagnostic results,
  not the corrected retrained policy's reward.
- The initial diagnostic playback recaptured an existing CUDA graph and failed.
  Its code and logs are archived. Diagnostics now attach before the original
  capture; the completed videos and evaluations use that repaired harness.
- **Franka cabinet is complete and reviewed.** Both runs finished 400 updates
  at 4,096 worlds and seed 42. Final-50 reward is **79.4728 baseline versus
  102.5287 candidate**; median throughput is **8,384 versus 15,852.5 FPS**
  (1.891x, including the smaller contact allocation). Final population return
  is **81.0380 versus 102.6454**; episode success is **100% versus 99.9756%**
  over 8,192 episodes each. Each population evaluation checks 20,000 finite
  internal substeps. Candidate contact capacity never saturates during all
  400 training updates or either final evaluation. Both videos and the full
  reward/FPS plot are reviewed. See `franka_cabinet_review.json` for the
  numerical differences and single-seed limitations.
- Earlier Cabinet previews and policy-transfer diagnostics remain preserved.
  The same baseline checkpoint returned 78.55 on baseline physics and 49.04
  on candidate physics; this transfer score did not predict the subsequently
  retrained candidate's higher reward. See `cabinet_policy200_transfer_cpu/`.
- The remaining ten Jacobi task pairs are now queued serially, starting with
  Franka reaching. Each pair retains its original training budget, both
  variants, final videos and population evaluations. Execution completion
  still requires result and video review. See `remaining_status.json` and
  `remaining_campaign.log`. Shadow Hand and APGD/PSPG remain excluded.
- Cabinet performance: capacity-wide contact work is costly. An isolated
  allocation-only change from 6,307,840 to 524,288 slots reduces control latency
  from 471.15 to 249.20 ms (**1.89x faster**). At 4096 worlds, baseline and
  candidate each pass 512 random controls across reset; baseline checkpoint50
  also passes 1000 playback controls. Peak demand across these checks is
  116,601 contacts, with no saturation and all 40,480 checked physics substeps
  finite. These intermediate checks do not complete reward validation.
- Forthcoming Cabinet runs use a continuous contact-capacity guard: record
  demand before every internal solve and reject saturation after each PPO
  update. Both two-update PPO guard checks and the actual candidate launcher's
  fresh three-update smoke passed. The main baseline retained its automatic
  allocation; its completed candidate used 128 slots per environment in a shared buffer.
  This difference is disclosed: throughput improvement includes allocation
  optimization. Both variants of the other three Cabinet tasks use the same
  guarded allocation and require their own full-population preflight. Physics,
  timestep, iterations and PPO budgets are unchanged. Evidence is in
  `cabinet_capacity_checks/summary.json` and `cabinet_performance_diagnosis.json`.
- Prismatic and gripper extensions: **67 CPU/CUDA tests passed**, plus the
  asset-inertia and articulation-dynamics regressions. This source-fingerprint
  gate passed before candidate training. The implementation extends the same
  scalar inertia rows and compact response, retaining authored drive gains
  and armature.
- Independent regressions exposed and corrected moving-axis slider geometry,
  spurious damping during common rotation, and missing limit moment arms.
  The actuation-disable setting now also disables its numerical drive tangent.
  Fail-before/pass-after logs are retained; no failed formulation is layered in.
- Reach/lift Jacobi presets resolve for eight selected task configurations.
  Configuration resolution is not a trained-task result. The experimental
  path now enforces one prismatic mimic relation per tree for Franka fingers,
  including contact/limit reactions; unsupported angular mimic pairs are
  rejected. Coordinates come from native poses to respect external FK resets.
  The actual Franka lift CPU preflight completed 100 zero-action steps with
  finite state and maximum finger-equality error 8.48e-8 m. Full learning
  validation remains pending.
- UR10's fully locked D6 root is treated as a fixed joint; movable D6 remains
  outside this experimental path. The real UR10 CPU preflight passes.
- Restored lazy Jacobian row-map initialization for OSC queries, with existing
  Panda/ANYmal dynamics and permutation-aware query assertions passing.
- Fixed a frictionless cone-projection defect that admitted negative scalar
  limit impulses. Two analytical regressions fail before and pass after.
  Franka reach requires a smaller Jacobi step for both joint-position and
  OSC control: their shared DVI preset uses omega=0.05 for both legacy and
  armature runs, with all physical parameters unchanged. Each passes a seeded
  8-world/512-step random-action CPU check before substep sanitization.
  These are preflights, not trained-task reward acceptance.
- Playback checks include every physics substep before state sanitization
  and resets. All 12 selected candidate tasks completed an initial random-action
  check. Direct cabinet remained finite but reached 3,300 m/s immediately after
  reset; it is not accepted on finite-state checks alone. A CLI-only Jacobi
  omega=0.05 trial reduced the same sequence's peak to 8.98 m/s. The campaign
  uses 0.05 for candidate cabinet workflows while retaining the original 0.15
  upstream baseline; this numerical-settings difference is recorded explicitly.
  All four candidate cabinet workflows pass the initial 0.05 exploration check
  (peak speeds 5.17–9.13 m/s). Every full cabinet run additionally requires a
  512-control check at 4096 GPU worlds in training configuration. The main
  cabinet baseline passed that check; its candidate also passed a separate
  guarded-capacity GPU check. The launch-time preflight and other workflows'
  checks remain required. Full cabinet reward comparisons and videos are pending.
- All 12 initial candidate exploration settings and their upstream controls
  completed 512 Gaussian-action controls at eight CPU worlds. Measurements
  are tabulated in `preflight_summary.md` and CSV in the campaign directory.
  These screens do not establish trained reward equivalence. The initial
  Allegro omega=0.15 screens are superseded by the corrected main Allegro
  omega=0.10 GPU checks. Direct Allegro's newly selected 0.10 setting still
  requires its own training-scale check. The ten task pairs after main
  Allegro and main cabinet have not completed full training. No APGD/PSPG
  environment runs are included in this phase.
- An exact fetched-upstream Newton source copy is retained for subsequent
  no-experiment controls. The initial Allegro legacy run loaded its source
  before slider edits and uses the unchanged legacy numerical path.

The completed locomotion results below refer to the archived pre-pull C1 v3
source, not the currently developing manipulation extension. No commits or
pushes have been made.

## Previous validated locomotion result — October 10, 2026 (Pacific)

The corrected relative-inertia implementation passes independent mechanics
checks and the original Jacobi training budgets for Ant, ANYmal C, H1, Go2 and
G1. All five completed 64-world videos and 4,096-world evaluations on the final
optimized source. The new regression cases fail on original HEAD; twenty focused
CPU/CUDA tests pass on the final implementation. Fresh GPU MuJoCo Warp torque,
relative-damping and moving-chain comparisons also pass the stated tolerances.

| Robot | Final-50 reward | Reference reward | Median training FPS | Speed at 4,096 worlds | Premature ends |
|---|---:|---:|---:|---:|---:|
| Ant | 89.48 | 74.93 / 76.64* | 293,726 | 5.386 m/s | 0 |
| ANYmal C | 24.00 | 25.22 | 70,431 | 0.497 m/s | 0 |
| H1 | 30.08 | 26.43 | 69,518 | 0.505 m/s | 21 |
| Go2 | 32.98 | 33.21 | 78,810 | 0.516 m/s | 0 |
| G1 | 32.36 | 27.89 | 33,367 | 0.495 m/s | 2 |

References are the original paper Jacobi results, except Ant (*matched current
legacy DVI / MJWarp controls; historical Ant reward units differ). Ant uses its
native world-forward objective; the others command 0.5 m/s body-forward speed.
Evaluations last 20 seconds for Ant and 24 seconds for the others. Every native
state check remains finite, including checks immediately before automatic reset.
All 64-world videos have zero premature endings. At 4,096 worlds, H1/G1 have rare
base-contact falls; upright fractions are 99.9915% and 99.9993%. Ends count events,
not unique robots. Repeat event audits found 20 H1 and 3 G1 startup falls, with
joint-anchor errors below 0.90 mm and hinge-axis errors below 0.07 degrees at
termination. These observations establish intact joints at those events, not
perfect balance in every initial condition.

Training uses seed 42, original PPO budgets and actuator gains. Ant retains
armature 0.05 and damping 0.1; Go2 explicitly uses armature 0.02 (its MJWarp preset).
G1 resumed after an external process interruption: 1,351 retained updates plus
149 resumed updates, with optimizer/policy restored and environment/RNG reset.
The logger warm-up dip remains in its curve. Source snapshots identify each
training run; final Ant/G1 and all population checks use `C1_compact_source_v3`.

The implementation represents ideal armature kinetic energy in joint-relative
rotation, applies the relative semi-implicit PD tangent, and includes the
incremental bilateral reaction in the common unilateral impulse response.
It uses the existing sparse joint factors and a compact `T W T^T` response.
There is no new dense mass/joint factorization or bounded-motor impulse law.
The opt-in settings used here are `use_armature_rows=true` and `precond_reg=0.0`;
raw joint regularization remains 1e-6. The legacy default is preserved.
Supported scope is enabled revolute/fixed/free small trees, with sparse
Jacobi/APGD/PSPG; large heterogeneous trees and exact stiff/saturated MJWarp
actuator semantics are not established by these tests.

Frozen Ant physics cost is 3.382 ms versus legacy 2.220 ms (1.52x), including the
captured solver/integration step but excluding collision and PPO. Final Ant
training is 293,726 FPS versus the initial full-response candidate's 227,862 FPS.
The G1 product optimization reduced its frozen step from 40.45 to 26.70 ms.
Historical plot FPS from other solvers is not a matched current timing comparison.

Only Jacobi reward/FPS data in the five existing paper PNGs were replaced;
titles, style, image dimensions and every unrelated solver array were verified
unchanged. The submission manuscript `main.tex` was not changed. APGD and PSPG
have analytical physics coverage; their full training awaits the requested
Jacobi review checkpoint.

Evidence and videos: `isaaclab-dvi/logs/dvi_armature_correctness/2026-10-09/`.
- `completion_audit.json`: source/checkpoint hashes, budgets, tests and artifacts.
- `jacobi_armature_validation.png`: comparison with original/reference curves.
- `final_rotor_mjwarp.json` and `final_passive_reference/`: final-source reference checks.
- `investigation.md`: failed candidates, fixes, interruptions and event audits.
- `Newton-DVI-tech-report/dvi_relative_armature_validation.pdf`: three-page equations/results note.

Rejected cached-mass source was archived and reverted before this candidate;
unsuccessful performance prototypes remain only as evidence, not active variants.
No commits or pushes were made. Entries below retain the investigation history.

# Release-3 Newton DVI integration and locomotion validation

### C1 Ant validation completed — 2026-10-09 (Pacific)

C0 was archived and reverted before this candidate. C1 retains the relative
armature energy, uses the existing sparse joint factors to assemble the
bilateral-consistent response once per step, and solves contacts and limits
with that response. It includes the relative PD tangent; bounded motor impulse
rows are not used. Exact formulation and artifact paths are in
`isaaclab-dvi/logs/dvi_armature_correctness/2026-10-09/C1_formulation.md`.

- 52 CPU/CUDA tests passed (46 existing, 6 new including graph initialization).
  The four new physics tests have 14 failing subcases on original HEAD.
- Full Ant Jacobi training: 4096 worlds, seed 42, 1000 PPO updates, contact
  iterations 10, one substep; armature 0.05 and damping 0.1 retained.
  Joint precond_reg=0 replaces the biased 1e-4 regularizer; joint_reg=1e-6 remains.
- Final-50 reward **89.0999**, median training **227,861.5 FPS**.
  Matched legacy and GPU MJWarp rewards were **74.93** and **76.6416**.
- Final-checkpoint evaluation, 64 worlds / 20 seconds: mean world-X speed
  **5.4124 m/s**, upright fraction **1.0**, **0** premature terminations,
  max first-world joint-anchor error **0.002286 m**, finite native state.
  Matched legacy and MJWarp speeds were **4.4078** and **4.3853 m/s**.
- Readable 10-second video with camera following world X/Y (physical poses and
  camera height unchanged): `ant_C1_follow.mp4` in the investigation directory.
  The initial stationary clip loses the robot and is retained only as an audit.
- Ant's paper PNG now has only its Jacobi data/FPS replaced; title, colors,
  layout and all unrelated curve arrays are preserved. Historical APGD/PSPG
  use a different reward scale and are not current matched comparisons.
- Performance remains under investigation: full-response prototype 5.397 ms
  versus legacy 2.209 ms on the same frozen 4096-world scene. A compact
  representation of the same response is being tested before the other robots.

Training source: `C1_source_fixed/` with hashes. First startup failed before
training because graph preparation read unassembled endpoint buffers; topology
now comes from the model and two new preparation regressions pass. The successful
run is `logs/rsl_rl/armature_correctness_C1/2026-10-10_05-20-37_ant_jacobi`.

## Armature correctness investigation — 2026-10-09 (Pacific)

The current goal is physics validation first, then Ant and the other established
DVI tasks. No new full training has started in this investigation.

Candidate C0 (cached mass, unchanged staged constraints) is rejected as a complete
solver implementation, **not** as evidence that relative armature energy is wrong.
Source archive: `/home/horde/Documents/DVI/rollback_archives/armature_correctness_start_20261010`.
Evidence/scripts: `isaaclab-dvi/logs/dvi_armature_correctness/2026-10-09/`.

- Pure torque, armature 0.05: the analytical speed is 0.15625 rad/s.
  Precision clarification from the final-source audit: GPU MJWarp differs from
  that value by 2.98e-8; current DVI differs from MJWarp by 1.49e-7 rad/s.
  The earlier wording incorrectly assigned the reference-only error to both.
- Moving two-hinge passive chain, 0.5 s: angle error against MJWarp falls from
  4.84e-5 at dt=1/120 to 1.10e-5 at dt=1/960 (armature 0.05).
- Actual Ant: pulled-back augmented mass matches independent reduced mass plus
  diagonal armature to 2.98e-7 relative error. The production joint regularizer
  causes 4.16% force-response error; precond_reg=0 gives 0.006%, two existing
  refinement steps give 0.003%. These are diagnostic configurations, not new
  training settings.
- Isolated two-link frictionless impact with 1 m/s closing speed: even 1000
  contact iterations leave 0.732 m/s closing after the final joint solve with
  armature 0.05 (one sweep), versus 0 for the exact rigid DVI solution. Two
  sweeps leave 0.535 m/s. The preceding joint impulse is propagated correctly;
  the contact operator omits the additional bilateral response to its impulse.
- The retained independent-body implicit PD tangent is not relative joint
  damping. Ant-like kd=0.1 gives a small isolated error; kd=5 makes it substantial.
  Clipping and stiffness integration also differ from MJWarp in some cases, so
  those are not attributed to physical armature or silently treated as matched.

C0 source changes have been archived and reverted before candidate C1. The
historical documents/results remain. C1 will use relative auxiliary inertia rows,
the existing sparse factorization, and the bilateral-consistent unilateral
response. No bounded motor impulse formulation is being restored. Its physics,
finite-iteration behavior and cost must pass before training.


## Cached mass only, on the restored solver — October 10, 2026 (UTC)

Following the rollback below, the requested physical mass augmentation is
available through the new opt-in field:

```text
env.sim.physics.solver_cfg.use_cached_armature_mass=true
```

It defaults to false. Ordinary presets and task/PPO/actuator parameters are
unchanged. Keep `use_armature_rows=false` (the two models are mutually exclusive).
Use the standard train/evaluate entrypoints; archived experimental launcher
monkeypatches must not be reused with this implementation.

Newton now adds only Sections 1–2 of `dvi_cached_armature_equations.tex`:
body-frame `H_b = M_b + G_b.T A G_b`, its cached inverse and matching gyroscopic
bias. It retains separate constraint-family sweeps, the sparse joint backend,
force-based actuation and existing implicit-gain treatment. No condensed joint
response, mixed family/motor solve, bounded motor impulse rows, or dense joint
backend is restored. It is the same physical mass model as the earlier cached
trial, with different constraint/actuator algorithms. The historical note's
later implementation sections do not describe this new path.

Validation completed:

- 59 focused Newton CPU/CUDA tests passed, including independent matrix checks,
  zero-armature equivalence, all three contact solvers, limits, cache refresh and
  CUDA graph replay. The 47 pre-existing focused checks also pass.
- Ant startup/rollout/PPO smoke passed: 64 environments, three iterations, 6,144
  transitions, seed 42, Jacobi-10, one coupling sweep with joint post-stabilization,
  one substep, semi-implicit actuation. Authored armature `0.05`, damping `0.1`
  and angular damping `0.01` remain. Saved configuration confirms the new flag.
- Run: `logs/rsl_rl/dvi_mass_only_smoke/2026-10-10_04-36-35_ant_jacobi10`.
- Frozen 4,096-Ant benchmark: **2.210 ms restored baseline vs 5.079 ms cached
  mass only (2.30×)**, 15,065 contacts. These are captured physics-step timings
  including integration, excluding collision/PPO/host monitoring; not training
  FPS. Physical mass is cached once, while cross-body products and the denser
  sparse joint system add per-step work. Both use the flat sparse backend.

Evidence, source snapshots and hashes: `logs/dvi_mass_only/2026-10-10/`
(`validation.json`, `focused_tests.log`, `ant_smoke.log`, `benchmark.py`, timing
and profiler JSON). This is not a trained walking result or a resolved speed
regression. No full training, paper-plot update, commit or push was performed.

### Existing-result review and smoke video — October 10, 2026 (UTC)

The user chose to share existing results and smoke playback only. No additional
training or M-based joint approximation was attempted; solver source is unchanged.
Smoke rewards after its three PPO updates are `-2.94`, `-4.03`, `-4.89`.
Ten-second playback of `model_2.pt` across 64 environments measured `0.0303 m/s`
mean world-forward speed over the last five seconds, 100% upright samples and
no episode endings. This nearly untrained policy mainly stands/moves slowly;
it provides no evidence of restored high reward or fast trained locomotion.

Video (one visible Ant, 1280×720, 30 FPS):
`logs/dvi_mass_only/2026-10-10/review/ant_smoke_3_updates.mp4`.
Playback metrics and paired frozen-scene profiling are in the same directory.
The changed joint structure stores 28 rather than 10 off-diagonal blocks per
Ant, and uses eight rather than six elimination levels. Profiled joint
factorization costs `0.283 → 0.762 ms`; triangular solves `0.678 → 1.130 ms`.
Captured physics replay remains `2.209 → 5.078 ms`. The profile and captured
replay use different launch modes and are not additive measurements.
See the Newton status file for the read-only H/M matrix comparison and its
limitations; no approximation was substituted into dynamics.

## Workspace rollback — October 10, 2026 (UTC)

The user's requested rollback is complete. Active Newton source/tests match
`d6098901ac6197ee1737707f3a0fd8a3165f4695`; the Isaac Lab DVI manager/configuration
match `c0c3f214369076c5d293abada286825cfb0e8b86`. The condensed contact/limit solve,
bounded motor rows and dense cached-mass backend were uncommitted experiments
after those revisions. Their source, configuration hooks, tests and unreleased
fragments have been archived and removed from the active implementation.

Restored behavior is separate constraint-family solves, force-based actuation
and the existing sparse joint solver. The committed optional revolute armature
rows remain available. The ordinary Ant preset disables them and uses the
legacy body-inertia approximation. Ant retains its original joint damping
`0.1`, authored armature `0.05`, body angular damping `0.01` and semi-implicit
actuation. The previous `0.005` armature was only a run override. Quaternion
normalization and body angular damping also predate the reverted experiments.

Validation: 47 focused Newton regression tests passed. The standard release
validation launcher also completed an Ant smoke run with 64 environments,
three PPO updates, seed 42, Jacobi-10 contacts and one substep. The saved runtime
configuration confirms sparse joints, no experimental coupling/motor fields,
`use_armature_rows=false`, and the original Ant parameters. This smoke test is
not evidence of trained walking performance; no full training was launched.

Rollback archive, including original files, patches, hashes and validation:
`/home/horde/Documents/DVI/rollback_archives/dvi_decoupled_20261010T041258Z`.
Smoke checkpoint/configuration:
`logs/rsl_rl/dvi_rollback_smoke/2026-10-10_04-15-12_ant_jacobi10`.
Earlier experiment launchers and records under `logs/` are historical and need
their archived solver source to reproduce those runs. Existing paper figures
remain their previously recorded experimental data; they were not relabeled
as rollback results. No manuscript/PNG changes, commit or push were made.

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

## October 8 matrix regression and armature investigation

At Newton `d6098901` / Lab `c0c3f214`, the requested Ant/ANYmal C/H1/Go2/G1
Jacobi → APGD → PSPG matrix completed Ant and ANYmal C Jacobi. H1 Jacobi failed
with non-finite observations after 12 updates; remaining cases have not started.
The campaign record is `logs/dvi_armature_matrix/2026-10-07/status.json`, phase
`diagnosing_regression`.

The matched 1000-update Ant legacy control keeps the imported armature 0.05 and
changes only its treatment and output names. Final-50 reward is 74.93 versus
22.42 for rotor rows, and 64-world evaluation speed is 4.408 versus 0.355 m/s.
Both remain upright; smaller joint-anchor errors do not imply recovered gait.

Further investigation found:

- The eight existing isolated armature CPU/CUDA tests still pass. The ideal
  reflected-inertia row is not disproved by the training failure.
- Limits, joints/armature and contacts do not reach a coupled solution at the
  current staged budgets. Frozen Ant systems reproduced in float64 reveal an
  additional contact error floor caused by restarting every 10-iteration inner
  solve from zero. Within-timestep reuse and a condensed joint/armature response
  improve convergence; neither has been accepted as a production training fix.
- A 32-update H1 diagnostic reproduces a native physics NaN at environment step
  103. Legacy and rotor-plus-final-joint-correction controls finish all 768
  environment steps finite. The latter changes one solver setting in an isolated
  diagnostic, not the campaign recipe, and is not a trained walking result.
- A separate analytic two-body probe shows that existing implicit PD omits the
  parent/child cross terms of relative joint damping. Chrono and Kamino source
  comparisons motivate a coupled armature/contact solve and a consistent joint
  dynamics row for implicit drives.

[Detailed report and engine/literature references](logs/dvi_armature_matrix/2026-10-07/armature_research/REPORT.md),
[new diagnostic convergence plot](logs/dvi_armature_matrix/2026-10-07/armature_research/ant_coupling_diagnostic.png),
and [matched Ant report](logs/dvi_armature_matrix/2026-10-07/ant_diagnostics/REPORT.md)
retain raw data, config comparisons and reproducible scripts under ignored logs.
The full matrix awaits solver validation. The existing Ant/ANYmal C paper PNG
updates are experimental measurements; this investigation changed no additional
paper figures, manuscript text, production source or campaign parameters.
No commits or pushes were made.

Requested Ant playback is available for the
[rotor-row policy](logs/dvi_armature_matrix/2026-10-07/armature_research/videos/ant_jacobi_rotor.mp4)
and [matched legacy policy](logs/dvi_armature_matrix/2026-10-07/armature_research/videos/ant_jacobi_legacy.mp4).
Both are 20-second 1080p/30-fps recordings with a following camera, original
physics settings and preserved physical world positions; all 600 frames decoded.

Follow-up clarification: joint impulses already reach the contact RHS correctly.
What is absent is their incremental response inside the contact mass operator.
Many outer sweeps are not the intended solution; one or two remain the target.
The report now distinguishes this from force forwarding and treats cold starts
as a secondary convergence diagnostic. A new isolated native actuator-row
prototype matches the intended `SEMI_IMPLICIT` two-body response while retaining
old-state PD torque; keeping that torque is valid when the row represents only
the incremental inertia and replaces, rather than duplicates, body augmentation.
This is a diagnostic construction, not a production solver change or RL result.

The requested seven-page formulation note is available as
[PDF](../Newton-DVI-tech-report/dvi_coupled_armature_proposal.pdf) and
[LaTeX source](../Newton-DVI-tech-report/dvi_coupled_armature_proposal.tex).
It defines the shared contact/limit/motor operator, mode-specific actuator rows
and finite-effort treatment, with source references and algebra checks.
Production implementation and further training await review of this formulation.

### October 8: approved formulation implemented and tested experimentally

Added opt-in `DVISolverCfg.use_coupled_constraints` and `use_drive_rows`, forwarded
to Newton DVI. Both default to false. Coupled contacts and limits share the
contact iteration budget and include incremental bilateral/rotor reactions.
Drive rows preserve the selected actuator convention and place bounded motor
impulses in the shared solve. The staged outer-sweep and final-stabilization
settings are unused in the coupled mode; asset parameters and PPO settings were
not adjusted. The existing campaign launcher still needs the new flags before
it can run this formulation.

Completed isolated trials, using seed 42 and 4096 environments:

| Trial | Budget | Outcome |
| --- | --- | --- |
| H1, Jacobi-10, coupling only | 32 updates / 768 steps | Finite; max angular component 46.35 rad/s |
| H1, Jacobi-10, coupled drives | 32 updates / 768 steps | Finite; max angular component 35.13 rad/s |
| Ant, Jacobi-10, coupled drives | 1000 updates / 32,000 steps | Finite; last-50 mean reward 46.73395 |

The previous H1 staged-rotor run failed around step 103. The new H1 runs establish
short-run stability, not learned walking. Matched Ant playback at 64 worlds for
20 seconds gives 2.29959 m/s world-forward speed, all worlds upright, and zero
premature terminations. Previous staged rotor rows gave 0.35520 m/s; the matched
legacy approximation gave 4.40780 m/s. Thus Ant improves substantially but still
trails the legacy control after the same training budget.

The [new Ant playback](logs/dvi_armature_matrix/2026-10-07/armature_research/coupled_implementation/ant_playback/ant_coupled.mp4)
is 1080p/30 fps, with all 600 frames decoded. The checkpoint is
`logs/rsl_rl/coupled_armature_trials/2026-10-08_17-22-58_ant_drives_jacobi_1000_4096/model_999.pt`.
Commands, solver source snapshots, test evidence, and metrics are in
[`coupled_implementation`](logs/dvi_armature_matrix/2026-10-07/armature_research/coupled_implementation/).

Performance is the main prototype limitation: matched captured physics-step
times are 2.338 ms for staged rotor rows, 6.459 ms for coupled rotor rows, and
12.100 ms for coupled drive rows (4096 Ant worlds, frozen 15,070-contact scene).
These are solver-step timings, not PPO training FPS. The monitored training run's
126,091.5 median FPS includes host state copies and is not a paper benchmark.
Newton's analytic/dense-reference and CPU/CUDA/graph checks passed; see the
[Newton status](../newton-dvi/release16_dvi_integration_status.md) for details.

Next work is response reuse and condensed preconditioning/convergence validation
at the unchanged iteration budgets, followed by the full environment/solver
campaign. No additional paper figures or manuscript text were changed, and no
commits or pushes were made.

### October 8: coupled-solver performance optimization

Newton now reuses mixed-Jacobi responses, including the final equality reaction,
omits an unused velocity scatter when refinement is disabled, fuses small
independent joint backsolves into one CUDA block per component, and reduces the
mixed residual within thread blocks. Physics parameters and solver budgets are
unchanged; configured refinement remains enabled for H1.

On identical frozen 4096-world scenes with Jacobi-10, full coupled-drive physics
steps decreased from **12.098 to 3.809 ms for Ant** and **48.176 to 12.693 ms for
H1**. The staged references are **2.320 ms** and **4.159 ms**, respectively.
The remaining penalties are **1.64x for Ant** and **3.05x for H1**. These timings
exclude collision, PPO, and monitoring and must not replace training FPS.

Forty Newton regressions passed, including CPU/CUDA, dense references, conditional
graphs, and the original LDL suite. Frozen-scene velocities/impulses agree within
float32 tolerances, with the same iteration counts; both 120-step evaluations
remained finite. The response-count regression fails before the optimization.
See the [Newton status](../newton-dvi/release16_dvi_integration_status.md) and
[`performance/summary.json`](logs/dvi_armature_matrix/2026-10-07/armature_research/coupled_implementation/performance/summary.json)
for evidence and remaining limitations.

The training campaign and paper update are still outstanding. H1's remaining
physics cost requires further work on the coupled mass response. This pass
produced no new training curves, changed no paper figures, and made no commits.

### October 8 campaign: Jacobi completed; APGD and PSPG await review

Completed Ant, ANYmal C, H1, Go2, and G1 with the optimized coupled solver,
4096 environments, seed 42, and the original current-task training budgets.
`use_armature_rows`, `use_coupled_constraints`, and `use_drive_rows` were enabled.
The task recipes, gains, armatures, contact budgets, semi-implicit actuator mode,
regularization, and refinement settings were retained. Go2 retains the explicit
DC-motor experiment with armature 0.02. Training had startup configuration/source
audits but no per-step host state monitoring. All 43 solver source hashes match
the campaign snapshot.

| Task | PPO updates | Last-50 mean reward | Median training FPS | Playback forward speed | Premature ends / 4096 worlds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ant | 1000 | 46.4168 | 289,853.5 | 2.2402 m/s | 0 |
| ANYmal C | 500 | 21.3736 | 61,619.0 | 0.4985 m/s | 1 |
| H1 | 1000 | -2.6258 | 65,104.5 | 0.2810 m/s | 7 |
| Go2 | 500 | 32.1602 | 75,457.5 | 0.5112 m/s | 5 |
| G1 (resumed) | 1500 | 6.7320 | 26,581.5 | 0.4828 m/s | 0 |

Playback ran for 1200 policy steps at both 64 and 4096 environments, checking
observations and native body poses/velocities after every policy step. All ten
evaluations remained finite. The table uses the 4096-world checks: 20 seconds
for Ant and 24 seconds for the other tasks. Ant speed is world-forward; other
speeds are body-forward with a fixed 0.5 m/s command. Premature ends exclude
timeouts and count resets, not unique robots. All 64-world checks had zero
premature ends. H1 therefore avoids the old early numerical failure but still
tracks the speed command poorly; finiteness does not establish restored learning
quality or solver convergence. These are one-seed, flat-ground evaluations.

Two interruptions are recorded rather than hidden. ANYmal C's first playback
completed, but the video-metadata wrapper then read an argument that the evaluator
had removed from `sys.argv`. The wrapper now captures the output path first;
playback was repeated successfully without retraining. The original artifacts
remain under `jacobi_anymal_c/first_playback_metadata_error/`.

G1's worker and controller disappeared around a chat interruption after logged
iteration 1459; no process exit status was available. Recovered policy, critic,
and optimizer from `model_1450.pt`, then performed updates 1451 through 1499.
Simulation and RNG state were initialized again with seed 42. The curve retains
original samples 0–1450 and resumed samples 1451–1499; the unsaved original tail
is excluded. Configuration comparison shows only output path/checkpoint changes.
The last-50 reward includes the restart transient: the 50 samples before recovery
averaged 11.0753, and the final ten resumed samples averaged 12.1219. G1's curve is
provisional for an uninterrupted paper benchmark. Recovery used a detached
controller; it has now exited with all five cases complete.

Updated only the five existing `*_jacobi_mjwarp_apgd_progress.png` panels' Jacobi
reward data and FPS. Titles, style, dimensions, and other solver curve arrays
were retained; the latter were checked for exact equality. All five 1080p videos
decoded all 600 frames. `main.tex` and the PDF/SVG panels remain unchanged.

The Ant panel is not yet a valid cross-solver reward comparison: its historical
logs use `Isaac-Ant-Direct-v0` from `isaaclab_tasks.direct.ant`, whereas the current
core task multiplies continuous rewards by `step_dt` and normalizes progress by
policy-step duration instead of physics-step duration. No numerical rescaling
was applied to hide that difference. The APGD/PSPG reruns and a matching MJWarp
reference must use the same current task definition before final paper use.

Evidence: [summary JSON](logs/dvi_coupled_matrix/2026-10-08/summary.json),
[summary CSV](logs/dvi_coupled_matrix/2026-10-08/summary.csv),
[commands and state](logs/dvi_coupled_matrix/2026-10-08/status.json), and
[Ant reward definitions](logs/dvi_coupled_matrix/2026-10-08/ant_reward_comparability.json).
Each case directory contains playback metrics, state arrays, video, checkpoint
hash, and artifact checks. Videos:
[Ant](logs/dvi_coupled_matrix/2026-10-08/jacobi_ant/jacobi_ant.mp4),
[ANYmal C](logs/dvi_coupled_matrix/2026-10-08/jacobi_anymal_c/jacobi_anymal_c.mp4),
[H1](logs/dvi_coupled_matrix/2026-10-08/jacobi_h1/jacobi_h1.mp4),
[Go2](logs/dvi_coupled_matrix/2026-10-08/jacobi_go2/jacobi_go2.mp4), and
[G1](logs/dvi_coupled_matrix/2026-10-08/jacobi_g1/jacobi_g1.mp4).

The campaign is `jacobi_complete_awaiting_review`. No training or playback worker
remains. APGD and PSPG were not started, as requested. Performance, H1 tracking,
G1's uninterrupted benchmark, and historical-reference comparability remain
open for the review. No commits or pushes were made.

## Ant cached-mass trial complete (2026-10-09)

The user selected **Ant first, then review**. The opt-in cached mass adapter
completed 1000 PPO updates with 4096 environments, seed 42 and Jacobi-10.
Resolved task and PPO settings match the preceding coupled run except output
names. The new backend uses two fixed internal residual corrections against
the unchanged parent-axis rotor/equality equations; these are recorded in
`cached_solver_audit.json`, not silently treated as an old configuration value.
The adapter caches mass elimination once but still factors changing geometric
constraints each step. It is an experimental implementation of the existing
model, not a new armature model or a default backend.

| Ant treatment | Final-50 mean reward | Median training FPS | World-forward speed |
|---|---:|---:|---:|
| Cached mass trial | 43.5013 | 185,912.5 | 2.0229 m/s |
| Previous coupled rows | 46.4168 | 289,853.5 | 2.2391 m/s |
| Legacy body armature control | 74.9338 | 354,975.5 | 4.4078 m/s |

Speed is the final-half mean from matched 20-second, 64-world playback. The
legacy control retains armature 0.05 through its earlier body-inertia
approximation. Both 64- and 4096-world cached playbacks had finite states,
upright fraction 1.0 and zero premature terminations. Independent final-system
checks in eight worlds per playback have maximum relative energy error
`4.051e-7` and relative residual `7.998e-6` against dense float64 solves.

Matched captured physics steps on one frozen 4096-world scene with 15,065
contacts take 7.680 ms for the cached prototype and 3.807 ms for the existing
coupled solver: **2.017x slower**. Timing includes integration, but not collision
detection, PPO, or monitoring. Maximum absolute single-step body-velocity
difference is `1.060e-4`. The fast isolated mass response therefore does not
establish a useful full-step performance improvement here.

The first full attempt was interrupted after update 327 to fix a global TF32
setting that affected PPO. It is archived and excluded. The reported run is a
fresh seed-42 run after scoping library/precision settings to physics and
verifying their restoration. Startup failures and the independent test's
failure-before/pass-after evidence are preserved in the trial directory.

Updated Ant's existing Jacobi PNG data/FPS only, preserving title/style,
dimensions and unrelated solver arrays. `main.tex` and all other robot PNGs
were checked unchanged. The separate review plot uses matched current-task
controls; historical solver curves retained in the paper still have the older
Ant reward definition. The proposal PDF's version 4 includes the new results.

Artifacts: [report](logs/dvi_cached_training/2026-10-09/REPORT.md),
[reward/FPS comparison](logs/dvi_cached_training/2026-10-09/ant_comparison.png),
[Ant video](logs/dvi_cached_training/2026-10-09/ant_cached.mp4), and
[full summary](logs/dvi_cached_training/2026-10-09/summary.json).
The 1080p video decoded successfully. The cached implementation completed
training but has not recovered the legacy behavior or improved solver speed.
All workers have exited at `ant_complete_awaiting_review`; no other robot,
APGD, or PSPG run was started. No commits or pushes were made.

## Ant mass and Jacobi audit (2026-10-09)

A frozen-state check after the cached Ant trial found a concrete convergence
issue: mixed Jacobi uses the coupled joint/armature response for its operator,
but a body-local inverse-mass diagonal for its updates. With the same ten
iterations, replacing only that diagonal with the coupled one reduces bounded
motor impulse vector error from **31.86–37.00% to 4.19–5.20%** against a converged
offline reference across eight captured Ant worlds. Contact residuals also
improve, but remain substantial. Independent matrix calculations reproduce
the native solver's impulses to `2.986e-7` maximum absolute error.

This is a preconditioning issue at the fixed budget, not proof of an incorrect
physical mass or a complete diagnosis of the learning regression. Newton and
the compiled MuJoCo model both use armature 0.05; intended generalized masses
agree at rest to `2.389e-7` relative error. No legacy body augmentation remains,
and the physical-inverse final velocity update agrees with the accumulated
forces/impulses to `4.768e-7`. A matched current-task MJWarp training control has
not been run in this audit.

The cached adapter targets the preceding coupled equations but adds dense
geometric factorization and repeated residual corrections. Retaining the
existing sparse backend and improving the mixed contact/motor preconditioner
is the next focused step, with convergence and cost measured before training.
No production solver code or paper PNG changed, and no training was launched.
The frozen systems have no active limits; other robots were not sampled.

Evidence and reproduction: [audit report](logs/dvi_mass_audit/2026-10-09/REPORT.md),
[all diagonal results](logs/dvi_mass_audit/2026-10-09/diagonal_results.json), and
[mass checks](logs/dvi_mass_audit/2026-10-09/model_audit.json).

## Standalone cached mass: Ant trial complete (2026-10-09)

The requested experiment uses the cached augmented mass as the actual operator
throughout the step, with zero auxiliary armature rows and zero corrections to
the previous parent-axis equations. Its fixed body-axis kinetic energy includes
cross coupling, and its gyroscopic term uses the full angular momentum.
Jacobi scaling remains unchanged. The runtime audit distinguishes this backend
from the temporary native solver used during environment startup.

Ant completed 1000 PPO updates, 4096 environments, seed 42, and Jacobi-10.
Resolved task/PPO YAML differs from the previous coupled run only in output names.

| Treatment | Final-50 reward | Median training FPS | Forward speed, 64 worlds |
|---|---:|---:|---:|
| Standalone cached mass | 46.4645 | 253,802.5 | 2.2344 m/s |
| Previous cached preconditioner | 43.5013 | 185,912.5 | 2.0229 m/s |
| Previous coupled rows | 46.4168 | 289,853.5 | 2.2391 m/s |
| Legacy body armature | 74.9338 | 354,975.5 | 4.4078 m/s |

Both 64- and 4096-environment 1200-step playbacks had finite observed native
states, upright fraction 1.0 and no premature terminations. The larger playback
averaged 2.2331 m/s. Physics-only frozen-step timing is 4.851 ms for standalone
mass, 3.807 ms for sparse coupled rows, and 7.677 ms for the previous cached
preconditioner, including integration and excluding collision/PPO. The new run
is faster than the previous cached trial, with reward/speed close to the earlier
coupled-row run; the legacy behavior is not recovered.

Pre-training floating-chain and full-size startup checks passed. The final
frozen audit measures force-balance velocity error at most `6.905e-5`, but also
exposes float32 geometric-condensation sensitivity: worst energy-operator error
0.4536 in strongly constrained directions, with overall Frobenius error below
0.001. Offline float64 setup with the same cached float32 mass, followed by
float32 storage, reduces this energy error to `1.338e-4`. No precision change
was made to the completed training, and it is not presented as a validated fix.

Updated Ant's paper PNG Jacobi data/FPS with original title/style and other
solver arrays retained. The manuscript and other robot PNGs are unchanged.
Proposal version 5 includes the standalone formulation and measured results.
All workers completed at the Ant review boundary. No APGD, PSPG, or other robot
training was started; no commits or pushes were made.

Artifacts: [report](logs/dvi_standalone_cached_mass/2026-10-09/REPORT.md),
[reward/FPS plot](logs/dvi_standalone_cached_mass/2026-10-09/ant_comparison.png),
[Ant video](logs/dvi_standalone_cached_mass/2026-10-09/ant_cached.mp4),
[summary](logs/dvi_standalone_cached_mass/2026-10-09/summary.json), and
[precision diagnostic](logs/dvi_standalone_cached_mass/2026-10-09/precision_diagnostic.json).

## Cached mass accuracy and fresh MJWarp Ant control (2026-10-10)

The latest Ant experiment retains the augmented mass in the shared dynamics
response for all impulses. Contact/limit laws remain independent of armature;
their body-local Jacobi diagonals, ten iterations, and all task/PPO parameters
are unchanged. No predictor-only split was introduced.

The condensed response now uses the algebraically equivalent stable sum
`C = P B P.T + K R K.T`, where `B = H^-1`, `K = B J.T (J B J.T + R)^-1`
and `P = I - K J`. This reduces float32 cancellation. A float64 prototype was
rejected for training at 21.992 ms/step. Stable float32 takes 5.614 ms on the
matched frozen 4096-world Ant scene (previous cached: 4.851 ms; sparse coupled:
3.806 ms). The new constrained-direction regression fails before the change
and passes afterward; both cached-mass tests and focused Ruff checks pass.
The sixteen final playback samples have worst energy-operator error 0.000787.
Force reporting still has a maximum force-balance velocity discrepancy of
0.001063; this remaining numerical limitation is recorded in the report.

Both Ant runs completed 1000 PPO updates, 4096 worlds and seed 42, sequentially:

| Solver | Final-50 reward | Median training FPS | 64-world forward speed | Premature terminations |
|---|---:|---:|---:|---:|
| Stable cached mass, Jacobi-10 | 42.6198 | 230,542 | 1.9581 m/s | 0 |
| Fresh MJWarp, current task | 76.6416 | 503,814 | 4.3853 m/s | 2 |

The numerical fix did not recover MJWarp walking performance. The fresh control
uses the same current task/PPO settings; native physics presets differ, including
contact behavior and joint-wrench observation availability. It avoids comparing
rewards against the historical task definition.

Follow-up comparisons use actual GPU MJWarp and identical authored inertias.
Applied-torque responses match analytic armature mechanics with and without
armature. For a light damped hinge, DVI-10 gives 0.090895 rad/s versus MJWarp's
0.162866; DVI-1000 gives 0.162867. A limit stop retains 0.8435 rad/s from an
incoming 1 rad/s after ten iterations, approaching zero when converged. A
single-contact stop retains -0.1963 m/s normal velocity after ten iterations,
approaching the near-rigid MJWarp result when converged. High iteration counts
are diagnostics, not changes to training parameters. Passive-chain trajectory
errors decrease with timestep, including nonzero armature.

Frozen Ant Jacobi-10 impulses are independently reproduced within 7.44e-9.
At unchanged mass, diagonal, relaxation and ten iterations, initializing motor
impulses from the clipped drive-force estimate reduces their error versus a
converged reference from 33.13–35.24% to 0.894–2.053%. Contact violations remain
substantial. This initialization is an offline experiment only; it has not
been installed or trained. Separate tests identify PD integration and saturation
semantics that differ from MJWarp even after convergence; the saturated external
force example is outside Ant's normal command bounds.

Ant's existing PNG received only the completed Jacobi and MJWarp reward/FPS
updates, preserving title/style and unrelated curves. The manuscript and other
robot PNGs are unchanged. Both playback videos decode successfully. All 43
archived solver source files match the training revision. No other robots,
APGD/PSPG training, commits or pushes were started.

Artifacts: [complete report](logs/dvi_cached_mass_precision/2026-10-10/REPORT.md),
[current-task reward/FPS plot](logs/dvi_cached_mass_precision/2026-10-10/ant_matched_comparison.png),
[DVI video](logs/dvi_cached_mass_precision/2026-10-10/ant_cached.mp4), and [MJWarp video](logs/dvi_cached_mass_precision/2026-10-10/ant_mjwarp.mp4).


## Ant armature reduction from 0.05 to 0.005 (2026-10-10)

Completed the requested controlled Ant ablation: cached augmented mass with Jacobi, 4,096 environments, seed 42, 1,000 PPO iterations. Only `scene.robot.actuators.body.armature` changes physically; output paths/names are also different. All 43 DVI solver source hashes and numerical solver settings match the preceding precision-stable cached-mass run. The loaded model and both playbacks verify 0.005 on all eight actuated joints per environment. No preconditioner, motor initialization, contact law or solver formulation changes were made.

| Run | Armature | Final-50 reward | Median training FPS | Forward speed (m/s) | Premature terminations |
|---|---:|---:|---:|---:|---:|
| DVI Jacobi, reduced armature | 0.005 | 99.241 | 236,476 | 6.362 | 0 |
| DVI Jacobi, previous armature | 0.05 | 42.620 | 230,542 | 1.958 | 0 |
| MJWarp reference | 0.05 | 76.642 | 503,814 | 4.385 | 2 |

Reward is the mean of the last 50 updates; FPS is median training throughput; speed and termination counts are from the 20-second, 64-environment playback. The additional 4,096-environment playback gives 6.361 m/s and 0 premature terminations. MJWarp remains the earlier reference at 0.05, not a matched reduced-armature trial.

Archive: `/home/horde/Documents/DVI/isaaclab-dvi/logs/dvi_reduced_armature/2026-10-10/REPORT.md`. This contains commands, resolved parameter differences, checkpoint, data, source hashes, mass checks, comparison plot and validated playback. Updated only Ant's Jacobi curve/FPS in the existing paper PNG, preserving its title/style and the other arrays. APGD/P-SPG-FB retain historical task data; the separate comparison plot is the interpretable current-task comparison. Paper `main.tex` and all other PNGs remain unchanged.

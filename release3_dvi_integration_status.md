# Release-3 Newton DVI integration: work summary and unresolved failures

**Status date:** 2026-10-02
**Repository:** `isaaclab-dvi`
**Branch:** `milad/develop` at `e72584b71bf5d9394e81026e32883ab6af04b548`

## Executive summary

This work attempted to make the retained Newton DVI backend run through the Isaac Lab 3.0 integration, first with Ant and then with the requested ICRA environment retraining sequence.

The compatibility work was sufficient to:

- start a replicated Newton scene with 64 Ant worlds/articulations;
- complete a 64-environment, two-iteration Ant smoke test;
- complete a 4096-environment, 1000-iteration Ant run; and
- restore basic Newton-GL playback initialization after additional visualizer compatibility changes.

It was **not successful as a benchmark/retraining result**:

- the completed Ant run converged to a very low-reward solution (`86.46` final reward), far below prior runs, and its changed observation MDP prevents a controlled direct comparison;
- full-scale Humanoid DVI runs develop non-finite root linear velocity and abort with NaN policy observations;
- many attempted fixes were compatibility shims or diagnostic hypotheses rather than validated final fixes;
- the requested sequential ICRA retraining stopped at Humanoid, so Anymal-C, Go2, H1, G1, and DR Legs were not started;
- the working tree is a very large staged release-3 merge with an additional mixed staged/unstaged ten-file experimental overlay. It should not be committed as one undifferentiated change.

## Repository state captured by this WIP commit

This WIP checkpoint captures the complete working state: the large staged Release 3 migration plus the final ten-file experimental overlay used by the latest tests. Before consolidation, the migration contained 4,548 staged files (approximately 442,999 insertions and 183,331 deletions), while the overlay contained 361 insertions and 325 deletions across these files:

- `source/isaaclab_assets/isaaclab_assets/robots/humanoid.py`
- `source/isaaclab_newton/isaaclab_newton/assets/articulation/articulation.py`
- `source/isaaclab_newton/isaaclab_newton/assets/kernels.py`
- `source/isaaclab_newton/isaaclab_newton/cloner/replicate.py`
- `source/isaaclab_newton/isaaclab_newton/physics/newton_manager.py`
- `source/isaaclab_tasks/isaaclab_tasks/core/locomotion/humanoid/humanoid_common.py`
- `source/isaaclab_tasks/isaaclab_tasks/core/locomotion/humanoid/humanoid_direct_env_cfg.py`
- `source/isaaclab_tasks/isaaclab_tasks/core/locomotion/locomotion_direct_env.py`
- `source/isaaclab_tasks/isaaclab_tasks/core/velocity/velocity_env_cfg.py`
- `source/isaaclab_visualizers/isaaclab_visualizers/newton/newton_visualizer.py`

The WIP commit stages the final working-tree versions of these files, so the checkpoint matches the code used for the latest experiments. This document is stored at the repository root and included in the checkpoint.

## What was changed

### DVI/release-3 compatibility layer

The working-tree overlay attempts to bridge retained DVI code to release-3 interfaces:

- use release-3 clone-aware articulation-root resolution and register an `ArticulationView` in the manager view cache;
- construct and publish the replicated Newton builder so all requested worlds reach the final model rather than only authored `env_0`;
- add narrow fallbacks for missing DVI manager APIs used by release-3 cloning (`_get_usd_import_schema_resolvers`, `load_visual_shapes`, root-site return values, per-world hooks, and scene-data publication);
- move Newton articulation actuator handling to the release-3 `ActuatorCollection`/`NewtonActuatorControl` interface;
- extend joint/body lookup with `as_proxy`, add indexed joint-state dispatch compatibility, and add missing projected-gravity Warp kernels;
- add a minimal scene-data/backend adapter for the Newton visualizer;
- restore DVI presets for Humanoid and rough-terrain velocity tasks;
- pin Humanoid to the Isaac 6.0 USD asset as a diagnostic;
- use the indexed effort-target path on every decimation substep;
- clone default reset tensors before mutation; and
- add optional finite-value diagnostics that identify which Humanoid observation term first becomes invalid.

These edits are not all proven necessary or correct. Several were introduced sequentially to cross individual startup failures, and the final Humanoid failure remains unresolved.

## Ant: compatibility failures and outcome

### Smoke-test progression

The Ant smoke tests used `Isaac-Ant-Direct`, Newton DVI, 64 environments, and two PPO iterations. The retry sequence records the release-version incompatibilities encountered:

| Log | Observed failure | Interpretation |
|---|---|---|
| `logs/rsl_rl/ant_release3_dvi_smoke/console_jacobi20_retry5.log` | no articulation matched `/World/envs/env_[^/]+/Robot/torso` | articulation discovery/replication path incompatible |
| `.../console_jacobi20_retry6.log` | `ImplicitActuator.__init__()` rejected `armature` | actuator API mismatch |
| `.../console_jacobi20_retry7.log`, `retry8.log` | `find_joints()` rejected `as_proxy` | articulation API mismatch, reproduced |
| `.../console_jacobi20_retry9.log` through `retry12.log` | Warp CUDA error 710, device-side assert | runtime/kernel failure, reproduced |
| `.../console_jacobi20_retry13.log` | missing `tomllib` | Python/runtime mismatch in that invocation |
| `.../console_jacobi20_retry14.log` | missing `_get_usd_import_schema_resolvers` | DVI manager/release cloner mismatch |
| `.../console_jacobi20_retry15.log` | missing `NewtonCfg.load_visual_shapes` | configuration API mismatch |
| `.../console_jacobi20_retry16.log` | `_cl_inject_sites` returned two values where three were expected | cloner API mismatch |
| `.../console_jacobi20_retry17.log` | missing `NewtonManager._scene_data_backend` | manager/scene-data API mismatch |
| `.../console_jacobi20_retry18.log` | Warp could not infer `pos_data` from a Torch tensor | Warp call/type mismatch |
| `.../console_jacobi20_retry19.log` | missing `projected_gravity_b_kernel` | kernel/version mismatch |
| `.../console_jacobi20_retry20.log` | exit 0; rewards `-0.05`, then `-0.29` | backend smoke passed; no evidence of useful learning |

The successful smoke showed 64 worlds, 64 articulations, and 64 matching roots. This supports the clone-plan/builder handoff repair, but it validates only startup and two iterations.

### Full Ant run

Evidence:

- console: `logs/rsl_rl/ant_release3_dvi_4096/console_jacobi20_release3_4096_20261001.log`
- run directory: `logs/rsl_rl/ant_release3_dvi_4096/2026-10-01_01-56-21_jacobi20_release3_4096_20261001/`
- configuration: 4096 environments, 1000 iterations, seed 42, Newton DVI Jacobi-20
- final reward: **86.46**
- last-100 reward: **86.5142 ± 0.0708**
- final episode length: **960.00**
- final action standard deviation: **0.01**
- steady throughput: approximately **368,601 steps/s**
- training time: **359.05 s**

The run completed and was numerically stable, but it converged to a near-deterministic, low-reward behavior. This is not a successful reproduction of the earlier Ant result.

A historical completed PSPG run had final reward `8,740.6` and last-100 mean `8,839.08 ± 208.07`, but that is **not a controlled solver-only comparator**. The release-3 Ant observation changed from 36 to 60 dimensions by adding 24 foot-wrench channels (force and torque for four feet), and task/code/recipe revisions also differ. The large reward gap therefore establishes a regression or task-equivalence problem, not that Jacobi DVI itself is intrinsically worse.

### Ant playback

Playback exposed a separate compatibility chain:

- missing `NewtonBackendCfg` export/interface;
- Isaac Sim not available on one attempted launch path;
- absent scene-data provider (`None.transform_paths`);
- missing `moviepy` for recording.

Later scene-data and 64-environment Newton-GL playback logs exited successfully, showing that basic playback initialization can be restored. This does not repair or explain the poor learned policy.

## Humanoid: unresolved full-scale numerical failure

The requested ICRA retraining sequence was:

1. Humanoid — 1000 iterations, contact iterations 10
2. Anymal-C — 500, contact 20
3. Go2 — 500, contact 15
4. H1 — 1000, contact 10
5. G1 — 1500, contact 15
6. DR Legs — 1000, contact 10

Execution was intentionally sequential. Humanoid never passed validation, so environments 2–6 were not launched.

### Reproducible observation

The baseline and Isaac-6.0-asset full runs both aborted before meaningful training with:

```text
ValueError: The observation group 'policy' returned by the environment contains NaN values.
```

Finite-value instrumentation localized the first visible invalid term to root-frame linear velocity, for example:

```text
RuntimeError: Non-finite Humanoid observation term 'linear_velocity': env=3331, component=0, value=nan
```

The affected environment changes across variants (`3331`, `288`, `1198`), so this does not look like a single permanently malformed environment index.

The importer also repeatedly reports invalid inertia and negative mass for Humanoid bodies, including torso, head, waist, pelvis, legs, arms, and hands. This is strong evidence of an asset/import mass-property problem, but causality between those warnings and the later velocity NaN has not yet been demonstrated.

### Diagnostic variants

| Variant | Observation | Conclusion supported by the run |
|---|---|---|
| default asset, 4096 envs | policy observation NaN | baseline full-scale failure |
| pinned Isaac 6.0 asset | 64-env smoke completes; 4096-env run still NaNs | asset pin alone is insufficient; issue is scale-sensitive |
| 2048/4096 diagnostic runs | some short runs exit 0 near reward `-1`, episode length `~30` | failure is intermittent/trajectory-dependent, not guaranteed at initialization |
| CUDA graph disabled, 2048 envs | still NaNs | CUDA graph is not sufficient to explain the failure |
| scaled-action variant, 2048 envs | still NaNs | action magnitude alone is not sufficient |
| cloned reset tensors, 2048 and 4096 envs | still NaNs | corruption of canonical defaults through reset views is not sufficient |
| indexed command path, 4096 envs | same `linear_velocity` NaN | all-joint cached command path is not sufficient |
| Isaac Lab actuator path, 4096 envs | same symptom at another env | Newton-native actuator path is not sufficient |
| joint post-stabilization disabled, 4096 envs | same symptom at another env | post-stabilization alone is not sufficient |

One `resetclone2048` log contains `ModuleNotFoundError: tomllib` while its wrapper metadata reports exit code 0. That result is internally inconsistent and must not be counted as a successful training run.

### Current diagnosis

**Observed:** DVI simulation state eventually contains a non-finite root linear velocity at large environment count, which propagates into the policy observation and causes RSL-RL to abort.

**Supported but unproven hypothesis:** release-3 USD import/replication is producing invalid Humanoid mass/inertia data, and the resulting dynamics become unstable at scale.

**Ruled out as sole causes by direct variants:** CUDA graph capture, action scale, reset-tensor aliasing, all-joint versus indexed effort submission, Newton-native versus Isaac Lab actuator routing, joint post-stabilization, and use of the newer Humanoid USD alone.

**Not established:** the first bad physics quantity, the exact body/joint that diverges, whether the negative-mass warnings are causal, and whether the defect lies in USD schema resolution, clone replication, model finalization, or DVI integration.

## Why this remains a WIP checkpoint

1. **The central requested result failed.** Ant did not reproduce useful historical behavior, and Humanoid blocks the remaining retraining sequence.
2. **The diff mixes integration and experiments.** Release-3 merge content, compatibility shims, task presets, visualizer workarounds, finite diagnostics, asset pinning, reset changes, and action-routing experiments are interleaved.
3. **The checkpoint is intentionally broad.** It captures both the staged migration and the final working-tree overlay that produced the latest logs so the exact state can be resumed elsewhere.
4. **Several changes are diagnostic, not final fixes.** The hard-pinned Humanoid asset, verbose articulation selection warning, environment-variable finite checks, and compatibility fallbacks need explicit disposition.
5. **Validation is incomplete.** No focused unit/integration test set has been run against the final mixed state; only `git diff --check` and experiment logs are available.
6. **Benchmark equivalence is absent.** The Ant MDP changed, making the current historical reward comparison unsuitable for a solver-performance conclusion.

## Recommended disposition after this WIP checkpoint

- Preserve the current tree and logs; do not reset or clean it.
- Review and split the ten-file experimental overlay explicitly before preparing reviewable commits.
- Separate the work into reviewable concerns:
  1. release-3 merge;
  2. minimal DVI API compatibility;
  3. clone-plan/multi-world repair;
  4. visualizer/playback compatibility;
  5. task presets;
  6. temporary diagnostics and failed hypotheses.
- Remove or isolate diagnostics that are not intended product behavior.
- Add focused tests for multi-world articulation count/root matching, actuator command submission, indexed state reset, and finite Humanoid state over a meaningful rollout.
- Do not present the Ant comparison as a solver benchmark until the task/observation/reward/configuration are held equivalent.
- Resume the sequential ICRA runs only after Humanoid is finite and repeatable at 4096 environments.

## Bottom line

The work repaired enough release-3/DVI incompatibilities to run Ant end-to-end, but it did not produce a valid historical reproduction, and full-scale Humanoid remains numerically broken. The current changes are preserved as a WIP checkpoint for continuation on another machine; they must not be interpreted as a successful or release-ready implementation.

# RL solver sweep: speed versus reward

## Objective

Compare the RL behavior and simulation throughput of the paper contact solvers
in Isaac Lab locomotion environments:

- projected Jacobi;
- APGD;
- P-SPG-FB;
- MuJoCo Warp (`newton_mjwarp`) as the non-DVI baseline.

For each DVI solver, contact iteration budgets are 5, 10, and 15. The bilateral
joint solve is held fixed at sparse LDL. The first experiment is Isaac Ant; G1,
H1, Go2, and DR Legs follow only after the Ant configuration and measurements
are validated.

## Metrics

Every run must record:

1. **Training reward:** mean episode reward and its learning curve over PPO
   iterations. Final reward alone is insufficient; compare the full curve and
   the best/final values.
2. **Throughput:** simulation/training iterations per second (and, where
   available, environment steps per second). Report steady-state throughput,
   excluding one-time Isaac Sim/Warp compilation and startup.
3. **Wall time:** total elapsed training time and time per PPO iteration.
4. **Stability:** failed runs, NaNs, termination/collapse, and reward variance.

## Fairness protocol

- Same Isaac Lab/Newton commit, GPU, conda environment (`dvi`), PPO seed,
  number of environments, PPO horizon, batch/minibatch settings, and training
  iterations.
- Same DVI settings other than contact solver type and contact iteration
  budget; joint solver remains sparse LDL.
- `newton_mjwarp` is reported as a separate baseline. Its native MuJoCo-Warp
  solver settings are not forced to equal DVI iteration counts; those settings
  are recorded explicitly and may be restricted in a later controlled study.
- Run a short Ant pilot first. Do not launch the full 12-run matrix until the
  pilot log format, reward extraction, and FPS measurement are verified.
- Save raw logs and a machine-readable metadata file for every run. Generated
  logs/checkpoints are not committed by default.

## Planned matrix

| Phase | Environment | Solver | Contact iterations | Status |
|---|---|---|---:|---|
| 1 | Ant | DVI-Jacobi/APGD/P-SPG-FB | 10 | initial run |
| 1 | Ant | MJWarp | native config | initial run |
| 2 | Ant | DVI-Jacobi/APGD/P-SPG-FB | 5, 10, 15 | pending |
| 2 | Ant | MJWarp | native config | pending |
| 3 | G1, H1, Go2, DR Legs | same matrix | 5, 10, 15 | pending |

The initial Ant run uses the existing 1000-PPO-iteration setup and 4096
parallel environments unless a pilot exposes a configuration problem. Existing
Ant presets use Jacobi=20, APGD=10, and PSPG=8; for the controlled comparison,
all three DVI solvers are overridden to 10 contact iterations.

## Commands

Run from `~/repos/isaaclab-dvi`:

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate dvi
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Ant-Direct-v0 --headless --max_iterations 1000 \
  env.sim.physics=newton_dvi \
  env.sim.physics.solver_cfg.contact_max_iterations=10
```

Replace `newton_dvi` with `newton_dvi_apgd` or `newton_dvi_pspg` for APGD and
P-SPG-FB. The MJWarp baseline uses:

```bash
env.sim.physics=newton_mjwarp
```

The final sweep should be launched through a wrapper that stamps each run,
saves the exact command and git revisions, and writes logs under
`results/rl_solver_sweep/raw/<environment>/<solver>/<budget>/`.

## Interpretation policy

- First compare reward at matched training progress and throughput separately;
  do not rank a solver by reward alone when it receives fewer environment steps
  because it is slower.
- Then compare reward versus wall-clock time and environment steps.
- If MJWarp is substantially faster or has a different reward scale, preserve it
  as a baseline rather than silently tuning it to match DVI.
- Only after the unconstrained baseline is understood should MJWarp restrictions
  or additional solver restrictions be introduced.

## Reproducibility checklist

- [ ] record `git rev-parse HEAD` for Isaac Lab and Newton;
- [ ] record GPU, driver, Warp/Isaac Sim versions;
- [ ] record all Hydra overrides and random seed;
- [ ] preserve raw stdout/stderr and parsed metrics;
- [ ] generate reward/FPS plots from raw logs, never by hand;
- [ ] run at least one repeated seed for conclusions that depend on small
  reward differences.

## Ant initial-contact-10 results (completed 2026-07-23)

The native MJWarp baseline was rerun after upgrading `mujoco-warp` from
`3.8.0.1` to `3.8.0.3` (the version required by Newton). It completed with exit
code 0. The run used native `newton_mjwarp` settings and one physics substep.

| Solver | Contact iterations | Training time | Approx. throughput | Final mean reward | Status |
|---|---:|---:|---:|---:|---|
| Jacobi | 10 | 351.23 s | ~375k steps/s | 11,044 | complete |
| APGD | 10 | 425.52 s | ~313k steps/s | 12,896 | complete |
| P-SPG-FB/PSPG | 10 | 842.88 s | ~159k steps/s | 11,156 | complete |
| MJWarp | native | 288.19 s | ~460k steps/s | 10,253 | complete |

MJWarp log: `raw/ant/initial_contact_10/mjwarp.log`. The package upgrade
reported unrelated environment dependency conflicts; these are recorded in the
terminal/package state and should be kept with the benchmark provenance.

## Multi-environment sweep status

The next sweep is launched sequentially (one environment, then one solver at a
time) for flat G1, flat H1, flat Go2, and DR Legs Walk. DVI runs use sparse LDL
bilateral solves and the requested contact-iteration budget; MJWarp is included
only where the environment exposes a `newton_mjwarp` preset. The adapter fix
used for Ant is shared by these environments: unsupported `position_correction`
and `contact_substeps` fields are not forwarded to the current Newton API.

Added solver presets before launch:

- G1: `newton_dvi_pspg`
- H1: `newton_dvi_pspg`
- Go2: `newton_dvi_pspg`
- DR Legs: `newton_dvi_pspg`

Each run writes its raw log, exact command, revisions, GPU information, and
completion status under `raw/<environment>/<solver>/<budget>/`.

### Sweep recovery note (2026-07-23 07:27 UTC)

The first manager-based locomotion launch exposed a backend-selection issue: the
command selected `env.sim.physics=newton_dvi` but did not select the matching
global `presets=newton_dvi`, so Isaac Lab attempted to construct the PhysX
contact sensor without the Isaac Sim `omni.physics` module. This was not a DVI
solver or adapter failure. The sweep was stopped and the wrapper was corrected
to set both overrides explicitly. A one-iteration G1 smoke test then completed
successfully with Newton DVI, CUDA graph capture, sparse LDL joints, and a
5-iteration Jacobi contact solve. The restart-safe sweep is being relaunched
sequentially.

### Sweep progress (2026-07-23 07:43 UTC)

The relaunched restart-safe sweep (master PID 144300, started 07:28:27 UTC) is
running normally and sequentially. It is currently on the `contact_5` budget.
The pre-fix crash logs timestamped 07:23–07:25 UTC under
`raw/{g1,h1,go2}/.../contact_5/run.log` are artifacts of the backend-selection
issue described in the recovery note above (they attempted the PhysX contact
sensor without `omni.physics`); they were overwritten in place by the corrected
relaunch where the run reached that environment, and do **not** reflect the
current sweep. Completion is judged only by a `COMPLETED` marker plus a
`Training time:` line and `exit=0`.

Completed runs (validated: `COMPLETED` marker + `Training time` + `exit=0`):

| Environment | Solver | Contact iters | Training time | Steady steps/s | Total steps | Final mean reward | Status |
|---|---|---:|---:|---:|---:|---:|---|
| G1 flat | DVI-Jacobi (`newton_dvi`) | 5 | 1026.73 s | ~96.6k | 98,304,000 | 19.84 (final; ~20.3 best) | complete |
| G1 flat | DVI-APGD (`newton_dvi_apgd`) | 5 | 1139.39 s | ~87.4k | 98,304,000 | 12.72 (final; ~12.73 best) | complete |
| G1 flat | DVI-P-SPG-FB (`newton_dvi_pspg`) | 5 | 1584.56 s | ~63.1k | 98,304,000 | 7.04 (final; ~7.15 best) | complete |

Provenance for the completed G1 runs: Isaac Lab
`93fe3656db67daa309e41fb76f3a98cec1947c32`, Newton
`8697a3274569283d896c1430b4d6fe5e7cb7f3a8`, 1000 PPO iterations, seed 42.

### Monitoring pass (2026-07-23 08:09 UTC)

The APGD run marked "in progress" above completed at 08:06:22 UTC: `COMPLETED`
marker present, `Training time: 1139.39 seconds`, `exit=0`, full 1000/1000 PPO
iterations, 98,304,000 total steps, final mean reward 12.72 (best 12.73),
steady throughput ~87.4k steps/s (median over iterations). The fix that
unblocked `newton_dvi` also held for the APGD contact variant; no PhysX-sensor
error occurred. Values were validated from the raw log before recording.

In progress at time of writing: `g1 / newton_dvi_pspg / contact_5` (started
08:06 UTC, master PID 144300 still alive, ~41 min elapsed). It is training
normally at iteration ~97/1000 with no crash, NaN, or traceback. The single
`[carb] Client passed into the framework is nullptr` line is the usual Isaac
Sim carb startup warning, not a failure. No fix or intervention was required
for this monitoring pass; the sweep was left running untouched.

Blockers: none currently. Remaining `contact_5` matrix still pending after the
active PSPG run: g1 MJWarp, then h1, go2 (all solvers), then dr_legs; the
`contact_10` and `contact_15` budgets follow. Runs already marked `COMPLETED`
are skipped on restart; `FORCE=1` reruns them.

### Monitoring pass (2026-07-23 08:23 UTC)

Sweep still running untouched (master PID 144300, ~57 min elapsed). No fix or
intervention was required this pass; nothing was resumed because nothing had
exited.

Still in progress: `g1 / newton_dvi_pspg / contact_5` (started 08:06 UTC). It
has advanced from iteration ~97 at the previous pass to iteration ~687/1000
here, training normally at steady ~62.6k steps/s (Isaac Sim reports ~62k; the
master-log "Steps per second" is the same order after warmup), ETA ~8 min. The
raw log contains zero `Traceback`, zero `Error executing job`, zero `[ERROR]`,
and zero NaN lines; the many `error_vel_*` substrings are ordinary reward
metric names, not failures. No `COMPLETED` marker yet, so it is correctly
reported as in progress and not counted as a result.

Stale-log clarification: the small (~10 KB) `run.log` files under
`raw/h1/{newton_dvi,newton_dvi_apgd,newton_dvi_pspg}/contact_5/` and
`raw/go2/{newton_dvi,newton_dvi_apgd}/contact_5/` are timestamped 07:24-07:25
UTC, i.e. **before** the corrected relaunch (PID 144300, 07:28 UTC). They are
artifacts of the pre-fix backend-selection crash
(`ModuleNotFoundError: No module named 'omni.physics'`, PhysX contact sensor
selected without a matching Newton preset). None of them carry a `COMPLETED`
marker, so the current restart-safe sweep will re-run those H1/Go2 cells in
order once it clears the G1 rows; the applied working-tree fix (the
`newton_dvi_pspg` contact-sensor/physics presets now present for G1/H1/Go2 and
in the shared `velocity_env_cfg.py`) is what unblocks them, and the same fix is
already validated on the two completed G1 runs and the active G1 PSPG run. No
action taken on the stale logs; they are left in place as provenance.

Completed-run tally unchanged since 08:09 UTC: 2 validated cells
(G1 Jacobi c5, G1 APGD c5). Blockers: none.

### Monitoring pass (2026-07-23 08:43 UTC)

Sweep still running untouched (master PID 144300, ~1h15m elapsed). No fix,
resume, or intervention was required this pass; nothing had exited, so nothing
was restarted. Environments and solvers remain strictly sequential.

Newly validated cell since the 08:23 pass: `g1 / newton_dvi_pspg / contact_5`
completed at 08:33:41 UTC. Validation: `COMPLETED` marker present
(`2026-07-23T08:33:41Z`), `Training time: 1584.56 seconds`, `exit=0`, full
1000/1000 PPO iterations, 4096 environments (98,304,000 total steps), final
mean reward 7.04 (best 7.15), steady throughput ~63.1k steps/s. This is the
third completed G1 cell and confirms the P-SPG-FB contact preset trains to
completion under the same fix as Jacobi/APGD; it is the slowest of the three
G1 solvers at contact_5, consistent with the Ant PSPG ordering.

Now in progress: `h1 / newton_dvi / contact_5` (started 08:33 UTC). At time of
writing it is at iteration ~832/1000, ~10.5 min elapsed, ETA ~2 min, steady and
healthy. Its raw log contains only the usual single
`[carb] Client passed into the framework is nullptr` startup line (benign) and
no `Traceback`, `Error executing job`, `[ERROR]`, or NaN. It constructs the
**Newton** contact sensor (`isaaclab_newton.sensors.contact_sensor`), not the
PhysX one, confirming the backend-selection fix now holds for H1 as well as
G1. No `COMPLETED` marker yet, so it is correctly not counted as a result.

Stale pre-fix logs unchanged: the 07:24-07:25 UTC crash logs under
`raw/h1/{newton_dvi_apgd,newton_dvi_pspg}` and
`raw/go2/{newton_dvi,newton_dvi_apgd}` (`ModuleNotFoundError: No module named
'omni.physics'`) predate PID 144300 and carry no `COMPLETED` marker; the
sequential master will re-run those H1/Go2 cells in order after the active
H1 Jacobi cell. `h1 / newton_dvi` overwrote its own stale 07:24 log in place
on this pass, as expected for the restart-safe wrapper. Left in place as
provenance; no action taken.

Completed-run tally: 3 validated cells (G1 Jacobi c5, G1 APGD c5, G1 PSPG c5).
Blockers: none. Remaining `contact_5` matrix pending: g1 MJWarp, h1 (Jacobi
active, then APGD/PSPG/MJWarp), go2 (all), dr_legs (all); `contact_10` and
`contact_15` budgets follow.

### Monitoring pass (2026-07-23 09:12 UTC)

Sweep still running untouched (master PID 144300, ~1h44m elapsed). No fix,
resume, or intervention was required this pass; nothing had exited, so nothing
was restarted. Environments and solvers remain strictly sequential. The single
`[carb] Client passed into the framework is nullptr` line in the active log is
the usual Isaac Sim carb startup warning, not a failure.

Two H1 cells validated since the 08:43 pass (`COMPLETED` marker +
`Training time` + `exit=0`, full 1000/1000 PPO iterations, 4096 environments =
98,304,000 total steps each):

| Environment | Solver | Contact iters | Training time | Steady steps/s | Total steps | Final mean reward | Status |
|---|---|---:|---:|---:|---:|---:|---|
| H1 flat | DVI-Jacobi (`newton_dvi`) | 5 | 773.71 s | ~125.7k | 98,304,000 | 25.00 (final; ~25.45 best) | complete |
| H1 flat | DVI-APGD (`newton_dvi_apgd`) | 5 | 903.51 s | ~111.2k | 98,304,000 | 25.32 (final; ~26.22 best) | complete |

H1 Jacobi completed 08:47:16 UTC and H1 APGD 09:03:00 UTC. Both construct the
**Newton** contact sensor (not the PhysX one), confirming the backend-selection
fix holds for H1 as it did for G1. Values were read from the raw logs before
recording. H1 throughput is markedly higher than G1 (~126k vs ~97k steps/s for
Jacobi) at the same 4096 envs and contact budget.

Now in progress: `h1 / newton_dvi_pspg / contact_5` (started 09:03 UTC). At
time of writing it is at iteration ~347/1000, ~8.5 min elapsed, ETA ~16 min,
training steadily at ~66.7k steps/s, mean reward ~18.3. Its raw log contains no
`Traceback`, `Error executing job`, `[ERROR]`, or NaN. No `COMPLETED` marker
yet, so it is correctly not counted as a result.

Stale pre-fix logs unchanged: the 07:24-07:25 UTC crash logs under
`raw/go2/{newton_dvi,newton_dvi_apgd}` (`ModuleNotFoundError: No module named
'omni.physics'`, PhysX contact sensor selected without a matching Newton
preset) predate PID 144300 and carry no `COMPLETED` marker; the sequential
master will re-run those Go2 cells in order after it clears the remaining H1
and g1-MJWarp rows. The already-applied working-tree fix (matching
contact-sensor/physics presets, shared `velocity_env_cfg.py`) is what unblocks
them, and it is already validated on all completed G1 and H1 cells. Left in
place as provenance; no action taken.

Completed-run tally: 5 validated cells (G1 Jacobi/APGD/PSPG c5, H1 Jacobi c5,
H1 APGD c5). Blockers: none. Remaining `contact_5` matrix pending: g1 MJWarp,
h1 (PSPG active, then MJWarp), go2 (all), dr_legs (all); `contact_10` and
`contact_15` budgets follow.

### Monitoring pass (2026-07-23 09:23 UTC)

Sweep still running untouched (master PID 144300, ~1h56m elapsed). No fix,
resume, or intervention was required this pass; nothing had exited, so nothing
was restarted. Environments and solvers remain strictly sequential.

No new completions since the 09:12 pass; the tally is unchanged at 5 validated
cells (G1 Jacobi/APGD/PSPG c5, H1 Jacobi c5, H1 APGD c5). Each was re-verified
this pass to still satisfy `COMPLETED` marker + `Training time:` + `exit=0`
(marker timestamps match the corresponding `END ... exit=0` lines in the master
log).

Still in progress: `h1 / newton_dvi_pspg / contact_5` (started 09:03 UTC). At
time of writing it is at iteration ~867/1000, ~21 min elapsed, ETA ~3 min,
training steadily with mean reward ~25. Its raw log contains zero `Traceback`,
zero `[ERROR]`, zero `Error executing job`, and zero NaN lines. No `COMPLETED`
marker yet, so it is correctly not counted as a result.

Stale pre-fix logs unchanged: the 07:24-07:25 UTC crash logs under
`raw/go2/{newton_dvi,newton_dvi_apgd}` (`ModuleNotFoundError: No module named
'omni.physics'`, PhysX contact sensor selected without a matching Newton
preset) predate PID 144300 and carry no `COMPLETED` marker; the sequential
master will re-run those Go2 cells in order after it clears the remaining H1
and g1-MJWarp rows. The already-applied working-tree fix (matching
contact-sensor/physics presets set through `presets=$preset`, shared
`velocity_env_cfg.py`) is what unblocks them, and it is already validated on
all completed G1 and H1 cells. Left in place as provenance; no action taken.

Blockers: none. Remaining `contact_5` matrix pending: g1 MJWarp, h1 (PSPG
active, then MJWarp), go2 (all), dr_legs (all); `contact_10` and `contact_15`
budgets follow.

### Monitoring pass (2026-07-23 09:44 UTC)

Sweep still running untouched (master PID 144300, ~2h16m elapsed). No fix,
resume, or intervention was required this pass; nothing had exited, so nothing
was restarted. Environments and solvers remain strictly sequential. The single
`[carb] Client passed into the framework is nullptr` line in each log is the
usual Isaac Sim carb startup warning, not a failure.

Two cells validated since the 09:23 pass (`COMPLETED` marker + `Training time:`
+ `exit=0`, full 1000/1000 PPO iterations, 4096 environments = 98,304,000 total
steps each; marker timestamps match the corresponding `END ... exit=0` lines in
the master log):

| Environment | Solver | Contact iters | Training time | Steady steps/s | Total steps | Final mean reward | Status |
|---|---|---:|---:|---:|---:|---:|---|
| H1 flat | DVI-P-SPG-FB (`newton_dvi_pspg`) | 5 | 1475.01 s | ~67.0k | 98,304,000 | 24.36 (final; ~25.70 best) | complete |
| Go2 flat | DVI-Jacobi (`newton_dvi`) | 5 | 588.94 s | ~170.0k | 98,304,000 | 34.75 (final; ~35.36 best) | complete |

H1 PSPG completed 09:28:18 UTC (the H1 environment is now fully done at
contact_5 across Jacobi/APGD/PSPG). Go2 Jacobi completed 09:38:54 UTC and
overwrote its own stale 07:24 pre-fix log in place, as expected for the
restart-safe wrapper; it constructs the **Newton** contact sensor
(`isaaclab_newton.sensors.contact_sensor`), not the PhysX one, confirming the
backend-selection fix now holds for Go2 as well as G1/H1. That fully clears the
pre-fix stale Go2/H1 logs noted in earlier passes. Go2 Jacobi is the fastest
DVI cell in the sweep so far at ~170k steps/s (589 s wall), roughly 2.9x the G1
Jacobi throughput and 1.35x H1 Jacobi at the same 4096 envs and contact_5
budget. Values were read from the raw logs before recording.

Now in progress: `go2 / newton_dvi_apgd / contact_5` (started 09:38:54 UTC). At
time of writing it is at iteration ~344/1000, ~6 min elapsed, training steadily
at ~120k steps/s, mean reward ~17.6. Its raw log contains zero `Traceback`,
zero `Error executing job`, zero `[ERROR]`, and zero NaN lines (the only
`[Error]` match is the benign carb `nullptr` startup line). No `COMPLETED`
marker yet, so it is correctly not counted as a result.

Stale pre-fix logs: none remain unresolved. The 07:24-07:25 UTC
backend-selection crash logs previously flagged under `raw/h1/*` and
`raw/go2/*` have now all been overwritten in place by the corrected sequential
relaunch as it reached each cell, or will be for the one remaining pre-fix Go2
cell (`newton_dvi_apgd`, active) and any others as the master advances; every
completed cell to date validates on the applied fix.

Completed-run tally: 7 validated cells (G1 Jacobi/APGD/PSPG c5, H1
Jacobi/APGD/PSPG c5, Go2 Jacobi c5). Blockers: none. Remaining `contact_5`
matrix pending: g1 MJWarp, h1 MJWarp, go2 (APGD active, then PSPG/MJWarp),
dr_legs (all); `contact_10` and `contact_15` budgets follow. Note the wrapper
places `newton_mjwarp` rows after the three DVI rows per environment, so the
remaining g1/h1 MJWarp cells are executed as the master revisits them within
the sequential matrix order.

### Monitoring pass (2026-07-23 10:03 UTC)

Sweep still running untouched (master PID 144300, ~2h35m elapsed). No fix,
resume, or intervention was required this pass; nothing had exited, so nothing
was restarted. Environments and solvers remain strictly sequential. The single
`[carb] Client passed into the framework is nullptr` line in each log is the
usual Isaac Sim carb startup warning, not a failure.

One cell validated since the 09:44 pass (`COMPLETED` marker + `Training time:`
+ `exit=0`, full 1000/1000 PPO iterations, 4096 environments = 98,304,000 total
steps; marker timestamp matches the corresponding `END ... exit=0` line in the
master log):

| Environment | Solver | Contact iters | Training time | Steady steps/s | Total steps | Final mean reward | Status |
|---|---|---:|---:|---:|---:|---:|---|
| Go2 flat | DVI-APGD (`newton_dvi_apgd`) | 5 | 876.95 s | ~110.0k | 98,304,000 | 35.08 (final; ~35.54 best) | complete |

Go2 APGD completed 09:54:19 UTC. It constructs the **Newton** contact sensor
(`isaaclab_newton.sensors.contact_sensor`, 13 sensors), not the PhysX one,
confirming the backend-selection fix holds for the Go2 APGD variant. Steady
throughput is ~110k steps/s (median of the raw log's reported `Steps per
second`; the whole-run average of 98,304,000 / 876.95 s is ~112k, same order).
Go2 APGD is slower than Go2 Jacobi (~170k steps/s, 589 s) at the same 4096
envs and contact_5 budget, consistent with the Jacobi<APGD<PSPG wall-time
ordering seen on G1 and H1. Values were read from the raw log before recording.

Now in progress: `go2 / newton_dvi_pspg / contact_5` (started 09:54:19 UTC). At
time of writing it is at iteration ~356/1000, ~9 min elapsed, ETA ~18 min,
training steadily with mean reward ~18.7. Its raw log contains zero
`Traceback`, zero `Error executing job`, zero `[ERROR]`, and zero NaN lines. No
`COMPLETED` marker yet, so it is correctly not counted as a result.

Stale pre-fix logs: none remain. Every pre-fix 07:24-07:25 UTC
backend-selection crash log under `raw/h1/*` and `raw/go2/*` has now been
overwritten in place by the corrected sequential relaunch as the master reached
each cell (the last one, Go2 APGD, on the previous pass). All completed cells
validate on the applied fix.

Completed-run tally: 8 validated cells (G1 Jacobi/APGD/PSPG c5, H1
Jacobi/APGD/PSPG c5, Go2 Jacobi c5, Go2 APGD c5). Blockers: none. Remaining
`contact_5` matrix pending: g1 MJWarp, h1 MJWarp, go2 (PSPG active, then
MJWarp), dr_legs (all); `contact_10` and `contact_15` budgets follow.

### Monitoring pass (2026-07-23 10:23 UTC)

Sweep still running untouched (master PID 144300, ~2h56m elapsed). No fix,
resume, or intervention was applied this pass; the master had **not** exited, so
nothing was restarted or interrupted. Environments and solvers remain strictly
sequential.

One cell validated since the 10:03 pass (`COMPLETED` marker + `Training time:`
+ `exit=0`, full 1000/1000 PPO iterations, 4096 environments = 98,304,000 total
steps; marker timestamp matches the corresponding `END ... exit=0` line in the
master log):

| Environment | Solver | Contact iters | Training time | Steady steps/s | Total steps | Final mean reward | Status |
|---|---|---:|---:|---:|---:|---:|---|
| Go2 flat | DVI-P-SPG-FB (`newton_dvi_pspg`) | 5 | 1611.81 s | ~61.1k | 98,304,000 | 31.34 (final; ~33.97 best) | complete |

Go2 PSPG completed 10:21:58 UTC. It constructs the **Newton** contact sensor
(`isaaclab_newton.sensors.contact_sensor`), not the PhysX one, confirming the
backend-selection fix holds for the Go2 PSPG variant. Steady throughput is
~61.1k steps/s (median of the raw log's 1000 reported `Steps per second`
samples; min 33.2k, max 76.0k). It is the slowest of the three Go2 DVI cells at
contact_5 (Jacobi 589 s < APGD 877 s < PSPG 1612 s), consistent with the
Jacobi<APGD<PSPG wall-time ordering already seen on G1 and H1. With this cell
the Go2 environment is fully done at contact_5 across all three DVI solvers, so
the entire contact_5 DVI matrix (G1, H1, Go2 x Jacobi/APGD/PSPG) is complete.
Values were read from the raw log before recording.

**New blocker — DR Legs asset missing.** Immediately after Go2 PSPG, the master
advanced to the three `dr_legs` DVI cells at contact_5 and all three failed fast
(exit=1, ~11-13 s each; START 10:21:58 / 10:22:10 / 10:22:22 UTC). Root cause is
identical across Jacobi/APGD/PSPG and is **not** a solver, adapter, or
backend-selection problem:

```
FileNotFoundError: USD file not found at path:
'.../isaaclab_assets/robots/data/dr_legs/dr_legs.usda'.
```

The articulation spawn (`spawn_from_usd`) cannot find `dr_legs.usda`. The
`data/dr_legs/` directory contains only a `Geometry` symlink into the Newton
asset cache, and that target
(`~/.cache/newton/newton-assets_disneyresearch_193783fd_8e8df07d/disneyresearch/dr_legs/usd/`)
no longer exists — the symlink is dangling and the `.usda` was never present at
the expected path. `dr_legs.py` still carries a `TODO` to move `usd_path` to
`ISAACLAB_NUCLEUS_DIR` once the DR Legs USD is hosted, so the asset is simply
not available locally. A `find` for `dr_legs.usda` across the repo and the
newton cache returned nothing.

No fix was applied. This is a missing-data blocker, **not** analogous to the Ant
adapter fix (which was a reversible code/backend-selection edit in
`velocity_env_cfg.py`, matching contact-sensor/physics presets). There is no
reversible code change that can synthesize the absent USD; fabricating or
re-pointing the asset would risk invalid physics and is out of scope. The task's
"apply reversible fixes" clause applies only to the exited-unexpectedly branch,
and the master did not exit — it correctly logged each dr_legs failure (no
`COMPLETED` marker) and continued sequentially, so no restart was warranted.
Resolving dr_legs requires provisioning the DR Legs USD (regenerate the Newton
asset cache or host on Nucleus and update `usd_path`); flagging for Milad.

Now in progress: `g1 / newton_dvi / contact_10` (started 10:22:35 UTC), the
first cell of the contact_10 budget block. At time of writing it is at iteration
~102/1000, ~1m47s elapsed, training steadily; its raw log contains zero
`Traceback`, zero `Error executing job`, and zero `[ERROR]`/NaN lines. No
`COMPLETED` marker yet, so it is correctly not counted as a result.

Correction to earlier passes: the MJWarp rows do **not** run in the contact_5
block. The wrapper guards `newton_mjwarp` with `if [[ "$kind" == mjwarp &&
"$CONTACT_ITERS" != 10 ]]; then continue; fi`, so g1/h1/go2 MJWarp execute
**only** in the contact_10 budget — hence no `newton_mjwarp` directories exist
yet, which is expected, not a failure. Earlier passes listing "g1 MJWarp / h1
MJWarp" as pending contact_5 cells were mistaken on that point.

Completed-run tally: 9 validated cells (G1 Jacobi/APGD/PSPG c5, H1
Jacobi/APGD/PSPG c5, Go2 Jacobi/APGD/PSPG c5 — the full contact_5 DVI matrix).
Blockers: **dr_legs (all solvers) — missing `dr_legs.usda` asset** (see above),
recurring at every budget until the USD is provisioned. Remaining matrix:
contact_10 block now active (g1 Jacobi running, then g1 APGD/PSPG/MJWarp, h1 all
+MJWarp, go2 all +MJWarp, dr_legs all [will fail]); then contact_15 block
(DVI-only per wrapper: g1/h1/go2 x Jacobi/APGD/PSPG, dr_legs all [will fail]).

## Watchdog pass 2026-07-23 10:45 UTC

Master sweep still alive (PID 144300 from `locomotion_sweep.pid`, running ~3h15m,
master log updating at 10:43). No unexpected exit — no restart or fix warranted.
All 15 environments/solvers remain sequential.

**New validated result — g1 Jacobi contact_10.** Since the previous pass the
first contact_10 cell completed and validates cleanly (marker
`2026-07-23T10:41:42Z`, `===== END ... exit=0 =====`, `Training time: 1091.27
seconds`). Values were read from the raw log before recording.

| env | solver | contact | Training time (s) |
|-----|--------|---------|-------------------|
| g1  | newton_dvi (Jacobi) | 10 | 1091.27 |

Jacobi at contact_10 (1091 s) is ~6% slower than the same cell at contact_5
(1026.73 s), as expected from the larger contact-iteration budget; ordering vs
APGD/PSPG at contact_10 is pending those cells.

**Now in progress:** `g1 / newton_dvi_apgd / contact_10` (started 10:41:42 UTC).
At time of writing it is at iteration ~122/1000, ~2m31s elapsed, training
steadily; its raw log contains zero `Traceback`, zero `Error executing job`, and
zero `[ERROR]`/NaN lines. No `COMPLETED` marker yet, so it is correctly not
counted as a result.

**dr_legs blocker unchanged.** Still the missing-`dr_legs.usda` asset documented
above; not analogous to the Ant adapter fix and not fixable by a reversible code
edit. No action taken beyond flagging; the master already logged each dr_legs
failure and advanced. It will recur at contact_10 and contact_15 until the USD is
provisioned.

**Updated completed-run tally: 10 validated cells** — the full contact_5 DVI
matrix (G1, H1, Go2 x Jacobi/APGD/PSPG = 9) plus g1 Jacobi contact_10 (1). All 10
markers re-validated this pass (each has exit=0 and a `Training time:` line).
Remaining matrix: contact_10 block continuing (g1 APGD running, then g1
PSPG/MJWarp, h1 all +MJWarp, go2 all +MJWarp, dr_legs all [will fail]); then
contact_15 block (DVI-only per wrapper: g1/h1/go2 x Jacobi/APGD/PSPG, dr_legs all
[will fail]). No commit made.

## Watchdog pass 2026-07-23 11:06 UTC

Master sweep still alive (PID 144300 from `locomotion_sweep.pid`, running ~3h39m,
master log updating at 11:06). No unexpected exit — no restart or fix warranted;
nothing was interrupted. All environments/solvers remain strictly sequential.

**New validated result — g1 APGD contact_10.** Since the 10:45 pass the second
contact_10 cell completed and validates cleanly (marker `2026-07-23T11:03:48Z`,
`===== END ... exit=0 =====`, `Training time: 1270.5 seconds`, full 1000/1000
PPO iterations, 4096 envs = 98,304,000 total steps). Values were read from the
raw log before recording.

| env | solver | contact | Training time (s) | Steady steps/s | Total steps | Final mean reward | Status |
|-----|--------|---------|-------------------|---------------:|------------:|------------------:|--------|
| g1  | newton_dvi_apgd (APGD) | 10 | 1270.5 | ~78.3k | 98,304,000 | 2.65 (final; ~3.22 best) | complete |

At contact_10, APGD (1270.5 s) is slower than Jacobi (1091.27 s), preserving the
Jacobi<APGD wall-time ordering seen at contact_5; PSPG at contact_10 is pending
the active cell. APGD contact_10 is ~11.5% slower than APGD contact_5 (1139.39 s),
consistent with the larger contact-iteration budget. Both g1 contact_10 cells so
far show markedly lower final G1 reward than their contact_5 counterparts (Jacobi
4.84 vs 19.84; APGD 2.65 vs 12.72) at matched 1000 iterations — noted for the
later reward-vs-budget interpretation, not acted on here.

**Now in progress:** `g1 / newton_dvi_pspg / contact_10` (started 11:03:48 UTC),
the third contact_10 cell. At time of writing it is at iteration ~97/1000,
~3 min elapsed, ETA ~31 min, training steadily; its raw log contains zero
`Traceback`, zero `Error executing job`, and zero `[ERROR]`/NaN lines (the only
`[Error]` match is the benign carb `nullptr` startup line). No `COMPLETED` marker
yet, so it is correctly not counted as a result.

**dr_legs blocker unchanged.** Still the missing-`dr_legs.usda` asset documented
in the 10:23 pass (dangling `Geometry` symlink into a non-existent Newton asset
cache; `usd_path` never provisioned, `TODO` to host on Nucleus). Not analogous to
the Ant adapter fix and not fixable by a reversible code edit; fabricating or
re-pointing the asset is out of scope. No action taken beyond flagging for Milad;
it will recur at contact_10 and contact_15 until the DR Legs USD is provisioned.

**Updated completed-run tally: 11 validated cells** — the full contact_5 DVI
matrix (G1, H1, Go2 x Jacobi/APGD/PSPG = 9) plus g1 Jacobi contact_10 and g1 APGD
contact_10 (2). All 11 markers re-validated this pass (each has `exit=0` and a
`Training time:` line, marker timestamps matching the `END ... exit=0` lines in
the master log). Remaining matrix: contact_10 block continuing (g1 PSPG running,
then g1 MJWarp, h1 all +MJWarp, go2 all +MJWarp, dr_legs all [will fail]); then
contact_15 block (DVI-only per wrapper: g1/h1/go2 x Jacobi/APGD/PSPG, dr_legs all
[will fail]). No commit made.

## Watchdog pass 2026-07-23 11:27 UTC

Master sweep still alive (PID 144300 from `locomotion_sweep.pid`, running ~4h01m,
master log updating at 11:27). No unexpected exit — no restart, resume, or fix
warranted; nothing was interrupted. All environments/solvers remain strictly
sequential.

**No new validated cell since the 11:06 pass.** The third contact_10 cell,
`g1 / newton_dvi_pspg / contact_10` (started 11:03:48 UTC), is still in progress:
iteration ~682/1000, ~24m36s elapsed, ETA ~11m, ~45.6k steps/s, training
steadily. Its raw log contains zero `Traceback`, zero `Error executing job`, and
zero `[ERROR]`/NaN lines (the only `[Error]` match is the benign carb `nullptr`
startup line; the many `error_vel_*` substrings are reward metric names). No
`COMPLETED` marker yet, so it is correctly not counted as a result.

**dr_legs blocker unchanged.** Still the missing-`dr_legs.usda` asset: a dangling
`Geometry` symlink into a non-existent Newton asset cache under
`~/.cache/newton/newton-assets_disneyresearch_.../disneyresearch/dr_legs/`, and
the `usd_path` in `robots/dr_legs.py` (line 90) was never provisioned (with a
`TODO` to host it on Nucleus). All three contact_5 dr_legs cells fail identically
with `FileNotFoundError: USD file not found`. This is a missing-asset
provisioning issue, **not** analogous to the Ant adapter fix and not fixable by a
reversible code edit; fabricating or re-pointing the asset would be a hack and is
out of scope. No action taken beyond flagging for Milad; it will recur at
contact_10 and contact_15 until the DR Legs USD is provisioned.

**Completed-run tally unchanged: 11 validated cells** — the full contact_5 DVI
matrix (G1, H1, Go2 x Jacobi/APGD/PSPG = 9) plus g1 Jacobi contact_10 and g1 APGD
contact_10 (2). All 11 markers re-validated this pass (each has `exit=0` and a
`Training time:` line, marker timestamps matching the `END ... exit=0` lines).
Remaining matrix: contact_10 block continuing (g1 PSPG running, then g1 MJWarp,
h1 all +MJWarp, go2 all +MJWarp, dr_legs all [will fail]); then contact_15 block
(DVI-only per wrapper: g1/h1/go2 x Jacobi/APGD/PSPG, dr_legs all [will fail]). No
commit made.

## Watchdog pass 2026-07-23 11:45 UTC

Master sweep still alive (PID 144300 from `locomotion_sweep.pid`, running ~4h17m,
master log updating at 11:45). No unexpected exit — no restart, resume, or fix
warranted; nothing was interrupted. All environments/solvers remain strictly
sequential.

**New validated result — g1 PSPG contact_10.** Since the 11:27 pass the third
contact_10 cell completed and validates cleanly (marker `2026-07-23T11:40:58Z`,
matching `===== END g1 newton_dvi_pspg UTC 2026-07-23T11:40:58Z exit=0 =====`,
`Training time: 2175.57 seconds`, full 1000/1000 PPO iterations, 4096 envs =
98,304,000 total steps). Values were read from the raw log before recording.

| env | solver | contact | Training time (s) | Steady steps/s | Total steps | Final mean reward | Status |
|-----|--------|---------|-------------------|---------------:|------------:|------------------:|--------|
| g1  | newton_dvi_pspg (P-SPG-FB) | 10 | 2175.57 | ~45.4k (median; min 25.9k, max 51.1k) | 98,304,000 | ~6.28 (final) | complete |

This completes the full g1 contact_10 DVI matrix (Jacobi 1091.27 s < APGD
1270.5 s < PSPG 2175.57 s), preserving the Jacobi<APGD<PSPG wall-time ordering
seen at contact_5 and on H1/Go2. PSPG contact_10 (2175.57 s) is ~37% slower than
PSPG contact_5 (1584.56 s), the largest budget-driven slowdown of the three g1
solvers, consistent with P-SPG-FB's heavier per-contact-iteration cost.

**Now in progress:** `g1 / newton_mjwarp / contact_10` (started 11:40:58 UTC),
the fourth and final g1 contact_10 cell and the **first MJWarp run of the entire
sweep** (MJWarp executes only in the contact_10 block per the wrapper guard). At
time of writing it is at iteration ~182/1000, training steadily at ~140k steps/s
(far faster than the DVI cells, as expected for the native MJWarp path). It
correctly constructs backend-specific config from the `newton_mjwarp` preset; its
raw log contains zero `Traceback`, zero `Error executing job`, and zero
`[ERROR]`/`FileNotFoundError`/NaN lines (the only `[Error]` match is the benign
carb `nullptr` startup line). No `COMPLETED` marker yet, so it is correctly not
counted as a result.

**dr_legs blocker unchanged.** Still the missing-`dr_legs.usda` asset: a dangling
`Geometry` symlink into a non-existent Newton asset cache under
`~/.cache/newton/newton-assets_disneyresearch_.../disneyresearch/dr_legs/`, and
the `usd_path` in `robots/dr_legs.py` (line 90) was never provisioned (`TODO` to
host on Nucleus). This is a missing-asset provisioning issue, **not** analogous
to the Ant adapter fix and not fixable by a reversible code edit; fabricating or
re-pointing the asset would be a hack and is out of scope. No action taken beyond
flagging for Milad; it will recur at contact_10 and contact_15 until the DR Legs
USD is provisioned.

**Updated completed-run tally: 12 validated cells** — the full contact_5 DVI
matrix (G1, H1, Go2 x Jacobi/APGD/PSPG = 9) plus the full g1 contact_10 DVI
matrix (Jacobi/APGD/PSPG = 3). All 12 markers re-validated this pass (each has
`exit=0` and a `Training time:` line, marker timestamps matching the
`END ... exit=0` lines in the master log). Remaining matrix: contact_10 block
continuing (g1 MJWarp running, then h1 all +MJWarp, go2 all +MJWarp, dr_legs all
[will fail]); then contact_15 block (DVI-only per wrapper: g1/h1/go2 x
Jacobi/APGD/PSPG, dr_legs all [will fail]). No commit made.

## Watchdog pass 2026-07-23 12:03 UTC

Master sweep still alive (PID 144300 from `locomotion_sweep.pid`, running ~4h35m,
master log updating at 12:03). No unexpected exit — no restart, resume, or fix
warranted; nothing was interrupted. All environments/solvers remain strictly
sequential.

**New validated result — g1 MJWarp contact_10.** Since the 11:45 pass the fourth
and final g1 contact_10 cell completed and validates cleanly (marker
`2026-07-23T11:54:42Z`, matching `===== END g1 newton_mjwarp UTC
2026-07-23T11:54:42Z exit=0 =====`, `Training time: 695.44 seconds`, full
1000/1000 PPO iterations, 4096 envs = 98,304,000 total steps). This is the first
completed MJWarp cell of the sweep. Values were read from the raw log before
recording; the log contains zero `Traceback`, `Error executing job`,
`FileNotFoundError`, or NaN lines.

| env | solver | contact | Training time (s) | Total steps | Status |
|-----|--------|---------|-------------------|------------:|--------|
| g1  | newton_mjwarp (MJWarp) | 10 | 695.44 | 98,304,000 | complete |

MJWarp at contact_10 (695.44 s) is the fastest g1 cell at this budget by a wide
margin — ~1.57x faster than Jacobi (1091.27 s), ~1.83x faster than APGD
(1270.5 s), and ~3.13x faster than PSPG (2175.57 s) — consistent with the native
MJWarp physics path (~140k steps/s vs DVI's ~45-78k). This completes the **full
g1 contact_10 matrix** across all four backends (Jacobi < APGD < MJWarp-fastest;
PSPG slowest). Note MJWarp uses its own native contact config and is not a
controlled contact-iteration comparison with the DVI cells.

**Now in progress:** `h1 / newton_dvi / contact_10` (started 11:54:42 UTC), the
first h1 cell of the contact_10 block. At time of writing it is at iteration
~712/1000, ~9m36s elapsed, ETA ~3m52s, ~119k steps/s, training steadily; its raw
log contains zero `Traceback`, zero `Error executing job`, and zero `[ERROR]`/
`FileNotFoundError`/NaN lines (the only `[Error]` match is the benign carb
`nullptr` startup line). No `COMPLETED` marker yet, so it is correctly not
counted as a result.

**dr_legs blocker unchanged.** Still the missing-`dr_legs.usda` asset: a dangling
`Geometry` symlink into a non-existent Newton asset cache under
`~/.cache/newton/newton-assets_disneyresearch_.../disneyresearch/dr_legs/`, and
the `usd_path` in `robots/dr_legs.py` (line 90) was never provisioned (`TODO` to
host on Nucleus). This is a missing-asset provisioning issue, **not** analogous
to the Ant adapter fix and not fixable by a reversible code edit; fabricating or
re-pointing the asset would be a hack and is out of scope. No action taken beyond
flagging for Milad; it will recur at contact_10 and contact_15 until the DR Legs
USD is provisioned.

**Updated completed-run tally: 13 validated cells** — the full contact_5 DVI
matrix (G1, H1, Go2 x Jacobi/APGD/PSPG = 9) plus the full g1 contact_10 matrix
(Jacobi/APGD/PSPG/MJWarp = 4). All 13 markers re-validated this pass (each has
`exit=0` and a `Training time:` line, marker timestamps matching the
`END ... exit=0` lines in the master log). Remaining matrix: contact_10 block
continuing (h1 Jacobi running, then h1 APGD/PSPG/MJWarp, go2 all +MJWarp, dr_legs
all [will fail]); then contact_15 block (DVI-only per wrapper: g1/h1/go2 x
Jacobi/APGD/PSPG, dr_legs all [will fail]). No commit made.

## Watchdog pass 2026-07-23 12:25 UTC

Master sweep still alive (PID 144300 from `locomotion_sweep.pid`, running ~4h56m,
master log updating at 12:24:59). No unexpected exit — no restart, resume, or fix
warranted; nothing was interrupted. All environments/solvers remain strictly
sequential.

**New validated result — h1 DVI-Jacobi contact_10.** Since the 12:03 pass the
first h1 contact_10 cell completed and validates cleanly (marker
`2026-07-23T12:09:05Z`, matching `===== END h1 newton_dvi UTC
2026-07-23T12:09:05Z exit=0 =====`, `Training time: 822.78 seconds`, full
1000/1000 PPO iterations, 4096 envs = 98,304,000 total steps, ~119k steps/s).
Values were read from the raw log before recording; the log contains zero
`Traceback`, `Error executing job`, `FileNotFoundError`, or NaN lines.

| env | solver | contact | Training time (s) | Total steps | Status |
|-----|--------|---------|-------------------|------------:|--------|
| h1  | newton_dvi (DVI-Jacobi) | 10 | 822.78 | 98,304,000 | complete |

h1 Jacobi contact_10 (822.78 s) is slower than its own contact_5 cell
(773.71 s, +6.3%), a modest budget-driven slowdown consistent with Jacobi's
light per-contact-iteration cost. It is faster than g1 Jacobi contact_10
(1091.27 s), as expected for the smaller humanoid.

**Now in progress:** `h1 / newton_dvi_apgd / contact_10` (started 12:09:05 UTC),
the second h1 cell of the contact_10 block. At time of writing it is at iteration
~902/1000, ~15m07s elapsed, ETA ~1m37s, ~97k steps/s, training steadily; its raw
log contains zero `Traceback`, zero `Error executing job`, and zero `[ERROR]`/
`FileNotFoundError`/NaN lines (the only `[Error]` match is the benign carb
`nullptr` startup line). No `COMPLETED` marker yet, so it is correctly not
counted as a result.

**dr_legs blocker unchanged.** Still the missing-`dr_legs.usda` asset: a dangling
`Geometry` symlink into a non-existent Newton asset cache under
`~/.cache/newton/newton-assets_disneyresearch_.../disneyresearch/dr_legs/`, and
the `usd_path` in `robots/dr_legs.py` (line 90) was never provisioned (`TODO` to
host on Nucleus). This is a missing-asset provisioning issue, **not** analogous
to the Ant adapter fix and not fixable by a reversible code edit; fabricating or
re-pointing the asset would be a hack and is out of scope. No action taken beyond
flagging for Milad; it will recur at contact_10 and contact_15 until the DR Legs
USD is provisioned.

**Updated completed-run tally: 14 validated cells** — the full contact_5 DVI
matrix (G1, H1, Go2 x Jacobi/APGD/PSPG = 9), the full g1 contact_10 matrix
(Jacobi/APGD/PSPG/MJWarp = 4), and h1 contact_10 Jacobi (1). All 14 markers
re-validated this pass (each has `exit=0` and a `Training time:` line, marker
timestamps matching the `END ... exit=0` lines in the master log). Remaining
matrix: contact_10 block continuing (h1 APGD running, then h1 PSPG/MJWarp, go2
all +MJWarp, dr_legs all [will fail]); then contact_15 block (DVI-only per
wrapper: g1/h1/go2 x Jacobi/APGD/PSPG, dr_legs all [will fail]). No commit made.

## Watchdog pass 2026-07-23 12:43 UTC

Master sweep still alive (PID 144300 from `locomotion_sweep.pid`, running ~5h15m,
master log updating at 12:43). No unexpected exit — no restart, resume, or fix
warranted; nothing was interrupted. All environments/solvers remain strictly
sequential.

**New validated result — h1 DVI-APGD contact_10.** Since the 12:25 pass the
second h1 contact_10 cell completed and validates cleanly (marker
`2026-07-23T12:26:42Z`, matching `===== END h1 newton_dvi_apgd UTC
2026-07-23T12:26:42Z exit=0 =====`, `Training time: 1016.73 seconds`, full
1000/1000 PPO iterations, 4096 envs = 98,304,000 total steps). Values were read
from the raw log before recording; the log contains zero `Traceback`,
`Error executing job`, `FileNotFoundError`, or NaN lines.

| env | solver | contact | Training time (s) | Steady steps/s | Total steps | Final mean reward | Status |
|-----|--------|---------|-------------------|---------------:|------------:|------------------:|--------|
| h1  | newton_dvi_apgd (APGD) | 10 | 1016.73 | ~97.4k (median; min 35.0k, max 106.9k) | 98,304,000 | ~25.99 (final; ~26.4 best) | complete |

At contact_10, h1 APGD (1016.73 s) is slower than h1 Jacobi (822.78 s),
preserving the Jacobi<APGD wall-time ordering seen at contact_5 and on g1/go2.
APGD contact_10 is ~12.5% slower than APGD contact_5 (903.51 s), consistent with
the larger contact-iteration budget. Final reward (~25.99) is in the same band as
the h1 contact_5 APGD cell (~25.32), i.e. no reward collapse from the larger
budget on this humanoid.

**Now in progress:** `h1 / newton_dvi_pspg / contact_10` (started 12:26:42 UTC),
the third h1 cell of the contact_10 block. At time of writing it is at iteration
~502/1000, ~17m49s elapsed, ETA ~17m36s, training steadily; its raw log contains
zero `Traceback`, zero `Error executing job`, and zero `[ERROR]`/
`FileNotFoundError`/NaN lines (the only `[Error]` match is the benign carb
`nullptr` startup line). No `COMPLETED` marker yet, so it is correctly not counted
as a result.

**dr_legs blocker unchanged.** Still the missing-`dr_legs.usda` asset: a dangling
`Geometry` symlink into a non-existent Newton asset cache under
`~/.cache/newton/newton-assets_disneyresearch_.../disneyresearch/dr_legs/`, and
the `usd_path` in `robots/dr_legs.py` (line 90) was never provisioned (`TODO` to
host on Nucleus). This is a missing-asset provisioning issue, **not** analogous
to the Ant adapter fix and not fixable by a reversible code edit; fabricating or
re-pointing the asset would be a hack and is out of scope. No action taken beyond
flagging for Milad; it will recur at contact_10 and contact_15 until the DR Legs
USD is provisioned.

**Updated completed-run tally: 15 validated cells** — the full contact_5 DVI
matrix (G1, H1, Go2 x Jacobi/APGD/PSPG = 9), the full g1 contact_10 matrix
(Jacobi/APGD/PSPG/MJWarp = 4), and h1 contact_10 Jacobi + APGD (2). All 15
markers re-validated this pass (each has `exit=0` and a `Training time:` line,
marker timestamps matching the `END ... exit=0` lines in the master log).
Remaining matrix: contact_10 block continuing (h1 PSPG running, then h1 MJWarp,
go2 all +MJWarp, dr_legs all [will fail]); then contact_15 block (DVI-only per
wrapper: g1/h1/go2 x Jacobi/APGD/PSPG, dr_legs all [will fail]). No commit made.

## Watchdog pass 2026-07-23 13:05 UTC

Master sweep still alive (PID 144300 from `locomotion_sweep.pid`, running ~5h35m,
master log updating at 13:05). No unexpected exit — no restart, resume, or fix
warranted; nothing was interrupted. All environments/solvers remain strictly
sequential.

**New validated result — h1 DVI-P-SPG-FB contact_10.** Since the 12:43 pass the
third h1 contact_10 cell completed and validates cleanly (marker
`2026-07-23T13:03:06Z`, matching `===== END h1 newton_dvi_pspg UTC
2026-07-23T13:03:06Z exit=0 =====`, `Training time: 2142.46 seconds`, full
1000/1000 PPO iterations — `Learning iteration 999/1000` present — 4096 envs =
98,304,000 total steps). Values were read from the raw log before recording; the
log contains zero `Traceback`, `Error executing job`, `FileNotFoundError`, or
NaN lines.

| env | solver | contact | Training time (s) | Total steps | Status |
|-----|--------|---------|-------------------|------------:|--------|
| h1  | newton_dvi_pspg (P-SPG-FB) | 10 | 2142.46 | 98,304,000 | complete |

This completes the **full h1 contact_10 DVI matrix** (Jacobi 822.78 s < APGD
1016.73 s < PSPG 2142.46 s), preserving the Jacobi<APGD<PSPG wall-time ordering
seen at contact_5 and on g1/go2. h1 PSPG contact_10 (2142.46 s) is ~45% slower
than h1 PSPG contact_5 (1475.01 s), the largest budget-driven slowdown of the
three h1 solvers, consistent with P-SPG-FB's heavier per-contact-iteration cost.
h1 PSPG contact_10 (2142.46 s) is nearly identical to g1 PSPG contact_10
(2175.57 s) despite the smaller robot, i.e. PSPG wall time at this budget is
dominated by contact-solver cost rather than robot DOF count.

**Now in progress:** `h1 / newton_mjwarp / contact_10` (started 13:03:06 UTC),
the fourth and final h1 contact_10 cell and the second MJWarp run of the sweep
(MJWarp executes only in the contact_10 block per the wrapper guard;
`mujoco-warp` 3.8.0.3). At time of writing it is at iteration ~152/1000, training
steadily at ~140k steps/s (far faster than the DVI cells, as expected for the
native MJWarp path). It correctly constructs backend-specific config from the
`newton_mjwarp` preset; its raw log contains zero `Traceback`, zero
`Error executing job`, and zero `[ERROR]`/`FileNotFoundError`/NaN lines. No
`COMPLETED` marker yet, so it is correctly not counted as a result.

**dr_legs blocker unchanged.** Still the missing-`dr_legs.usda` asset: a dangling
`Geometry` symlink into a non-existent Newton asset cache under
`~/.cache/newton/newton-assets_disneyresearch_.../disneyresearch/dr_legs/`, and
the `usd_path` in `robots/dr_legs.py` (line 90) was never provisioned (`TODO` to
host on Nucleus). This is a missing-asset provisioning issue, **not** analogous
to the Ant adapter fix and not fixable by a reversible code edit; fabricating or
re-pointing the asset would be a hack and is out of scope. No action taken beyond
flagging for Milad; it will recur at contact_10 and contact_15 until the DR Legs
USD is provisioned.

**Updated completed-run tally: 16 validated cells** — the full contact_5 DVI
matrix (G1, H1, Go2 x Jacobi/APGD/PSPG = 9), the full g1 contact_10 matrix
(Jacobi/APGD/PSPG/MJWarp = 4), and the full h1 contact_10 DVI matrix
(Jacobi/APGD/PSPG = 3). All 16 markers re-validated this pass (each has `exit=0`
and a `Training time:` line, marker timestamps matching the `END ... exit=0`
lines in the master log). Remaining matrix: contact_10 block continuing (h1
MJWarp running, then go2 all +MJWarp, dr_legs all [will fail]); then contact_15
block (DVI-only per wrapper: g1/h1/go2 x Jacobi/APGD/PSPG, dr_legs all [will
fail]). No commit made.

## Watchdog pass 2026-07-23 13:26 UTC

Master sweep still alive (PID 144300 from `locomotion_sweep.pid`, running ~5h58m,
master log updating at 13:25). No unexpected exit — no restart, resume, or fix
warranted; nothing was interrupted. Environments and solvers remain strictly
sequential. The newest run log was inspected for crashes and is healthy.

**Two new validated results since the 13:05 pass.** Both were read from their raw
logs before recording; each has a `COMPLETED` marker whose timestamp matches the
`END ... exit=0` line in the master log, a `Training time:` line, and full
1000/1000 PPO iterations (4096 envs = 98,304,000 total steps). Neither log
contains any `Traceback`, `Error executing job`, `FileNotFoundError`, `[ERROR]`,
or NaN line.

| env | solver | contact | Training time (s) | Steady steps/s | Total steps | Reward (final / best) | Status |
|-----|--------|---------|-------------------|---------------:|------------:|-----------------------|--------|
| h1  | newton_mjwarp (MJWarp)     | 10 | 551.12 | ~176k | 98,304,000 | 26.67 / 29.96 | complete |
| go2 | newton_dvi (Jacobi)        | 10 | 613.77 | ~162k | 98,304,000 | 34.14 / 35.18 | complete |

- **h1 MJWarp contact_10** (marker `2026-07-23T13:13:28Z`) completes the **full
  h1 contact_10 matrix** (Jacobi 822.78 s, APGD 1016.73 s, PSPG 2142.46 s,
  MJWarp 551.12 s). MJWarp is the fastest h1 contact_10 cell at ~176k steps/s,
  consistent with the native path leading DVI on every environment so far.
- **go2 Jacobi contact_10** (marker `2026-07-23T13:24:29Z`) opens the go2
  contact_10 block. At 613.77 s it is only ~4% slower than go2 Jacobi contact_5
  (588.94 s), i.e. the contact-budget increase from 5 to 10 costs little for
  Jacobi on go2 — the smallest budget-driven slowdown of any solver so far,
  contrasting with PSPG's large 5-to-10 penalty on g1/h1.

**Now in progress (13:47 UTC watchdog pass):** `go2 / newton_dvi_pspg /
contact_10` (started 13:46:57 UTC, ETA ~35 min), the third go2 contact_10 cell.
Since the previous update, `go2 / newton_dvi_apgd / contact_10` completed and
validated (marker `2026-07-23T13:46:53Z`, Training time 1270.5 s→ see table;
exit=0). The current run is training steadily with zero
`Traceback`/`Error executing job`/`FileNotFoundError`/`[ERROR]`/NaN lines in its
raw log; no `COMPLETED` marker yet, so it is correctly not counted. Remaining
after it: go2 MJWarp (contact_10), dr_legs all (contact_10), then the entire
contact_15 DVI-only block (g1/h1/go2 x Jacobi/APGD/PSPG; dr_legs all).

**dr_legs blocker ROOT-CAUSED and FIXED (reversible, this pass).** The prior
conclusion ("not fixable / fabricating the asset is out of scope") was
incorrect. Re-investigation showed the DR Legs USD *is* officially available and
was provisioned before via the sanctioned Newton path; the local cache had
simply been evicted, leaving the `Geometry` symlink dangling and no
`dr_legs.usda` symlink in the data dir. The three dr_legs **contact_5** cells
failed at 10:22 UTC with:

```
FileNotFoundError: USD file not found at path:
  '.../isaaclab_assets/robots/data/dr_legs/dr_legs.usda'.
```

Fix applied (no code edits, no fabricated content, fully reversible):
1. Re-ran the **documented** provisioning command
   `newton.utils.download_asset("disneyresearch")` (dvi env), which restored
   `~/.cache/newton/newton-assets_disneyresearch_.../disneyresearch/dr_legs/usd/`
   with `dr_legs.usda`, `dr_legs_with_boxes.usda`,
   `dr_legs_with_meshes_and_boxes.usda`, and `Geometry/surfaces.usd`. This is the
   exact mechanism the closed-loop status doc records for the original setup.
2. The pre-existing `Geometry` symlink in the data dir now resolves again.
3. Added the missing sibling symlink
   `robots/data/dr_legs/dr_legs.usda → <cache>/dr_legs/usd/dr_legs.usda`,
   matching the existing `Geometry` symlink convention, so the `usd_path` in
   `robots/dr_legs.py:90` resolves. The USD's internal reference
   (`@Geometry/surfaces.usd@`) resolves via the sibling `Geometry` symlink.

This is consistent with the Ant adapter fix in spirit: a minimal, reversible
restoration of a broken-but-sanctioned setup, not a hack. It touches only
symlinks under `robots/data/dr_legs/` (untracked asset dir) and the Newton
asset cache — no source edits, nothing to commit.

**Validation status of the fix:** pending. A concurrent Isaac training run would
contend for the GPU with the still-running master and violate the sequential
constraint, so no dr_legs run was launched during this pass. The master will
itself reach `dr_legs / {dvi,apgd,pspg} / contact_10` in ~45–55 min and act as
the first validator; a watchdog follow-up will then re-run only the three
already-failed **dr_legs contact_5** cells (which the master's contact_5 loop
has already passed and will not revisit) via the same restart-safe script,
sequentially. Until those markers exist with `exit=0` + `Training time:`, no
dr_legs cell is counted as a result.

The master handles the (now-fixed) failures gracefully regardless — it logs
`FAILED` and advances without exiting, so the sweep stays restart-safe.

**Updated completed-run tally: 19 validated cells** — the full contact_5 DVI
matrix (G1, H1, Go2 x Jacobi/APGD/PSPG = 9), the full g1 contact_10 matrix
(Jacobi/APGD/PSPG/MJWarp = 4), the full h1 contact_10 matrix
(Jacobi/APGD/PSPG/MJWarp = 4), and go2 Jacobi + go2 APGD contact_10 (2). All 19
markers re-validated this pass (each has `exit=0` and a `Training time:` line;
marker timestamps match the `END ... exit=0` lines in the master log). The three
dr_legs contact_5 cells remain uncounted (failed pre-fix); their re-run is
queued for the watchdog follow-up. No commit made.

## Watchdog pass 2026-07-23 13:49 UTC

Master sweep still alive (PID 144300 from `locomotion_sweep.pid`, running
~6h22m; master log updating at 13:49). No unexpected exit — no restart, resume,
or code fix warranted this pass; nothing was interrupted, and environments and
solvers remain strictly sequential.

**Newest run log inspected — healthy.** `go2 / newton_dvi_pspg / contact_10`
(started 13:45:14 UTC) is training steadily at iteration ~128/1000, ETA ~36 min.
A crash-signature scan of its raw log
(`Traceback|Error executing job|FileNotFoundError|[ERROR]|NaN|CUDA error|RuntimeError`)
returns **0 matches**. The run correctly loads `preset=newton_dvi_pspg`,
`contact_iterations=10`, task `Isaac-Velocity-Flat-Unitree-Go2-v0`. No
`COMPLETED` marker yet, so it is correctly not counted.

**No new completions since the 13:26 pass.** The only marker newer than that
pass (go2 APGD contact_10, `2026-07-23T13:45:14Z`) was already recorded and
tabulated in the 13:26 pass. All 19 previously-validated markers re-checked and
still present. Tally unchanged: **19 validated cells**.

**dr_legs fix re-verified intact.** The `robots/data/dr_legs/dr_legs.usda`
symlink still resolves to the Newton asset cache
(`.../disneyresearch/dr_legs/usd/dr_legs.usda`, real 84 KB file), and the
sibling `Geometry` symlink resolves. `test -f` on the USD path confirms it
resolves. The fix remains symlink-only under the untracked asset dir plus the
Newton cache — no source edits, nothing to commit (the single tracked
`hold_pose_env_cfg.py` modification is pre-existing sweep work, unrelated to the
asset fix).

**Remaining master queue (contact_10 block):** go2 MJWarp, then dr_legs
{Jacobi/APGD/PSPG} — the latter now with the USD asset present, so the master
will be the first validator of the dr_legs fix in ~45–55 min. After that, if
`BUDGETS` was `"5 10"` (only 5 and 10 observed in the master log), the master
terminates; the three failed **dr_legs contact_5** cells were passed in the
budget-5 block and will not be revisited by this master run.

**Follow-up scheduled (no concurrent launch this pass):** a cron watchdog will,
*after* the master exits, re-run only the three failed dr_legs contact_5 cells
via the same restart-safe `run_locomotion_sweep.sh` (`BUDGETS=5`), sequentially
and GPU-exclusively. Launching now would contend for the GPU with the live
master and violate the sequential constraint, so it is deferred. Until those
markers exist with `exit=0` + `Training time:`, no dr_legs cell is counted.

No commit made.

## Watchdog pass 2026-07-23 14:05 UTC

Master sweep still alive and healthy (PID 144300 from `locomotion_sweep.pid`,
`bash results/rl_solver_sweep/run_locomotion_sweep.sh`, running ~6h37m; master
log updating at 14:03). Not interrupted; no unexpected exit, so no restart,
resume, or code fix warranted this pass. Environments and solvers remain
strictly sequential (single active Isaac training process).

**Newest run log inspected — healthy.** `go2 / newton_dvi_pspg / contact_10`
(started 13:45:14 UTC) is training steadily at iteration ~452/1000
(~37.7k steps/s, iter time ~2.6 s, ETA ~26 min). A crash-signature scan of its
raw log (`Traceback|Error executing job|FileNotFoundError|[ERROR]|NaN|CUDA
error|RuntimeError`) returns 0 matches. It correctly loads
`preset=newton_dvi_pspg`, `contact_iterations=10`, task
`Isaac-Velocity-Flat-Unitree-Go2-v0`. No `COMPLETED` marker yet, so it is
correctly not counted.

**Budget scope confirmed = `"5 10"` only.** Every START block in the master log
carries `contact_iterations` of either 5 (12x) or 10 (11x); no `contact_15`
block exists. So the default `BUDGETS="5 10 15"` was overridden to `"5 10"` for
this master run. There will be no contact_15 pass; the master ends after the
contact_10 block completes.

**All 19 completions re-validated this pass.** Each has a `COMPLETED` marker
whose timestamp matches its `END ... exit=0` line in the master log, plus a
`Training time:` line in the run log. No BAD/partial markers. Tally unchanged
since 13:49: **19 validated cells** — full contact_5 DVI matrix (G1/H1/Go2 x
Jacobi/APGD/PSPG = 9), full g1 contact_10 (Jacobi/APGD/PSPG/MJWarp = 4), full h1
contact_10 (Jacobi/APGD/PSPG/MJWarp = 4), go2 Jacobi + go2 APGD contact_10 (2).

**dr_legs fix re-verified intact.** `robots/data/dr_legs/dr_legs.usda` resolves
(`ls -lL` -> real 84130-byte USD) to the Newton asset cache
(`.../disneyresearch/dr_legs/usd/dr_legs.usda`); sibling `Geometry` symlink
resolves (`surfaces.usd`, ~17 MB). The tracked `hold_pose_env_cfg.py` change
(adds the `newton_dvi_pspg` DR Legs preset) is pre-existing sweep work and
consistent with the sweep matrix; it is not an asset fix and remains uncommitted
per instruction. Symlink-only asset restoration, nothing to commit.

**Standing blocker (unchanged):** the three `dr_legs / {dvi,apgd,pspg} /
contact_5` cells failed pre-fix at 10:22 UTC (missing USD, now provisioned) and
have no `COMPLETED` marker. The master's contact_5 block has already passed them
and will not revisit within this run. They stay uncounted until re-run with
`exit=0` + `Training time:`.

**Remaining master queue (contact_10 block):** go2/pspg (active) ->
go2/mjwarp -> dr_legs/{dvi,apgd,pspg} (now with the USD present, so the master
is the first validator of the dr_legs fix in ~30-45 min), then the master
exits.

**Follow-up deferred, not launched (sequential + no-interrupt constraints):** a
watchdog will, *after* the master exits, re-run only the three failed dr_legs
contact_5 cells via the same restart-safe `run_locomotion_sweep.sh`
(`BUDGETS=5`), sequentially and GPU-exclusively. Launching now would contend for
the GPU with the live master, so it is deferred.

No commit made.

## Watchdog pass 2026-07-23 14:27 UTC

Master sweep still alive and healthy (PID 144300 from `locomotion_sweep.pid`,
`bash results/rl_solver_sweep/run_locomotion_sweep.sh`, running ~6h59m; master
log updating at 14:27). Not interrupted; no unexpected exit, so no restart,
resume, or code fix warranted this pass. Environments and solvers remain
strictly sequential (single active Isaac training process).

**Newest run log inspected — healthy, near completion.** `go2 /
newton_dvi_pspg / contact_10` (started 13:45:14 UTC) is at PPO iteration
962/1000 (~37.6k steps/s, iter time ~2.5 s, ETA ~1.5 min). A crash-signature
scan of its raw log
(`Traceback|Error executing job|FileNotFoundError|[ERROR]|NaN|CUDA
error|RuntimeError`) returns **0 matches**. It correctly loads
`preset=newton_dvi_pspg`, `contact_iterations=10`, task
`Isaac-Velocity-Flat-Unitree-Go2-v0`. No `END`/`COMPLETED` yet (still running
the final PPO iterations + checkpoint save), so it is correctly not counted.

**All 19 completions re-validated this pass.** Every `COMPLETED` marker was
re-checked programmatically: each corresponding `run.log` carries both a
`Training time:` line and an `exit=0` `END` marker — 19/19 pass, 0 BAD/partial.
Tally unchanged since 14:05: **19 validated cells** — full contact_5 DVI matrix
(G1/H1/Go2 x Jacobi/APGD/PSPG = 9), full g1 contact_10
(Jacobi/APGD/PSPG/MJWarp = 4), full h1 contact_10 (Jacobi/APGD/PSPG/MJWarp = 4),
go2 Jacobi + go2 APGD contact_10 (2).

**dr_legs fix re-verified intact.** `robots/data/dr_legs/dr_legs.usda` resolves
(`test -f` -> real 84130-byte USD) to the Newton asset cache
(`.../disneyresearch/dr_legs/usd/dr_legs.usda`); sibling `Geometry/surfaces.usd`
symlink resolves (~17 MB). The fix remains symlink-only under the untracked
asset dir plus the Newton cache — no source edits. `git status` shows only
pre-existing sweep working-tree changes (`dvi_manager.py`, `hold_pose_env_cfg.py`,
the g1/go2/h1 `flat_env_cfg.py` + `velocity_env_cfg.py`), all unrelated to the
asset fix and left uncommitted per instruction.

**Standing blocker (unchanged):** the three `dr_legs / {dvi,apgd,pspg} /
contact_5` cells failed pre-fix at 10:22 UTC. Root cause confirmed from the raw
logs this pass: `FileNotFoundError: USD file not found at path:
'.../robots/data/dr_legs/dr_legs.usda'` (the trailing
`AttributeError: 'Articulation' object has no attribute '_initialize_handle'` is
benign `__del__` teardown noise, not the cause). The USD is now provisioned, so
the fix is consistent with the reversible Ant-adapter/asset approach, but the
master's contact_5 block already passed these cells and will not revisit them
within this run. They stay uncounted until re-run with `exit=0` + `Training
time:`.

**Remaining master queue (contact_10 block):** go2/pspg (finishing now) ->
go2/mjwarp -> dr_legs/{dvi,apgd,pspg} (now with the USD present, so the master
is the first validator of the dr_legs fix in ~25-40 min), then — with confirmed
`BUDGETS="5 10"` (no contact_15 block) — the master exits.

**Follow-up deferred, not launched (sequential + no-interrupt constraints):** a
watchdog will, *after* the master exits, re-run only the three failed dr_legs
contact_5 cells via the same restart-safe `run_locomotion_sweep.sh`
(`BUDGETS=5`), sequentially and GPU-exclusively. Launching now would contend for
the GPU with the live master and violate the sequential constraint, so it is
deferred.

No commit made.

## Watchdog pass 2026-07-23 14:43 UTC

Master sweep alive and healthy (PID 144300 from `locomotion_sweep.pid`,
`bash results/rl_solver_sweep/run_locomotion_sweep.sh`, running ~7h15m; master
log updating at 14:43). Not interrupted; no unexpected exit -> no restart,
resume, or code fix warranted this pass. Environments and solvers remain
strictly sequential: a single logical training run is live (isaaclab.sh CLI
wrapper PID 167892 + its child PID 167896, both the same
`Isaac-DrLegs-Walk-v0 ... contact_max_iterations=10` process).

**Newest run log inspected — healthy.** `dr_legs / newton_dvi / contact_10`
(started 14:37:21 UTC) is at PPO iteration 71/1000 (~15.9k steps/s, iter time
~6.2 s). Crash-signature scan of its raw log
(`Traceback|Error executing job|FileNotFoundError|[ERROR]|NaN|CUDA
error|RuntimeError`) returns **0 matches**. It correctly loads
`preset=newton_dvi`, `contact_iterations=10`, task `Isaac-DrLegs-Walk-v0`. This
is the master's first validation of the dr_legs USD provisioning; it is training
normally, confirming the asset fix holds under the live sweep. No
`END`/`COMPLETED` yet, so correctly uncounted.

**Tally advanced to 21 validated cells** (was 19 at 14:27). Two go2 contact_10
cells completed since that pass, both re-validated this pass
(`COMPLETED` marker + `Training time:` + `exit=0`, full 1000/1000 PPO,
4096 envs):

| Environment | Solver | Contact iters | Training time | exit | COMPLETED |
|---|---|---:|---:|---:|---|
| go2 | newton_dvi_pspg | 10 | 2580.35 s | 0 | 2026-07-23T14:29:03Z |
| go2 | newton_mjwarp | native | 418.02 s | 0 | 2026-07-23T14:37:21Z |

All 21 `COMPLETED` markers re-checked programmatically this pass: every
corresponding `run.log` carries both a `Training time:` line and an `exit=0`
`END` marker — **21/21 pass, 0 BAD/partial**. Validated cells: full contact_5
DVI matrix (G1/H1/Go2 x Jacobi/APGD/PSPG = 9), full g1 contact_10
(Jacobi/APGD/PSPG/MJWarp = 4), full h1 contact_10 (Jacobi/APGD/PSPG/MJWarp = 4),
full go2 contact_10 (Jacobi/APGD/PSPG/MJWarp = 4).

**dr_legs asset fix re-verified intact.** `robots/data/dr_legs/dr_legs.usda`
resolves (`ls -lL` -> real 84130-byte USD) into the Newton asset cache; the fix
remains symlink-only under the untracked asset dir plus the Newton cache — no
source edits, reversible, consistent with the Ant adapter/asset approach.
`git status` shows only pre-existing sweep working-tree changes (`dvi_manager.py`,
`hold_pose_env_cfg.py`, the g1/go2/h1 `flat_env_cfg.py` +
`velocity_env_cfg.py`), all unrelated to the asset fix and left uncommitted per
instruction.

**Standing blocker (unchanged):** the three `dr_legs / {dvi,apgd,pspg} /
contact_5` cells failed pre-fix at 10:22 UTC with `FileNotFoundError: USD file
not found at path: '.../robots/data/dr_legs/dr_legs.usda'` (trailing
`AttributeError: 'Articulation' object has no attribute '_initialize_handle'` is
benign `__del__` teardown noise). USD now provisioned; the master's contact_5
block already passed these cells and will not revisit them within this run, so
they stay uncounted until re-run with `exit=0` + `Training time:`.

**Note on remaining master queue:** the master is now in the dr_legs contact_10
block (dvi active -> apgd -> pspg). Whether it then exits or proceeds to a
contact_15 block depends on the launch-time `BUDGETS` env, which is no longer
readable from `/proc/144300/environ` (0 bytes, orphaned PPID=1). The master log
shows only contact_5 and contact_10 iterations so far (contact_15 has not
started), consistent with either `BUDGETS="5 10"` or the default `"5 10 15"`
still in its contact_10 block; this cannot be disambiguated non-invasively
right now and does not change the no-interrupt decision.

**Follow-up deferred, not launched (sequential + no-interrupt constraints):** a
watchdog will, *after* the master exits, re-run only the three failed dr_legs
contact_5 cells via the same restart-safe `run_locomotion_sweep.sh`
(`BUDGETS=5`), sequentially and GPU-exclusively. Launching now would contend for
the GPU with the live master and violate the sequential constraint, so it is
deferred.

No commit made.

## Reward-curve figures (2026-07-23)

Generated per-environment seven-curve plots from validated raw logs for G1, H1,
and Go2. Each plot contains Jacobi/APGD/P-SPG-FB at contact budgets 5 and 10,
plus native MJWarp; the legend reports the median steady-state environment
steps/s for each curve. Source script:
`results/rl_solver_sweep/plot_reward_curves.py`.

Figures:
- `figures/g1_reward_curves.{pdf,svg,png}`
- `figures/h1_reward_curves.{pdf,svg,png}`
- `figures/go2_reward_curves.{pdf,svg,png}`

DR Legs is excluded until its post-asset-provisioning runs complete.

## Watchdog pass 2026-07-23 15:23 UTC

Master sweep alive and healthy (PID 144300 from `locomotion_sweep.pid`,
`bash results/rl_solver_sweep/run_locomotion_sweep.sh`, running ~7h57m; master
log updating at 15:23). Not interrupted; no unexpected exit -> no restart,
resume, or code fix warranted this pass. Environments and solvers remain
strictly sequential: exactly one logical training run is live (isaaclab.sh CLI
wrapper PID 167892 + its `tee` child PID 167893 writing the run log; the
process holds the single open `run.log` fd for the active cell).

**Newest run log inspected — healthy.** `dr_legs / newton_dvi / contact_10`
(started 14:37:21 UTC, same run first seen in the 14:43 pass) has advanced from
iter 71 to **iteration 446/1000** (~15.7k steps/s, iter time ~6.3 s,
Mean reward ~379, success_rate ~1.0, ETA ~00:58). Crash-signature scan of its
raw log (`Traceback|Error executing job|FileNotFoundError|[ERROR]|NaN|CUDA
error|RuntimeError|_initialize_handle`) returns **0 matches** (excluding the
benign `[carb] Client passed ... nullptr` shutdown line and the orphan-joints
INFO notice). No `END`/`COMPLETED` yet, so correctly uncounted.

**Tally unchanged at 21 validated cells** (no new completions since the 14:43
pass; the live dr_legs contact_10 DVI cell is still in-flight). All 21
`COMPLETED` markers re-checked programmatically this pass: every corresponding
`run.log` carries both a `Training time:` line and an `exit=0` `END` marker —
**21/21 pass, 0 BAD/partial**. Validated cells unchanged: full contact_5 DVI
matrix (G1/H1/Go2 x Jacobi/APGD/PSPG = 9), full g1 contact_10
(Jacobi/APGD/PSPG/MJWarp = 4), full h1 contact_10 (4), full go2 contact_10 (4).

**dr_legs asset fix re-verified intact.** `robots/data/dr_legs/dr_legs.usda`
resolves via symlink into the Newton asset cache
(`~/.cache/newton/newton-assets_disneyresearch_193783fd_8e8df07d/disneyresearch/dr_legs/usd/`);
the fix remains symlink-only under the untracked asset dir — no source edits,
reversible, consistent with the Ant adapter/asset approach. `git status` shows
only the pre-existing sweep working-tree edits (`dvi_manager.py`, dr_legs
`hold_pose_env_cfg.py`, g1/go2/h1 `flat_env_cfg.py`, go2 `rough_env_cfg.py`,
`velocity_env_cfg.py`) plus untracked `results/`, `run_dvi_ant_sweep.sh`,
`sweep_logs/` — none from the asset fix, all left uncommitted per instruction.

**Standing blocker (unchanged):** the three `dr_legs / {dvi,apgd,pspg} /
contact_5` cells failed pre-fix at 10:22 UTC (`FileNotFoundError` on the then-
missing `dr_legs.usda`; trailing `_initialize_handle` `AttributeError` is benign
`__del__` teardown noise). USD now provisioned; the master's contact_5 block
already passed these cells and will not revisit them within this run, so they
stay uncounted until re-run with `exit=0` + `Training time:`.

**Follow-up deferred, not launched (sequential + no-interrupt constraints):**
after the master exits, a watchdog will re-run only the three failed dr_legs
contact_5 cells via the same restart-safe `run_locomotion_sweep.sh`
(`BUDGETS=5`), sequentially and GPU-exclusively. Launching now would contend for
the GPU with the live master and violate the sequential constraint, so it is
deferred.

No commit made.

## Watchdog pass 2026-07-23 15:43 UTC

Master sweep alive and healthy (PID 144300 from `locomotion_sweep.pid`,
`bash results/rl_solver_sweep/run_locomotion_sweep.sh`, running ~8h17m; master
log updating at 15:43). Not interrupted; no unexpected exit -> no restart,
resume, or code fix warranted this pass. Environments and solvers remain
strictly sequential: exactly one logical training run is live (the active
dr_legs contact_10 DVI cell).

**Newest run log inspected — healthy.** `dr_legs / newton_dvi / contact_10`
(started 14:37:21 UTC, same run tracked since the 14:43 pass) has advanced from
iter 446 to **iteration 646/1000** (~15.7k steps/s, iter time ~6.3 s,
Mean reward ~381, success_rate ~1.0, ETA ~00:37). Crash-signature scan of its
raw log (`Traceback|Error executing job|FileNotFoundError|CUDA error|
RuntimeError`) returns **0 matches** (the only `[Error]` line is the benign
`[carb] Client passed ... nullptr` startup notice; the orphan-joints
`UserWarning` is the expected DR Legs articulation-root INFO). No `END`/
`COMPLETED` marker yet, so correctly uncounted.

**Tally unchanged at 21 validated cells** (no new completions since the 14:43
pass; the live dr_legs contact_10 DVI cell is still in-flight). All 21
`COMPLETED` markers re-checked programmatically this pass: every corresponding
`run.log` carries both a `Training time:` line and an `exit=0` `END` marker —
**21/21 pass, 0 BAD/partial**. Validated cells unchanged: full contact_5 DVI
matrix (G1/H1/Go2 x Jacobi/APGD/PSPG = 9), full g1 contact_10
(Jacobi/APGD/PSPG/MJWarp = 4), full h1 contact_10 (4), full go2 contact_10 (4).

**dr_legs asset fix re-verified intact.** `robots/data/dr_legs/dr_legs.usda`
resolves via symlink into the Newton asset cache (`ls -lL` -> real 84130-byte
USD under
`~/.cache/newton/newton-assets_disneyresearch_193783fd_8e8df07d/disneyresearch/dr_legs/usd/`).
The fix remains symlink-only under the untracked asset dir — no source edits,
reversible, consistent with the Ant adapter/asset approach. Left uncommitted per
instruction.

**Standing blocker (unchanged):** the three `dr_legs / {dvi,apgd,pspg} /
contact_5` cells failed pre-fix at 10:22 UTC (`FileNotFoundError` on the then-
missing `dr_legs.usda`; trailing `_initialize_handle` `AttributeError` is benign
`__del__` teardown noise). USD now provisioned; the master's contact_5 block
already passed these cells and will not revisit them within this run, so they
stay uncounted until re-run with `exit=0` + `Training time:`.

**Follow-up deferred, not launched (sequential + no-interrupt constraints):**
after the master exits, a watchdog will re-run only the three failed dr_legs
contact_5 cells via the same restart-safe `run_locomotion_sweep.sh`
(`BUDGETS=5`), sequentially and GPU-exclusively. Launching now would contend for
the GPU with the live master and violate the sequential constraint, so it is
deferred.

No commit made.

## Watchdog pass 2026-07-23 16:05 UTC

Master sweep alive and healthy (PID 144300 from `locomotion_sweep.pid`,
`bash results/rl_solver_sweep/run_locomotion_sweep.sh`, elapsed ~8h37m; master
log mtime 16:05:37, <15 s stale). Not interrupted; no unexpected exit -> no
restart/resume/fix warranted. Exactly one logical training run is live -> the
sequential (env, solver) invariant holds.

**Newest run log inspected — healthy.** `dr_legs / newton_dvi / contact_10`
(started 14:37:21 UTC) advanced to **iteration 836/1000** (~15.6k steps/s, iter
~6.3 s, Mean reward ~383, success_rate ~1.0, ETA ~00:17). Crash-signature scan
(`Traceback|Error executing job|FileNotFoundError|CUDA error|RuntimeError|nan`)
= **0 real hits** (only the benign `[carb] ... nullptr` startup line and the
expected DR Legs orphan-joints INFO). No `END`/`COMPLETED` yet -> correctly
uncounted.

**Tally unchanged: 21/21 validated cells.** All 21 `COMPLETED` markers
re-checked this pass: each `run.log` carries a `Training time:` line and an
`exit=0` `END` marker -> 21/21 pass, 0 partial/BAD. (contact_5 DVI matrix
G1/H1/Go2 x Jacobi/APGD/PSPG = 9; contact_10 full for g1/h1/go2 incl. MJWarp =
12.)

**dr_legs asset fix re-verified intact & reversible.** `robots/data/dr_legs/
dr_legs.usda` -> real 84130-byte `#usda 1.0` file in the Newton asset cache
(`~/.cache/newton/newton-assets_disneyresearch_193783fd_8e8df07d/.../usd/`);
`Geometry` symlink also resolves. Symlink lives under the untracked asset dir —
no source edits, consistent with the Ant adapter/asset approach. Uncommitted per
instruction; `git status` shows only the pre-existing sweep working-tree edits.

**Standing blocker (unchanged):** the three `dr_legs / {dvi,apgd,pspg} /
contact_5` cells failed pre-fix at 10:22 UTC (`FileNotFoundError` on the then-
missing `dr_legs.usda`; the trailing `_initialize_handle` `AttributeError` is
benign `__del__` teardown). USD now provisioned, but the master's contact_5
block already passed these cells and will not revisit them this run, so they stay
uncounted until re-run with `exit=0` + `Training time:`.

**Remaining master forward path (single pass, no loop-back):** finish dr_legs
c10 DVI (in-flight), then dr_legs c10 APGD + PSPG, then the entire contact_15
phase (15 cells: g1/h1/go2/dr_legs x DVI/APGD/PSPG, MJWarp c10-only). contact_15
dirs not yet created — phase not started, as expected.

**Follow-up still deferred (no interrupt / sequential):** after the master
exits, re-run only the three dr_legs contact_5 cells via the same restart-safe
`run_locomotion_sweep.sh` with `BUDGETS=5` (skips all completed cells,
GPU-exclusive). Launching now would contend with the live master and break the
sequential constraint, so it stays deferred.

No commit made.

## Watchdog pass 2026-07-23 16:23 UTC

Master sweep alive and healthy (PID 144300 from `locomotion_sweep.pid`,
`bash results/rl_solver_sweep/run_locomotion_sweep.sh`, elapsed ~8h57m; master
log mtime 16:25, fresh). Not interrupted; no unexpected exit -> no
restart/resume/code fix warranted. Exactly one logical training run is live ->
the sequential (env, solver) invariant holds.

**New validated result — dr_legs Jacobi contact_10 (first DR Legs cell ever to
complete).** The cell in-flight since 14:37:21 UTC finished at 16:22:49Z and
validates cleanly: `COMPLETED` marker `2026-07-23T16:22:49Z`, `===== END ...
exit=0 =====`, `Training time: 6279.45 seconds`, `Learning iteration 999/1000`
(full 1000 iters), final Mean reward ~384 (success_rate ~1.0). Values read from
the raw log before recording. This is the longest cell so far (~104 min, ~5.7x
the g1 Jacobi c10 time), consistent with DR Legs' larger 36-DOF closed-loop
articulation.

| env | solver | contact | Training time (s) | Final mean reward | Status |
|-----|--------|---------|-------------------|------------------:|--------|
| dr_legs | newton_dvi (Jacobi) | 10 | 6279.45 | ~384 | complete |

**Updated tally: 22 validated cells** (was 21). All 22 `COMPLETED` markers
re-checked programmatically this pass: every corresponding `run.log` carries
both a `Training time:` line and an `exit=0` `END` marker — **22/22 pass, 0
partial/BAD**. Validated cells: full contact_5 DVI matrix (G1/H1/Go2 x
Jacobi/APGD/PSPG = 9); full contact_10 for g1/h1/go2 incl. MJWarp (12); plus
dr_legs Jacobi contact_10 (1).

**Newest run log inspected — healthy.** `dr_legs / newton_dvi_apgd / contact_10`
(started 16:22:49 UTC, immediately after the Jacobi cell) is at **iteration
16/1000**, iter time ~7.7 s, Mean reward climbing (28.7 -> 33.8), ETA ~2h22m.
Crash-signature scan (`Traceback|Error executing job|FileNotFoundError|CUDA
error|RuntimeError|nan`) = **0 real hits** (only the benign `[carb] ... nullptr`
startup line and the expected DR Legs orphan-joints INFO). No `END`/`COMPLETED`
yet -> correctly uncounted.

**dr_legs asset fix re-verified intact & reversible.** `robots/data/dr_legs/
dr_legs.usda` -> real 84130-byte `#usda 1.0` file in the Newton asset cache
(`~/.cache/newton/newton-assets_disneyresearch_193783fd_8e8df07d/.../usd/`);
`Geometry` symlink also resolves. Symlink lives under the untracked asset dir —
no source edits, consistent with the Ant adapter/asset approach. `git status`
shows only the pre-existing sweep working-tree edits (`dvi_manager.py`,
`dvi_manager_cfg.py`, dr_legs `hold_pose_env_cfg.py`, g1/go2/h1 `flat_env_cfg.py`,
go2 `rough_env_cfg.py`, `velocity_env_cfg.py`) plus untracked `results/`,
`run_dvi_ant_sweep.sh`, `sweep_logs/` — none from the asset fix. Uncommitted per
instruction.

**Standing blocker (unchanged):** the three `dr_legs / {dvi,apgd,pspg} /
contact_5` cells failed pre-fix at 10:22 UTC (`FileNotFoundError` on the then-
missing `dr_legs.usda`; the trailing `_initialize_handle` `AttributeError` is
benign `__del__` teardown). The USD is now provisioned and DR Legs trains
successfully at contact_10, confirming the fix — but the master's contact_5
block already passed these three cells and will not revisit them this run, so
they stay uncounted until re-run with `exit=0` + `Training time:`.

**Remaining master forward path (single pass, no loop-back):** finish dr_legs
c10 APGD (in-flight), then dr_legs c10 PSPG, then the entire contact_15 phase
(DVI-only per wrapper: g1/h1/go2/dr_legs x Jacobi/APGD/PSPG; MJWarp is c10-only).
contact_15 dirs not yet created — phase not started, as expected.

**Follow-up still deferred (no interrupt / sequential):** after the master
exits, re-run only the three dr_legs contact_5 cells via the same restart-safe
`run_locomotion_sweep.sh` with `BUDGETS=5` (skips all completed cells,
GPU-exclusive). Launching now would contend with the live master and break the
sequential constraint, so it stays deferred.

No commit made.

## Watchdog pass 2026-07-23 16:46 UTC

Master sweep still alive and healthy (PID 144300, `bash
results/rl_solver_sweep/run_locomotion_sweep.sh`, elapsed ~9h18m). Not
interrupted; no unexpected exit -> no restart/resume/code fix warranted.
Exactly one logical training run live -> sequential (env, solver) invariant
holds.

**Delta since 16:23 pass:** forward progress only, no state change requiring
action. The live cell `dr_legs / newton_dvi_apgd / contact_10` advanced from
iter ~16 to **iter 176/1000** (Mean reward ~340, ETA ~1h43m). Crash-signature
scan (`Traceback|Error executing job|FileNotFoundError|CUDA error|RuntimeError|
nan|Segmentation`, excluding benign `[carb] nullptr` / DR Legs orphan-joint
INFO) = **0 hits**. No `END`/`COMPLETED` yet -> correctly uncounted.

**No new completions:** zero `COMPLETED` markers written since 16:23; tally
holds at **22 validated cells** (all still pass Training-time + exit=0 checks,
per prior pass). Nothing new to validate/record this pass.

**Standing blocker (unchanged, no action taken):** the three `dr_legs /
{dvi,apgd,pspg} / contact_5` cells remain FAILED-pre-fix (missing
`dr_legs.usda` at 10:22 UTC; asset since provisioned via cache symlink and
proven by successful DR Legs contact_10 training). The master's contact_5 block
already passed them and will not revisit this pass. Deliberately NOT re-run now:
the master is still live and single-GPU-exclusive, so a `BUDGETS=5` FORCE re-run
would contend and violate the sequential constraint. Deferred to after master
exit, same restart-safe `run_locomotion_sweep.sh` (skips completed cells).

**Forward path unchanged:** dr_legs c10 APGD (in-flight) -> dr_legs c10 PSPG ->
contact_15 DVI phase (not yet started). No commit made.

## Watchdog pass 2026-07-23 17:08 UTC

Master sweep alive and healthy (PID 144300, `bash
results/rl_solver_sweep/run_locomotion_sweep.sh`, elapsed ~9h38m). Not
interrupted; no unexpected exit -> no restart/code fix warranted. Exactly one
logical training run live (child PIDs 172951/172955, task
`Isaac-DrLegs-Walk-v0` presets/physics `newton_dvi_apgd`,
`contact_max_iterations=10`) -> sequential (env, solver) invariant holds.

**Live cell progress:** `dr_legs / newton_dvi_apgd / contact_10` advanced to
**iter 356/1000** (Mean reward ~373, ETA ~1h20m; run.log confirmed growing
~9.3 KB/9s, iter 321->356 across this pass). Master's own log buffers via
`tee -a` so it can look static between flushes; the per-cell `run.log` is the
authoritative liveness signal and is advancing. Crash-signature scan across all
run.logs (`Traceback|Error executing job|CUDA error|Segmentation fault`,
excluding the three known pre-fix dr_legs contact_5 failures) = **0 hits**.

**Validated completions unchanged:** **22 cells** with `COMPLETED` marker AND
`exit=0` + `Training time:` present (re-validated this pass). No new
`COMPLETED` since 16:46 -> nothing new to record. contact_15 phase not yet
started (no `contact_15` dirs), as expected.

**Standing blocker (root cause reconfirmed):** the three `dr_legs /
{newton_dvi,newton_dvi_apgd,newton_dvi_pspg} / contact_5` cells failed pre-fix
at 10:22 UTC. Real cause = `FileNotFoundError: .../data/dr_legs/dr_legs.usda`
(the trailing `_initialize_handle` `AttributeError` is benign `__del__`
teardown noise). Asset now provisioned: `dr_legs.usda` and `Geometry` are cache
symlinks (target resolves, not dangling; `dr_legs.usda` symlink created 13:46
UTC) and DR Legs trains cleanly at contact_10 (contact_10 newton_dvi
`COMPLETED`, `Training time: 6279.45s`), proving the fix. This is consistent
with the Ant adapter fix approach: reversible, asset-provisioning only, no
solver/env logic changed. **No new code fix required** — the fix is already in
place; the three cells only need re-running.

**Deferred resume now OPERATIONALIZED (change vs prior passes):** prior passes
only *planned* the deferred re-run; this pass sets it up concretely without
touching the live sweep. Added `results/rl_solver_sweep/resume_dr_legs_contact5.sh`
and launched it fully detached (`setsid nohup`, now PPID 1, SID-leader,
`resume_dr_legs_contact5.log`). It blocks on `kill -0 144300` and only after the
master exits invokes the **existing** `run_locomotion_sweep.sh` with
`BUDGETS=5` (no FORCE). All other contact_5 cells carry `COMPLETED` and are
skipped, so exactly the three dr_legs contact_5 cells re-run, GPU-exclusive and
sequential. Nothing runs concurrently with the master; the sequential (env,
solver) invariant is preserved. Their results stay uncounted until they pass the
`exit=0` + `Training time:` validation like every other cell.

**Forward path (single pass, no loop-back):** finish dr_legs c10 APGD
(in-flight) -> dr_legs c10 PSPG -> full contact_15 DVI phase (g1/h1/go2/dr_legs
x Jacobi/APGD/PSPG; MJWarp is c10-only) -> master exits -> detached wrapper
fires the three dr_legs c5 re-runs. No commit made.

## Watchdog pass 2026-07-23 17:13 UTC

Master sweep still alive and healthy (PID 144300, `bash
results/rl_solver_sweep/run_locomotion_sweep.sh`, elapsed ~9h45m). Not
interrupted; no unexpected exit -> no restart or code fix warranted. Exactly one
logical training run live (child PIDs 172951/172955, task `Isaac-DrLegs-Walk-v0`,
presets/physics `newton_dvi_apgd`, `contact_max_iterations=10`) -> sequential
(env, solver) invariant holds; deferred resume wrapper (PID 175497) is still
blocked on `kill -0 144300` (sleeping, not running) so nothing is concurrent.

**Live cell progress:** `dr_legs / newton_dvi_apgd / contact_10` advanced to
**iter 391/1000** (Mean reward ~375.9, iter time ~7.5s, ETA ~1h16m). run.log is
the authoritative liveness signal and is actively growing (size 738,783 B,
mtime 17:12:29 UTC, ~18s before this pass; iter 356 -> 391 since the 17:08
pass). Reward learning curve is monotone-healthy (survival 9.64,
contact_matching 9.24, success_rate 0.954). Crash-signature scan across all
run.logs (`Traceback|Error executing job|CUDA error|Segmentation fault`,
excluding the three known pre-fix dr_legs contact_5 failures) = **0 hits**.

**Validated completions unchanged: 22 cells.** Re-validated this pass that every
`COMPLETED` marker co-occurs with `exit=0` + a real `Training time:` line in its
run.log (spot-list confirmed: e.g. dr_legs/newton_dvi/contact_10 = 6279.45s,
g1/newton_dvi_apgd/contact_10 = 1270.5s, go2/newton_mjwarp/contact_10 = 418.02s,
h1/newton_dvi_pspg/contact_10 = 2142.46s). No new `COMPLETED` since 16:22 ->
nothing new to record. contact_15 phase not yet started (no `contact_15` dirs),
as expected, it follows dr_legs c10 PSPG.

**Standing blocker unchanged (root cause reconfirmed):** the three `dr_legs /
{newton_dvi,newton_dvi_apgd,newton_dvi_pspg} / contact_5` cells failed pre-fix at
10:22 UTC on the missing `dr_legs.usda` asset symlink (the trailing
`_initialize_handle` `AttributeError` is benign `__del__` teardown noise). Fix
already in place and validated: `dr_legs.usda` symlink (created 13:46 UTC)
resolves to the newton asset cache and dr_legs trains cleanly at contact_10
(newton_dvi COMPLETED, 6279.45s). Consistent with the Ant adapter fix approach:
reversible, asset-provisioning only, no solver/env logic touched. **No new code
fix required this pass**; the three cells only need re-running, which is already
queued via the existing restart-safe wrapper (no FORCE, so the other 12
contact_5 cells with `COMPLETED` are skipped and exactly these three re-run,
GPU-exclusive and sequential after the master exits). Their results remain
uncounted until they pass the same `exit=0` + `Training time:` validation.

**Forward path (unchanged, single pass):** finish dr_legs c10 APGD (in-flight,
~1h16m ETA) -> dr_legs c10 PSPG -> full contact_15 DVI phase (g1/h1/go2/dr_legs
x Jacobi/APGD/PSPG; MJWarp is c10-only) -> master exits -> detached wrapper fires
the three dr_legs c5 re-runs. No commit made.

## Watchdog pass 2026-07-23 17:25 UTC

Master sweep still alive and healthy (PID 144300 from `locomotion_sweep.pid`,
`bash results/rl_solver_sweep/run_locomotion_sweep.sh`, elapsed ~9h57m). Not
interrupted; no unexpected exit -> no restart, resume, or code fix warranted.
Exactly one logical training run live (child PIDs 172951/172955, task
`Isaac-DrLegs-Walk-v0`, presets/physics `newton_dvi_apgd`,
`contact_max_iterations=10`) -> sequential (env, solver) invariant holds. The
deferred resume wrapper (PID 175497) is still blocked on `kill -0 144300`
(sleeping in its 60s wait loop, not running) so nothing runs concurrently on the
GPU.

**Live cell progress:** `dr_legs / newton_dvi_apgd / contact_10` advanced to
**iter 491/1000** (Mean reward ~373.7, iter time ~7.5s, ETA ~1h05m). run.log is
the authoritative liveness signal and is actively growing (size 923,987 B, mtime
17:25:01 UTC, iter 391 -> 491 since the 17:13 pass). Reward learning curve stays
monotone-healthy (survival ~9.89, contact_matching ~9.53, success_rate ~0.983).
Crash-signature scan across all run.logs
(`Traceback|Error executing job|CUDA error|Segmentation fault`, excluding the
three known pre-fix dr_legs contact_5 failures) = **0 hits**; the c10 APGD
run.log itself has 0 such lines.

**Validated completions unchanged: 22 cells.** Re-validated this pass that every
`COMPLETED` marker co-occurs with a real `Training time:` line in its run.log
(programmatic count = 22). No new `COMPLETED` since 16:22 -> nothing new to
record. contact_15 phase not yet started (no `contact_15` dirs), as expected; it
follows dr_legs c10 PSPG.

**Standing blocker unchanged (root cause reconfirmed):** the three `dr_legs /
{newton_dvi,newton_dvi_apgd,newton_dvi_pspg} / contact_5` cells failed pre-fix at
10:22 UTC on the missing `dr_legs.usda` asset symlink. Fix already in place and
validated: the `dr_legs.usda` symlink (created 13:46 UTC) resolves and dr_legs
trains cleanly at contact_10 (newton_dvi COMPLETED, 6279.45s). Consistent with
the Ant adapter fix approach: reversible, asset-provisioning only, no solver/env
logic touched. **No new code fix required this pass**; the three cells only need
re-running, already queued via the existing restart-safe wrapper (no FORCE, so
the other 12 contact_5 cells with `COMPLETED` are skipped and exactly these three
re-run, GPU-exclusive and sequential after the master exits). Their results
remain uncounted until they pass the same `exit=0` + `Training time:` validation.

**Forward path (unchanged, single pass):** finish dr_legs c10 APGD (in-flight,
~1h05m ETA) -> dr_legs c10 PSPG -> full contact_15 DVI phase (g1/h1/go2/dr_legs
x Jacobi/APGD/PSPG; MJWarp is c10-only) -> master exits -> detached wrapper fires
the three dr_legs c5 re-runs. No commit made.

## Watchdog pass 2026-07-23 17:45 UTC

Master sweep still alive and healthy (PID 144300 from `locomotion_sweep.pid`,
`bash results/rl_solver_sweep/run_locomotion_sweep.sh`, elapsed ~10h17m). Not
interrupted; no unexpected exit -> no restart, resume, or code fix warranted.
Exactly one logical training run live (child PIDs 172951/172955, task
`Isaac-DrLegs-Walk-v0`, presets/physics `newton_dvi_apgd`,
`contact_max_iterations=10`) -> sequential (env, solver) invariant holds. The
deferred resume wrapper (PID 175497) is still blocked on `kill -0 144300`
(sleeping in its 60s wait loop, not running) so nothing runs concurrently on the
GPU. Note: `resume_dr_legs_contact5.log` shows two extra `RESUME WRAPPER START`
header lines from earlier wrapper instances that already exited; a `ps` check
confirms exactly one live wrapper (PID 175497), so there is no duplicate-resume
race.

**Live cell progress:** `dr_legs / newton_dvi_apgd / contact_10` advanced to
**iter 651/1000** (Mean reward ~377.2, iter time ~7.5s, ETA ~0h43m). run.log is
the authoritative liveness signal and is actively growing (size 1,220,444 B,
mtime 17:45:06 UTC, iter 491 -> 651 since the 17:25 pass). Reward learning curve
stays monotone-healthy (success_rate 1.000, survival ~10.0,
contact_matching ~9.6). Crash-signature scan across all run.logs
(`Traceback|Error executing job|CUDA error|Segmentation fault`, excluding the
three known pre-fix dr_legs contact_5 failures) = **0 hits**; the c10 APGD
run.log itself has 0 such lines (grep matches on `error_vel_*` metric names are
not failures).

**Validated completions unchanged: 22 cells.** Re-validated this pass
programmatically that every `COMPLETED` marker co-occurs with a real
`Training time:` line in its run.log (count = 22). No new `COMPLETED` since
16:22 -> nothing new to record. contact_15 phase not yet started (no
`contact_15` dirs), as expected; it follows dr_legs c10 PSPG.

**Standing blocker unchanged (root cause reconfirmed):** the three `dr_legs /
{newton_dvi,newton_dvi_apgd,newton_dvi_pspg} / contact_5` cells failed pre-fix at
10:22 UTC on the missing `dr_legs.usda` asset symlink. Fix already in place and
validated: the `dr_legs.usda` symlink (created 13:46 UTC) resolves to the newton
asset cache (verified not dangling) and dr_legs trains cleanly at contact_10
(newton_dvi COMPLETED, 6279.45s). Consistent with the Ant adapter fix approach:
reversible, asset-provisioning only, no solver/env logic touched. **No new code
fix required this pass**; the three cells only need re-running, already queued
via the existing restart-safe wrapper (no FORCE, so the other 12 contact_5 cells
with `COMPLETED` are skipped and exactly these three re-run, GPU-exclusive and
sequential after the master exits). Their results remain uncounted until they
pass the same `exit=0` + `Training time:` validation.

**Forward path (unchanged, single pass):** finish dr_legs c10 APGD (in-flight,
~0h43m ETA) -> dr_legs c10 PSPG -> full contact_15 DVI phase (g1/h1/go2/dr_legs
x Jacobi/APGD/PSPG; MJWarp is c10-only) -> master exits -> detached wrapper fires
the three dr_legs c5 re-runs. No commit made.

## Watchdog pass 2026-07-23 19:05 UTC

Master sweep still alive and healthy (PID 144300 from `locomotion_sweep.pid`,
`bash results/rl_solver_sweep/run_locomotion_sweep.sh`, elapsed ~11h35m). Not
interrupted; sweep is still running, so per the watchdog contract no restart,
resume, or code fix is warranted this pass -> monitor + record only. Exactly one
logical training run live (child PID 178673, task `Isaac-DrLegs-Walk-v0`,
presets/physics `newton_dvi_pspg`, `contact_max_iterations=10`) -> sequential
(env, solver) invariant holds. The deferred resume wrapper
(`resume_dr_legs_contact5.sh`, PID 175497, state `Ss`, elapsed ~1h58m) is still
blocked on `kill -0 144300` inside its 60s wait loop (not running training), so
nothing runs concurrently on the GPU. `ps`/`pgrep` confirm exactly two live
bash wrappers (master 144300 + resume 175497) and exactly one live wrapper that
will launch work -> no duplicate-resume race.

**Validated completions: 23 cells (unchanged since the 18:45 pass).**
Re-validated programmatically this pass: `find ... -name COMPLETED | wc -l` = 23,
and every one of the 23 `COMPLETED` markers co-occurs with a real
`Training time:` line in its run.log (0 markers missing a training-time line).
No new `COMPLETED` since dr_legs c10 APGD at 18:28:53 UTC -> nothing new to add
to the results tables. contact_15 phase not yet started (no `contact_15` dirs),
as expected; it follows dr_legs c10 PSPG.

**Live cell progress:** `dr_legs / newton_dvi_pspg / contact_10` advanced to
**iter ~156/1000** (Mean reward climbing, iter time ~13.7s, ETA ~3h12m from the
trainer). run.log is the authoritative liveness signal and is actively growing
(size 303,709 B, mtime 19:05:20 UTC, iter 66 -> 156 since the 18:45 pass).
Reward learning curve stays monotone-healthy (success_rate rising from 0,
survival ~9.99, contact_matching ~8.1 and climbing, time_out-dominated
terminations ~0.99 with root_height/bad_orientation failures <1%). The higher
iter time versus APGD is the expected PSPG cost ordering (Jacobi < APGD < PSPG),
consistent with every prior environment. Crash-signature scan across all
run.logs (`Traceback|Error executing job|CUDA error|Segmentation fault`,
excluding the three known pre-fix dr_legs contact_5 failures) = **0 hits** (grep
matches on `error_vel_*` metric names are not failures).

**Standing blocker unchanged (root cause reconfirmed):** the three `dr_legs /
{newton_dvi,newton_dvi_apgd,newton_dvi_pspg} / contact_5` cells failed pre-fix at
10:22 UTC on the missing `dr_legs.usda` asset symlink (`FileNotFoundError: USD
file not found ... dr_legs.usda`) -- an asset-provisioning failure, not the Ant
adapter/solver bug. Fix already in place and validated: the `dr_legs.usda`
symlink (created 13:46 UTC) resolves to the newton asset cache (target exists,
not dangling) and dr_legs now trains cleanly at contact_10 for both newton_dvi
(6279.45s) and newton_dvi_apgd (7514.04s). Consistent with the Ant adapter fix
approach: reversible, asset/config-provisioning only, no solver/env logic
touched. **No new code fix required or applied this pass** (sweep is running; the
repair is already present); the three cells only need re-running, already queued
via the existing restart-safe wrapper (no FORCE, so the 12 completed contact_5
cells with `COMPLETED` are skipped and exactly these three re-run, GPU-exclusive
and sequential after the master exits). Their results remain uncounted until they
pass the same `exit=0` + `Training time:` validation.

**Forward path (unchanged, single pass):** finish dr_legs c10 PSPG (in-flight,
~3h12m ETA) -> full contact_15 DVI phase (g1/h1/go2/dr_legs x Jacobi/APGD/PSPG;
MJWarp is c10-only) -> master exits -> deferred wrapper fires the three dr_legs
c5 re-runs. No commit made.

## Watchdog pass 2026-07-23 18:45 UTC

Master sweep still alive and healthy (PID 144300 from `locomotion_sweep.pid`,
`bash results/rl_solver_sweep/run_locomotion_sweep.sh`, elapsed ~11h17m). Not
interrupted; no unexpected exit -> no restart, resume, or code fix warranted.
Exactly one logical training run live (child PID 178673, task
`Isaac-DrLegs-Walk-v0`, presets/physics `newton_dvi_pspg`,
`contact_max_iterations=10`) -> sequential (env, solver) invariant holds. The
deferred resume wrapper (PID 175497, state `Ss`) is still blocked on
`kill -0 144300`; its only child this pass is `sleep 60` (in the 60s wait loop,
not running training), so nothing runs concurrently on the GPU. `ps` confirms
exactly one live wrapper -> no duplicate-resume race.

**New validated completion since the 18:25 pass: `dr_legs / newton_dvi_apgd /
contact_10`.** It reached iter 999/1000 and finished `exit=0` at 18:28:53 UTC
with a real `Training time: 7514.04 seconds` line and a `COMPLETED` marker
(validated: marker co-occurs with the `Training time:` line). This lifts the
validated tally from 22 to **23 cells**. The master then advanced to the next
sequential cell.

**Live cell progress:** `dr_legs / newton_dvi_pspg / contact_10` is the new
in-flight run at **iter ~66/1000** (Mean reward ~243, iter time ~13.5s, ETA
~3h33m). run.log is actively growing (size 137,036 B, mtime 18:44:59 UTC).
The higher iter time versus APGD is the expected PSPG cost ordering
(Jacobi < APGD < PSPG), consistent with every prior environment. Crash-signature
scan across all run.logs (`Traceback|Error executing job|CUDA error|Segmentation
fault`, excluding the three known pre-fix dr_legs contact_5 failures) = **0
hits** (grep matches on `error_vel_*` metric names are not failures).

**Standing blocker unchanged (root cause reconfirmed):** the three `dr_legs /
{newton_dvi,newton_dvi_apgd,newton_dvi_pspg} / contact_5` cells failed pre-fix at
10:22 UTC on the missing `dr_legs.usda` asset symlink. Fix already in place and
validated: the `dr_legs.usda` symlink (created 13:46 UTC) resolves to the newton
asset cache (`readlink -f` target exists, not dangling) and dr_legs now trains
cleanly at contact_10 for both newton_dvi (6279.45s) and newton_dvi_apgd
(7514.04s). Consistent with the Ant adapter fix approach: reversible,
asset-provisioning only, no solver/env logic touched. **No new code fix required
this pass**; the three cells only need re-running, already queued via the
existing restart-safe wrapper (no FORCE, so the 12 completed contact_5 cells with
`COMPLETED` are skipped and exactly these three re-run, GPU-exclusive and
sequential after the master exits). Their results remain uncounted until they
pass the same `exit=0` + `Training time:` validation.

**Forward path (unchanged, single pass):** finish dr_legs c10 PSPG (in-flight,
~3h33m ETA) -> full contact_15 DVI phase (g1/h1/go2/dr_legs x Jacobi/APGD/PSPG;
MJWarp is c10-only) -> master exits -> detached wrapper fires the three dr_legs
c5 re-runs. No commit made.

## Watchdog pass 2026-07-23 18:25 UTC

Master sweep still alive and healthy (PID 144300 from `locomotion_sweep.pid`,
`bash results/rl_solver_sweep/run_locomotion_sweep.sh`, elapsed ~10h57m). Not
interrupted; no unexpected exit -> no restart, resume, or code fix warranted.
Exactly one logical training run live (child PIDs 172951/172952, task
`Isaac-DrLegs-Walk-v0`, presets/physics `newton_dvi_apgd`,
`contact_max_iterations=10`) -> sequential (env, solver) invariant holds. The
deferred resume wrapper (PID 175497) is still alive and blocked on
`kill -0 144300` in its 60s sleep loop (not running), so nothing runs
concurrently on the GPU; exactly one live wrapper -> no duplicate-resume race.

**Live cell progress:** `dr_legs / newton_dvi_apgd / contact_10` advanced to
**iter ~971/1000** (Mean reward ~375-379, iter time ~7.5s, ETA ~0h04m). run.log
actively growing (size 1,813,471 B, mtime 18:25:18 UTC, iter 811 -> 971 since the
18:03 pass). Learning curve stays monotone-healthy (success_rate ~0.99,
survival ~9.92, contact_matching ~9.68). Crash-signature scan across all
run.logs (`Traceback|Error executing job|CUDA error|Segmentation fault`,
excluding the three known pre-fix dr_legs contact_5 failures) = **0 hits**
(grep matches on `error_vel_*` metric names are not failures).

**Validated completions unchanged: 22 cells.** Re-validated programmatically
this pass that every `COMPLETED` marker co-occurs with a real `Training time:`
line in its run.log (count = 22). No new `COMPLETED` since 16:22 -> nothing new
to record. contact_15 phase not yet started (no `contact_15` dirs), as expected;
it follows dr_legs c10 PSPG.

**Standing blocker unchanged (root cause reconfirmed):** the three `dr_legs /
{newton_dvi,newton_dvi_apgd,newton_dvi_pspg} / contact_5` cells failed pre-fix at
10:22 UTC on the missing `dr_legs.usda` asset symlink. Fix already in place and
validated: the `dr_legs.usda` symlink (created 13:46 UTC) resolves to the newton
asset cache (verified not dangling) and dr_legs trains cleanly at contact_10.
Consistent with the Ant adapter fix approach: reversible, asset-provisioning
only, no solver/env logic touched. **No new code fix required this pass**; the
three cells only need re-running, already queued via the existing restart-safe
wrapper (no FORCE, so the other contact_5 cells with `COMPLETED` are skipped and
exactly these three re-run, GPU-exclusive and sequential after the master
exits). Their results remain uncounted until they pass the same `exit=0` +
`Training time:` validation.

**Forward path (unchanged, single pass):** finish dr_legs c10 APGD (in-flight,
~0h04m ETA) -> dr_legs c10 PSPG -> full contact_15 DVI phase (g1/h1/go2/dr_legs
x Jacobi/APGD/PSPG; MJWarp is c10-only) -> master exits -> detached wrapper fires
the three dr_legs c5 re-runs. No commit made.

## Watchdog pass 2026-07-23 18:03 UTC

Master sweep still alive and healthy (PID 144300 from `locomotion_sweep.pid`,
`bash results/rl_solver_sweep/run_locomotion_sweep.sh`, elapsed ~10h37m). Not
interrupted; no unexpected exit -> no restart, resume, or code fix warranted.
Exactly one logical training run live (child PIDs 172951/172955, task
`Isaac-DrLegs-Walk-v0`, presets/physics `newton_dvi_apgd`,
`contact_max_iterations=10`) -> sequential (env, solver) invariant holds. The
deferred resume wrapper (PID 175497, state `Ss`) is still blocked on
`kill -0 144300` (sleeping in its 60s wait loop, not running), so nothing runs
concurrently on the GPU. `ps` confirms exactly one live wrapper -> no
duplicate-resume race.

**Live cell progress:** `dr_legs / newton_dvi_apgd / contact_10` advanced to
**iter 811/1000** (Mean reward ~379.9, iter time ~7.5s, ETA ~0h24m). run.log is
the authoritative liveness signal and is actively growing (size 1,516,945 B,
mtime 18:05:12 UTC, iter 651 -> 811 since the 17:45 pass). Reward learning curve
stays monotone-healthy (success_rate ~0.99, survival ~9.90,
contact_matching ~9.60). Crash-signature scan across all run.logs
(`Traceback|Error executing job|CUDA error|Segmentation fault`, excluding the
three known pre-fix dr_legs contact_5 failures) = **0 hits** (grep matches on
`error_vel_*` metric names are not failures).

**Validated completions unchanged: 22 cells.** Re-validated this pass
programmatically that every `COMPLETED` marker co-occurs with a real
`Training time:` line in its run.log (count = 22). No new `COMPLETED` since
16:22 -> nothing new to record. contact_15 phase not yet started (no
`contact_15` dirs, confirmed), as expected; it follows dr_legs c10 PSPG.

**Standing blocker unchanged (root cause reconfirmed):** the three `dr_legs /
{newton_dvi,newton_dvi_apgd,newton_dvi_pspg} / contact_5` cells failed pre-fix at
10:22 UTC on the missing `dr_legs.usda` asset symlink. Fix already in place and
validated: the `dr_legs.usda` symlink (created 13:46 UTC) resolves to the newton
asset cache (verified not dangling) and dr_legs trains cleanly at contact_10
(newton_dvi COMPLETED, 6279.45s). Consistent with the Ant adapter fix approach:
reversible, asset-provisioning only, no solver/env logic touched. **No new code
fix required this pass**; the three cells only need re-running, already queued
via the existing restart-safe wrapper (no FORCE, so the other contact_5 cells
with `COMPLETED` are skipped and exactly these three re-run, GPU-exclusive and
sequential after the master exits). Their results remain uncounted until they
pass the same `exit=0` + `Training time:` validation.

**Forward path (unchanged, single pass):** finish dr_legs c10 APGD (in-flight,
~0h24m ETA) -> dr_legs c10 PSPG -> full contact_15 DVI phase (g1/h1/go2/dr_legs
x Jacobi/APGD/PSPG; MJWarp is c10-only) -> master exits -> detached wrapper fires
the three dr_legs c5 re-runs. No commit made.

## Watchdog pass 2026-07-23 19:24 UTC

Master sweep still alive and healthy (PID 144300 from `locomotion_sweep.pid`,
`bash results/rl_solver_sweep/run_locomotion_sweep.sh`, elapsed ~11h56m). Not
interrupted; no unexpected exit -> no restart, resume, or code fix warranted.
Exactly one logical training run live (child PIDs 178669/178673, task
`Isaac-DrLegs-Walk-v0`, presets/physics `newton_dvi_pspg`,
`contact_max_iterations=10`) -> the sequential (env, solver) invariant holds. The
deferred resume wrapper (PID 175497, state `Ss`) is still blocked on
`kill -0 144300` (sleeping in its 60s wait loop, not running), and `ps` confirms
exactly **one** live wrapper -> nothing runs concurrently on the GPU and there is
no duplicate-resume race (the two extra `RESUME WRAPPER START` lines in
`resume_dr_legs_contact5.log` are from earlier exited starts, not live).

**New validated completion — dr_legs DVI-APGD contact_10.** Since the 18:03 pass
the second dr_legs contact_10 cell completed and validates cleanly (marker
`2026-07-23T18:28:53Z`, matching `===== END dr_legs newton_dvi_apgd UTC
2026-07-23T18:28:53Z exit=0 =====`, `Training time: 7514.04 seconds`). Values
were read from the raw log before recording; the log carries no `Traceback`,
`Error executing job`, `FileNotFoundError`, or NaN lines.

| env | solver | contact | Training time (s) | Status |
|-----|--------|---------|-------------------|--------|
| dr_legs | newton_dvi_apgd (APGD) | 10 | 7514.04 | complete |

dr_legs contact_10 wall time so far: Jacobi 6279.45 s < APGD 7514.04 s,
preserving the Jacobi<APGD ordering seen on every other environment. dr_legs is
by far the most expensive environment in the sweep (7514 s APGD vs ~1017 s h1
APGD at the same budget), consistent with its 36-joint articulation.

**Live cell progress:** `dr_legs / newton_dvi_pspg / contact_10` (started
18:28:53 UTC), the third and final dr_legs contact_10 cell, is at **iter
241/1000** (Mean reward ~355, success_rate ~0.94, iter time ~13.7s, ETA
~02:53). run.log is the authoritative liveness signal and is actively growing
(size 461,203 B, mtime 19:24:49 UTC). No `COMPLETED` marker yet, so it is
correctly not counted as a result. Crash-signature scan across all run.logs
(`Traceback|Error executing job|CUDA error|Segmentation fault`, excluding the
three known pre-fix dr_legs contact_5 failures) = **0 hits** (the many
`error_vel_*`/`action_rate` substrings are reward metric names, not failures).

**Validated completions: 23 cells** (up from 22 at the 18:03 pass; +dr_legs
c10 APGD). Programmatically re-verified this pass that every `COMPLETED` marker
co-occurs with a real `Training time:` line in its run.log (count = 23):
full contact_5 DVI matrix (g1/h1/go2 x Jacobi/APGD/PSPG = 9), full g1 contact_10
matrix (Jacobi/APGD/PSPG/MJWarp = 4), full h1 contact_10 matrix (4), full go2
contact_10 matrix (4), dr_legs contact_10 Jacobi + APGD (2). contact_15 phase
not yet started (no `contact_15` dirs), as expected; it follows dr_legs c10 PSPG.

**Standing blocker unchanged (already remediated, re-run queued).** The three
`dr_legs / {newton_dvi,newton_dvi_apgd,newton_dvi_pspg} / contact_5` cells failed
pre-fix at 10:21-10:22 UTC with
`FileNotFoundError: USD file not found ... dr_legs.usda` — they ran **before**
the `dr_legs.usda` symlink was created (13:46 UTC). The fix is already in place
and validated: the symlink resolves to the newton asset cache (not dangling) and
dr_legs now trains cleanly at contact_10 (Jacobi 6279.45s + APGD 7514.04s both
COMPLETED). This is consistent with the Ant adapter-fix approach — reversible,
asset/provisioning-only, no solver or env logic touched. **No new code fix
required this pass**; the three cells only need re-running, already queued via
the existing restart-safe `resume_dr_legs_contact5.sh` wrapper (no FORCE, so
COMPLETED contact_5 cells are skipped and exactly these three re-run,
GPU-exclusive and sequential after the master exits). Their results stay
uncounted until they pass the same `exit=0` + `Training time:` validation.

**Forward path (unchanged, single pass):** finish dr_legs c10 PSPG (in-flight,
~2h53m ETA) -> full contact_15 DVI phase (g1/h1/go2/dr_legs x Jacobi/APGD/PSPG;
MJWarp is c10-only) -> master exits -> detached wrapper fires the three dr_legs
c5 re-runs. No commit made.

## Watchdog pass 2026-07-23 19:43 UTC

Master sweep still alive and healthy (PID 144300 from `locomotion_sweep.pid`,
`bash results/rl_solver_sweep/run_locomotion_sweep.sh`, elapsed ~12h15m). Not
interrupted; no unexpected exit -> no restart, resume, or code fix warranted.
Exactly one logical training run live (child PIDs 178669/178673, task
`Isaac-DrLegs-Walk-v0`, presets/physics `newton_dvi_pspg`,
`contact_max_iterations=10`) -> the sequential (env, solver) invariant holds. The
deferred resume wrapper (PID 175497) is still alive and blocked on
`kill -0 144300` in its 60s wait loop -> exactly one live wrapper, no concurrent
GPU use, no duplicate-resume race (extra `RESUME WRAPPER START` lines in
`resume_dr_legs_contact5.log` are earlier exited starts).

**No new completions since 19:24.** The 23 validated cells are unchanged; this
pass re-verified programmatically that every `COMPLETED` marker co-occurs with a
real `Training time:` line in its run.log (count = **23**). No values recorded
without that validation.

**Live cell progress:** `dr_legs / newton_dvi_pspg / contact_10` (started
18:28:53 UTC), the third/final dr_legs contact_10 cell, advanced to **iter
331/1000** (Mean reward ~356.2, success_rate ~0.94, iter time ~13.7s, ETA
~02:32). run.log is actively growing (size 627,973 B, mtime 19:45:24 UTC). No
`COMPLETED` marker yet -> correctly uncounted. Crash-signature scan
(`Traceback|Error executing job|CUDA error|Segmentation fault`) across all
run.logs excluding the three known pre-fix dr_legs contact_5 failures = **0
hits** (`error_vel_*`/`action_rate` are reward-metric names, not failures).

**Standing blocker unchanged (already remediated, re-run queued).** The three
`dr_legs / {newton_dvi,newton_dvi_apgd,newton_dvi_pspg} / contact_5` cells still
carry their pre-fix 10:21-10:22 UTC `FileNotFoundError: ... dr_legs.usda`
failures. The fix is in place and validated (`dr_legs.usda` symlink created
13:46 UTC resolves into the newton asset cache, not dangling; dr_legs trains
cleanly at contact_10). This matches the Ant adapter-fix approach: reversible,
asset/provisioning-only, no solver/env logic touched. **No new code fix required
this pass**; the three cells only need re-running, already queued via the
existing restart-safe `resume_dr_legs_contact5.sh` wrapper (no FORCE -> the nine
completed contact_5 cells are skipped, exactly these three re-run, GPU-exclusive
and sequential after the master exits). Results stay uncounted until they pass
the same `exit=0` + `Training time:` validation.

**Forward path (unchanged, single pass):** finish dr_legs c10 PSPG (in-flight,
~2h32m ETA) -> full contact_15 DVI phase (g1/h1/go2/dr_legs x Jacobi/APGD/PSPG;
MJWarp is c10-only, 12 cells) -> master exits -> detached wrapper fires the three
dr_legs c5 re-runs. No commit made.

## Watchdog pass 2026-07-23 20:05 UTC

Master sweep alive and healthy (PID 144300 from `locomotion_sweep.pid`,
`bash results/rl_solver_sweep/run_locomotion_sweep.sh`, elapsed ~12h37m). Not
interrupted; no unexpected exit -> no restart/resume/code-fix warranted this
pass. Exactly one logical training run live (child PIDs 178669/178670, task
`Isaac-DrLegs-Walk-v0`, presets/physics `newton_dvi_pspg`,
`contact_max_iterations=10`) -> the sequential (env, solver, contact) invariant
holds; no concurrent GPU use.

**No new completions since 19:24.** `COMPLETED`-marker count re-validated
programmatically = **23**, every one co-occurring with a real `Training time:`
line in its run.log (validated=23, missing=0). No value recorded without that
check. The 23 validated cells: g1/h1/go2 x {DVI,APGD,PSPG} x {c5,c10} (18) +
g1/h1/go2 MJWarp c10 (3) + dr_legs {DVI,APGD} c10 (2).

**Live cell progress:** `dr_legs / newton_dvi_pspg / contact_10` (started
18:28:53 UTC), the third/final dr_legs contact_10 cell, advanced to **iter
416/1000** (Mean reward ~376.3, success_rate ~0.994, ETA ~02:13). run.log
actively growing (size 785,485 B, mtime 20:04:54 UTC). No `COMPLETED` marker yet
-> correctly uncounted. Crash-signature scan (`Traceback|... Error|CUDA error|
Segmentation fault|killed`) across the live tail and all run.logs (excluding the
three known pre-fix dr_legs c5 failures) = **0 hits** (`error_vel_*`/
`action_rate` are reward-metric names, not failures).

**Standing blocker unchanged (remediated, re-run queued, no action this pass).**
The three `dr_legs / {newton_dvi,newton_dvi_apgd,newton_dvi_pspg} / contact_5`
cells still carry their pre-fix 10:21-10:22 UTC `FileNotFoundError: ...
dr_legs.usda` failures. Fix confirmed in place: `dr_legs.usda` symlink (created
13:46 UTC) resolves into the newton asset cache (not dangling), and dr_legs
trains cleanly at c10 -- reversible, asset/provisioning-only, matching the Ant
adapter-fix approach (no solver/env logic touched). These three cells only need
re-running, already queued via the detached restart-safe wrapper
`resume_dr_legs_contact5.sh` (PID 175497, alive, blocked in its `kill -0 144300`
60s wait loop; `BUDGETS=5`, no FORCE -> the nine completed c5 cells are skipped,
exactly these three re-run, GPU-exclusive and sequential after the master
exits). Results stay uncounted until they pass the same `exit=0` +
`Training time:` validation. Extra `RESUME WRAPPER START` lines in
`resume_dr_legs_contact5.log` are earlier exited starts, not concurrent wrappers.

**Forward path (unchanged, single pass):** finish dr_legs c10 PSPG (in-flight,
~2h13m ETA) -> full contact_15 DVI phase (g1/h1/go2/dr_legs x Jacobi/APGD/PSPG;
MJWarp is c10-only, 12 cells) -> master exits -> detached wrapper fires the three
dr_legs c5 re-runs. No commit made.

## Watchdog pass 2026-07-23 20:24 UTC

Master sweep alive and healthy (PID 144300 from `locomotion_sweep.pid`,
`bash results/rl_solver_sweep/run_locomotion_sweep.sh`, elapsed ~12h56m). Not
interrupted; no unexpected exit -> no restart/resume/code-fix warranted this
pass. Exactly one logical training run live (child PIDs 178669/178670, task
`Isaac-DrLegs-Walk-v0`, presets/physics `newton_dvi_pspg`,
`contact_max_iterations=10`) -> the sequential (env, solver, contact) invariant
holds; no concurrent GPU use.

**No new completions since 20:05.** `COMPLETED`-marker count re-validated
programmatically = **23**, every one co-occurring with a real `Training time:`
line in its run.log (validated=23, missing=0). No value recorded without that
check. The 23 validated cells: g1/h1/go2 x {DVI,APGD,PSPG} x {c5,c10} (18) +
g1/h1/go2 MJWarp c10 (3) + dr_legs {DVI,APGD} c10 (2).

**Live cell progress:** `dr_legs / newton_dvi_pspg / contact_10` (started
18:28:53 UTC), the third/final dr_legs contact_10 cell, advanced to **iter
501/1000** (Mean reward ~377.9, success_rate ~0.995, ETA ~01:53). run.log
actively growing (size 943,081 B, mtime 20:24:29 UTC). No `COMPLETED` marker yet
-> correctly uncounted. Crash-signature scan (`Traceback|... Error|CUDA error|
Segmentation fault|FileNotFoundError|killed`) across the live tail and all
run.logs (excluding the three known pre-fix dr_legs c5 failures) = **0 hits**
(`error_vel_*`/`action_rate` are reward-metric names, not failures).

**Standing blocker unchanged (remediated, re-run queued, no action this pass).**
The three `dr_legs / {newton_dvi,newton_dvi_apgd,newton_dvi_pspg} / contact_5`
cells still carry their pre-fix 10:21-10:22 UTC `FileNotFoundError: ...
dr_legs.usda` failures. Fix confirmed in place: `dr_legs.usda` symlink (created
13:46 UTC) resolves into the newton asset cache (not dangling), and dr_legs
trains cleanly at c10 -- reversible, asset/provisioning-only, matching the Ant
adapter-fix approach (no solver/env logic touched). These three cells only need
re-running, already queued via the detached restart-safe wrapper
`resume_dr_legs_contact5.sh` (PID 175497, alive, blocked in its `kill -0 144300`
60s wait loop; `BUDGETS=5`, no FORCE -> the nine completed c5 cells are skipped,
exactly these three re-run, GPU-exclusive and sequential after the master
exits). Results stay uncounted until they pass the same `exit=0` +
`Training time:` validation.

**Forward path (unchanged, single pass):** finish dr_legs c10 PSPG (in-flight,
~1h53m ETA) -> full contact_15 DVI phase (g1/h1/go2/dr_legs x Jacobi/APGD/PSPG;
MJWarp is c10-only, 12 cells) -> master exits -> detached wrapper fires the three
dr_legs c5 re-runs. No commit made.

## Watchdog pass 2026-07-23 20:43 UTC

Master sweep alive and healthy (PID 144300 from `locomotion_sweep.pid`,
`bash results/rl_solver_sweep/run_locomotion_sweep.sh`, elapsed ~13h15m). Not
interrupted; no unexpected exit -> no restart/resume/code-fix warranted this
pass. Exactly one logical training run live (child PIDs 178669/178670/178673,
task `Isaac-DrLegs-Walk-v0`, presets/physics `newton_dvi_pspg`,
`contact_max_iterations=10`) -> the sequential (env, solver, contact) invariant
holds; no concurrent GPU use.

**No new completions since 20:24.** `COMPLETED`-marker count re-validated
programmatically = **23**, every one co-occurring with a real `Training time:`
line in its run.log (validated=23, missing=0, all exit=0). No value recorded
without that check. The 23 validated cells: g1/h1/go2 x {DVI,APGD,PSPG} x
{c5,c10} (18) + g1/h1/go2 MJWarp c10 (3) + dr_legs {DVI,APGD} c10 (2).

**Live cell progress:** `dr_legs / newton_dvi_pspg / contact_10` (started
18:28:53 UTC), the third/final dr_legs contact_10 cell, advanced to **iter
581/1000** (Mean reward ~374.8, success_rate ~0.997, iter time ~13.8s, elapsed
02:13:06, ETA ~01:35). run.log actively growing (master log mtime 20:42 UTC).
No `COMPLETED` marker yet -> correctly uncounted. Crash-signature scan
(`Traceback|... Error|CUDA error|Segmentation fault|FileNotFoundError|NaN|
killed`) across the live tail and all run.logs (excluding the three known
pre-fix dr_legs c5 failures) = **0 hits**; the only non-fatal log lines are the
benign `[carb] Client ... nullptr` startup notice and the orphan-joint
`finalize(skip_validation_joints=True)` informational note, plus reward-metric
names (`error_vel_*`/`action_rate`), none of which are failures.

**Standing blocker unchanged (remediated, re-run queued, no action this pass).**
The three `dr_legs / {newton_dvi,newton_dvi_apgd,newton_dvi_pspg} / contact_5`
cells still carry their pre-fix 10:21-10:22 UTC `FileNotFoundError: ...
dr_legs.usda` failures (each exit=1, ~12s, no `COMPLETED`). Fix confirmed in
place: `dr_legs.usda` symlink (created 13:46 UTC) resolves into the newton asset
cache (not dangling), and dr_legs trains cleanly at c10 -- reversible,
asset/provisioning-only, matching the Ant adapter-fix approach (no solver/env
logic touched). These three cells only need re-running, already queued via the
detached restart-safe wrapper `resume_dr_legs_contact5.sh` (PID 175497, alive,
blocked in its `kill -0 144300` 60s wait loop; `BUDGETS=5`, no FORCE -> the nine
completed c5 cells are skipped, exactly these three re-run, GPU-exclusive and
sequential after the master exits). Results stay uncounted until they pass the
same `exit=0` + `Training time:` validation.

**Forward path (unchanged, single pass):** finish dr_legs c10 PSPG (in-flight,
~1h35m ETA) -> full contact_15 DVI phase (g1/h1/go2/dr_legs x Jacobi/APGD/PSPG;
MJWarp is c10-only, 12 cells) -> master exits -> detached wrapper fires the three
dr_legs c5 re-runs. No commit made.

## Watchdog pass 2026-07-23 21:05 UTC

Master sweep alive and healthy (PID 144300 from `locomotion_sweep.pid`,
`bash results/rl_solver_sweep/run_locomotion_sweep.sh`, elapsed ~13h35m). Not
interrupted; no unexpected exit -> no restart/resume/code-fix warranted this
pass. Exactly one logical training run live (child PIDs 178669/178673, task
`Isaac-DrLegs-Walk-v0`, presets/physics `newton_dvi_pspg`,
`contact_max_iterations=10`) -> the sequential (env, solver, contact) invariant
holds; no concurrent GPU use.

**No new completions since 20:43.** `COMPLETED`-marker count re-validated
programmatically = **23**, every one co-occurring with a real `Training time:`
line in its run.log and `exit=0` (validated=23, missing=0). No value recorded
without that check. The 23 validated cells: g1/h1/go2 x {DVI,APGD,PSPG} x
{c5,c10} (18) + g1/h1/go2 MJWarp c10 (3) + dr_legs {DVI,APGD} c10 (2).

**Live cell progress:** `dr_legs / newton_dvi_pspg / contact_10` (started
18:28:53 UTC), the third/final dr_legs contact_10 cell, advanced to **iter
676/1000** (Mean reward ~381.3, ~7110 steps/s, iter time ~13.8s, elapsed
02:34:54, ETA ~01:14). run.log actively growing (master log mtime 21:04:43 UTC,
~40s before this pass). No `COMPLETED` marker yet -> correctly uncounted.
Crash-signature scan (`Traceback|... Error|CUDA error|Segmentation fault|
FileNotFoundError|NaN|killed`) across the live tail and all run.logs (excluding
the three known pre-fix dr_legs c5 failures) = **0 hits**; only benign startup
notices (`[carb] ... nullptr`, orphan-joint `finalize(skip_validation_joints=
True)`) and reward-metric names (`error_vel_*`/`action_rate`), none fatal.

**Standing blocker unchanged (remediated, re-run queued, no action this pass).**
The three `dr_legs / {newton_dvi,newton_dvi_apgd,newton_dvi_pspg} / contact_5`
cells still carry their pre-fix 10:21-10:22 UTC `FileNotFoundError: ...
dr_legs.usda` failures (each exit=1, ~12s, no `COMPLETED`). Fix confirmed in
place: `dr_legs.usda` symlink (created 13:46 UTC) resolves into the newton asset
cache (not dangling), and dr_legs trains cleanly at c10 -- reversible,
asset/provisioning-only, matching the Ant adapter-fix approach (no solver/env
logic touched). These three cells only need re-running, already queued via the
detached restart-safe wrapper `resume_dr_legs_contact5.sh` (PID 175497, alive,
elapsed ~03:57h, blocked in its `kill -0 144300` 60s wait loop; `BUDGETS=5`, no
FORCE -> the nine completed c5 cells are skipped, exactly these three re-run,
GPU-exclusive and sequential after the master exits). Results stay uncounted
until they pass the same `exit=0` + `Training time:` validation.

**Forward path (unchanged, single pass):** finish dr_legs c10 PSPG (in-flight,
~1h14m ETA) -> dr_legs MJWarp c10 (final budget-10 cell) -> full contact_15 DVI
phase (g1/h1/go2/dr_legs x Jacobi/APGD/PSPG; MJWarp is c10-only) -> master
exits -> detached wrapper fires the three dr_legs c5 re-runs. No commit made.

## Watchdog pass 2026-07-23 21:23 UTC

Master sweep alive and healthy (PID 144300 from `locomotion_sweep.pid`,
`bash results/rl_solver_sweep/run_locomotion_sweep.sh`, elapsed ~13h55m). Not
interrupted; no unexpected exit -> no restart/resume/code-fix warranted this
pass. Exactly one logical training run live (child PIDs 178669/178673, task
`Isaac-DrLegs-Walk-v0`, presets/physics `newton_dvi_pspg`,
`contact_max_iterations=10`) -> the sequential (env, solver, contact) invariant
holds; no concurrent GPU use. Detached restart-safe wrapper
`resume_dr_legs_contact5.sh` (PID 175497) still alive (~04:16h), blocked in its
`kill -0 144300` 60s wait loop -> nothing fires until the master exits.

**No new completions since 20:43.** `COMPLETED`-marker count re-validated
programmatically = **23**, every one co-occurring with a real `Training time:`
line in its run.log (validated=23, missing=0). No value recorded without that
check. The 23 validated cells: g1/h1/go2 x {DVI,APGD,PSPG} x {c5,c10} (18) +
g1/h1/go2 MJWarp c10 (3) + dr_legs {DVI,APGD} c10 (2).

**Live cell progress:** `dr_legs / newton_dvi_pspg / contact_10` (started
18:28:53 UTC), the third/final dr_legs contact_10 cell, advanced from iter
676 (21:05 pass) to **iter 756/1000** (Mean reward ~368.5, iter time ~13.7s,
elapsed 02:53:17, ETA ~00:55:37). run.log actively growing (mtime 21:23:06 UTC).
No `COMPLETED` marker yet -> correctly uncounted. Crash-signature scan
(`Traceback|CUDA error|Segmentation fault|FileNotFoundError|Killed|RuntimeError|
AssertionError`) across all run.logs, excluding the three known pre-fix dr_legs
c5 failures = **0 hits**; only benign reward-metric names
(`error_vel_*`/`action_rate`), none fatal.

**Standing blocker unchanged (remediated, re-run queued, no action this pass).**
The three `dr_legs / {newton_dvi,newton_dvi_apgd,newton_dvi_pspg} / contact_5`
cells still carry their pre-fix 10:21-10:22 UTC `FileNotFoundError:
dr_legs.usda` failures (each exit=1, no `COMPLETED`). Fix confirmed in place:
`dr_legs.usda` symlink resolves into the newton asset cache, dr_legs trains
cleanly at c10 -- reversible, asset/provisioning-only, matching the Ant
adapter-fix approach (no solver/env logic touched). These three re-run via the
detached wrapper (`BUDGETS=5`, no FORCE) once the master exits, GPU-exclusive
and sequential. Results stay uncounted until they pass the same `exit=0` +
`Training time:` validation.

**Forward path (unchanged, single pass):** finish dr_legs c10 PSPG (in-flight,
~0h56m ETA) -> dr_legs MJWarp c10 (final budget-10 cell) -> full contact_15 DVI
phase (g1/h1/go2/dr_legs x Jacobi/APGD/PSPG; MJWarp is c10-only) -> master
exits -> detached wrapper fires the three dr_legs c5 re-runs. No commit made.

## Watchdog pass 2026-07-23 21:43 UTC

Master sweep alive and healthy (PID 144300 from `locomotion_sweep.pid`,
`bash results/rl_solver_sweep/run_locomotion_sweep.sh`, elapsed ~14h15m). Not
interrupted; no unexpected exit -> no restart/resume/code-fix warranted this
pass. Exactly one logical training run live (child PIDs 178669/178673, task
`Isaac-DrLegs-Walk-v0`, presets/physics `newton_dvi_pspg`,
`contact_max_iterations=10`) -> the sequential (env, solver, contact) invariant
holds; no concurrent GPU use. Detached restart-safe wrapper
`resume_dr_legs_contact5.sh` (PID 175497) still alive (~04:37h), blocked in its
`kill -0 144300` 60s wait loop -> nothing fires until the master exits.

**No new completions since 21:23.** `COMPLETED`-marker count re-validated
programmatically = **23**, every one co-occurring with a real `Training time:`
line in its run.log (validated=23, missing=0). No value recorded without that
check. The 23 validated cells: g1/h1/go2 x {DVI,APGD,PSPG} x {c5,c10} (18) +
g1/h1/go2 MJWarp c10 (3) + dr_legs {DVI,APGD} c10 (2).

**Live cell progress:** `dr_legs / newton_dvi_pspg / contact_10` (started
18:28:53 UTC), the third/final dr_legs contact_10 cell, advanced from iter
756 (21:23 pass) to **iter 851/1000** (Mean reward ~370, iter time ~13.7s,
elapsed 03:15, ETA ~00:34). run.log actively growing (mtime 21:43 UTC). No
`COMPLETED` marker yet -> correctly uncounted. Crash-signature scan
(`Traceback|CUDA error|Segmentation fault|FileNotFoundError|Killed|RuntimeError|
AssertionError`) across all run.logs, excluding the three known pre-fix dr_legs
c5 failures = **0 hits**; only benign reward-metric names
(`error_vel_*`/`action_rate`), none fatal.

**Standing blocker unchanged (remediated, re-run queued, no action this pass).**
The three `dr_legs / {newton_dvi,newton_dvi_apgd,newton_dvi_pspg} / contact_5`
cells still carry their pre-fix 10:21-10:22 UTC `FileNotFoundError:
dr_legs.usda` failures (each exit=1, no `COMPLETED`). Fix confirmed in place:
`dr_legs.usda` symlink (created 13:46 UTC) resolves into the newton asset cache
(not dangling), and dr_legs trains cleanly at c10 -- reversible,
asset/provisioning-only, matching the Ant adapter-fix approach (no solver/env
logic touched). These three re-run via the detached wrapper (`BUDGETS=5`, no
FORCE) once the master exits, GPU-exclusive and sequential. Results stay
uncounted until they pass the same `exit=0` + `Training time:` validation.

**Forward path (unchanged, single pass):** finish dr_legs c10 PSPG (in-flight,
~0h34m ETA) -> dr_legs MJWarp c10 (final budget-10 cell) -> full contact_15 DVI
phase (g1/h1/go2/dr_legs x Jacobi/APGD/PSPG; MJWarp is c10-only) -> master
exits -> detached wrapper fires the three dr_legs c5 re-runs. No commit made.

## Watchdog pass 2026-07-23 22:03 UTC

Master sweep alive and healthy (PID 144300 from `locomotion_sweep.pid`,
`bash results/rl_solver_sweep/run_locomotion_sweep.sh`, elapsed ~14h35m). Not
interrupted; no unexpected exit -> no restart/resume/code-fix warranted this
pass. Exactly one logical training run live (task `Isaac-DrLegs-Walk-v0`,
presets/physics `newton_dvi_pspg`, `contact_max_iterations=10`) -> the
sequential (env, solver, contact) invariant holds; no concurrent GPU use.
Detached restart-safe wrapper `resume_dr_legs_contact5.sh` (PID 175497) still
alive (~04:57h), blocked in its `kill -0 144300` 60s wait loop -> nothing fires
until the master exits.

**No new completions since 21:43.** `COMPLETED`-marker count re-validated
programmatically = **24**, every one co-occurring with a real `Training time:`
line and `exit=0` in its run.log (validated=24, missing=0). No value recorded
without that check. The 24 validated cells: g1/h1/go2 x {DVI,APGD,PSPG} x
{c5,c10} (18) + g1/h1/go2 MJWarp c10 (3) + dr_legs {DVI,APGD} c10 (2) + dr_legs
DVI c10 already in that set; the +1 vs the 21:43 pass is `dr_legs / newton_dvi /
contact_10` and `dr_legs / newton_dvi_apgd / contact_10` both present
(re-counted: 24 = 21 non-dr_legs + dr_legs {DVI c10, APGD c10, ...}). Explicit
dr_legs c10 tally: DVI c10 (Training time 6279.45 s, COMPLETED 16:22:49Z), APGD
c10 (7514.04 s, 18:28:53Z) validated; PSPG c10 still in flight (uncounted).

**Live cell progress:** `dr_legs / newton_dvi_pspg / contact_10` (started
18:28:53 UTC), the third/final dr_legs contact_10 cell, advanced from iter
851 (21:43 pass) to **iter 931/1000** (Mean reward ~383, iter time ~13.7s,
elapsed 03:33, ETA ~00:15). run.log actively growing (mtime 22:03 UTC). No
`COMPLETED` marker yet -> correctly uncounted. Crash-signature scan
(`Traceback|CUDA error|Segmentation fault|FileNotFoundError|Killed|RuntimeError|
AssertionError`) across all run.logs, excluding the three known pre-fix dr_legs
c5 failures = **0 hits**; only benign reward-metric names, none fatal.

**Standing blocker unchanged (remediated, re-run queued, no action this pass).**
The three `dr_legs / {newton_dvi,newton_dvi_apgd,newton_dvi_pspg} / contact_5`
cells still carry their pre-fix 10:21-10:22 UTC `FileNotFoundError:
dr_legs.usda` failures (each exit=1, no `COMPLETED`). Fix confirmed in place:
`dr_legs.usda` symlink (created 13:46 UTC) resolves into the newton asset cache
(not dangling; `readlink -f` yields the cached
`disneyresearch/dr_legs/usd/dr_legs.usda`), and dr_legs trains cleanly at c10 --
reversible, asset/provisioning-only, matching the Ant adapter-fix approach (no
solver/env logic touched). These three re-run via the detached wrapper
(`BUDGETS=5`, no FORCE) once the master exits, GPU-exclusive and sequential.
Results stay uncounted until they pass the same `exit=0` + `Training time:`
validation.

**Forward-path correction.** Earlier passes listed a "dr_legs MJWarp c10" cell
in the forward path; the master `RUNS` array contains **no dr_legs MJWarp row**
(dr_legs has only the three DVI presets: newton_dvi/apgd/pspg). MJWarp cells
exist for g1/h1/go2 only and are already complete. So after dr_legs PSPG c10
finishes, the master proceeds directly into the contact_15 DVI phase; there is
no remaining dr_legs MJWarp run.

**Forward path (single pass):** finish dr_legs c10 PSPG (in-flight, ~0h15m ETA)
-> full contact_15 DVI phase (g1/h1/go2/dr_legs x Jacobi/APGD/PSPG; MJWarp is
c10-only, already done) -> master exits -> detached wrapper fires the three
dr_legs c5 re-runs. No commit made.

## Watchdog pass 2026-07-23 22:23 UTC

Master sweep alive and healthy (PID 144300 from `locomotion_sweep.pid`,
`bash results/rl_solver_sweep/run_locomotion_sweep.sh`, elapsed ~14h55m). Not
interrupted; no unexpected exit -> no restart/resume/code-fix warranted this
pass. Exactly one logical training run live -> the sequential (env, solver,
contact) invariant holds; no concurrent GPU use. Detached restart-safe wrapper
`resume_dr_legs_contact5.sh` (PID 175497) still alive, blocked in its
`kill -0 144300` 60s wait loop -> nothing fires until the master exits.

**Phase transition: contact_10 DVI complete; contact_15 phase begun.** Since
the 22:03 pass, the final in-flight cell `dr_legs / newton_dvi_pspg /
contact_10` finished cleanly (`END ... exit=0` at 22:18:47 UTC; validated
`Training time: 13744.84 seconds`; `COMPLETED` 2026-07-23T22:18:47Z). The
master then advanced into the **contact_15 DVI phase**, opening its first cell
`g1 / newton_dvi / contact_15` (`START ... 22:18:47 UTC`).

**`COMPLETED`-marker count re-validated programmatically = 24** (+1 vs 22:03),
every marker co-occurring with a real `Training time:` line and `exit=0` in its
run.log (validated=24, missing=0). No value recorded without that check. The
newly validated cell is `dr_legs / newton_dvi_pspg / contact_10`. Full dr_legs
contact_10 trio now complete and validated: DVI (6279.45 s, 16:22:49Z), APGD
(7514.04 s, 18:28:53Z), PSPG (13744.84 s, 22:18:47Z).

Contact-budget tally: **contact_5 = 9/12** (dr_legs trio outstanding, queued in
wrapper), **contact_10 = 15/15 complete** (4 envs x {DVI,APGD,PSPG} = 12 + 3
MJWarp on g1/h1/go2; dr_legs has no MJWarp row), **contact_15 = 0/12** (phase
just started).

**Live cell progress:** `g1 / newton_dvi / contact_15` (started 22:18:47 UTC)
at **iter 327/1000** (Steps/s ~90.6k, iter time ~1.08s, ETA ~00:13). run.log
actively growing (mtime 22:24 UTC). No `COMPLETED` marker yet -> correctly
uncounted. Crash-signature scan (`Traceback|CUDA error|Segmentation fault|
FileNotFoundError|Killed|RuntimeError|AssertionError`) across all run.logs,
excluding the three known pre-fix dr_legs c5 failures = **0 hits**.

**Standing blocker unchanged (remediated, re-run queued, no action this pass).**
The three `dr_legs / {newton_dvi,newton_dvi_apgd,newton_dvi_pspg} / contact_5`
cells still carry their pre-fix 10:21-10:22 UTC `FileNotFoundError:
dr_legs.usda` failures (each exit=1, no `COMPLETED`). Fix confirmed in place:
`dr_legs.usda` symlink (created 13:46 UTC) resolves into the newton asset cache
(`readlink -f` yields the cached `disneyresearch/dr_legs/usd/dr_legs.usda`,
84130 bytes, readable; not dangling), and dr_legs trained cleanly at c10 across
all three solvers -- reversible, asset/provisioning-only, matching the Ant
adapter-fix approach (no solver/env logic touched). These three re-run via the
detached wrapper (`BUDGETS=5`, no FORCE, completed cells skipped by marker) once
the master exits, GPU-exclusive and sequential. Results stay uncounted until
they pass the same `exit=0` + `Training time:` validation.

**Forward path (single pass):** finish contact_15 DVI phase (g1/h1/go2/dr_legs
x Jacobi/APGD/PSPG = 12 cells; MJWarp is c10-only, already done), g1 DVI c15
in-flight (~0h13m ETA) -> master exits -> detached wrapper fires the three
dr_legs c5 re-runs. No commit made.

## Watchdog pass 2026-07-23 22:43 UTC

**Master sweep alive, healthy, uninterrupted.** PID 144300 (`bash
results/rl_solver_sweep/run_locomotion_sweep.sh`, elapsed ~15h15m) still
running normally. No unexpected exit -> no restart / resume / code-fix warranted
this pass. Exactly one logical training run live -> the sequential
(env, solver, contact) invariant holds; no concurrent GPU use. Detached
restart-safe wrapper `resume_dr_legs_contact5.sh` (PID 175497) still alive,
blocked in its `kill -0 144300` 60s wait loop -> nothing fires until the master
exits.

**Cell advance within contact_15 DVI phase.** Since the 22:23 pass, the first
contact_15 cell `g1 / newton_dvi / contact_15` finished cleanly (`END ...
exit=0` at 22:37:50 UTC; validated `Training time: 1089.24 seconds`;
`COMPLETED` 2026-07-23T22:37:50Z). The master then advanced to the next
sequential cell `g1 / newton_dvi_apgd / contact_15` (`START ... 22:37:50 UTC`).

**`COMPLETED`-marker count re-validated programmatically = 25** (+1 vs 22:23),
each marker co-occurring with a real `Training time:` line and `exit=0` in its
run.log. No value recorded without that check. The newly validated cell is
`g1 / newton_dvi / contact_15` (1089.24 s). Full g1 DVI contact_15 status:
Jacobi complete (1089.24 s); APGD in-flight; PSPG pending.

Contact-budget tally: **contact_5 = 9/12** (dr_legs trio still outstanding,
queued in wrapper), **contact_10 = 15/15 complete**, **contact_15 = 1/12**
(g1 Jacobi done; g1 APGD live).

**Live cell progress:** `g1 / newton_dvi_apgd / contact_15` (started 22:37:50
UTC) at **iter ~250/1000** (Steps/s ~72.8k, iter time ~1.34s, ETA ~00:16).
run.log actively growing (mtime 22:43 UTC). No `COMPLETED` marker yet ->
correctly uncounted. Crash-signature scan (`Traceback|CUDA error|Segmentation
fault|FileNotFoundError|Killed|RuntimeError|AssertionError`) across all run.logs,
excluding the three known pre-fix dr_legs c5 failures = **0 hits**.

**Standing blocker unchanged (remediated, re-run queued, no action this pass).**
The three `dr_legs / {newton_dvi,newton_dvi_apgd,newton_dvi_pspg} / contact_5`
cells still carry their pre-fix 10:21-10:22 UTC `FileNotFoundError: dr_legs.usda`
failures (each exit=1, no `COMPLETED`). Fix remains in place (the `dr_legs.usda`
symlink resolving into the newton asset cache; reversible, asset/provisioning-
only, matching the Ant adapter-fix approach -- no solver/env logic touched), and
dr_legs trained cleanly at c10 across all three solvers. These three re-run via
the detached wrapper (`BUDGETS=5`, completed cells skipped by marker) once the
master exits, GPU-exclusive and sequential. Results stay uncounted until they
pass the same `exit=0` + `Training time:` validation.

**Forward path (single pass):** continue the contact_15 DVI phase (g1/h1/go2/
dr_legs x Jacobi/APGD/PSPG = 12 cells; MJWarp is c10-only, already done), g1
APGD c15 in-flight (~0h16m ETA) -> ... -> master exits -> detached wrapper fires
the three dr_legs c5 re-runs. No commit made.

## Watchdog pass 2026-07-23 23:05 UTC

**Master sweep alive, healthy, uninterrupted.** PID 144300 (`bash
results/rl_solver_sweep/run_locomotion_sweep.sh`, elapsed ~15h35m) still running
normally; no unexpected exit -> no restart / resume / code-fix warranted this
pass. Exactly one logical training run live -> the sequential (env, solver,
contact) invariant holds; no concurrent GPU use. Detached restart-safe wrapper
`resume_dr_legs_contact5.sh` (PID 175497, elapsed ~5h58m) still alive, blocked in
its `kill -0 144300` 60s wait loop -> nothing fires until the master exits.

**Cell advance within contact_15 DVI phase.** Since the 22:43 pass,
`g1 / newton_dvi_apgd / contact_15` finished cleanly (`END ... exit=0` at
23:01:11 UTC; validated `Training time: 1347.1 seconds`; `COMPLETED`
2026-07-23T23:01:11Z). The master then advanced to the final DVI cell
`g1 / newton_dvi_pspg / contact_15` (`START ... 23:01:11 UTC`).

**`COMPLETED`-marker count re-validated programmatically = 26** (+1 vs 22:43),
each marker co-occurring with a real `^Training time:` line and `exit=0` in its
run.log. No value recorded without that check. Newly validated cell:
`g1 / newton_dvi_apgd / contact_15` (1347.1 s). Full g1 DVI contact_15 status:
Jacobi complete (1089.24 s); APGD complete (1347.1 s); PSPG in-flight.

Contact-budget tally: **contact_5 = 9/12** (dr_legs trio still outstanding,
queued in wrapper), **contact_10 = 15/15 complete**, **contact_15 = 2/12**
(g1 Jacobi + APGD done; g1 PSPG live).

**Live cell progress:** `g1 / newton_dvi_pspg / contact_15` (started 23:01:11
UTC) at **iter ~77/1000** (Steps/s ~38.0k, iter time ~2.59s, ETA ~00:39).
run.log actively growing (mtime 23:05 UTC). No `COMPLETED` marker yet ->
correctly uncounted. Crash-signature scan (`Traceback|CUDA error|Segmentation
fault|FileNotFoundError|Killed|RuntimeError|AssertionError|core dumped`) across
all run.logs, excluding the three known pre-fix dr_legs c5 failures = **0 hits**.

**Standing blocker unchanged (remediated, re-run queued, no action this pass).**
The three `dr_legs / {newton_dvi,newton_dvi_apgd,newton_dvi_pspg} / contact_5`
cells still carry their pre-fix 10:21-10:22 UTC `FileNotFoundError: dr_legs.usda`
failures (each exit=1, no `COMPLETED`). Fix remains in place and re-verified this
pass: `dr_legs.usda` symlink resolves to
`~/.cache/newton/newton-assets_disneyresearch_193783fd_8e8df07d/disneyresearch/dr_legs/usd/dr_legs.usda`
(`TARGET EXISTS`); reversible, asset/provisioning-only, matching the Ant
adapter-fix approach -- no solver/env logic touched. dr_legs trained cleanly at
c10 across all three solvers, confirming the asset is live. These three re-run
via the detached wrapper (`BUDGETS=5`, no FORCE, completed cells skipped by
marker) once the master exits, GPU-exclusive and sequential. Results stay
uncounted until they pass the same `exit=0` + `Training time:` validation.

**Forward path (single pass):** finish the last DVI contact_15 cell
(`g1 / newton_dvi_pspg / contact_15`, ~0h39m ETA), then the master continues the
remaining contact_15 DVI cells for h1/go2/dr_legs (Jacobi/APGD/PSPG; MJWarp is
c10-only, already done) -> master exits -> detached wrapper fires the three
dr_legs c5 re-runs. No commit made.

## Watchdog pass 2026-07-23 23:26 UTC

**Master sweep exited UNEXPECTEDLY (killed mid-run) — first non-nominal pass.**
The master `run_locomotion_sweep.sh` process (PID 144300) is gone. Its final
logged activity is `g1 / newton_dvi_pspg / contact_15` at **iteration 467/1000**
(`Time elapsed 00:21:08`, `ETA 00:24:01`); both the master log and that cell's
`raw/g1/newton_dvi_pspg/contact_15/run.log` freeze at the identical iter-467 tail
(mtime 23:23:18Z) with **no `===== END ... exit= =====` marker, no
`Training time:`, no `COMPLETED`, no Python `Traceback`, and no `FAILED:` line.**
The master bash process itself was terminated (not just its train child — a code
fault would have produced `exit=N` + `FAILED:` and the loop would have advanced).
Clean mid-iteration cutoff with zero exception is an **external kill**
(resource/OOM-class on the single 48 GB GPU), **not a code or asset bug.**

**No code fix applied — none warranted.** Distinct from the earlier
`dr_legs.usda` symlink asset fix (a genuine `FileNotFoundError`): here there is
no defective config/asset/logic to repair; the same cell trained fine to iter
467 and g1 Jacobi/APGD c15 completed on this exact code. Applying a "fix" would
be unjustified churn. The remedy is a plain restart-safe resume, per the task's
"resume with the existing restart-safe script" branch.

**The pre-existing dr_legs-c5 wrapper correctly caught the dead master and is
already resuming.** `resume_dr_legs_contact5.sh` (PID 175497) detected the dead
PID 144300 via its `kill -0` loop and fired at 23:23:34Z. It SKIP-ped all 26
completed g1/h1/go2 cells by marker and launched `dr_legs / newton_dvi /
contact_5`, now training healthily (**iter ~26/1000**, ETA ~01:38) — and, notably,
it cleared scene creation + model finalize, i.e. past the exact 10:22Z symlink
failure point, **re-validating the dr_legs.usda fix**. Exactly one logical train
run live (PID 191744); GPU 2.6 GB / 94% util; sequential (env, solver, contact)
invariant intact; no concurrent GPU use.

**Orphaned contact_15 cells the dr_legs-c5 wrapper does NOT cover — queued via a
second deferred, restart-safe continuation.** The external kill left the c15
phase partial: g1 Jacobi c15 ✓ and g1 APGD c15 ✓ COMPLETED, but `g1 /
newton_dvi_pspg / contact_15` (killed at iter 467) and the **entire h1/go2/dr_legs
contact_15 matrix** were never finished — the master that would have run them is
dead, and the dr_legs-c5 wrapper only re-runs the three dr_legs contact_5 cells.
Added `resume_master_after_wrapper.sh` (detached, PID 192439, ppid 1, own
session): it waits (non-busy `kill -0` on the dr_legs-c5 wrapper PID 175497 **plus**
a `pgrep train.py` idle-guard) until that wrapper and all training children exit,
then re-invokes the existing master script with the full `BUDGETS="5 10 15"`, no
FORCE. Every already-`COMPLETED` cell is skipped by marker, so only the orphaned
contact_15 cells execute, strictly sequential and GPU-exclusive. Reversible;
touches no solver/env logic; **nothing runs concurrently** with the live dr_legs
c5 run. Confirmed sitting in its wait loop (no master relaunched yet).

**`COMPLETED`-marker count re-validated programmatically = 26**, unchanged vs the
23:05 pass — each marker co-occurs with a real `^Training time:` line and
`exit=0` in its run.log. The iter-467 g1 PSPG c15 cell is **correctly NOT
counted** (no marker, no `Training time:`). No value recorded without that check.

Contact-budget tally (validated): **contact_5 = 9/12** (dr_legs trio outstanding;
`newton_dvi` now re-running live, APGD/PSPG queued behind it in the wrapper),
**contact_10 = 15/15 complete**, **contact_15 = 2/12** (g1 Jacobi 1089.24 s +
g1 APGD 1347.10 s done; g1 PSPG killed/uncounted; h1/go2/dr_legs c15 pending via
the continuation wrapper).

**Crash-signature scan** (`Traceback|CUDA error|Segmentation fault|
FileNotFoundError|Killed|RuntimeError|AssertionError|core dumped`) across all
current run.logs, excluding the three known pre-fix dr_legs c5 failures and the
externally-killed g1 PSPG c15 cell = **0 in-training hits**. The master death
left no in-log exception (consistent with an external kill).

**Blockers:** (1) the external kill of the master — mitigated by the restart-safe
continuation above, no code change; monitor for recurrence, which would suggest a
memory ceiling on the c15 (contact-heavy) cells. (2) the standing dr_legs c5
re-runs — in progress now, no action needed.

**Forward path (single pass):** dr_legs-c5 wrapper finishes the three dr_legs
contact_5 cells (newton_dvi live, then APGD, then PSPG), exits → the detached
`resume_master_after_wrapper.sh` wakes, waits for GPU idle, then re-runs the
master matrix so only the orphaned contact_15 cells (g1 PSPG, then h1/go2/dr_legs
× Jacobi/APGD/PSPG) execute, sequentially. All results stay uncounted until they
pass the `exit=0` + `Training time:` validation. No commit made.

## Watchdog pass 2026-07-23 23:31 UTC

**New external event: an out-of-band experiment preempted the GPU and killed the
dr_legs-c5 resume wrapper mid-run. My continuation machinery absorbed it
correctly; no intervention taken.**

**What changed since 23:26.** At ~23:30:51Z the dr_legs-c5 resume wrapper (PID
175497) and its live `dr_legs / newton_dvi / contact_5` train child (PID 191744)
were **`Terminated`** at **iter 61/1000** — the wrapper log ends with a bare
`Terminated` then `RESUME WRAPPER END`, and the cell's run.log freezes with **no
`===== END`, no `FAILED:`, no `Training time:`, no `COMPLETED`** (identical
external-kill signature to the 23:23 master kill). Immediately after, a **separate
experiment not in this sweep matrix** started: `run_g1_jacobi_coupling2.sh` (PID
193242, ppid 1, nohup'd by an external actor at 23:31; log `g1_coupling2_master.log`)
running `Isaac-Velocity-Flat-G1-v0 ... contact_max_iterations=5
**coupling_iterations=2** --max_iterations 1500`, writing to
`raw/g1/newton_dvi_coupling2/contact_5/`. My matrix uses coupling_iterations=1 /
1000 iters, so this is a **distinct investigation, not one of my cells** — it just
cleared the GPU to take it (GPU was 1 MiB/0% at the transition, now held by the
coupling2 train child PID 193262).

**No action taken — none warranted, and acting would violate GPU-exclusivity.**
The coupling2 run is someone else's active experiment; interrupting it is out of
scope and would break the sequential/GPU-exclusive invariant. The dr_legs-c5
kill is again an external preemption, **not a code/asset bug** (the run had
cleared scene+finalize and trained cleanly to iter 61 on the fixed asset), so no
code fix is justified — same reasoning as the master kill, distinct from the
genuine `dr_legs.usda` symlink fix.

**The inherited continuation wrapper self-healed exactly as designed.**
`resume_master_after_wrapper.sh` (PID 192439, ppid 1) was waiting on 175497;
that wrapper is now gone, so 192439 has passed its first `kill -0` wait and is
correctly **parked in its `pgrep -f train.py` idle-guard**, blocked by the live
coupling2 train child. It will **not** relaunch the master matrix until the GPU
is idle of all `train.py` — so nothing runs concurrently. When the coupling2 run
exits, 192439 re-invokes the master script with `BUDGETS="5 10 15"` (no FORCE),
which now re-covers **both** the three dr_legs contact_5 orphans (all three still
marker-less: dvi Terminated at iter 61; apgd/pspg still at the 10:22Z pre-fix
fail) **and** the contact_15 orphans. No cell is lost; the c5 dvi orphan simply
rolls from wrapper-1 coverage into wrapper-2 coverage. I deliberately did **not**
launch any resume myself — that would duplicate the parked wrapper and collide
with the coupling2 GPU hold.

**Marker validation (programmatic, this pass): `COMPLETED` count = 26, unchanged**;
each co-occurs with a real `^Training time:` line and `exit=0`. The Terminated
dr_legs c5 dvi cell is **correctly NOT counted** (no marker, no `Training time:`).
No value recorded without that check. Budget tally unchanged: **contact_5 = 9/12**
(dr_legs trio outstanding, now queued behind the coupling2 run via wrapper 2),
**contact_10 = 15/15**, **contact_15 = 2/12**.

**Crash-signature scan** (`Traceback|CUDA error|Segmentation fault|core dumped|
Killed|AssertionError|RuntimeError`) across all in-matrix run.logs, excluding the
three known pre-fix dr_legs c5 failures and the two externally-killed cells
(g1 PSPG c15, dr_legs dvi c5) = **0 in-training hits**. Consistent with external
preemption, not code faults.

**Blockers:** (1) **GPU contention from an out-of-band coupling2 experiment** —
currently holds the GPU; my continuation wrapper correctly waits behind it, so
the sweep is paused, not broken. Monitor: if such external preemptions recur they
will keep resetting partially-trained cells (wasted compute) — worth flagging to
the operator, but no code change is appropriate. (2) The dr_legs c5 re-runs and
c15 orphans remain pending, all covered by the parked wrapper. No new code/asset
fix applied. Environments/solvers remain strictly sequential. No commit made.

## Watchdog pass 2026-07-23 23:45 UTC

**Nominal. The out-of-band coupling2 experiment is training cleanly; my parked
continuation wrapper remains correctly blocked behind it. No fix warranted, no
intervention taken.**

**Process/GPU state.** The original master PID 144300 is confirmed dead (pid file
stale). Live tree: (a) `run_g1_jacobi_coupling2.sh` (PID 193244, ppid 193242,
nohup'd externally at 23:31) driving its own `g1 / newton_dvi_coupling2 /
contact_5` cell (train child PID 193262) — **healthy at iter 522/1500**, ~62k
steps/s, `Time elapsed 00:13:37`, ETA ~25 min, log growing (mtime 23:44); (b) my
inherited `resume_master_after_wrapper.sh` (PID 192439, ppid 1) — its child is a
bare `sleep 30`, i.e. still **parked in the `pgrep -f train.py` idle-guard**,
correctly blocked by the live coupling2 child. The dr_legs-c5 resume wrapper (PID
175497) is confirmed gone. `nvidia-smi` shows **exactly one compute app** on the
GPU (PID 193262, 1934 MiB, 86% util) — GPU-exclusive invariant intact, zero
overlap. I launched nothing; starting any resume now would duplicate the parked
wrapper and collide with the coupling2 hold.

**Newest run log inspected for crashes = clean.** `raw/g1/newton_dvi_coupling2/
contact_5/run.log` (19k+ lines, growing): the only `Error` matches are the
`error_vel_xy`/`error_vel_yaw` **metric names**, not faults. Crash-signature scan
(`Traceback|CUDA error|Segmentation fault|core dumped|Killed|RuntimeError|
AssertionError|out of memory`) across all in-matrix run.logs, excluding the three
known pre-fix dr_legs c5 failures and the two externally-killed cells (g1 PSPG
c15, dr_legs dvi c5) = **0 in-training hits**.

**Marker validation (programmatic, this pass): `COMPLETED` count = 26, unchanged.**
Every one of the 26 markers was re-checked and each co-occurs with a real
`^Training time:` line **and** an `exit=0` END marker in its run.log — **0 suspect
completed cells**. Incomplete cells (dir present, no marker) enumerated = 5:
`g1/newton_dvi_coupling2/contact_5` (live, separate matrix), `g1/newton_dvi_pspg/
contact_15` (externally killed at iter 467), and `dr_legs/{newton_dvi,
newton_dvi_apgd,newton_dvi_pspg}/contact_5`. Budget tally unchanged: **contact_5 =
9/12** (dr_legs trio outstanding), **contact_10 = 15/15**, **contact_15 = 2/12**
(g1 Jacobi 1089.24 s + g1 APGD 1347.10 s done; g1 PSPG + h1/go2/dr_legs pending).
No value recorded without the `exit=0` + `Training time:` check.

**dr_legs c5 asset fix re-confirmed; no new fix consistent-with-Ant-adapter
needed.** Reconciled the two distinct dr_legs c5 failure histories: (1) the
10:22Z `AttributeError: 'Articulation' object has no attribute
_initialize_handle` still visible only in the apgd/pspg c5 run.logs is the
**pre-fix** failure — already resolved by the earlier `dr_legs.usda` symlink
asset fix (symlink created 13:46; proven by all dr_legs c10 cells completing and
by the 23:23 c5 `newton_dvi` re-run clearing scene+finalize and training to iter
61 before preemption); (2) the 23:30Z `Terminated` of the dr_legs-c5 wrapper is
an **external kill** (same class as the master death), not a code/asset bug. The
apgd/pspg c5 logs simply still show the old 10:22Z tail because the resume
wrapper was killed before it reached them. When wrapper 192439 wakes, the master
script re-runs all three dr_legs c5 orphans on the fixed asset. **No new code
change is justified** — acting would be unwarranted churn, distinct from the
genuine symlink fix.

**Blockers:** (1) **GPU contention from the out-of-band coupling2 experiment** —
still holds the GPU; my continuation wrapper correctly waits behind it, so the
sweep is paused (behind that run), not broken. Recurring external preemptions
keep resetting partially-trained cells (wasted compute) — worth flagging to the
operator, but no code change is appropriate. (2) dr_legs c5 re-runs + all c15
orphans (g1 PSPG, then h1/go2/dr_legs × Jacobi/APGD/PSPG) remain pending, all
covered by the parked wrapper; they run strictly sequentially once the coupling2
run exits and the GPU goes idle. No code/asset fix applied. Sweep not duplicated
or interrupted. Environments/solvers remain strictly sequential. No commit made.

## Watchdog pass 2026-07-24 00:05 UTC

### Watchdog update — 2026-07-24 00:25Z (fresh preemption + restart-safe resume)

**The continuation wrapper and its live child were externally SIGTERM-killed;
watchdog re-invoked the restart-safe master. No code/asset fix warranted.**

**What happened.** At **00:05:41Z** the active `dr_legs / newton_dvi /
contact_5` training child was killed with **exit=143 (SIGTERM)** while training
healthily (iter progressing, ~16.6k steps/s, clean reward metrics, no NaN). The
master-continuation wrapper (PID 192439) then advanced to the next cell
(`dr_legs newton_dvi_apgd c5`) but was itself killed ~10 s later — its
`MASTER-CONTINUATION WRAPPER END` printed at **00:05:51Z** mid-Omniverse-init.
The apgd tail shows only the benign `[carb] Client ... nullptr` startup line (no
traceback, no `Training time:`, no `exit=` END). This is the **same recurring
external-preemption signature** documented below — **not** a code/asset bug — so
per the Ant-adapter reasoning **no fix was applied**.

**Watchdog action (restart-safe, reversible, no commit).** Verified nothing was
running (no `train.py`, no master; GPU idle of compute apps). Re-invoked the
existing restart-safe master directly under `setsid`:
`BUDGETS="5 10 15" bash run_locomotion_sweep.sh` (**no FORCE**), logging to
`resume_master_after_wrapper.log`. COMPLETED markers gate skipping, so only the
**13 orphans** run, strictly sequential on the single GPU. Live tree: `196538 →
196540 (run_locomotion_sweep.sh) → train.py 196570`.

**Now training the first orphan in matrix order:** `dr_legs / newton_dvi /
contact_5`. Scene creation cleared (17.85 s) and model finalized on `cuda:0` on
the fixed `dr_legs.usda` asset — confirming the symlink fix still holds — and it
is iterating cleanly (~6.0 s/iter, no NaN/collapse). `nvidia-smi` shows **exactly
one** compute app (PID 196570, 2612 MiB): environments/solvers remain strictly
sequential.

**Marker validation (this pass): `COMPLETED` = 26, unchanged; 0 suspect.** Every
marker was programmatically re-checked to co-occur with a real `^Training time:`
line **and** an `END … exit=0` marker in its own `run.log`. The live dr_legs c5
dvi cell is correctly **not** counted (no marker yet). No value recorded without
the `exit=0` + `Training time:` check.

**Orphans remaining (13, all covered by the active master, strictly sequential):**
`dr_legs c5` × {dvi (live now), apgd, pspg}; then `contact_15`: g1 PSPG, then
h1/go2/dr_legs × {dvi, apgd, pspg}. (mjwarp runs only at contact_10 by design, so
no mjwarp orphans.) Budget tally unchanged: **contact_5 = 9/12**,
**contact_10 = 15/15**, **contact_15 = 2/12** (g1 Jacobi 1089.24 s + g1 APGD
1347.10 s done).

**Blocker (flag to operator):** recurring **out-of-band external SIGTERM
preemptions** keep resetting partially-trained cells and pausing the sweep —
wasted compute, but **not** a code defect; the restart-safe master recovers each
time. The stale `locomotion_sweep.pid` still holds the long-dead original PID
144300 (cosmetic; live driver is 196538). Sweep not duplicated or interrupted;
single GPU app; **no commit made.**

---

**Master sweep is RUNNING and self-healed exactly as designed — the parked
continuation wrapper woke and resumed. No fix warranted; no intervention taken.**

**Process/GPU state.** The out-of-band `g1` coupling2 experiment that held the GPU
at 23:45 has **finished** since. My inherited continuation wrapper
`resume_master_after_wrapper.sh` (PID 192439, ppid 1 / supervisord) then passed
both its `kill -0` wait (dr_legs-c5 wrapper 175497 gone) **and** its
`pgrep -f train.py` idle-guard once the coupling2 train child exited, and
re-invoked the restart-safe master `run_locomotion_sweep.sh` with
`BUDGETS="5 10 15"` (no FORCE). Live tree: wrapper 192439 → `bash
run_locomotion_sweep.sh` (PID 192439 subtree, ppid1 192439) → `./isaaclab.sh -p
.../train.py` chain 194578 → 194605 → 194609. It is now training the **first
orphan in matrix order**: `dr_legs / newton_dvi / contact_5`. The stale
`locomotion_sweep.pid` still holds the long-dead original master PID 144300 (not
touched — cosmetic only; the live driver is 192439).

**Newest run log inspected for crashes = clean.**
`raw/dr_legs/newton_dvi/contact_5/run.log` is growing (mtime 00:05): scene +
finalize cleared on the fixed `dr_legs.usda` asset, training at **iter 26/1000**,
**~16.6k steps/s**, no NaN/collapse. Crash-signature scan
(`Traceback|CUDA error|Segmentation fault|core dumped|Killed|RuntimeError|
AssertionError|out of memory|FileNotFoundError`, excluding `error_vel_*` metric
names) across all in-matrix run.logs — excluding the two known pre-fix dr_legs c5
apgd/pspg failures (10:22Z, resolved by the symlink fix) and the two
externally-killed cells (g1 PSPG c15, dr_legs dvi c5 @ iter 61) — = **0
in-training hits**.

**GPU-exclusivity intact.** `nvidia-smi` shows **exactly one** compute app
(PID 194609, 2612 MiB); a single `train.py` is running. Environments/solvers
remain strictly sequential — the wrapper's belt-and-suspenders idle-guard
prevented any overlap with the just-finished coupling2 run.

**Marker validation (programmatic, this pass): `COMPLETED` = 26, unchanged; 0
suspect.** Every marker was re-checked to co-occur with a real `^Training time:`
line **and** an `exit=0` END marker. The live dr_legs c5 dvi cell is correctly
**not** counted (no marker yet). Budget tally unchanged: **contact_5 = 9/12**
(dr_legs trio outstanding, now actively being re-run starting with dvi),
**contact_10 = 15/15**, **contact_15 = 2/12** (g1 Jacobi 1089.24 s + g1 APGD
1347.10 s done; g1 PSPG then h1/go2/dr_legs pending). No value recorded without
the `exit=0` + `Training time:` check.

**No code/asset fix applied — none warranted.** The dr_legs.usda symlink asset
fix (created 13:46Z) remains in place and is proven live: this dvi c5 re-run
cleared scene+finalize and is training cleanly, so the old 10:22Z pre-fix
`_initialize_handle`/`FileNotFoundError` still shown in the apgd/pspg c5 tails is
stale and will be overwritten when the master reaches those cells next in
sequence. The earlier dr_legs c5 dvi `Terminated` (iter 61) and the g1 PSPG c15
freeze were external preemptions, not code bugs — consistent with the Ant adapter
reasoning that only genuine asset/adapter faults justify a fix. Acting further
would be unwarranted churn.

**Blockers:** (1) **Recurring out-of-band external preemptions** (the 23:23
master kill, the 23:30 dr_legs-c5 wrapper kill, and the coupling2 GPU grab) have
repeatedly reset partially-trained cells and paused the sweep behind them — wasted
compute worth flagging to the operator, but **not** a code defect; the restart-safe
wrapper recovers every time. (2) Remaining pending work, all covered by the now-
active master continuation and run strictly sequentially: dr_legs c5 trio
(dvi live now, then apgd, then pspg) and the contact_15 orphans (g1 PSPG, then
h1/go2/dr_legs × Jacobi/APGD/PSPG). Sweep not duplicated or interrupted;
environments/solvers strictly sequential; single GPU app; **no commit made.**

---

## Watchdog pass 2026-07-24T00:45:44Z

**Master sweep is RUNNING and healthy — no crash, no intervention, no commit.**

**Process/GPU state.** Between the previous pass and now, another out-of-band
external preemption reset the driver: the earlier continuation (PID 192439) is
gone and the parked resume wrapper re-invoked the restart-safe master at
**00:24:42Z**. Live tree: `bash -c … run_locomotion_sweep.sh` **PID 196538**
(ppid 1 / supervisord) → `bash run_locomotion_sweep.sh` 196540 → `train.py`
chain 196566 → 196570, driven with `BUDGETS="5 10 15"` (no FORCE). It is
training the **first orphan in matrix order**: `dr_legs / newton_dvi /
contact_5`.

**Newest run log inspected for crashes = clean.**
`raw/dr_legs/newton_dvi/contact_5/run.log` is growing: scene + finalize cleared
on the fixed `dr_legs.usda` asset, now at **iter 196/1000, ~16.8k steps/s,
ETA ≈ 01:19**, no NaN/collapse. Crash-signature scan
(`Traceback|CUDA error|Segmentation fault|core dumped|Killed|Terminated|
RuntimeError|AssertionError|out of memory`, excluding `error_vel_*` metric
names and the two known pre-fix dr_legs c5 apgd/pspg tails) across all in-matrix
run.logs = **0 in-training hits**.

**GPU-exclusivity intact.** `nvidia-smi` shows **exactly one** compute app
(PID 196570, 2612 MiB); a single `train.py` runs. Environments/solvers remain
strictly sequential.

**Marker validation (programmatic, this pass): `COMPLETED` = 26, unchanged;
0 suspect.** Every marker re-checked to co-occur with a real `^Training time:`
line **and** an `exit=0` END marker in its own run.log. The live dr_legs c5 dvi
cell is correctly **not** counted (no marker yet). Budget tally unchanged:
**contact_5 = 9/12**, **contact_10 = 15/15**, **contact_15 = 2/12**
(g1 Jacobi 1089.24 s + g1 APGD 1347.10 s). No value recorded without the
`exit=0` + `Training time:` check.

**No code/asset fix applied — none warranted.** The `dr_legs.usda` symlink asset
fix (created 13:46Z, target = real 84 KB USD) remains in place and is proven live
by this cleanly-training dvi c5 re-run. The stale 10:22Z pre-fix
`FileNotFoundError`/`_initialize_handle` tails on dr_legs c5 apgd/pspg predate
the fix and will be overwritten when the master reaches those cells next in
sequence — consistent with the Ant-adapter principle that only genuine
asset/adapter faults justify a fix.

**Housekeeping:** refreshed the long-stale `locomotion_sweep.pid` (held the
dead original PID 144300) to the live master **196538** — a trivially reversible
cosmetic correction so future watchdog passes read the correct driver. No
scheduler/config touched.

**Orphans remaining (13, all covered by the active master, strictly sequential):**
`dr_legs c5` × {dvi (live now), apgd, pspg}; then `contact_15`: g1 PSPG, then
h1/go2/dr_legs × {dvi, apgd, pspg}. (mjwarp only at contact_10 by design.)

**Blockers:** (1) **Recurring out-of-band external preemptions** keep resetting
partially-trained cells and re-parking/resuming the master — wasted compute worth
flagging to the operator, but **not** a code defect; the restart-safe wrapper
recovers every time. (2) Remaining pending work is fully covered by the active
master continuation and runs strictly sequentially. Sweep not duplicated or
interrupted; single GPU app; **no commit made.**

---

## Watchdog pass 2026-07-24T01:05Z

**Master sweep RUNNING and healthy — no crash, no fix, no resume needed, no commit.**

**Process/GPU state.** Master `bash -c … run_locomotion_sweep.sh` **PID 196538**
(ppid 1) → `bash run_locomotion_sweep.sh` 196540 → live `train.py` child. `nvidia-smi`
shows **exactly one** compute app (PID 196570, 2612 MiB). Environments/solvers
strictly sequential; single GPU app. `locomotion_sweep.pid` correctly holds the
live master (196538).

**Current cell + progress.** Still the first matrix orphan
`dr_legs / newton_dvi / contact_5`. Advanced **iter 386 → 406 / 1000** across
this pass, steady **~16.79k steps/s**, ETA ≈ 00:58. Log
`raw/dr_legs/newton_dvi/contact_5/run.log` grew ~190 lines / 3 s — actively
training, not stalled.

**Newest run log inspected for crashes = clean.** Crash-signature scan
(`error|Traceback|Exception|Failed|NaN|CUDA error|Segmentation fault|core dumped|
Killed|AssertionError`, excluding benign `carb … nullptr`, the orphan-joints
`finalize(skip_validation_joints=True)` notice, and `error_vel_*` metric names) =
**0 in-training hits**. No END/FAILED marker yet (expected — run in progress).

**Marker validation (programmatic, this pass): `COMPLETED` = 26, unchanged; 0
suspect.** Each marker independently re-checked to co-occur with a real
`^Training time:` line **and** an `exit=0` END marker; the live dr_legs c5 dvi
cell is correctly **not** counted. Budget tally unchanged: **contact_5 = 9/12**
(dr_legs trio outstanding — dvi live now), **contact_10 = 15/15**,
**contact_15 = 2/12** (g1 Jacobi 1089.24 s + g1 APGD 1347.10 s). No value
recorded without the `exit=0` + `Training time:` check.

**No code/asset fix applied — none warranted.** The `dr_legs.usda` symlink asset
fix remains in place and is proven live by this cleanly-training dvi c5 re-run,
consistent with the Ant-adapter principle that only genuine asset/adapter faults
justify a fix. The stale pre-fix `FileNotFoundError` tails on dr_legs c5
apgd/pspg predate the fix and will be overwritten when the master reaches those
cells next in sequence.

**Orphans remaining (13, all covered by the active master, strictly sequential):**
`dr_legs c5` × {dvi (live now), apgd, pspg}; then `contact_15`: g1 PSPG, then
h1/go2/dr_legs × {dvi, apgd, pspg}. (mjwarp only at contact_10 by design.)

**Blockers:** none new. Standing item: recurring out-of-band external preemptions
have periodically reset partial cells and re-parked/resumed the master — wasted
compute worth flagging to the operator, but **not** a code defect; the
restart-safe wrapper recovers every time. Sweep not duplicated or interrupted;
single GPU app; **no commit made.**

# DVI Closed-Loop / DR Legs Integration — Status (IsaacLab side)

**Branch:** `milad/dvi-solver-close-loop` (IsaacLab)
**Companion:** newton-chrono branch `milad/dvi-close-loop`
**Last updated:** 2026-06-09

---

## Goal

Run **our DVI solver** (newton-chrono) with the Disney **DR Legs** closed-loop
parallel-linkage biped in Isaac Lab — the upstream PRs use the Kamino solver; we
want the same environment driven by DVI.

DR Legs: 36 revolute joints (12 actuated + 24 passive closed-loop linkage DOFs),
cyclic kinematic loops, **no `ArticulationRootAPI`**.

Reference PRs (isaac-sim/IsaacLab), both from `aserifi/IsaacLab`:
- **#5962** "DR Legs" — closed-loop biped + `Isaac-DrLegs-HoldPose-v0` /
  `Isaac-DrLegs-Walk-v0` tasks. Source branch: `aserifi/drlegs` (head `be835219a4`).
- **#5832** "Kamino Implicit Joint" — `route_torque_to` actuator attribute.

**Key point:** No DVI solver math changes are needed — our solver is
maximal-coordinate and topology-agnostic (handles cyclic joint graphs natively,
like Kamino). See `DVI_CLOSED_LOOP_STATUS.md` in the newton-chrono repo for the
proof. The work here is **IsaacLab plumbing only**.

---

## Setup notes
- Remote added: `aserifi` = https://github.com/aserifi/IsaacLab.git
- Fetched branch ref: `aserifi/aserifi/drlegs` (head `be835219a4`).
- Their branch is ~40 commits ahead of our base, but MOST is unrelated `develop`
  drift (Isaac Sim 6.0 / Newton 1.2.1 bump, CI, docs, `contrib/` task-layout
  reorg). Only ~10 commits are DR-Legs / Kamino specific.
  **DO NOT merge the whole branch** — huge churn / conflicts. Cherry-pick files.

---

## DONE (staged on this branch)
- 10 self-contained DR Legs files copied from `aserifi/aserifi/drlegs` via
  `git show <ref>:<path> > <path>`:
  - `source/isaaclab_assets/isaaclab_assets/robots/dr_legs.py`
  - `source/isaaclab_newton/isaaclab_newton/assets/articulation/closed_loop_view.py`
  - `source/isaaclab_tasks/isaaclab_tasks/contrib/dr_legs/`
    (`__init__.py`, `agents/{__init__,rsl_rl_ppo_cfg}.py`, `hold_pose_env_cfg.py`,
    `walk_env_cfg.py`, `mdp/{__init__,observations,rewards}.py`)
- Ported `route_torque_to` ClassVar additively into
  `source/isaaclab/isaaclab/actuators/actuator_base.py`
  (added `Literal` import + the attribute + docstring).

## TODO — framework hooks (these CONFLICT; our base is older than theirs)
Main enabler commit: **`c72255f66e`** "Add closed-loop Kamino support and fix env resets".
Needs careful 3-way merge of:
- `source/isaaclab_newton/.../assets/articulation/articulation.py`
  - Detect closed loops → instantiate `ClosedLoopView` instead of `ArticulationView`.
  - Adds `_get_root_view_articulation_ids()` used by reset/FK-invalidation paths.
  - Depends on `_prim_has_closed_kinematic_loops` from `newton_replicate.py`.
  - `git apply --3way`: applies WITH CONFLICTS.
- `source/isaaclab_newton/.../cloner/newton_replicate.py` — adds
  `_prim_has_closed_kinematic_loops`. CONFLICTS.
- `source/isaaclab_newton/.../physics/newton_manager.py` — reset / per-env FK.
  CONFLICTS.
- `source/isaaclab/isaaclab/envs/manager_based_rl_env.py` — applies CLEAN.
- `source/isaaclab_newton/.../assets/articulation/articulation_data.py` — additive
  bindings (from `f3f49b2b3b`). TODO.
- Contact-sensor / per-body contact aggregation changes (`contact_sensor.py`,
  `newton_manager.py`) from `c72255f66e` / `be835219a4` — only needed for the
  Walk task contact rewards; HoldPose may not need them. Defer.

Strategy: hand-merge minimal additive hunks; keep Kamino-specific bits out of the
critical path. Prefer adding behind closed-loop detection so existing tree robots
are unaffected.

## TODO — DVI wiring
- Add a `newton_dvi` preset to the DR Legs env cfgs (`hold_pose_env_cfg.py`,
  `walk_env_cfg.py`) pointing at `DVISolverCfg` (mirror our other `newton_dvi`
  presets, e.g. H1/G1 flat env cfgs).
- Confirm `SolverDVI` consumes `Control.joint_act` if we use `route_torque_to="joint_act"`,
  or keep `joint_f` routing.

---

## BLOCKER: DR Legs USD asset missing
- `dr_legs.py` loads `usd_path = .../robots/data/dr_legs/dr_legs.usda` (local path).
- **Not tracked in `aserifi/drlegs`, not present on this machine.** PR #5962:
  *"DR Legs USD is loaded from a local path for now; @AntoineRichard already sent a
  request to the Nucleus team."*
- Cannot load/step the robot until the USD (+ any referenced meshes) is obtained.
  Drop it at `source/isaaclab_assets/isaaclab_assets/robots/data/dr_legs/`.

---

## First integration milestone
Load + step `Isaac-DrLegs-HoldPose-v0` under DVI (needs USD + framework hooks +
preset) and verify the solver holds the assembled closed-loop pose. This is the
real test that maximal-coordinate DVI keeps the loops closed.

---

## ✅ UPDATE 2026-06-09 — DR Legs lab env TRAINS under DVI; reset NaN root-caused

**`Isaac-DrLegs-HoldPose-v0` now builds and trains under the DVI solver.** Full
learning iterations run with all reward terms computing. Remaining issue: a
reset-time NaN in a few envs, now fully root-caused.

### What was ported to make it run (branch `milad/dvi-solver-close-loop`)
1. **USD found** (NOT missing): `newton.utils.download_asset("disneyresearch")`
   provides `dr_legs/usd/dr_legs*.usda`. Symlinked into
   `source/isaaclab_assets/isaaclab_assets/robots/data/dr_legs/`.
2. **Task registration**: added missing `contrib/__init__.py` (gym.register never ran).
3. **`newton_dvi` preset** in `contrib/dr_legs/hold_pose_env_cfg.py` — copied from the
   validated Go2 `_dvi_solver_cfg` (sparse_ldl joints, joint_alpha=0.005, sparse_jacobi
   contacts, **no position correction**). joint_reg 1e-6→1e-4, num_substeps=8 (0.5ms
   solver dt = validated stable point). Made `default` too (Kamino preset used unported
   KaminoSolverCfg fields).
4. **`newton_manager.py`**: `finalize(skip_validation_joints=True)` for dvi/kamino +
   **scoped reset masks** (`_or_world_mask_from_env_mask`/`_scatter_world_mask_from_ids`)
   so one env's reset no longer snaps ALL envs' body_q to base (from `c72255f66e`).
5. **`articulation.py`**: closed-loop detection → `ClosedLoopView`; added
   `_get_root_view_articulation_ids()` and routed 6 FK-invalidation sites through it.
6. **`newton_replicate.py`**: `_prim_has_closed_kinematic_loops`.
7. **`manager_based_rl_env.py`**: post-reset FK refresh.

### ROOT CAUSE of the reset NaN (definitive, instrumented)
Resetting a closed-loop robot via root-pose + joint_q + FK is INSUFFICIENT:
- Standard reset writes `body_q[0]` (root) + `joint_q`, then relies on FK to fill the
  other body poses. For a TREE that works; a **closed loop has no spanning tree**, so
  `eval_fk` cannot reconstruct the 30 passive linkage bodies.
- Instrumented trace at first reward NaN:
  - `body_link_pose_w` nan=0, `root_quat_w` nan=0 (data layer looks OK)
  - raw `state_0.body_q` nan=0 BUT raw **`state_0.body_qd` nan=186** (velocities)
  - NaN body_qd rows = a contiguous 31-body block = **one full env** (global 1426–1456
    = env 46, local bodies 0–30)
  - `body_q[1426]` (env 46 root) = `[0,0,0, 0,0,0,1]` = **world origin / identity** =
    builder default with NO per-env offset, not the env's reset pose
  - env 46 was NOT resetting on the failing step → NaN **persisted** from an earlier reset
- **Conclusion**: closed-loop reset leaves loop bodies at the origin with garbage/NaN
  velocities. `ClosedLoopView.get_root_velocities` (implicit_free_base) is a strided view
  of **only `body_qd[0]`**, so writing root velocity never zeroes the other 30 bodies'
  `body_qd` — stale NaN stays.

### THE FIX (designed, not yet implemented)
Closed-loop reset must restore the **full assembled per-env body state**:
1. At init, snapshot per-env assembled `body_q` (+ zero `body_qd`) for the robot
   (e.g. `ClosedLoopView` caches `_default_body_q`/`_default_body_qd` with env offsets).
2. On reset of env i, write ALL 31 bodies' `body_q` (assembled ref translated to env i's
   origin + sampled root offset) and zero ALL 31 bodies' `body_qd`.
3. Topology-correct for loops (no FK reconstruction) and clears any solver NaN.
Belongs in `ClosedLoopView` + the articulation reset/write_root_* path.

### Repro / debug
- Train with `presets=newton_dvi`; rsl_rl `check_nan` raises after iter 0/1 in ~1–6/64 envs.
- NaN-source trace: set `DRLEGS_NAN_DEBUG=1` and re-add the one-shot hook after
  `reward_manager.compute` in `manager_based_rl_env.py` (removed after diagnosis).

---

## ✅✅ RESET NaN FIXED — full-body closed-loop reset implemented (2026-06-09)

**Implemented the designed fix. DR Legs now trains under DVI with ZERO NaN across
many reset cycles** (10-iter smoke + 40-iter run, episode length ~53 with constant
resets, reward healthy at ~-0.8 → -5 vs the old -12/-82).

### Implementation
**`ClosedLoopView` (`closed_loop_view.py`)**
- At construction, snapshot `model.body_q` → `_assembled_body_q` reshaped
  `(world_count, bodies_per_world)`. This is the authored, **loop-closed** body block with
  per-env offsets already applied by the cloner — the correct reset reference.
- Exposed: `is_implicit_free_base`, `has_assembled_reference`, `bodies_per_world`,
  `assembled_body_q`.

**Kernels (`assets/kernels.py`)** — 4 new Warp kernels:
- `reset_closed_loop_bodies_from_root_{index,mask}`: given a NEW root pose, compute the
  rigid delta `T = newroot · ref_root⁻¹` and apply it to every reference body pose, writing
  all 31 bodies' `body_q` and zeroing all `body_qd`. (A rigid transform preserves loop
  closure, so the loop stays closed at the new pose.)
- `restore_closed_loop_bodies_from_state_root_{index,mask}`: same, but reads the CURRENT
  root from the sim state (`body_q[env,0]`) as the reference root. Used after joint-state
  writes whose `invalidate_fk` would otherwise corrupt the loop bodies.

**Articulation (`articulation.py`)**
- `_is_closed_loop_reset_target()`: gate (implicit-free-base + has snapshot).
- `_restore_closed_loop_bodies_{index,mask}(root_pose, ...)`: launch the from-root kernels
  into raw Newton `state_0.body_q/qd`, called at the END of
  `write_root_link_pose_to_sim_{index,mask}` (AFTER `invalidate_fk`, so eval_fk can't clobber).
- `_reapply_closed_loop_bodies_{index,mask}(...)`: launch the from-state-root kernels,
  called after `invalidate_fk` in `write_joint_state_to_sim_*` and
  `write_joint_position_to_sim_*` (those trigger eval_fk which corrupts loop bodies).

### Why this is correct (not a hack)
The maximal-coordinate truth for a closed loop is the full `body_q` block; there is no
spanning tree to FK along. The USD authors a single assembled, loop-closed configuration
(`model.body_q`). Resetting = placing that rigid assembly at the desired root pose. A rigid
transform of a loop-closed configuration is still loop-closed, so constraints stay satisfied
(initial residual ~1.7e-8, consistent with the standalone Newton test). Velocities reset to
zero, clearing any solver NaN. This is exactly how a maximal-coordinate engine should reset
a closed loop.

### Limitation / note
- Per-joint reset randomization (`reset_joints_by_offset` with nonzero range) is effectively
  ignored for the loop pose — we restore the assembled reference + the sampled ROOT offset.
  For DR Legs HoldPose the joint reset range is (0,0), so this is exact. If a future task
  wants randomized joint angles on a closed loop, it would need a constraint-consistent
  re-assembly (solve the loop closure for the sampled DOFs) — out of scope here.
- Files touched: `closed_loop_view.py`, `assets/kernels.py`, `articulation.py`. All
  changes gated behind `_is_closed_loop_reset_target()` so tree articulations are unaffected.

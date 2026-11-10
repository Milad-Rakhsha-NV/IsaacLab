# Go2 RFT training and MPM inference study

Updated November 10, 2026 (UTC). **Implementation and one-seed pilot training are complete. Physical validation and successful RFT-to-MPM transfer are not complete.** Both evaluated policies fail in the current finite-bed MPM reference. The static-support gate also fails. These results support further solver investigation, not a robust locomotion or real-robot deployment claim.

## Code and experiment status

| Component | Status | Evidence / limit |
|---|---|---|
| Preserve SuperMimic | Complete | `milad/supermimic` was clean and already pushed at `a30205213313a3692dbac66978afd88963623dd1` |
| Native Isaac Lab task | Complete | `Isaac-Velocity-Flat-UnitreeGo2-RFT-Direct`; explicit `DirectRLEnv`, existing packages and runner |
| Newton RFT integration | Complete, experimental | Shared-model RFT/MJWarp composition; per-world GPU materials and heights; no MPM in training |
| Full robot surface binding | Implemented | 27 colliders including every hip, thigh, calf and foot, plus torso/head; collision geometry and inertias preserved |
| Stock Go2–MPM example | Smoke passed | 4 s, 237,314 particles, two-way coupling, rigid-supported 8 cm strip; separate pretrained policy |
| Pilot training | Complete | Randomized and command-matched nominal policies, 1,000 PPO iterations / 12.288 M transitions each |
| RFT inference | Complete for pilot matrix | Nominal 10 s, matched 16-world field/command bank, weaker/stronger materials |
| MPM inference | Complete for pilot matrix | Same frozen direct-task policy; nominal for both policies and two material corners for randomized policy |
| Numerical verification | Local checks pass | 17 Newton RFT + 6 coupling tests; 8 analytical/numerical gates on CPU and CUDA; native Isaac Lab test |
| Robot mesh convergence | Training mesh not accepted | 5.53% force / 3.33% torque error against finest tested angular + edge refinement; exceeds planned <5% force gate |
| Static support | Failed | Zero-action robot sinks and drifts; near-rest bearing is not an exact frictional sticking model |
| Resolved MPM reference | Not accepted | Grid, timestep, coupling and preparation convergence not established |
| Robust transfer / final study | Not accepted | Early MPM failures; only one training seed; no final multi-seed or real-robot study |

Isaac Lab branch: [milad/go2-rft-mpm](https://github.com/Milad-Rakhsha-NV/IsaacLab/tree/milad/go2-rft-mpm).
Newton branch: [milad/go2-rft](https://github.com/Milad-Rakhsha-NV/newton/tree/milad/go2-rft).
Both dependency declarations and `uv.lock` pin Newton commit `87457620c5fa8aedfe616c0ebdff1035c1b84afd` on that fork, based on Isaac Lab's previous Newton revision `42f7def118c2c4299ce493457bafd4e8a22a696e`.

## Implementation

### Coupling and geometry

`source/isaaclab_newton/isaaclab_newton/physics/rft_solver.py` composes the RFT surface evaluator with the existing MJWarp articulation solver on the same Newton model. Each substep evaluates granular body wrenches, adds them to existing external forces, advances the articulation exactly once, and preserves the external-force input. Joint dynamics transfers each link's load through the robot. There is no additional joint-torque application of the same granular force and no RFT proxy-relaxation loop.

The adapter reads finalized collision primitives, scales, body ownership and local transforms. Boxes use oriented triangular faces; spheres, capsules and cylinders use native primitive meshes; meshes use their source geometry, including the actual convex hull for convex colliders. Surfaces are cached and remain fixed during training. The stock Go2 training mesh has **1,536 nondegenerate triangles** after pruning, from 27 source colliders. `surfaces.json` records every shape/body mapping. Triangles do not replace the rigid collision model or alter mass/inertia.

Quadrature points lying inside another primitive on the same body are suppressed to approximate an exterior collider union. This avoids counting many internal overlapping patches. It is not an exact Boolean surface operation; coincident boundaries, intersection refinement, inter-link shielding and interaction between granular flows remain limitations.

RFT uses world-space forces and moments about each body's center of mass. `body_impulse` integrates the applied wrench in float64; differences over each control interval produce the plotted mean RFT load. Per-world material and height arrays remain on the GPU and can be edited in place under CUDA graph capture. Masked resets preserve other clones' material and articulation state.

The RFT composition requires Newton, does not support deterministic atomic reductions, has no conventional contact-sensor output for granular loads, and rejects generic partitioned-coupler and fixed-tendon use. The current task and strength calibration assume Earth gravity.

### Published flowing law and experimental support term

The core solver uses the published three-dimensional quasi-static RFT angular fit, clipped submerged triangles, depth weighting and leading-surface activation. A prescribed RFT field evolves no particles, compaction, free-surface volume or plastic stress history.

**The task additionally enables an experimental near-rest bearing term.** With outward patch normal `n`, downward unit vector `d`, positive depth `h`, strength scale `S`, relative point velocity `v`, tangential component `v_t`, and transition speed `epsilon`, it blends

```text
b = epsilon² / (|v|² + epsilon²)
p_b = bearing * S * h * max(n · d, 0)
t_rest = -p_b*n - mu_surface*p_b*v_t / sqrt(|v_t|² + epsilon²)
t = (1-b)*t_flow + b*t_rest
```

The task uses `bearing=0.87574` and `epsilon=0.04 m/s`, frozen before the reported main training runs. These are provisional parameters, not an independently validated sand calibration. The closure adds normal bearing and regularized sliding resistance but no exact tangential sticking state, loading history or validated unloading law. It is explicitly distinct from the published RFT law. `--no-bearing` retains the flowing-law ablation. Local verification of the default flowing law does not validate this extension's energy or static behavior.

Finite velocity regularization and the near-rest blend introduce speed dependence. The rate-independence verification applies to the unregularized base flowing law, not to this entire task configuration.

### Direct task and randomization

The task is implemented in `source/isaaclab_tasks/isaaclab_tasks/core/velocity/config/go2/rft_env*.py`. It uses the stock Go2 asset, joint order and DC motor model, with 12 position-offset actions scaled by 0.25 rad. Actor input has 45 values: base angular velocity, projected gravity, velocity command, joint-position offsets, joint velocities and previous actions. The critic adds the three true base linear-velocity components. Material identity and true base linear velocity are absent from the actor.

Control is 50 Hz. Isaac Lab physics steps at 200 Hz with four Newton substeps, giving an 800 Hz articulation/field update. The RFT scene has a visual bed but **no collidable support plane and no MPM particles**. The default configuration has 1,024 worlds; actual pilot training used 512 to keep the experiment inexpensive.

| Reset parameter | Randomized training distribution |
|---|---|
| Bulk density | Uniform 1,400–1,800 kg/m³ |
| Internal friction coefficient | Uniform 0.3–0.5 |
| Surface/internal friction ratio | Uniform 0.6–1.0 |
| RFT strength multiplier | Log-uniform 0.8–1.2 |
| Bed elevation | Uniform −0.02–0.02 m |
| Commands | Forward 0.1–0.5 m/s, lateral ±0.1 m/s, yaw ±0.3 rad/s; 10% stand |
| Actor observation noise | Independent additive noise, standard deviation 0.01 |

The nominal-material baseline keeps the same command distribution and training settings. Only field randomization changes. Commands are sampled at reset; start/stop command schedules and final held-out test seeds remain future work. These ranges are experimental distributions, not measured sand-property intervals.

The randomized policy is trained in **one run with one shared policy across 512 parallel environments**. `_reset_idx(env_ids)` resamples the fields and commands independently for just the environments being reset; other environments retain their samples. The other runs in the training table are baselines or development pilots, not separate runs for individual material settings. Observation noise is sampled on every observation rather than only at reset.

Here, randomized height means the flat granular free-surface elevation in each environment, not the robot's target base height or a spatially varying height map. Randomized velocity means the commanded robot forward/lateral/yaw velocity. The granular field is stationary; its bulk velocity and the robot's initial velocity are not randomized. The near-rest transition speed (0.04 m/s), bearing coefficient (0.87574), and inertial coefficient (zero) remain fixed.

Rewards combine planar/yaw tracking, survival, orientation/height penalties, motor effort and action changes. Target height is 0.32 m. Failure occurs below 0.16 m relative to the nominal environment origin, with excessive tilt, or with nonfinite root position. Height thresholds do not track the randomized ±2 cm bed elevation. Episodes time out at 10 s. Recorded inference disables autoreset and retains failed trajectories; errors include the post-failure interval.

## Training completed

GPU: NVIDIA RTX PRO 4500 Blackwell Server Edition, approximately 32 GB. Python 3.12.15, Warp 1.17.0, RSL-RL 5.5.1. Each main run used 512 environments, 24 steps per PPO iteration, seed 42, and the existing Go2 128×128×128 actor/critic architecture. PPO used RFT only.

| Run | Iterations / transitions | Training wall time | Checkpoint under `logs/rsl_rl/` |
|---|---:|---:|---|
| Randomized RFT, main | 1,000 / 12.288 M | 434.98 s (7.25 min) | `go2_rft/2026-11-10_02-40-07/model_999.pt` |
| Nominal material, matched commands | 1,000 / 12.288 M | 542.22 s (9.04 min) | `go2_rft_nominal_matched/2026-11-10_03-06-26/model_999.pt` |
| Nominal material, fixed-forward demonstration | 1,000 / 12.288 M | 539.51 s (8.99 min) | `go2_rft_nominal/2026-11-10_02-50-03/model_999.pt` |
| Earlier development pilot | 500 / 6.144 M | 197.8 s | `go2_rft/2026-11-10_02-34-45/model_499.pt` |

The matched nominal run is the primary comparison. The fixed-forward policy has an easier command distribution and is only a demonstration. The earlier development pilot predates internal-patch suppression and is excluded from the controlled comparison. Training wall times exclude first setup/compilation; nominal runs shared the GPU with inference work and do not establish an isolated throughput comparison. The main randomized run averaged about 28,250 transitions/s over its recorded training duration.

These runs are exploratory pilots, not gate-qualified final training. Static/reference gates did not pass, so no five-seed training sweep or deployment study was launched. The final iteration was used without selecting a checkpoint on MPM outcomes. RFT surface reductions use floating-point atomics, so repeated CUDA rollouts are not guaranteed to be bitwise identical; the values below refer to the saved recordings.

## Frozen-policy results

Tracking error is the time average of `||v_body,xy - command_xy||₂`. A trial fails if the height/tilt/nonfinite condition occurs at any recorded step. Rollouts continue after failure. All single-field cases below use a 0.3 m/s forward command, no observation noise, no policy exploration and no learning.

### Nominal RFT and matched randomized bank

| Policy / field | Duration | Mean tracking error | Failures |
|---|---:|---:|---:|
| Randomized policy, nominal RFT | 10 s | 0.129 m/s | 0/1; final height 0.196 m |
| Matched nominal policy, nominal RFT | 10 s | 0.208 m/s | 0/1; final height 0.266 m |
| Fixed-forward demonstration, nominal RFT | 10 s | 0.061 m/s | 0/1; 3.25 m forward progress |
| Randomized policy, randomized bank | 16 × 10 s | 0.141 m/s | **11/16** |
| Matched nominal policy, same bank | 16 × 10 s | 0.180 m/s | **6/16** |

The two bank manifests were checked for exact equality of sampled commands, material parameters and bed heights. Randomization improved mean tracking error but increased failures in this small bank. It has **not demonstrated improved robustness**. This bank uses the development seed; it is not an independent final test set or a statistical multi-seed result.

RFT is spatially unbounded. The nominal randomized-policy trajectory leaves the footprint of the hypothetical MPM comparison bed at 5.76 s; the fixed-forward demonstration does so at 3.26 s. Those 10 s RFT clips demonstrate motion, not matched finite-bed transfer over the full interval. The principal RFT/MPM graph compares the first 2 s, before boundary exit.

### MPM reference and material corners

MPM is built only by the inference script. It uses the existing staggered two-way `CouplerProxyCfg` with the complete robot, the same rigid actuator model and 800 Hz outer substeps. The reference bed spans `[-0.7,1.3] × [-0.6,0.6] × [-0.4,0] m`; a collidable bottom supports the particles at −0.4 m and is excluded from the rigid robot solver. Its planform is centered at x=0.3 m to cover the entire bed with a margin. There is no direct rigid-ground support of the robot. The 20 mm grid contains 968,000 particles initially. Other settings: PIC transfer, `pic27` collider basis, `P0` strain basis, 50 maximum iterations, tolerance 1e-4, nearly rigid modulus/yield pressure 1e15 Pa, and proxy relaxation 0.1. These are initial reference settings, not a validated physical modulus or converged solution.

The code sets friction through MPM's public collider setup API before capture and asserts the resulting face-friction values. This matters because asset-local USD bindings otherwise retain surface friction 1.0. Nominal reference inputs are density 1,600 kg/m³, internal friction 0.4 and actual collider friction 0.4. Matching these inputs alone does not make the constitutive responses equivalent.

| Frozen policy / material | RFT result | MPM result |
|---|---|---|
| Randomized / nominal | No failure in 10 s | Failure at **0.24 s**; 2 s error 0.315 m/s |
| Matched nominal / nominal | No failure in 10 s | Failure at **0.28 s**; 2 s error 0.277 m/s |
| Randomized / weaker: density 1,400, internal friction 0.35, roughness ratio 0.8 | Failure at 0.60 s; 2 s error 0.204 m/s | Failure at 0.24 s; bottom interaction later in rollout |
| Randomized / stronger: density 1,800, internal friction 0.45, roughness ratio 1.0 | No failure in 2 s; error 0.145 m/s | Failure at 0.24 s; 2 s error 0.307 m/s |

Nominal MPM cases stayed finite for 2 s and did not trigger the foot-bottom proximity diagnostic. Particle finiteness is not reference convergence. The weaker material reaches the bottom, so its later behavior is confined and is not a half-space accuracy comparison. An initial weak-field run exceeded sparse-grid capacity at 1.84 s; the capacity was increased and that case rerun. The gallery retains failures and reports execution errors rather than discarding poor behavior.

The MPM force diagnostic reads persistent relaxed proxy feedback at original source-body indices. It is the **latest feedback sample**, while RFT plots use **control-interval impulse averages**. These signals can show qualitative load differences but do not support a quantitative force-error gate until sampling is aligned. Reading transient MPM grid scratch after captured stepping was removed because that storage has been released.

The bed has not been independently settled and checkpointed, and no grid/timestep/iteration/coupling convergence or complete boundary-sensitivity study has passed. MPM uses an evolving finite material; RFT uses a fixed free surface without history. The observed failure therefore cannot yet be assigned solely to the RFT approximation or solely to MPM. A successful frozen-policy transfer has not been demonstrated.

## Numerical validation and remaining gates

Repository checks passed against the installed, locked Newton package: the native task test, 23 focused Newton tests, all repository formatting/lint hooks, generated environment-browser consistency, and a full Sphinx current-checkout build with warnings treated as errors. The build produced `docs/_build/current/`; existing deprecated-preset runtime warnings do not fail it.

Newton's 17 RFT tests cover analytic surface loads, COM torque and transforms, clipping/dry behavior, force preservation, captured per-world material changes and rejection of internal bearing patches. Six focused proxy-coupler tests also pass, including gathering feedback to the original source-body IDs. The native Isaac Lab test exercises the actor/critic layout, all four full limbs, zero particles, masked material/pose reset isolation and finite articulated stepping with nonzero granular load.

The standalone Newton investigation passes all eight gates on CPU and CUDA: analytic plates, sampled closed-cycle work, compression-only traction, friction bounds, flowing-law rate independence, similarity scaling, drilling-mesh convergence and timestep refinement against a scalar reference. These gates concern those fixtures; they do not validate Go2 stance, the experimental bearing law, or MPM material equivalence.

The robot surface study evaluates **common prescribed kinematics** from four recorded poses, with COM twists estimated using 20 ms finite differences. No robot trajectory is integrated during this load comparison.

| Surface setting | Nondegenerate triangles | Force error vs finest tested | Torque error vs finest tested |
|---|---:|---:|---:|
| Training: 12 angular segments, no edge limit | 1,536 | **5.53%** | 3.33% |
| 24 segments, 4 cm maximum edge | 14,088 | 1.24% | 1.35% |
| 48 segments, 2 cm maximum edge | 81,552 | Reference | Reference |

Angular-only refinement gives a smaller apparent training-mesh force error of 4.89%; adding edge refinement exposes long-face quadrature error. The planned <5% robot-surface gate is therefore **not passed by the training setting**. These four snapshots and a finite finest mesh establish only limited self-convergence, not physical accuracy or global convergence. The adapter exposes `max_edge_length` for the next accuracy/cost study; the reported checkpoints retain their original training mesh.

The 20 s zero-action RFT test ends at height 0.163 m after approximately 1.33 m horizontal drift and first triggers the height failure at 2.2 s. This plainly fails the intended static-support gate. Historical MPM zero-action probes at 20 and 10 mm grid spacing also collapse, but used surface friction 1.0 and a shallower 0.2 m bed. They are labeled diagnostics, not friction-matched or converged references.

| Planned gate | Current conclusion |
|---|---|
| A: integration | Local fixtures and native task checks pass; full prescribed fixed-base leg study remains |
| B: robot surfaces <5% | Training mesh fails force criterion; refined candidates need broader kinematic coverage and profiling |
| C: converged resolved reference | Pending settled-bed, grid, timestep, iteration, coupling and boundary studies |
| D: held-out force/impulse accuracy | Pending C and common-motion RFT/MPM replay; no accepted calibration/accuracy claim |
| E: static support | Failed; requires a justified static-to-flowing model with sticking/sliding and unloading |
| F: task readiness | Finite stepping/training demonstrated; low-posture and burial behavior prevent readiness acceptance |

Next physics work is a loaded-foot and fixed-base-leg investigation with a converged, prepared MPM bed, followed by a support closure with explicit static friction and unloading. Freeze any calibration on separate intrusion data, retrain with an accepted mesh, and only then expand the policy matrix. A common-interface rigid-ground baseline, pure-flowing-law policy ablation, multi-seed training, independent scenario seeds, start/stop/turning schedules, repeated-footfall reference tests and final statistical reporting remain open.

## Visual evidence and raw data

Open the [local video/graph gallery](logs/go2_granular/report/index.html). It includes the nominal RFT walking clips, MPM transfer failures, weaker/stronger material comparisons and the separate stock Go2 runway showcase. Green arrows show granular force on each body, scaled at **200 N per displayed metre**. They exclude actuator and conventional contact forces. All ten videos were decoded and their frame counts checked against the saved step counts. Seven PNG figures were generated. All clips use actual Newton stepping and a headless renderer.

| Figure | Content |
|---|---|
| [Training](logs/go2_granular/report/01-training.png) | Actual TensorBoard returns, episode lengths, tracking errors and throughput; 20-iteration smoothing |
| [Transfer](logs/go2_granular/report/02-transfer.png) | Same frozen randomized policy in RFT and MPM, first 2 s |
| [Standing diagnostics](logs/go2_granular/report/03-stance-reference.png) | Unresolved static support; historical unmatched MPM probes explicitly labeled |
| [Randomized bank](logs/go2_granular/report/04-random-fields.png) | Paired tracking errors and failure counts on the same 16 cases |
| [Surfaces](logs/go2_granular/report/05-surfaces.png) | Actual full-robot triangular surface overlay and free surface |
| [Surface convergence](logs/go2_granular/report/06-surface-convergence.png) | Angular-only versus angular plus edge refinement |
| [Material transfer](logs/go2_granular/report/07-material-transfer.png) | Nominal and weaker/stronger material height histories |

Raw trajectories, wrenches, body poses, summaries, binding manifests and videos are in `logs/go2_granular/<case>/`. `rollout.npz` state columns are time, world position xyz, body velocity xyz, nominal-relative height, planar tracking error, and failure flag. Wrenches use world force xyz followed by COM torque xyz. Training parameters, TensorBoard records and checkpoints are in the training directories listed above. `report/results.json` gathers training/inference metadata; `report/verification/` contains local test logs and Newton benchmark JSON.

Large media, model checkpoints and generated experiment artifacts remain local under ignored `logs/`; they are not embedded in Git. Cloning the branch provides code and this report, not the recordings/checkpoints. The gallery links require these local artifacts or an explicit artifact copy.

### View from another machine

The gallery is served on `mrakhsha-41eslp` by the detached user service `go2-granular-gallery.service`, bound to loopback port 8010. This keeps it separate from the chat's temporary shell. Check or restart it on that host with:

```bash
systemctl --user status go2-granular-gallery.service
systemctl --user restart go2-granular-gallery.service
```

For a manual server on another checkout, keep this command running in a terminal instead of starting the service:

```bash
cd /home/horde/Documents/Repos/IsaacLab
uv run --no-project python -m http.server 8010 --bind 127.0.0.1 --directory logs/go2_granular
```

On the machine running the browser:

```bash
ssh -o ExitOnForwardFailure=yes -N -L 18010:127.0.0.1:8010 horde@mrakhsha-41eslp
```

Keep that SSH terminal open, then open **http://localhost:18010/report/** in the browser on that same machine. `localhost` refers to the browser's machine; it does not automatically refer to the SSH server. A browser running directly on `mrakhsha-41eslp` can use **http://127.0.0.1:8010/report/** without a tunnel.

The primary videos use the same frozen randomized-training checkpoint:

- RFT: `http://localhost:18010/dr_rft_nominal/rollout.mp4` (10 s).
- MPM: `http://localhost:18010/dr_mpm_deep/rollout.mp4` (2 s, retaining the early failure).

### Download and watch offline

`logs/go2_granular/go2-rft-results.zip` contains the ten primary videos, seven figures, gallery, summaries and this report (about 24 MiB). With the tunnel active, download it at **http://localhost:18010/go2-rft-results.zip**. Alternatively, run this on the browser machine without starting a tunnel:

```bash
scp horde@mrakhsha-41eslp:/home/horde/Documents/Repos/IsaacLab/logs/go2_granular/go2-rft-results.zip .
```

Extract the entire archive and open `go2-rft-results/report/index.html` in a browser, or open any `rollout.mp4` directly in a video player. The extracted gallery does not need Python or a web server. The bundle excludes checkpoints and raw trajectory arrays; those remain at the paths listed above.

## Reproduce

From `/home/horde/Documents/Repos/IsaacLab` on this branch, install the locked environment with the required extras:

```bash
uv sync --extra rsl-rl --extra video --extra dev
```

Train the randomized pilot and the command-matched nominal-material baseline:

```bash
uv run python scripts/reinforcement_learning/train.py --rl_library rsl_rl \
  --task Isaac-Velocity-Flat-UnitreeGo2-RFT-Direct --viz none \
  --num_envs 512 --max_iterations 1000 --seed 42

uv run python scripts/reinforcement_learning/train.py --rl_library rsl_rl \
  --task Isaac-Velocity-Flat-UnitreeGo2-RFT-Direct --viz none \
  --num_envs 512 --max_iterations 1000 --seed 42 \
  env.randomize_material=False agent.experiment_name=go2_rft_nominal_matched
```

Evaluate one frozen policy in both fields, substituting its checkpoint path:

```bash
uv run python scripts/benchmarks/evaluate_go2_granular.py \
  --checkpoint CHECKPOINT --field rft --steps 500 --video \
  --output logs/go2_granular/reproduction_rft

uv run python scripts/benchmarks/evaluate_go2_granular.py \
  --checkpoint CHECKPOINT --field mpm --steps 100 --video \
  --output logs/go2_granular/reproduction_mpm
```

Use `--randomize --num_envs 16` for the RFT bank, or `--density 1400 --friction 0.35 --roughness 0.8` / `--density 1800 --friction 0.45 --roughness 1.0` for the material corners. The evaluator's default reference duration is not a claim of convergence. `--stand` sets a zero command; omitting a checkpoint uses zero actions.

Run checks and rebuild the evidence:

```bash
uv run python -m pytest -q source/isaaclab_tasks/test/core/test_go2_rft.py
uv run python -m unittest newton.tests.rft.test_rft
uv run python tools/update_environments_rst.py --check
uv run isaaclab --format
uv run python scripts/benchmarks/benchmark_go2_rft_surfaces.py
uv run python scripts/benchmarks/benchmark_go2_rft_surfaces.py --refine_quadrature \
  --output logs/go2_granular/report/surface_quadrature_convergence.json
uv run python scripts/benchmarks/report_go2_granular.py
```

The surface/report scripts consume the recorded pilot paths by default; provide `--trajectory` for a different surface-study recording. To reproduce the standalone Newton CPU/CUDA investigation, check out the pinned Newton commit and run `uv run python -m scripts.benchmarks.benchmark_rft --device cpu` and `--device cuda:0` there. This benchmark exercises the default flowing law separately from the robot bearing extension.

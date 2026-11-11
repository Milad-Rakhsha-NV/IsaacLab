# Go2 granular material investigation

Updated November 11, 2026 (Pacific). No policy was retrained for this investigation.

Completed: **50/50 force probes**, **6/6 material-policy trials** and **8/8 coupling trials**. Pending cases are excluded from the results below.

The selected MPM trials favor the RFT-trained policy over the rigid-ground policy, but **the current MPM configuration is not an accepted numerical reference**. Matching density, friction and bed depth does not establish that the two models represent the same sand. The force study also finds substantial differences outside a nominal foot-force fit.

[Interactive report and graphs](http://localhost:18010/investigation/) · [Four clearly labeled policy videos](http://localhost:18010/report/)

## What corresponds between training and MPM

| Quantity | RFT training | MPM evaluation |
|---|---|---|
| Bulk density | Reset-uniform 1,400–1,800 kg/m³ | Particle mass divided by represented material volume |
| Internal friction μ | Reset-uniform 0.3–0.5 | Native frictional yield parameter |
| Stress convention | τ = √(s:s/2), pressure P = −tr(σ)/3 | Same invariant on the low-pressure friction branch |
| Surface/internal friction ratio | Reset-uniform 0.6–1.0 | Collider friction = ratio × internal μ |
| Initial layer thickness | Reset-uniform 0–118 mm, over a fixed floor | Finite particle bed over a rigid floor |
| Strength multiplier | Reset-log-uniform 0.8–1.2 | No direct counterpart; a response-calibration parameter |
| Velocity regularization | Fixed 0.04 m/s | No corresponding constitutive input |
| Near-rest bearing term | Experimental, fixed coefficient 0.87574 | Emerges from the native material/contact solve |
| Elastic modulus / Poisson ratio | Absent | E = 10¹⁵ Pa, ν = 0.3 |
| Compressive yield cap | No independent input | 10¹⁵ Pa |
| Cohesion, viscosity, hardening, dilatancy | No independently calibrated terms | All zero in this study |
| Material history / free surface | Fixed local field, no evolving packing or free surface | Advected particles and native material history |

The RFT strength scale is `ρ g (894 μ³ − 386 μ² + 89 μ) × strength_multiplier`. Matching ρ and μ does not force this fitted local closure to reproduce an arbitrary MPM boundary-value problem. MPM particles are continuum quadrature samples, not individual sand grains.

A direct native constitutive check gives **τ = 399.99997 Pa at P = 1,000 Pa and μ = 0.4**, against the expected 400 Pa. Thus the friction convention agrees. This checks the yield invariant, not the full stress evolution or intruder force law. RFT's explicit inertial coefficient remains zero; the MPM momentum equation retains inertia.

The original videos use nominal values 1,600 kg/m³, μ = 0.4, surface/internal ratio 1.0 and a 118 mm bed. MPM stiffness and transfer controls were not randomized during RFT training. The three material-bank settings below lie within the RFT input ranges, but do not cover the entire joint randomization distribution. Forward/lateral/yaw commands were sampled at reset during training; the granular field itself was stationary.

## Numerical benchmark

The 50-case native MPM probe study covers prescribed sphere penetration, horizontal drag, a held-out plate, grid/timestep refinement, stress-solver budget, contact projection, horizontal domain size, PIC versus APIC transfer, material inputs, speed, and insertion/hold/withdrawal. The sphere radius is 22 mm, matching the Go2 foot collider; the plate is 50 × 50 × 5 mm. This is separate from the 236 mm knee-to-sole reference used to set the 118 mm training depth cap.

The nominal probe bed is 0.4 × 0.3 × 0.118 m, with a rigid bottom and frictionless vertical walls. Particles fill the exact volume at approximately two samples per grid-cell axis; density and total mass stay fixed as the grid changes. Their radius is half the cube root of represented cell volume because native MPM uses `(2r)^3` for volume, including for non-cubic cells. Assertions check both total volume and particle density before simulation. The 20, 10 and 5 mm grids contain 14,400, 115,200 and 921,600 particles. The bed settles for 0.25 s before loading. Horizontal-domain checks enlarge both dimensions by 1.5, with fixed depth.

Forces sum **every native collider impulse over a 5 ms output frame**, then divide by frame duration. Penetration force is averaged from 5 mm sole penetration to the prescribed maximum depth. Horizontal force is averaged after the first 10 mm of a 40 mm drag. Unless varied, sphere depth is 40 mm and speed is 0.1 m/s. Drag insertion uses 0.1 m/s before the prescribed horizontal speed.

The convergence target is at most **10% change relative to the finer result** for both loading and drag, for both geometries under spatial refinement and for the sphere under time refinement. These are two-level sensitivity checks, not an estimate of asymptotic discretization error.

| Refinement / force component | Coarser result | Finer result | Change relative to finer | 10% gate |
|---|---:|---:|---:|---|
| APIC sphere penetration, 10 → 5 mm grid | 4.003 N | 4.380 N | 8.6% | Pass |
| APIC sphere drag, 10 → 5 mm grid | 2.690 N | 2.636 N | 2.1% | Pass |
| APIC plate penetration, 10 → 5 mm grid | 18.633 N | 16.266 N | 14.6% | Fail |
| APIC plate drag, 10 → 5 mm grid | 3.503 N | 2.546 N | 37.6% | Fail |
| APIC sphere penetration, 0.3125 → 0.15625 ms | 4.003 N | 4.501 N | 11.1% | Fail |
| APIC sphere drag, 0.3125 → 0.15625 ms | 2.690 N | 2.931 N | 8.2% | Pass |

The completed plate and timestep refinements fail the full convergence gate. PIC also shows grid/time sensitivity; the complete case records and plots retain both transfer schemes.

Tightening the stress solve from 200 iterations / tolerance 10⁻⁴ to 500 / 10⁻⁶ changes nominal APIC sphere penetration from 4.003 N to 4.022 N (0.5%).

Across the 50 completed probes, the largest particle-center penetration is **0.222 mm into the floor** and **0.696 mm into the probe**. The largest case-mean momentum residual is **0.0189% of the bed gravitational impulse per frame**. These diagnostics do not establish force convergence or validate physical sand.

![MPM force convergence](http://localhost:18010/investigation/convergence.png)

## Provisional RFT force calibration

The fit uses only nominal APIC sphere penetration and drag on a 10 mm grid with a 0.3125 ms solver step. Every third qualifying output frame is retained, independently of its measured force. It evaluates the actual training sphere surface quadrature: 6 latitudes and 12 longitudes. Forty-five candidates span μ = 0.3–0.5 and surface/internal ratio = 0.6–1.0; each candidate receives a least-squares strength factor clipped to 0.8–1.2. Density stays fixed at 1,600 kg/m³.

RFT loading forces are averaged over the same prescribed substep poses as each 5 ms MPM impulse interval; horizontal translation leaves the uniform RFT field unchanged during steady drag. Each trace is divided by its measured three-component force L2 norm before fitting, giving penetration and drag equal weight. Error is the L2 norm of the vector residual divided by the measured trace norm; the reported fit error is the RMS of these normalized trace errors. The exploratory response target is 20% on the fit and every held-out trace, in addition to numerical convergence of the MPM reference.

The best tested candidate is **μ = 0.300, surface/internal ratio = 1.00, strength factor = 0.872**, with **21.6% fit error**. The nominal μ = 0.4, ratio = 1 candidate would need an unconstrained strength factor of **0.432**, outside the training range; its bounded fit error is 85.6%. No fitted values were applied to the frozen policies.

Completed held-out traces: **6/6**.

| Held-out trace | Fitted RFT vector error |
|---|---:|
| Plate: penetration, depth 40 mm, speed 0.1 m/s | 45.1% |
| Plate: horizontal drag, depth 40 mm, speed 0.1 m/s | 133.7% |
| Foot sphere: penetration, depth 60 mm, speed 0.2 m/s | 10.4% |
| Foot sphere: horizontal drag, depth 20 mm, speed 0.05 m/s | 75.9% |
| Foot sphere: penetration, depth 20 mm, speed 0.05 m/s | 20.8% |
| Foot sphere: horizontal drag, depth 60 mm, speed 0.2 m/s | 25.6% |

Drag errors include the vertical force component. Both fitted and held-out errors use the full three-component force vector.

Separate fits within individual material settings are diagnostic and are not independent validation of a material mapping.

- lower-material: 37.5% fit error.
- higher-material: 17.2% fit error.

For vertical motion at fixed 40 mm maximum depth, speeds [0.05, 0.1, 0.2] m/s give MPM mean resisting forces [3.762, 4.003, 4.466] N.

For drag motion at fixed 40 mm maximum depth, speeds [0.05, 0.1, 0.2] m/s give MPM mean resisting forces [2.539, 2.69, 2.926] N.

During the final part of the fixed-depth hold, MPM gives **-0.019 N** vertical load versus **13.77 N** from nominal RFT. The probe is held kinematically, without an imposed robot-weight load. Rate differences can include unresolved numerical effects; these tests do not establish a physical rate law.

![MPM and RFT force responses](http://localhost:18010/investigation/force-response.png)
![Rate and material-history checks](http://localhost:18010/investigation/rate-and-history.png)

## Coupling and preparation corrections

Multiple proxy passes solve the same physical interval. The initial implementation restored positions and velocities but retained advanced MPM APIC/material history from the previous pass. It now snapshots and restores all persistent namespaced state before each restart, while retaining coupling feedback. A native eight-particle APIC regression fails before the change and passes afterward; all 111 coupled-solver tests pass. Newton fix: `32f455bc4430f0954ae564fb40681fbaca95e7f3`.

Bed preparation now advances the **actual MPM entry state**, preserving its APIC, elastic and plastic history for inference. The robot colliders are held fixed with zero effective collider inertia during preparation; their original properties are restored afterward. Assertions verify that preparation does not move robot poses. Prepared particle positions/velocities are synchronized to the parent scene without replacing material history.

The original four videos use one coupling pass and no preparation. They do not exercise the multi-pass restart bug or the corrected preparation path. They remain labeled as original exploratory recordings. Superseded pre-fix sensitivity results are archived outside the final comparison.

## Frozen-policy comparison

The final comparison uses the same two existing checkpoints, seed 123, and 0.50 m/s forward / zero lateral / zero yaw command. Both policies were trained with 4,096 environments; the rigid checkpoint has 600 iterations and the RFT checkpoint 1,500. Their different training budgets and single training seed mean this is not a controlled causal ablation of material training alone. MPM policy trials last 3 s and begin after a 0.5 s preparation. The coarse original MPM discretization is retained to isolate changes in the material inputs: PIC, 20 mm grid, 4 substeps, 50 stress iterations, one proxy pass and relaxation 0.1. This is exploratory parameter sensitivity, not a converged transfer benchmark.

| Material | Density | Internal μ | Surface/internal ratio | Initial depth |
|---|---:|---:|---:|---:|
| Lower density / friction | 1,400 kg/m³ | 0.3 | 0.6 | 60 mm |
| Nominal | 1,600 kg/m³ | 0.4 | 1.0 | 118 mm |
| Higher density / friction | 1,800 kg/m³ | 0.5 | 0.8 | 100 mm |

| Material | Rigid policy / MPM error | RFT policy / MPM error | Same RFT policy / RFT error | Body-clearance check in MPM |
|---|---:|---:|---:|---|
| Lower density / friction | 0.378 m/s | 0.050 m/s | 0.053 m/s | Neither policy fails in 3 s |
| Nominal | 0.438 m/s | 0.240 m/s | 0.091 m/s | Rigid policy below threshold at 0.66 s |
| Higher density / friction | 0.431 m/s | 0.180 m/s | 0.111 m/s | Rigid policy below threshold at 0.66 s |

All six MPM runs have finite particle states, no bed-edge exits and no rigid-contact buffer overflow. The RFT-trained policy remains above the height/tilt failure thresholds in each trial. An independent audit of the saved body poses finds no foot-placement violation: the minimum signed lateral foot margin is 81.6–110.9 mm across the six trials, above the task's 10 mm threshold. Left/right margins are measured in the base frame; negative values would indicate centerline crossing. The same postprocessing matches the recorded task metric on the three RFT reference trials. This geometric check does not certify loaded-foot slip or swing/landing quality.

The RFT policy has lower tracking error at all three selected settings. Short trials and one evaluation seed cannot establish a success rate. The earlier RFT standing, slip and stepping acceptance failures also remain open.

![Frozen-policy tracking and support forces](http://localhost:18010/investigation/policy-and-support.png)

Whole-robot momentum accounting separates applied granular feedback from rigid-contact impulses after removing gravity. A rigid-floor standing control returns 147.334 N mean upward support over its full 2 s against approximately 147.34 N robot weight, with zero granular load; its mean after 0.5 s is 147.639 N. This supports the force signs/units and aggregation used by the audit. Policy-comparison means exclude the first 0.5 s. The backing floor may support most of the robot in these shallow-layer tests, so improved walking can reflect handling granular resistance while also contacting a rigid floor.

For each accepted physical substep, the audit computes total robot linear momentum `P = Σ mᵢ vᵢ` and the actually applied granular impulse `I_granular`. The inferred rigid-contact resultant is

\[
\overline{\mathbf F}_{\mathrm{rigid}}
=\frac{\Delta\mathbf P-M\mathbf g\,\Delta t-\mathbf I_{\mathrm{granular}}}{\Delta t}.
\]

Internal joint reactions cancel in this whole-robot balance. For multiple coupling passes, only the final re-solve contributes to the accepted substep. The inferred resultant can also include numerical momentum error from the rigid integration; the rigid-floor control checks its aggregate scale.

For the RFT policy in the 3 s MPM trials, the backing floor supplies approximately **90%, 76% and 68%** of the mean upward contact support in the lower, nominal and higher material settings. The nominal 2 s coupling-baseline trial gives 78%. These percentages use rigid vertical support divided by rigid plus granular vertical support over the stated averaging window. The experiment therefore demonstrates handling a shallow sand layer with substantial backing-floor contact, rather than weight support by a deep granular bed.

Relaxation 0.1 means `feedback = 0.9 × previous + 0.1 × new`; it is not a constant 90% removal of the converged force. Its effective lag changes with timestep. The separate 2 s coupling sweep varies relaxation, pass count and substeps without changing the policy, and uses the corrected material-history restart. The fixed-α substep cases therefore measure combined sensitivity to time discretization and feedback filtering, rather than a pure continuum convergence test. To preserve a one-pass filter time constant τ when refining the solver step Δt, use `α = 1 − exp(−Δt/τ)`.

The standalone probe momentum check excludes this relaxed coupling. The robot audit measures the feedback actually applied to the articulation and its rigid-contact resultant; it does not certify whole-system instantaneous momentum conservation across the filtered proxy exchange.

## Reproduction and next gate

From the Isaac Lab feature branch with the pinned dependency installed:

```bash
uv run --no-sync python scripts/benchmarks/validate_go2_materials.py --phase numerics
uv run --no-sync python scripts/benchmarks/validate_go2_materials.py --phase transfer
uv run --no-sync python scripts/benchmarks/validate_go2_materials.py --phase calibration
uv run --no-sync python scripts/benchmarks/validate_go2_materials.py --phase rate
uv run --no-sync python scripts/benchmarks/evaluate_go2_material_bank.py --phase coupling
uv run --no-sync python scripts/benchmarks/evaluate_go2_material_bank.py --phase materials
uv run --no-sync python scripts/benchmarks/evaluate_go2_material_bank.py --phase reference
uv run --no-sync python scripts/benchmarks/report_go2_materials.py --final
uv run --no-sync python scripts/benchmarks/report_go2_granular.py
```

The separate tight-tolerance probe uses `investigate_go2_granular.py --output logs/go2_granular/investigation/tolerance/sphere-vertical-apic-tight --voxel_size 0.01 --substeps 16 --iterations 500 --tolerance 1e-6 --transfer_scheme apic`. The native invariant check and before/after regression logs are retained with the investigation artifacts. Generated traces, PNGs and source snapshots stay under ignored `logs/go2_granular/investigation/`; the scripts and this report are version controlled. Case manifests record worker hashes, material/numerical settings, checkpoint paths and the Newton dependency pin for policy studies.

The next acceptance gate is a converged MPM contact/transfer configuration for sphere and plate loading/drag. Only after that gate should force matching define equivalent materials, followed by longer multi-seed frozen-policy tests. The current results support useful RFT training exposure to granular resistance; they do not yet establish quantitatively matched sand or a fully validated RFT policy.

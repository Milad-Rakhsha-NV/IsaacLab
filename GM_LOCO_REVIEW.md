# GM-Loco comparison and implications for the Go2 RFT study

Reviewed 13 November 2026: the supplied GM-Loco manuscript, its
[project page](https://humanoid-gm-locomotion.github.io/HUMANOID-GM/), and the released Isaac Lab branch at
commit [`14bb11e96a7ad7aa87d6dc889d74e92d87cdee30`](https://github.com/jnskkmhr/IsaacLab/tree/14bb11e96a7ad7aa87d6dc889d74e92d87cdee30).
Our comparison uses the Go2 study's Newton implementation `6cf5b879d576c19279183ee9da0dc4de039a080f`.

The subsequent [Go2 contact-model investigation](GO2_RFT_LOW_SPEED.md) records the fixed-link geometry
correction, matched MPM probes and low-speed integration diagnostics completed after this review.

The most useful change to investigate is the low-speed constitutive treatment. Terrain adaptation and a
broader, calibrated material distribution are also promising. The released implementation contains
coefficient and convention differences that require independent checks. It does not establish a solution
to the Go2 study's prolonged-support failure.

## 1. Shared foundation and important differences

Both implementations use the Agarwal–Goldman–Kamrin 3D RFT framework: integrate local depth-dependent
tractions, with angular dependence and a material-strength scale

\[
\xi=\rho_{\mathrm{eff}}g f(\mu_{\mathrm{int}}),\qquad
f(\mu)=894\mu^3-386\mu^2+89\mu.
\]

The quantity called terrain stiffness in GM-Loco is this intrusion-strength scale, with units N/m³. It
is distinct from an elastic Young's modulus. Their model reduces surface forces to a net body wrench,
which our solver already does.

| Feature | Released Newton G1 task | Current Go2 study candidate |
|---|---|---|
| Force application | Explicit external wrenches through a physics callback | Explicit RFT composition with MJWarp |
| Force update interval | 5 ms physics callback; two rigid substeps | 1.25 ms; four substeps per 5 ms physics tick |
| Contact surfaces | Two ankle/foot box approximations | All selected robot colliders, including feet, calves and thighs |
| Foot quadrature | 5 × 5 points per box face: 150 total, 25 on the sole | 128 spherical parameter triangles; 36 Gauss points per triangle |
| Small-speed magnitude factor | No multiplier that fades the traction magnitude to zero | Flow multiplied by speed/sqrt(speed² + 0.04²) |
| Extra near-rest bearing term | No separate bearing term in the selected 3D backend | Experimental normal-pressure blend, coefficient 0.87574 |
| Force filtering | EMA available, disabled in the linked task | No equivalent EMA |
| Policy memory | Ten observation frames in the linked vanilla PPO task | Current Go2 actor uses one observation frame |

Sources: [task actions and geometry](https://github.com/jnskkmhr/IsaacLab/blob/14bb11e96a7ad7aa87d6dc889d74e92d87cdee30/source/isaaclab_tasks/isaaclab_tasks/contrib/velocity/config/g1_29dof_soft_contact_newton/env_cfg/action_cfg.py),
[physics settings](https://github.com/jnskkmhr/IsaacLab/blob/14bb11e96a7ad7aa87d6dc889d74e92d87cdee30/source/isaaclab_tasks/isaaclab_tasks/contrib/velocity/config/g1_29dof_soft_contact_newton/env_cfg/physics_cfg.py),
[task README](https://github.com/jnskkmhr/IsaacLab/blob/14bb11e96a7ad7aa87d6dc889d74e92d87cdee30/source/isaaclab_tasks/isaaclab_tasks/contrib/velocity/config/g1_29dof_soft_contact_newton/README.md),
and [force kernels](https://github.com/jnskkmhr/IsaacLab/blob/14bb11e96a7ad7aa87d6dc889d74e92d87cdee30/source/isaaclab_tasks/isaaclab_tasks/contrib/soft_contact/_impl/kernels.py).

## 2. Low-speed response deserves priority over more randomization

Our flowing traction has the magnitude factor

\[
a(v)=\frac{\lVert v\rVert}{\sqrt{\lVert v\rVert^2+\varepsilon^2}},\qquad
\varepsilon=0.04\;\mathrm{m/s}.
\]

The separate near-rest contribution is blended with weight
\(b(v)=\varepsilon^2/(\lVert v\rVert^2+\varepsilon^2)\). Its tangential resistance is also regularized
with this velocity scale. Consequently the low-speed response is substantially different from the
published flowing RFT law, in both normal support and resistance to lateral slip.

GM-Loco's selected backend retains a finite depth-scaled traction at low speed. Its normalization uses
`v/(norm(v)+1e-6)` and its angle construction supplies a particular value at exactly zero velocity.
That normalization also changes the angle calculation near zero; it is not an exact mathematical
extension of a unit velocity direction to rest.

I evaluated the original released Warp kernels on matched isolated surface patches. The probe uses
50 mm depth, effective density 1,600 kg/m³, internal friction 0.4 and interface friction 0.4. Area is
1 m², so the reported force per area is a stress; these values are not whole-foot loads.

For a downward-facing patch sliding horizontally at 1 mm/s:

| Treatment | Upward stress | Tangential resistance magnitude |
|---|---:|---:|
| Our current bearing/regularization treatment | 21.33 kPa | 0.213 kPa |
| Our published flowing fit, regularization and bearing disabled | 2.69 kPa | 1.022 kPa |
| Released GM-Loco implementation | 3.56 kPa | 1.367 kPa |

The large normal-response difference comes mainly from the added bearing blend; the small tangential
response indicates a possible source of excessive low-speed slipping in our model. This is a patch-level
observation, not proof of the cause of a robot fall. The plot additionally shows the GM-Loco code after
substituting the published coefficient discussed below, to separate that issue from low-speed handling.

The current Go2 hold test demonstrates creep and substrate contact. It does not identify explicit
coupling as the sole cause. A controlled force-law comparison should precede a larger solver redesign.
GM-Loco's explicit implementation makes that comparison particularly useful.

## 3. Preserve the validated coefficient table and audit conventions

One released coefficient differs from the original source. For the term \(p_2p_3^2\) in \(c_3\),
Table S3 gives **−0.02730**; the released `Material3DRFTCfg.coef_3[15]` contains **−0.20770**. The other
59 cubic coefficients agree with our table. The original value was checked in the downloaded arXiv v3
supplement and visually on SI page 11.

This is consequential. For a horizontal plate moving downward, the dimensionless vertical coefficient
changes from **0.87574 to 1.05614**, about **20.6%**. In a second matched probe, an upward-facing patch
moving upward receives a positive upward stress of about 2.47 kPa from the released code. Substituting
the published coefficient changes it to about −1.92 kPa, agreeing with our unregularized fit in that case.

Sources: [released coefficient table](https://github.com/jnskkmhr/IsaacLab/blob/14bb11e96a7ad7aa87d6dc889d74e92d87cdee30/source/isaaclab_tasks/isaaclab_tasks/contrib/soft_contact/_impl/material.py)
and [original RFT source archive, including SI Table S3](https://arxiv.org/src/2205.14920v3).
These findings concern the inspected code revision; they do not establish which exact revision produced
the manuscript's results, or invalidate its reported hardware outcomes.

Further checks are needed before adopting numerical details:

- The released friction projection preserves the sign of the normal component. Our compressive-only
  projection should remain. At exactly zero velocity the released convention can also produce outward
  force on an upper horizontal face, including with the published coefficient substituted.
- Some oblique-patch probes differ even after correcting that coefficient. Check normal reflection,
  signed tilt, twist, and nearly vertical velocity limits against direct invariant evaluation.
- Their torque is accumulated about the link/ankle origin. Their wrench-composer API assumes forces
  with no supplied position act at COM. This flags an application-point audit: when transferring an
  anchor wrench, use \(M_{\mathrm{COM}}=M_a-(x_{\mathrm{COM}}-x_a)\times F\). Our current COM convention
  and impulse checks should be preserved.
- Their box sampler uses equally weighted endpoint grids; its moment and partially submerged response
  still requires convergence checks. Its sphere helper samples uniform polar angles with equal weights,
  which does not implement uniform spherical area. Preserve our analytic curved normals and area weights
  for Go2 rather than copying that sphere rule. The sphere helper is not used by the linked G1 box task.

These are reasons to use independent plate and body benchmarks, rather than numerical equality to the
released implementation as the acceptance criterion.

## 4. Material coverage can be made broader and better balanced

The linked Newton task samples material parameters at environment resets:

| Parameter | Released Newton task | Existing Go2 training distribution |
|---|---|---|
| Internal friction | 0.1–0.9 | 0.3–0.5 |
| Interface friction | 0.1–1.0, independently sampled | Internal friction × roughness 0.6–1.0 |
| Density input | 1,000–3,000 kg/m³, multiplied by packing 0.5–1.0 | Bulk density 1,400–1,800 kg/m³ |
| Additional strength multiplier | No corresponding sampled multiplier | 0.8–1.2 |

Their effective density spans 500–3,000 kg/m³. Combined with their internal-friction range, the implied
strength scale spans approximately **0.029–12.34 MN/m³**, a factor of 424. Our old Go2 distribution spans
approximately **0.177–1.266 MN/m³**, a factor of 7.2. Those are theoretical range endpoints, not evidence
that every endpoint matches a physical sand or a converged MPM material.
Nominal sand at effective density 1,600 kg/m³ and internal friction 0.4 has strength about 0.487 MN/m³,
already inside our range. A wider distribution alone does not explain a failure on that nominal material.

Source: [reset events](https://github.com/jnskkmhr/IsaacLab/blob/14bb11e96a7ad7aa87d6dc889d74e92d87cdee30/source/isaaclab_tasks/isaaclab_tasks/contrib/velocity/config/g1_29dof_soft_contact_newton/env_cfg/event_cfg.py)
and [effective-density update](https://github.com/jnskkmhr/IsaacLab/blob/14bb11e96a7ad7aa87d6dc889d74e92d87cdee30/source/isaaclab_tasks/isaaclab_tasks/contrib/soft_contact/_impl/soft_contact_model_warp.py).
The project page lists a slightly different upper internal-friction bound; the comparison above uses
the inspected task code.

For Go2, derive ranges from a calibrated MPM material bank, then balance coverage in log strength,
interface roughness and layer depth. Sample independently per environment at reset, within one training
run. A material curriculum can expand these ranges while retaining full velocity commands up to 1 m/s.
The user's no-command-curriculum requirement remains applicable.

Keep density conventions explicit: our bulk density already represents mass per occupied continuum
volume, so an additional packing multiplier must not count porosity twice. Preserve the fully rough
interface convention or justify any change with calibration; the original continuum treatment uses
\(\mu_{\mathrm{surf}}\leq\mu_{\mathrm{int}}\).

## 5. Curriculum and adaptation are useful, separate contributions

The released task has equal hard and soft terrain columns. Success advances the soft columns' rigid
floor from z=0 toward z=−0.12 m while the flat RFT surface remains at z=0. This is a substrate-depth
curriculum. Its existence does not prove the resulting gait obtains all its support from the sand.
Continue reporting granular and substrate impulses separately in our evaluations.

Sources: [scene and floor settings](https://github.com/jnskkmhr/IsaacLab/blob/14bb11e96a7ad7aa87d6dc889d74e92d87cdee30/source/isaaclab_tasks/isaaclab_tasks/contrib/velocity/config/g1_29dof_soft_contact_newton/env_cfg/scene_cfg.py)
and [terrain progression](https://github.com/jnskkmhr/IsaacLab/blob/14bb11e96a7ad7aa87d6dc889d74e92d87cdee30/source/isaaclab_tasks/isaaclab_tasks/contrib/velocity/config/g1_29dof_rigid/mdp/curriculums.py).
The manuscript's free-body comparison is a roughly one-second 35 kg box landing/sliding experiment.
That complements the long loaded holds; it does not replace them.

The manuscript also describes a teacher encoder with privileged terrain information, followed by a
student that infers terrain context from proprioceptive history. It supervises normalized strength
\(\eta=\xi/\xi_{\max}\). This supplies a concrete explanation for improved clearance adaptation under
changing terrain. Even their linked vanilla PPO baseline has observation history, an important difference
from our single-frame Go2 actor.

For an inexpensive Go2 study, start with short history and a small context encoder, comparing it against
the current policy under the same validated physics. Strength, interface response and layer depth are
separate context variables; a scalar strength estimate cannot represent every material difference.
Full VAE teacher/student distillation can follow if the cheap baseline establishes its value.

The manuscript uses 20,000 teacher and 20,000 student iterations. The linked Newton vanilla PPO config
uses 20,000 iterations; it is a separate configuration from the released compositional teacher/student
pipeline. Those budgets are much larger than our 1,500-iteration Go2 runs. Their convergence cannot be
inferred for our runs from matching the number of parallel environments.

## 6. Recommended Go2 sequence

1. **Isolate the low-speed law.** On identical prescribed states, compare the current blend, published
   flowing RFT, and a candidate treatment for stationary contact. Keep coefficients, geometry, forces,
   frames and all integration settings fixed. Include slow lateral sliding and unloading.
2. **Validate support and time integration.** Use preloaded holds, release, tangential loading, work cycles,
   timestep refinement and the zero-layer rigid limit. Near rest, a candidate closure should select a
   load-balancing reaction from an admissible set, rather than prescribe one fixed force at zero speed.
   An implicit or projected reaction treatment is a candidate to test, not a fix established by this review.
3. **Match Go2 MPM responses.** Compare the same sphere/lower-leg shapes, load histories and velocities,
   including the paper's landing/sliding style of test. Use a deep-bed control to separate substrate
   confinement from finite-layer behavior. Validate the MPM reference and actual two-way impulses.
4. **Choose the lowest-cost accepted force evaluation.** Retain the 128-triangle foot budget. Investigate
   lower/adaptive quadrature and 200/400/800 Hz force updates separately. A held body wrench can reduce
   cost, but requires new stability and work checks. Body-wrench aggregation is already implemented.
5. **Run one 4,096-environment training experiment** with calibrated reset randomization, a material-only
   curriculum, full commands and explicit zero-layer anchors. Test short policy history before committing
   to the full teacher/student training budget. Evaluate frozen weights in the accepted MPM configurations.

Go2 remains the target. The review supplies source inspection and isolated-patch measurements; it does
not change the production solver, rerun training, or establish that the support gate now passes.

## Diagnostic artifacts

The live review folder contains `patch-comparison.json`, `patch-comparison.npz`, the standalone probe,
and `low-speed-comparison.png`. The probe executes the released Warp functions, then the same functions
with only the published coefficient substituted, alongside our current and unregularized flowing laws.
It performs no body integration, MPM run or RL training. Geometry area, depth and material values are
fixed across treatments. Small-speed and angular differences therefore concern the traction evaluation.

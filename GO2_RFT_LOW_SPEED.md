# Go2 contact-model correction and low-speed validation

13 November 2026. [Portable report and numerical records](results/go2-contact-refinement/low-speed/index.html).
This follows the [GM-Loco review](GM_LOCO_REVIEW.md). Go2 remains the target; G1 is on hold.

The fixed-link geometry correction and an opt-in articulated near-rest reaction solve are implemented.
**No combined material profile is accepted for training.** Numerical support checks pass, but the
insertion-calibrated material misses independent lateral MPM response. No new policy training has
started. The next run retains 4096 environments, full commands through 1.0 m/s, reset randomization,
a material-only curriculum and short observation history. G1 remains on hold.

## Articulated reaction solve

[Current report, support video and numerical records](results/go2-contact-refinement/low-speed/static-contact/index.html).
Newton `e73147c247a5df0a366548aaaac709e882b9f1cf` adds bounded normal reactions and Coulomb friction
near rest. A projected iteration resolves forces against the complete rigid solver, including joint
drives and rigid contacts. It uses incipient normal-intrusion pressure as a capacity estimate;
this is an experimental constitutive extension, not the published moving law's exact static yield set.
The default explicit path is preserved. Reactions are numerical warm starts, not elastic strain or
terrain history. Only selected worlds' warm starts clear on reset.

Repeated trials previously advanced MuJoCo native time, filtered actuator state and delay history
multiple times. The coupling interface now saves and restores native integration state between trials.
Only one accepted interval advances time and contributes granular impulse. A native CPU/GPU regression
with filtered actuation and two rigid substeps fails before this correction and passes afterwards,
including CUDA graph replay. Public Newton state and native integration history are restored separately.

The new mode is controlled by `static_contact_iterations`, `static_contact_speed`,
`static_contact_strength` and `static_contact_relaxation`. Legacy bearing must be zero. Prescribed
`evaluate()` still reports moving-law loads; static reactions require the load and dynamics context in
`step()`. [Newton's solver documentation](https://github.com/Milad-Rakhsha/newton/blob/milad/rft-mpm/docs/solvers/rft.rst)
defines the equations, force units, iteration and residual.

### Numerical support checks

- A freely moving loaded flat plate holds under sub-limit lateral force, slips above the Coulomb limit,
  releases when lifted and dissipates an unforced lateral velocity. CPU/GPU and two timesteps pass.
- The isolated 128-triangle Go2 sphere sustains 18.42 N without contacting the floor. Initializing near
  the settled MPM depth yields approximately 82.47 mm depth with four or sixteen reaction iterations.
  The ten-second shallow-start case also stops creeping, but settles too deeply: 106.37 mm versus
  79.73 mm in the fine MPM reference. Removing numerical creep does not match loading history.
- Full Go2 with fixed joint PD targets holds in an **unbounded RFT half-space, with no backing floor**.
  A 0.20 m/s transition and four reaction iterations settle to a body height of 290.78 mm; sixteen
  iterations give 291.12 mm. Halving the four-iteration timestep gives 291.09 mm. Final support is
  147.34 N, equal to robot weight, and foot speeds are below 1 micrometre/s.
- A 0.02 m/s transition still chatters. Transition sensitivity is an explicit validation limit.
  Zero projected residual on inactive/flowing patches does not certify stable dynamics.

The ten-second support video uses nominal RFT strength and a stationary PD controller. It does not
show a newly trained walking policy or an MPM-matched finite bed. The weaker insertion-fit material
also reaches rest without a floor, but at deeper immersion than the intended 118 mm layer.

Passed: **33 focused Newton cases**, **50 additional proxy-coupling regressions** and **10 Isaac Lab
cases**. The focused Newton/Isaac checks also pass in a separate runtime installed from the pinned
Newton commit, without a working-tree source overlay. Coverage includes empty-layer articulated rigid
equivalence, masked native/RFT history reset and graph replay. Historical runtimes remain separate.

### Independent MPM loading checks

The new sphere fixture permits vertical and forward translation while constraining orientation.
It maintains 18.42 N downward load, then applies 1 N or 10 N sideways after one second. MPM is slowly
preloaded to 52.32 mm and retains its material history. RFT begins near the previously settled MPM
depth and is held for one second; the two depths are close when lateral loading starts. This tests
response from similar loaded poses, not identical hidden granular states.

| Sideways load | Method | Displacement over next second | Final sole depth | Backing-floor support |
|---|---|---:|---:|---|
| 1 N | RFT, strength scale 0.373 | 0.00019 mm | 83.95 mm | No |
| 1 N | MPM, 10 mm cells | 0.100 mm | 85.25 mm | No |
| 10 N | RFT, strength scale 0.373 | 34.17 mm | 118.05 mm | Yes |
| 10 N | MPM, 10 mm cells | 12.75 mm | 100.74 mm | No |
| 10 N | MPM, 5 mm cells | 17.13 mm | 95.19 mm | No |

The stronger lateral case remains a failed material comparison. The slow-insertion fit alone
underestimates resistance during combined loading; widening reset ranges cannot establish a nominal
material match. The 5 mm-grid check changes lateral displacement by 4.38 mm (34%) and also changes
the loaded starting depth from about 84.3 to 77.2 mm; the RFT trial starts near 84.0 mm. The reference
is therefore not yet grid converged or perfectly state matched. These short fixtures do not certify
whole-robot MPM convergence. Unloading changes the applied load to 5 N upward: both models
leave the bed and granular force vanishes once the whole sphere is clear. Closed-surface compressive
withdrawal drag is distinct from adhesion on a withdrawing lower face.

Reset randomization now interpolates from configured nominal density, friction, roughness and
strength, preserving a calibrated nominal material at zero curriculum difficulty. The previous
hard-coded nominal values would discard that calibration on reset. Commands retain their full
distribution throughout the material curriculum.

### Cost and remaining gate

At 4096 environments, four reaction iterations measure **5734 transitions/s**, or **0.714 s per control
step**, using about **4.83 GiB** GPU memory. The corrected explicit profile measured 21,347
transitions/s: the new solve costs 3.72 times as much. Collection for 1500 PPO iterations of 24 steps
would take about 7.1 hours at this rate, excluding optimizer work. Force evaluation itself stays near
11–21 ms on the common pose; repeated rigid trials account for the increased step cost.

The combined lateral/normal material closure and collection cost remain gates before retraining.
`scripts/benchmarks/configs/go2_rft_static_contact.json` is explicitly marked `training_ready: false`.
The existing policy videos and checkpoints retain their original solver provenance.

Rebuild the portable report from the checked-in numerical archive:

```bash
uv run python scripts/benchmarks/report_go2_static_contact.py \
  --input results/go2-contact-refinement/low-speed/static-contact \
  --output /tmp/go2-static-contact-report
```

## Fixed foot/calf boundaries

Go2's spherical foot and calf cylinders occupy separate bodies connected by a fixed joint. Previously,
RFT rejected internal patches only among colliders belonging to one body. Consequently an overlapping
calf surface inside the foot could produce an external granular force.

Newton now groups colliders connected by enabled fixed joints and tests containment in the owning
collider's current body frame. Planar and analytic curved integration use the same rule. Collision
shapes, inertial properties and triangle counts are unchanged. Each whole spherical foot retains **128
triangles**, including 64 on its lower hemisphere. Movable joints remain separate integration boundaries;
fixed-link topology is construction-time data.

In one prescribed buried Go2 pose:

| Motion | Previous vertical force | Corrected vertical force |
|---|---:|---:|
| Downward translation at 0.1 m/s | 243.15 N | 190.38 N |
| Horizontal translation at 0.1 m/s | 38.79 N | 29.06 N |
| Resting bearing extension | 186.72 N | 132.70 N |

The rest case removes **54.02 N of internal-patch support**. This is a controlled geometry comparison.
A training ablation is required to determine its effect on policy transfer.

The new regression fails on the previous implementation with 36.65 N on an entirely enclosed fixed-link
sphere. It passes after the correction, with planar and curved quadrature. An exposed fixed-link sphere
and a separate hinged sphere remain active, checking connectivity and coordinate-frame handling.
All **23 Newton RFT cases** and **six Isaac Lab geometry/native-coupling cases** pass, including the
zero-layer comparison with rigid dynamics.

The final profile uses order-six quadrature on feet and distal calf colliders: 36 integration nodes per
parameter triangle, including 4608 over one whole foot. The RFT surface retains 128 triangles;
the collision sphere is unchanged.
Against order-24
integration on 64 recorded poses crossed with 16 field cases, pooled force/torque differences are
0.52%/0.94% for feet and 0.53%/0.48% for calves. Global 95th-percentile force differences are 1.43% and
1.80%; these do not bound every individual case. At 4096 environments the profile measures about
21,347 zero-action transitions/s. Common-state RFT evaluation takes 11.6 ms for shallow immersion and
23.9 ms for deep immersion. These measurements describe simulation cost before PPO training.

## Matched MPM foot probes

The probe isolates the actual 22 mm-radius spherical collider. It has one vertical DOF, 1 kg inertial
mass and an independently applied load. It is slowly inserted, briefly held, and then released. There
is no robot controller or calf. Material inputs are bulk density 1600 kg/m³, internal friction 0.4,
interface friction 0.4, zero cohesion/dilatancy and the stiff elastic limit. MPM uses APIC and eight
substeps per 5 ms frame. Sparse-grid overflow and finite particle state are checked.

| Bed depth | Cell size | Coupling passes | Iteration cap | Applied load | Final sole depth |
|---|---|---|---|---|---|
| 118 mm | 10 mm | 1 | 200 | 18.42 N | 77.93 mm |
| 118 mm | 10 mm | 4 | 200 | 18.42 N | 77.76 mm |
| 118 mm | 5 mm | 1 | 200 | 18.42 N | 79.81 mm |
| 236 mm | 10 mm | 1 | 200 | 18.42 N | 86.19 mm |
| 118 mm | 5 mm | 1 | 400 | 18.42 N | 79.73 mm |

The shallow coarse cases run about ten seconds after release. The finer and deep cases run three
seconds and reach negligible final velocity. All five sustain the load without rigid floor contact.
The deep case records the force delivered to the rigid solver independently of the foot's momentum;
the 95th-percentile vertical balance error is about 1.4 micro-Newtons. Trial coupling solves contribute
only their final accepted source force to this diagnostic.

Coupling refinement changes settled depth by 0.17 mm. Grid refinement changes it by 1.88 mm and changes
the insertion-fit strength by about 9%; further force convergence remains relevant. Doubling the MPM
iteration cap changes fine-grid settled depth by only 0.078 mm and the insertion-fit scale from 0.376 to
0.373 in this fixture. The deeper bed
also changes the response. A finite RFT clipping plane does not model substrate confinement.

A separate coarse-grid probe under the nominal quarter-robot load, 36.83 N, reaches 109.43 mm depth
after three seconds and is still settling at 0.33 mm/s. The 8.6 mm floor clearance is below one grid
cell, so this case does not certify a converged nominal-load hold.

## Material matching and static reaction

For insertion at 0.02 m/s over 15–50 mm sole depth, least-squares fits require **0.34–0.38** of nominal
published RFT strength in the shallow cases and **0.28** in the coarse deep bed. The finer shallow fit
uses 0.376 with 4.7% normalized fitting error. The pressure/shear invariant conventions were inspected:
Newton's MPM friction parameter uses the same shear-to-pressure ratio as the RFT continuum model.
Equal density/friction inputs nevertheless produce different measured intrusion forces in these cases.

These are insertion-only fits for one sphere. Tangential sliding, unloading, lower-leg geometry,
bed-depth sensitivity and reference convergence must inform the training distribution. The fitted
scale lies below the earlier 0.8–1.2 strength range at the same density/friction. Altering friction also
changes angular resistance, so a strength match alone does not establish an equivalent material.

A separate scalar diagnostic starts the sphere at rest at an admissible RFT depth in an unbounded bed.
Explicit near-rest forces produce net sinking. A discrete implicit reaction selects

\[
 F=\operatorname{clip}\left(W-\frac{m v_n}{\Delta t},\;F_{\uparrow}(z_n),\;F_{\downarrow}(z_n)\right),
 \qquad m(v_{n+1}-v_n)=\Delta t(F-W).
\]

Here upward velocity and force are positive, and downward applied load is positive. The two bounds are
computed from the published directional flowing law. An admissible load gives zero next velocity;
overload gives yielding motion. Holds at four and eight substeps have zero drift and zero positive
granular work. Load removal, reversal and overload are recorded separately.

**This scalar oracle is not an articulated solver.** Direct implicit and projected articulated
prototypes were explored; they have not met the coupling, unloading and cost acceptance checks.
Published RFT coefficients are retained. The geometry correction alone does not establish static
sticking, material equivalence or improved learned transfer.

## Reproduction and provenance

Newton's correction is commit `f59677bd8c5037d68d4196c323ea0b4459c18a9c` on `milad/rft-mpm`, pushed to
`milad-rakhsha`. Isaac Lab pins this revision. Historical checkpoint runtimes are preserved.

The benchmark entry points are:

- `scripts/benchmarks/investigate_g1_loaded_foot.py --geometry sphere`: Go2 dynamic-load and MPM preload fixture.
- `scripts/benchmarks/benchmark_go2_rft_low_speed.py`: scalar holds; `--load_cycle` adds removal/reversal/overload.
- `scripts/benchmarks/benchmark_go2_contact_profile.py --mode compound`: controlled fixed-link boundary comparison.
- The same script's `--mode motion` retains fixed-link connectivity in batched recorded-motion fixtures.
- `scripts/benchmarks/report_go2_rft_low_speed.py`: portable report, fitted-force traces, plots and artifact hashes.

The contact profile `scripts/benchmarks/configs/go2_rft_fixed_union.json` retains the triangle budget
and refines distal-calf quadrature at the new intersection boundary. Its `training_ready` flag remains
false. The archived policy report and videos retain their original solver and training geometry.

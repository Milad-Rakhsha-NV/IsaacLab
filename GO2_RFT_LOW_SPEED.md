# Go2 contact-model correction and low-speed validation

13 November 2026. [Portable report and numerical records](results/go2-contact-refinement/low-speed/index.html).
This follows the [GM-Loco review](GM_LOCO_REVIEW.md). Go2 remains the target; G1 is on hold.

The fixed-link geometry correction is implemented and tested. **A replacement near-rest coupling model
has not been accepted, and no new policy training has started.** The next training run retains 4096
environments, full forward commands through 1.0 m/s, reset randomization and a material-only curriculum.

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

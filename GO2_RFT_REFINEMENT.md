# Go2 RFT contact-resolution investigation

The selective surface implementation and numerical measurements are complete. **The candidate is not
cleared for training:** prolonged static support fails, calf force percentiles exceed the proposed target
in some material cases, and direct force-evaluation cost exceeds the initial twofold target. G1 remains on hold.

The [HTML report](results/go2-contact-refinement/index.html) includes geometry, accuracy, cost and settling
plots. The [portable results folder](results/go2-contact-refinement) contains raw traces, full common-motion
recordings, profiles, reproduction commands and artifact checksums. The live report uses the existing SSH
tunnel at [localhost:18010/go2-contact-refinement/](http://localhost:18010/go2-contact-refinement/).

## Geometry and integration

| Collider on each leg | Previous retained triangles | Candidate triangles | Curved integration |
|---|---:|---:|---|
| Entire spherical foot, radius 22 mm | 132 | **128** | Order 6, 36 Gauss points per triangle |
| Upper calf cylinder | 48 | 48 | Previous planar rule |
| Middle calf cylinder | 48 | 128 | 16 circumferential × 3 axial panels, order 3 |
| Bottom calf cylinder | 48 | 96 | 16 circumferential × 2 axial panels, order 3 |
| Other robot colliders | Existing counts | Unchanged | Previous planar rule |

The foot has 64 triangles on its lower hemisphere. The count of 128 covers the entire sphere. A cylinder's
count includes side panels and minimally triangulated caps. All selected legs, calves and thighs remain
coupled, with their original collision dimensions, local transforms, masses and inertias.

The full robot has 2,032 retained integration triangles, compared with 1,536 previously. More accurate
curved normals are essential: subdividing an existing planar facet preserves its geometric normal error.
The new rule projects integration points onto the analytic sphere/cylinder, applies the mapping's area
Jacobian and evaluates normal activation using the sphere-center or cylinder-axis velocity. Tangential
spin therefore adds no spurious radial normal velocity. Parameter triangles and Gauss points are separate
budgets. Immutable point templates are shared across environment clones.

`RFTSurfaceResolutionCfg` matches finalized collider labels. Conflicting and unmatched rules raise errors.
The opt-in profile is [go2_rft_contact_128.json](scripts/benchmarks/configs/go2_rft_contact_128.json).
Historical task defaults and trained checkpoint geometry are preserved. Same-body primitive overlaps are
rejected; shielding and overlaps between articulated links remain a model limitation.

## Numerical results

The motion audit prescribes identical poses and exact COM twists from the accepted rigid controls at
0.5 and 1.0 m/s. It selects 32 frames from each ten-second recording and evaluates 16 combinations of four
heights, two bulk frictions and two interface ratios, at density 1,800 kg/m³ and strength multiplier 1.2.
No policy inference, dynamics integration or parameter fitting occurs in the force comparison.

Against order-24 integration on the same curved parameter patches:

| Foot surface | Pooled force difference | Pooled torque difference |
|---|---:|---:|
| Previous UV sphere | 6.09% | 12.25% |
| 128 planar facets | 5.22% | 9.86% |
| 128 analytic curved patches | **0.51%** | **0.94%** |

Normalized errors use 1 N and 0.01 Nm denominator floors. The worst material-case foot 95th percentiles
are 1.78% force and 3.94% torque. Individual foot maxima are 3.82% and 15.35%; the higher torque percentages
include small moments. The calf force percentile reaches 3.20%, above the proposed 2% target. Raising
quadrature order improves it but requires another cost/accuracy selection. Unselected surfaces retain
their original resolution, so this is a contact-region audit rather than whole-robot convergence.

The order-16 to order-24 foot change is 0.15% pooled force and 0.26% pooled torque, with 0.37%/0.72%
95th-percentile changes. Its largest normalized torque change remains about 7%; the numerical reference
is not certified pointwise in every low-load case. Prescribed sphere probes and independent analytic
sphere/cylinder bearing integrals provide additional checks. None establishes physical sand equivalence.

At 4,096 native Go2 environments on one RTX PRO 4500:

| Measurement | Previous | Candidate |
|---|---:|---:|
| Zero-action environment transitions/s | 60,929 | 33,611 |
| RFT evaluation, identical shallow motion | 3.53 ms | 8.12 ms |
| RFT evaluation, identical deep motion | 3.47 ms | 11.98 ms |
| Total reported GPU use | 3.61 GiB | 3.86 GiB |

Whole-environment cost rises about 1.81 times. Direct RFT evaluation rises 2.30–3.45 times and misses the
initial twofold target. Zero-action trajectories can differ; the direct force timings use identical
recorded states at two immersion offsets. PPO updates are excluded. GPU use includes the CUDA context
and other processes.

## Static-support gate

A vertical prismatic fixture applies independent loads of 18.42, 36.83 and 55.25 N to 1 kg intruders in a
118 mm layer. Nominal load is one quarter of Go2's weight. The two supported loads start at rest at
52.32 and 99.13 mm depth, with independently checked bearing capacities of 20.26 and 40.52 N: 110% of their
applied loads. This removes landing impact from the support measurement. The spherical intruders use the
new 128-triangle curved rule for ten seconds. At nominal load, the foot reaches the rigid substrate.
At half load, penetration from two to ten seconds grows by 20.3, 17.8 and 12.8 mm for 4, 8 and 16 substeps.

The proposed support gate permits at most 2 mm additional penetration while remaining clear of the
substrate. It fails. A small depth change after floor contact is not successful granular support. This
fixture has no matched Go2 MPM hold reference yet; it isolates the RFT time-integration issue.

The next solver stage must select a bounded near-rest reaction consistently with articulated dynamics.
Prolonged holds, overload, tangential loading, unloading work, impulse balance, timestep convergence and
the zero-layer rigid limit must pass before matching Go2 MPM forces and deriving reset-randomization ranges.
Another RFT training run waits for these gates. It will retain 4,096 environments, independent reset
randomization, a material curriculum and full commands up to 1.0 m/s without a command curriculum.

## Verification and provenance

All 21 Newton CPU/GPU RFT cases and six Isaac Lab geometry/native-coupling cases pass, including masked
resets, all-leg coupling, articulated impulse balance and the rigid zero-layer limit. Disabling curved
integration fails the independent analytic-load regression. A separate cache regression fails if
per-collider resolution is omitted from the geometry-cache key. Repository formatting and lint checks pass.

Newton implementation: `6cf5b879d576c19279183ee9da0dc4de039a080f`, branch `milad/rft-mpm` in the personal fork.
Isaac Lab pins that immutable revision in `pyproject.toml` and `uv.lock`. The new runtime is
`/home/horde/.cache/rft-study/go2-contact128-venv`; archived Go2 and G1 runtimes keep their original Newton
revisions. Measurements were collected against the same source before the Newton commit and retain
source/kernel hashes. The archived recordings include their original checkpoint and robot COM metadata.

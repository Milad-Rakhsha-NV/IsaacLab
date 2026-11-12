# RFT foot-resolution study

[View the HTML report](index.html) · [Full follow-up methods](../../G1_RFT_REFINEMENT.md)

This study measures how the RFT representation of a foot affects integrated force and torque. It separates surface resolution from granular-material validity and learned policy performance. Existing task defaults and checkpoints retain their coarse surfaces. No policy has been retrained with these finer meshes.

## G1: integration over a flat foot

The current collision box has **2 triangles on each sole** and 12 triangles over each closed foot. The audit uses 240 recorded current-policy poses with exact COM twists, swept over 41 material/height combinations. Independent batched foot queries reproduce the original full-model foot wrenches within 0.000004 N or Nm. Moments use the actual foot body COM.

Compared with a 10 mm maximum-edge reference, the larger of the left/right pooled sole errors is:

| Maximum edge | Sole triangles per foot | Whole-foot triangles | Force error | Torque error |
|---|---:|---:|---:|---:|
| Current, no subdivision | 2 | 12 | 5.09% | 18.01% |
| 40 mm | 128 | 528 | 0.27% | 0.58% |
| 20 mm | 512 | 2,112 | 0.10% | 0.26% |

The flat geometry is exact before refinement; subdivision improves sampling of varying depth, velocity and activation during foot rotation. The raw record also includes complete-foot reactions, percentiles and maximum errors. Its loaded-sample percentiles use 20 N / 1 Nm denominator floors.

## Go2: geometry and integration on a curved foot

Go2's collider is a **22 mm radius sphere**. The current UV mesh has 144 authored faces, 132 retained faces and maximum edges of 15.6 mm. Small nearly degenerate pole faces remain in the solver count. Refining its curvature requires regenerating the sphere surface; subdividing flat facets retains their original normals.

The prescribed benchmark uses 150 centered-sphere states at four bulk/interface friction combinations: 1–44 mm penetration, stationary support, insertion, translation, withdrawal, 8 rad/s spin and three orientations. The corrected bearing law is used. Torques are about the sphere center. The 384-segment UV reference has about 0.51 mm maximum edges; the 192 → 384 refinement changes pooled force and torque by about 0.10% and 0.49%.

| Foot surface | Triangles per foot | Maximum edge | Pooled force error | Pooled torque error |
|---|---:|---:|---:|---:|
| Current UV mesh | 132 | 15.6 mm | 8.74% | 18.89% |
| Uniform sphere, 3 subdivisions | 1,280 | 3.62 mm | 0.92% | 3.34% |
| Uniform sphere, 4 subdivisions | 5,120 | 1.82 mm | 0.32% | 1.36% |
| Uniform sphere, 5 subdivisions | 20,480 | 0.91 mm | 0.10% | 0.43% |

Pooled errors weight larger reactions more strongly. Difficult fully immersed spinning cases retain larger errors; a good pooled value does not certify every motion. The JSON includes 95th percentiles by depth, with denominator floors of 1 N and 0.01 Nm. These are deliberately retained alongside the pooled values. The diagrams classify lower-half triangles by their centroids at the nominal pose; the actual submerged leading surface changes during motion.

## Implementation requirements and limits

Refinement should be configurable per foot collider and cached when constructing the model. The same finer RFT surface is reused at every physics substep. Rigid/MPM collision shapes, authored transforms, masses and inertias remain fixed, with all selected leg and thigh surfaces still coupled.

The G1 128-triangle sole and Go2 1,280-triangle uniform sphere are candidates to compare with finer references. Actual Go2 foot trajectories, per-depth force/torque errors and throughput/memory at **4,096 environments** must determine the final settings. Material calibration should be checked with the selected surface before retraining and repeating the MPM comparisons.

These measurements establish numerical self-convergence within the RFT law. They do not validate real sand, establish equivalence to MPM, or fix the separate near-rest coupling problem. The prolonged G1 load-controlled fixture already uses a 20 mm edge (512 sole triangles) and still exhibits creep. The larger coarse-policy MPM rendering batch is stopped while foot resolution is investigated; its completed recordings remain historical measurements.

## Files

- `index.html` and three PNGs: standalone report, G1 sole subdivisions, Go2 lower-surface meshes and force/torque convergence.
- `foot-quadrature-convergence.json`: exact G1 foot surfaces, sampled frames, material cases, input hashes and errors.
- `g1-motion-0.5.npz`, `g1-motion-1.npz` and corresponding summaries: exact recorded poses and COM twists. File hashes match the trajectory records in the G1 audit.
- `geometry.json`, `rft-profile.json`: authored G1 probe geometry and frozen exploratory material profile. The profile remains explicitly unaccepted as an equivalent material.
- `go2-foot-resolution.json`, `go2-foot-resolution.npz`: synthetic sphere cases, numerical errors and raw wrenches. `wrench_i` corresponds to `results[i]` in the JSON; `body_pose` and `body_twist` record the common prescribed input. Materials follow JSON order.
- `manifest.json`: source revision, computational Newton revision and expanded-file SHA-256 hashes.

## View and reproduce

Fetch Git LFS data and serve the results from the Isaac Lab repository root:

```bash
git lfs pull --include="results/rft-foot-resolution/**" --exclude=""
uv run --no-project python -m http.server 8000 --bind 127.0.0.1 --directory results
```

Open <http://localhost:8000/rft-foot-resolution/>. For a remote server, forward its port with `ssh -N -L 18000:127.0.0.1:8000 USER@HOST` and use local port 18000.

The computational Newton revision is `277c30238619349d343104191d8e40582fa7ac70`, with kernel SHA-256 stored in both numerical records. In an Isaac Lab environment containing that revision:

```bash
uv run --no-sync python scripts/benchmarks/benchmark_go2_foot_resolution.py \
  --output /tmp/go2-foot-resolution.json

uv run --no-sync python scripts/benchmarks/benchmark_g1_rft_quadrature.py \
  --profile results/rft-foot-resolution/rft-profile.json \
  --trajectories results/rft-foot-resolution/g1-motion-0.5.npz \
                 results/rft-foot-resolution/g1-motion-1.npz \
  --output /tmp/g1-foot-resolution.json
```

The G1 command loads the robot asset and evaluates prescribed states; it does not load a policy or retrain. The Go2 command constructs centered analytic-radius sphere probes and needs no robot environment. Reproduced results may differ slightly through floating-point reductions. Hashes verify the archived original files, rather than requiring bitwise-identical reruns.

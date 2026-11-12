# G1 force matching and transfer investigation

The original study is preserved in `results/g1-rft-mpm/`. This follow-up separates numerical reference quality, RFT force approximation and learned robustness. MPM supplies no training transitions.

[Live report](http://localhost:18010/g1/refinement/) · [Learning curves](http://localhost:18010/g1/refinement/training.html) · [Original comparison](http://localhost:18010/g1/baseline-report/)

## Force-model correction

The experimental near-rest bearing extension supplied pressure on withdrawing surface patches. A prescribed closed 40 mm cycle on the exact G1 foot produced net input work of **−0.662 J**, returning more energy than supplied. Newton commit `277c30238619349d343104191d8e40582fa7ac70` releases bearing pressure on outward-moving patches. The same cycle now dissipates **3.876 J**. All 19 RFT regression tests pass, including CPU/CUDA coverage; the focused work-cycle regression fails before the fix and passes afterward.

Original Go2/G1 checkpoints used Newton `32f455bc4430f0954ae564fb40681fbaca95e7f3`. Their recordings remain historical observations. The defect could enable learned exploitation, but does not establish the sole cause of transfer failure. The correction concerns the added bearing extension, rather than the published flowing RFT law.

## Matched fixtures and numerical reference

Both solvers use the finalized G1 collision mesh, approximately **203 × 65 × 19 mm**, with its authored ankle offset. Winding is outward and enclosed volume positive. Prescribed insertion, unloaded hold, drag, pitch, roll and withdrawal apply identical poses and velocities. Forces and moments sum every substep's collider impulse about the ankle origin. Direct rigid foot/floor reactions are excluded. Material is prepared for 0.5 s before loading.

The 10 → 5 mm grid change fails the fixed 10% phase-mean force/torque gate; **5 → 3.33 mm passes**. Timestep (8 → 16 substeps), stress-iteration (200 → 400) and domain-size checks pass. Normalization floors of 20 N and 1 Nm allow 2 N and 0.1 Nm near zero. Reducing sampling from eight particles per voxel to one fails strongly and is unsuitable as a cheap replacement. These are isolated-foot self-convergence results, not validation of whole-robot MPM or physical sand.

The foot-only quadrature audit uses 240 poses with exact recorded COM twists from both final-policy command trials, swept over 41 material/height combinations. A batched evaluation agrees with the original full-model foot wrenches within 0.000004 N or Nm. The field's free-surface and bottom planes are retained. Counts below are **per foot**, without torso or leg surfaces. Pooled L2 errors report the larger of the left/right foot errors relative to a 10 mm maximum-edge reference.

| Maximum triangle edge | Sole triangles | Whole-foot triangles | Sole force / torque error | Whole-foot force / torque error |
|---|---:|---:|---:|---:|
| Current, no subdivision | 2 | 12 | 5.09% / 18.01% | 5.08% / 16.20% |
| 40 mm | 128 | 528 | 0.27% / 0.58% | 0.26% / 0.51% |
| 20 mm | 512 | 2,112 | 0.10% / 0.26% | 0.09% / 0.23% |
| 10 mm reference | 2,048 | 8,448 | — | — |

Two sole triangles represent the flat collision geometry exactly; spatially varying velocity, clipping and activation still require adequate integration. The 40 mm edge is a practical refinement candidate. Existing checkpoints retain the coarse mesh. Refining quadrature does not fix the separate static-coupling failure. Raw records also retain loaded-sample percentiles and maxima, so the pooled errors do not conceal individual difficult poses.

A vertical prismatic fixture separates inertial mass from applied load. At 158.13 N, fine MPM settles to 95.4 mm at a 5 mm grid and 101.5 mm at 3.33 mm, a 6.1% change, without backing-floor contact. At a 10 mm grid, inertial masses of 16 kg and 1 kg settle stably under the same load. This does not prove whole-biped coupling stability.

## Force-informed profile

The corrected RFT fit uses moving insertion/drag wrenches and a simulated loaded-foot response at two seconds. Unloaded holds, rotations and withdrawal remain held out. Selected strength multiplier: **0.7102**; bearing multiplier: **0.45**; bulk friction: **0.4**; effective interface/internal ratio: **0.3**; velocity regularization: **0.04 m/s**. At two seconds, corrected RFT reaches 97.4 mm with four substeps, 4.1% from fine MPM. Eight substeps give 93.1 mm, a 4.6% change from the four-substep result.

**Held-out wrench matching still fails.** This is an exploratory force-informed approximation, not a validated equivalent material. The fitted RFT interface ratio is effective; MPM retains ratio 1.0. A local memoryless closure cannot represent evolving cavities or plastic loading history.

## Prolonged static-hold failure

The two-second agreement is a transient response fit. With the same 158.13 N load, corrected RFT keeps creeping: four substeps reach the backing floor around 7.7 s. At ten seconds, depths are approximately 118.1, 109.9, 100.2 and 94.7 mm with 4, 8, 16 and 32 substeps. Early drift decreases from about 4.75 to 2.61, 1.26 and 0.65 mm/s. This approximately linear timestep dependence implicates explicit near-rest force coupling. The constitutive work-cycle correction does not supply an implicit static reaction selection.

**Static support remains unvalidated.** The retrained policy uses an exploratory profile that fails this prolonged-hold gate. Rigid backing-floor support can contribute substantially in finite-layer locomotion. Further randomization alone does not resolve this numerical sticking problem; an implicit reaction/sticking treatment requires a separate solver change and renewed validation.

The load-controlled RFT fixture already uses a 20 mm maximum triangle edge, corresponding to 512 sole triangles on the foot. Its creep is present with fine foot quadrature, independently of the coarse walking mesh's torque error.

The hold criterion permits at most 2 mm depth change between 2 and 10 s, without backing-floor proximity. All four RFT timestep settings fail. A 10 mm MPM long-hold control changes only about 0.03 mm over that interval and remains clear of the floor. Its coarser depth is grid-biased, so this additional control assesses hold stability rather than supplying the fine calibration target.

## Training and inference protocol

Both continuations start from the same original `model_2998.pt` actor and critic, reset the optimizer, use seed 42 and **4,096 environments**, and receive **1,000 additional PPO iterations**. Commands remain **0.1–1.0 m/s** throughout. Full material ranges apply from the first reset; there is no command curriculum.

- Additional-training control: original Newton revision, bearing and material ranges.
- Corrected force-informed variant: corrected Newton revision and the fitted nominal profile. At reset, density is 1,400–1,800 kg/m³, internal friction 0.3–0.5, interface/internal ratio 0.1–1.0, depth 0–118 mm, strength multiplier 0.355–1.065 and bearing multiplier 0.225–0.675. The ±50% multiplier ranges are experimental choices, not confidence intervals inferred from data.

The observation/action contract, actuators and rewards are unchanged. Native checks cover standing, 0.5/1.0 m/s tracking and three banks of 128 randomized environments. MPM inference compares nominal, shallow/loose and dense materials at both commands. Fixed-policy videos are published only after their complete frames and commands are verified. Whole-robot MPM remains a feasible coarse reference; isolated-foot convergence does not establish whole-robot convergence.

The material-bank diagnostics use a maximum 3 s observation window and end 0.5 s after a first body-failure threshold. Failure time is the primary transfer outcome; the material plot reports tracking before failure, while raw summaries retain full-rollout tracking. The full-length showcase protocol uses 10 s recordings. Completed videos state their actual durations; diagnostic previews remain where a replacement has not completed. The MPM showcases use a 4 × 8 m bed and continue after falls, with no early stop or reset. Videos contain actual simulated frames; their durations and settings are stated beside each recording.

The source audit confirms the same bulk yield invariant, `sqrt(0.5*s:s) <= mu*P`, in both solvers and exact-volume cell-centered MPM particle initialization. MPM and coupled-proxy source are unchanged between the original and corrected Newton revisions. MPM includes material inertia and evolving loading history. RFT's optional inertial correction remains disabled, and the calibration at 0.1 m/s does not establish accuracy at every walking-foot velocity.

Both 1,000-iteration continuations have completed. The revised run's final 100-iteration mean return is 12.66 versus 12.44 in the preceding window, a 1.75% increase; final mean episode length is 488.9 of 500 steps. These are descriptive single-seed training trends. The different material distributions make held-out trials more informative than comparing training returns alone.

| Native RFT check | Extra-training control | Revised profile |
|---|---:|---:|
| Standing planar error, 20 s × 16 environments | 0.035 m/s; first threshold at 10.04 s | 0.004 m/s; no threshold |
| 0.50 m/s planar error, 20 s × 16 environments | 0.071 m/s | 0.056 m/s |
| 1.00 m/s planar error, 20 s × 16 environments | 0.098 m/s | 0.119 m/s |
| Acceptance gates passed | 4/8 | 5/8 |

The revised policy has no body failures across its six native trials, including three independent 128-environment randomization banks. One of the 384 randomized environments violates foot placement. Maximum-command tracking and the loaded-foot velocity proxy also fail their gates. The proxy is not measured grain-relative slip.

Late in the revised standing trial (15–20 s), recorded upward granular force averages **23.0 N against 316.3 N weight**, about **7.3%**. Exact transformed foot colliders remain within 1 mm of the backing floor throughout that window. Under steady momentum balance, most remaining support is inferred to come from the substrate; rigid-contact force is not directly recorded in this trial. The low drift therefore does not establish granular static support.

The native recordings, three-material MPM bank and 20/10/8 mm whole-robot grid checks are complete. The remaining coarse-policy full-length recordings and original-checkpoint ablation are stopped while foot resolution is investigated. One revised-policy MPM recording at 0.50 m/s is complete for ten seconds in the wider bed; its heading drift is reported below.

## Reproduction and outputs

Source fixtures and analysis are under `scripts/benchmarks/`: `export_g1_granular_probe.py`, `investigate_g1_granular.py`, `investigate_g1_loaded_foot.py`, `benchmark_g1_bearing.py`, `benchmark_g1_rft_quadrature.py`, `benchmark_go2_foot_resolution.py`, `run_g1_foot_study.py`, `calibrate_g1_loaded_rft.py`, `run_g1_rft_refinement.py`, `run_g1_transfer_checks.py` and `report_g1_refinement.py`.

Live data is under ignored `logs/go2_granular/g1/refinement/`. Training uses separate locked environments to retain the original-model control. The corrected runtime is `/home/horde/.cache/rft-study/g1-fixed-venv`; launch with `UV_PROJECT_ENVIRONMENT` pointing there and `uv run --no-sync`. `fixed-runtime.json` records the immutable Newton revision and kernel hash. Training and inference status records contain exact commands and checkpoints. The [foot-resolution archive](results/rft-foot-resolution/README.md) contains the completed geometry/quadrature evidence. The complete follow-up policy archive remains pending.

## Curved Go2 foot resolution

The Go2 foot collider is a 22 mm radius sphere. Its current 12-segment UV mesh has 144 authored faces, 132 retained by the RFT solver, and maximum edges of 15.6 mm. Small nearly degenerate pole faces remain in those solver counts. Improving curved geometry requires regenerating the sphere surface; splitting its existing flat facets retains their coarse normals.

A separate preliminary benchmark prescribes 150 centered-sphere states at four internal/interface friction combinations. It covers 1–44 mm penetration, stationary support, insertion, translation, withdrawal, 8 rad/s rotation and three orientations. It uses the corrected bearing law, with moments about the sphere center. A 384-segment UV sphere with 0.51 mm maximum edges supplies the numerical reference. The 192 → 384 refinement changes pooled force and torque by approximately 0.10% and 0.49%.

| Foot mesh | Solver triangles per foot | Maximum edge | Pooled force error | Pooled torque error |
|---|---:|---:|---:|---:|
| Current UV sphere | 132 | 15.6 mm | 8.74% | 18.89% |
| Uniform sphere, 3 subdivisions | 1,280 | 3.62 mm | 0.92% | 3.34% |
| Uniform sphere, 4 subdivisions | 5,120 | 1.82 mm | 0.32% | 1.36% |
| Uniform sphere, 5 subdivisions | 20,480 | 0.91 mm | 0.10% | 0.43% |

These pooled numbers weight larger reactions more strongly. Deeply immersed spinning cases retain larger errors; per-depth percentiles and the raw wrenches are recorded. Percentiles use denominator floors of 1 N and 0.01 Nm. This benchmark isolates geometry and quadrature, rather than validating a robot policy or real granular material.

## Requirements for refined training surfaces

- Configure RFT resolution separately for foot colliders. Use the measured G1 candidate of 128 sole triangles, with 512 as a finer comparison. For Go2, compare uniform 1,280- and 5,120-triangle spherical feet against a finer reference.
- Retain the authored collision shapes, transforms, masses and inertias. Construct and cache the finer RFT surfaces once at model creation; every simulation substep reuses them. All selected leg and thigh surfaces remain coupled.
- Record triangle counts, minimum/maximum edge lengths and integrated forces/torques per foot. Whole-robot triangle totals do not establish foot accuracy.
- Check identical prescribed and recorded foot motion across penetration, sliding, rotation, withdrawal and material/depth extremes. Assess both pooled and per-depth errors, with explicit absolute normalization floors.
- Measure throughput and memory at 4,096 environments before selecting the training mesh. Mesh quality must be established independently of policy reward or additional randomization.
- Verify material calibration with the selected surfaces, retrain, and repeat the RFT/MPM comparisons. Existing checkpoints retain their coarse-mesh provenance.

The remaining coarse-policy MPM rendering batch is stopped while foot-resolution requirements are investigated. Its completed 0.50 m/s wide-bed recording is retained: no body threshold occurs in ten seconds, but yaw-rate error averages 0.262 rad/s and the robot follows a curved path. Both exact foot meshes remain inside the bed at the recorded control frames, with a minimum vertex margin of about 75 mm. Refinement of foot quadrature does not resolve the separately measured static-support failure.

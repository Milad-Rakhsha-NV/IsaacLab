# G1 force matching and transfer investigation

The original study is preserved in `results/g1-rft-mpm/`. This follow-up separates numerical reference quality, RFT force approximation and learned robustness. MPM supplies no training transitions. Commands retain their full 0.1–1.0 m/s range.

[Live investigation](http://localhost:18010/g1/refinement/) · [Original comparison](http://localhost:18010/g1/report/)

## Current evidence

The exact finalized G1 foot collider is approximately 203 × 65 × 19 mm, with its authored offset from the ankle. Its exported mesh has outward triangle winding and positive enclosed volume. The fixture uses the same mesh for MPM and RFT, prescribed ankle poses and velocities, and sums every substep's collider impulse and moment. No rigid-body integrator or direct foot/floor contact contributes to the measured wrench. All material is prepared for 0.5 s with the foot out of the bed.

The first 10 mm APIC fixture has finite particle states and a momentum-balance residual of about 2.2 × 10⁻⁵ of the bed-weight impulse. During insertion, mean vertical forces are approximately 76 N in MPM and 85 N in the original RFT closure. During the subsequent prescribed hold, MPM relaxes to approximately zero while the original experimental bearing term supplies about 227 N upward. This is preliminary evidence; refinement checks remain in progress. A prescribed hold is distinct from a freely loaded foot reaching equilibrium.

The first 10-to-5 mm comparison misses the fixed 10% phase-mean convergence criterion in insertion force and two small torque measurements. Halving the timestep at 5 mm changes phase-mean forces and torques by less than 3.1%; the timestep, iteration-budget and domain-size checks pass. The 8 mm grid still misses one phase-force check. A 3.33 mm check is running. The gate normalizes near-zero phase means by 20 N and 1 Nm, giving absolute allowances of 2 N and 0.1 Nm. Both force and torque must pass. Pointwise histories remain available for inspection; this gate establishes numerical self-convergence only.

On four saved G1 walking snapshots, the original RFT triangulation differs from the finest surface integration by about 2.0% in force and 5.8% in torque. The probe preserves the original field's free-surface and bottom planes; omitting those planes invalidates this finite-layer check. This is evidence about numerical integration, separate from force-model accuracy.

An exploratory fit selects parameters using insertion, hold and drag. Rotation and unloading are held out. The initial fit rejects the large near-rest bearing contribution but still fails held-out force/torque agreement. These results are not a validated calibration and do not justify treating MPM as physical ground truth.

## Work status

| Stage | Status |
|---|---|
| Matched G1-foot insertion, hold, drag, pitch, roll and withdrawal | Seven numerical profiles complete; finer-grid check running |
| MPM convergence and RFT calibration | Time, iteration and domain checks pass; grid and force-fit acceptance pending |
| More full-range RFT training and held-out MPM transfer | Original-model control is running with 4,096 environments and 1,000 additional iterations |

Both training continuations start from the original `model_2998.pt` actor and critic with a reset optimizer and seed 42. The original-model control retains the initial force law and ranges. The force-informed variant will combine the candidate closure and wider strength/interface-response ranges; this combination is exploratory. Both receive the complete material ranges at every reset from the first episode and retain the same commands, observations, actuators and rewards. The 64-environment continuation smoke test passes. A separate eight-environment profile smoke check confirms independent reset samples, zero bearing and the requested strength/interface ranges.

The fixtures and analysis live in `scripts/benchmarks/export_g1_granular_probe.py`, `investigate_g1_granular.py`, `run_g1_foot_study.py` and `calibrate_g1_rft.py`. Raw records and the live page are under ignored `logs/go2_granular/g1/refinement/`. The original robot environment and archived numerical results have not been overwritten.

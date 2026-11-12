# G1: rigid, RFT and MPM policy study

The G1 study is running on `milad/rft-mpm`. The rigid baseline trains first with 4,096 environments; RFT training follows only after the rigid policy passes its held-out standing, walking and randomized-command checks. Final G1 recordings and archived results are not yet complete. The native tests pass with self-collision enabled, including the 37-joint action/observation contract, thigh/calf/foot RFT binding, clone-local reset and rigid/RFT/MPM drive parity before contact.

[Live G1 report](http://localhost:18010/g1/report/) · [Generated G1 status](http://localhost:18010/g1/STATUS.md) · [Published Go2 results](results/go2-rft-mpm/README.md)

Both fresh policies use **4,096 environments, seed 42 and a 1,500-iteration initial PPO budget**. Forward commands are reset-uniform over 0.1–1.0 m/s, with lateral ±0.1 m/s, yaw ±0.3 rad/s and 10% standing. There is no command curriculum. The rigid baseline must pass its held-out standing, tracking, foot-placement, slip, stepping and numerical-capacity checks before RFT training proceeds. Training-budget completion is reported separately from these acceptance checks.

Both G1 configurations now use implicit-fast integration for their implicit joint drives. The pre-fix regression produced joint speeds above 240 rad/s for small position commands; the corrected rigid/RFT/MPM free-space trajectories remain bounded and agree. The final focused test file passes two tests. This verifies drive behavior and geometry/reset contracts, not trained locomotion.

Self-collision is explicitly enabled for both tasks. The maintained minimal asset disables it by default; enabling it makes the added leg colliders resist cross-leg interpenetration. An initial short run without this setting is retained only under an ignored archive directory and is excluded from the study/report. The corrected run starts from fresh weights.

The direct G1 tasks share the reset, observation, reward and curriculum infrastructure used by Go2. G1 uses its maintained minimal asset with 37 actuated joints and 123 actor/critic inputs, including base linear velocity as in the maintained G1 locomotion configuration. Both policies use identical G1 mechanics, actuators and state inputs. The gait term rewards biped single support; it uses collision contact timing on rigid ground and combined collision/granular-load timing in RFT. The body-clearance failure threshold is 0.4 m above the initial material surface.

The minimal G1 asset supplies foot and torso colliders. Four additional invisible capsules cover the thighs and calves: each spans the parent frame to its knee or ankle joint anchor, with radii 45 mm and 35 mm respectively. These are deliberate collision approximations. They preserve authored masses, inertia, joints and visible meshes and are identical in rigid, RFT and MPM simulations. RFT triangulates these same collision surfaces. The initial 0–118 mm layer range is retained for comparison with Go2; it is below half the approximately 350 mm G1 knee-to-sole length.

RFT samples materials independently for resetting environments: density 1,400–1,800 kg/m³, internal friction 0.3–0.5, surface/internal friction ratio 0.6–1.0 and a log-uniform strength factor 0.8–1.2. A performance-gated material curriculum widens these distributions from nominal values. Layer thickness spans 0–118 mm from the first reset over a fixed rigid substrate. RFT has no particles during training and retains the experimental bearing coefficient 0.87574, velocity regularization 0.04 m/s and zero inertial correction.

A future study would use four terrain panels with **two 10 s videos** each, commanding 1.0 and 0.5 m/s with zero lateral velocity and yaw, using the same frozen checkpoint and evaluation seed 123:

- Rigid-trained policy on a rigid floor.
- RFT-trained policy in RFT sand.
- RFT-trained policy in MPM sand.
- Rigid-trained policy in MPM sand.

MPM is inference only. The recordings use a prepared 12 × 3.2 × 0.118 m bed, PIC transfer, a 20 mm grid, four substeps, 50 stress iterations and one coupling pass with feedback factor 1.0. The earlier Go2 main recordings used 0.1. Controlled 2 s G1 checks at both commands compare one-pass factors 0.1 and 1.0 and four-pass refinement at factor 1.0, holding the checkpoint, material, prepared 5 × 3.2 × 0.118 m bed and seed fixed. These checks precede the long MPM videos and do not establish numerical convergence.

Preparation advances the actual MPM material history for 0.5 s while holding the robot fixed. Material density is 1,600 kg/m³, internal friction 0.4 and interface/internal ratio 1.0. Matching inputs do not establish RFT/MPM material equivalence. The earlier MPM force-convergence gates remain failed, and G1-specific force calibration is unvalidated. A sparse-grid capacity of 1,048,576 active cells is used with overflow checks.

The study also compares three material settings at 0.5 m/s with support-force and tracking plots. The report overlays rigid/RFT learning curves and groups tracking analysis into 0.50 and 1.00 m/s subsections. Support includes rigid contacts after collapse, so backing-floor support alone does not show successful walking. Native contact sensors are unavailable in the MPM proxy coupler; MPM inference disables gait rewards/sensors and measures articulated state and momentum-based support instead. The actor remains unchanged and has no contact-sensor inputs.

Registered tasks:

- `Isaac-Velocity-Flat-G1-Granular-Rigid-Direct`.
- `Isaac-Velocity-Flat-G1-Granular-RFT-Direct`.

The durable controller is `scripts/benchmarks/run_g1_granular_study.py`; the report generator is `scripts/benchmarks/report_g1_granular.py`. Generated checkpoints, videos, traces and plots remain under ignored `logs/` directories. Browser access uses the existing SSH tunnel to port 18010.

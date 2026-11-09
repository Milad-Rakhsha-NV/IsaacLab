# SuperMimic implementation plan

Branch: `milad/supermimic`. Target: this IsaacLab checkout, Newton/MJWarp,
29-DOF Unitree G1, headless CUDA training. Paper: https://arxiv.org/abs/2610.28480.

## Scope and ownership

Implement original research code under `projects/supermimic/`, using public
IsaacLab scene, articulation, and simulation APIs. Keep the framework and its
standard PPO entrypoints unchanged. Use the paper and the MIT-licensed original
BeyondMimic robot configuration as references; do not copy unlicensed unofficial
SuperMimic implementations. The package is a downstream experiment retained in
this branch at the user's request.

## Implementation

1. Prepare a uv environment and verify the imported IsaacLab source path.
2. Download G1-retargeted LAFAN1 data, pin its revision, and convert CSV motions
   to 50 Hz root/joint states. Record provenance, quaternion convention, joint
   order, clip boundaries, and content hashes. No synthetic motion substitutes.
3. Construct a G1 flat-ground scene with 5 ms physics and four substeps per
   control action. Match low-gain PD control and explicitly map joints by name.
   Read terminal successor states before reset. Keep timeouts and failures separate.
4. Implement differentiable kinematics and tracking/observation functions.
   Independently compare kinematics and body velocities against the simulator.
5. Implement episode-safe replay with observation histories and executed actions;
   normalized residual dynamics trained on eight-step autoregressive windows;
   a deterministic future-conditioned actor; and a bootstrapping value network.
6. Alternate simulator collection, supervised dynamics updates, 32-step imagined
   actor updates, and detached lambda-return critic updates. Freeze model/value
   parameters during actor updates without detaching their input gradients.
7. Add resumable checkpoints containing networks, optimizers, normalizers, RNG,
   resolved settings and data provenance; JSONL metrics; simulator evaluation;
   and a readiness report tied to the configuration and dataset.

## Validation gates before a full training launch

- CPU tests: motion conventions/interpolation and clip isolation, replay terminal
  boundaries/history, rotation/FK correctness, and analytic actor gradients.
- GPU simulator sanity: finite states/rewards, action ordering, commanded reset
  accuracy, FK/body-velocity agreement, terminal transition capture, and repeatable
  evaluation without noise/randomization.
- Dynamics sanity: fit real simulator transitions, show held-out improvement and
  acceptable multi-step prediction errors. Check gradients remain finite.
- Training sanity: short end-to-end run, nonzero finite actor gradients, bounded
  critic values, actual simulator evaluation, checkpoint/resume, peak memory and
  throughput. Stop on nonfinite states, gradients or losses.
- Run formatting/lint and targeted tests; document any backend/environment limits.
- Only launch the full run after these gates pass. Start with LAFAN1 on the local
  32 GB GPU; preserve horizon/model structure and scale batch sizes to memory.
  Do not describe this as a reproduction of the full BONES-SEED benchmark.

## Initial experiment and deviations

Use 256--512 simulator environments, a 512 imagined batch, 256 world-model windows,
256 replay steps/environment, eight-step model fits and 32-step imagination.
Establish fixed-physics motion tracking first; domain randomization, downstream
velocity interfaces, BONES-SEED scaling, matched PPO and DVI comparisons follow.
Without domain randomization this experiment does not establish sim-to-real transfer.
The paper leaves state parameterization and numerical details unspecified; document
our choices and report measured results rather than assuming the headline speedup.

## Execution record

- Created the requested branch from `f644e1f8f` with a clean checkout.
- Environment synchronization started with `uv sync --extra test`.
- Implementation and gate results will be recorded here as work proceeds.

### Implementation completed

- Implemented motion conversion, clip-safe sampling, episode-safe GPU replay,
  differentiable tree kinematics, analytic tracking objectives, residual dynamics,
  future-conditioned actor, critic, and online BPTT training under
  `projects/supermimic/`.
- Downloaded all 40 G1 LAFAN1 clips at pinned revision
  `ce1572906efe6157840e8474d5a0d7aa87481e74`; converted data carries source hashes.
- The shipped robot exposes 43 joints. A temporary derived USD welds the 14 finger
  joints before import; the simulator now exposes exactly 29 moving joints. Source
  assets are left untouched. Body/floor friction is explicitly 1.0.
- Checkpoint/resume, JSONL metrics, bounded values, nonfinite guards and a
  source/config/data/hardware-specific readiness gate are implemented. Full
  training continues from the validated warm-start checkpoint, whose hash is
  recorded in the report. It cannot bypass failed readiness checks.
- PyTorch compilation is supported and exercised on the local Blackwell GPU.
- All four CPU contract tests pass; targeted pre-commit hooks and Ruff pass.

### Development validation

The compiled development run (before the final explicit-friction adjustment)
passed 200 actor/critic updates. Held-out eight-step root-position RMSE decreased
from 0.0718 m to 0.0190 m, joint-position RMSE from 0.180 rad to 0.0616 rad, and
normalized dynamics error from 26.90 to 3.95 (persistence baseline: 25.77).

Independent FK agreement: position below 5e-6 m and body velocity below 6e-6 in
their respective units. Deterministic evaluation return differed by less than
1e-6 between repeated trials. Five-second first-frame evaluation across all 40
clips improved from 0.0760 to 0.2277, with failed windows decreasing from 40 to 36.
These are short-window coverage diagnostics, not full-clip success results.

The minimum actor gradient norm was 0.0051. Checkpoint reload and a resumed
optimizer update passed. Peak allocated VRAM was 2.63 GiB, with 27.18 GiB free.
After compilation, typical smoke updates took approximately 0.6--0.7 seconds;
the first compilation took several minutes.

### Final readiness and training launch

Final readiness **passed** on the final code and physics settings. Its report is
`projects/supermimic/runs/readiness/readiness.json`, with console output alongside.
The terminal successor also passed the explicit mutation/reset isolation check.

- Held-out eight-step root-position RMSE: 0.05553 -> 0.01435 m.
- Held-out joint-position RMSE: 0.20082 -> 0.04703 rad.
- Normalized dynamics error: 29.48 -> 2.25; persistence baseline: 27.45.
- Five-second evaluation return across 40 clips: 0.07599 -> 0.44876.
- Failed five-second windows: 40 -> 15.
- Checkpoint reload and a resumed optimizer update passed.
- Peak allocated VRAM: 2.62 GiB; free VRAM: 27.25 GiB.
- Four CPU tests and all targeted pre-commit hooks passed on the final source.

The full LAFAN1 experiment was launched only after this report passed and its
source/config/data/GPU fingerprint was checked. It continues from the validated
checkpoint with 256 simulator environments, a 512 imagined batch, ten 256-window
world-model batches per update, eight-step fits, 32-step imagination and compilation.
The target is **320,000 additional simulator iterations** (0.4 actor/model updates
per iteration), with checkpoints every 500 iterations and five-second evaluations
every 2,000. This remains the LAFAN1 fixed-physics experiment, not BONES-SEED.

- Run directory: `projects/supermimic/runs/lafan1_full/` (ignored by Git).
- PID: `1294213`; also recorded in `train.pid` and `launch.json`.
- Console log: `train.log`; structured progress: `metrics.jsonl`.
- Checkpoint: `last.pt`, written atomically; full source snapshot: `source/`.
- A copy of the exact passing readiness report is retained with the run.
- Initial evaluation in the fresh process: return 0.44849; 16 failed windows.
  The slight difference from the readiness evaluation occurs near failure thresholds.
- Initial 150 iterations produced finite gradients/losses and used 2.34 GiB peak
  allocated VRAM. Initial throughput suggests approximately one day for the budget.
- Verified the first periodic checkpoint by loading `last.pt` on CPU: iteration
  500, 400 actor updates including readiness, all 40 clips recorded. Training
  continued beyond iteration 625 with finite gradients and losses.

Monitor from the repository root:

```bash
tail -f projects/supermimic/runs/lafan1_full/train.log
```

To stop gracefully and save at the next iteration boundary:

```bash
kill -TERM "$(cat projects/supermimic/runs/lafan1_full/train.pid)"
```

Resume using the README command with `--compile`, the run's `readiness.json`, and
`--resume projects/supermimic/runs/lafan1_full/last.pt`. Source changes require a new
matching readiness report. No commit or push was performed.

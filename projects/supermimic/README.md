# SuperMimic on IsaacLab

Original research implementation of the learned-world-model motion-tracking loop
in [SuperMimic](https://arxiv.org/abs/2610.28480). Uses this checkout's public
IsaacLab scene/articulation APIs and Newton/MuJoCo Warp; no simulator derivatives.
No code is copied from the unlicensed unofficial SuperMimic reproductions.

For tensor shapes, network diagrams, control/state conventions and the exact
training objectives, see [the architecture document](../../SUPERMIMIC_ARCHITECTURE.md).

## Scope

The first experiment is fixed-physics tracking of the 40 G1-retargeted LAFAN1
motions. It is **not** the paper's BONES-SEED benchmark, a PPO comparison, or a
sim-to-real reproduction. A successful readiness check establishes numerical and
physical consistency and the ability to train; it does not establish convergence.

Run all commands from the IsaacLab checkout root. Dependencies come from its
`pyproject.toml` and `uv.lock` (Python 3.12, CUDA-enabled PyTorch, Newton, Warp,
MuJoCo Warp, NumPy/SciPy). Isaac Sim, mjlab and a pretrained policy are unnecessary.
The existing `test` extra supplies pytest; `huggingface_hub` is already available
through the checkout's dependencies.

```bash
uv sync --extra test
uv run python -m projects.supermimic.motions --download \
  --output projects/supermimic/data/lafan1.npz
uv run python -m pytest projects/supermimic/tests/test_contracts.py -q
uv run python -m projects.supermimic.sanity \
  --motions projects/supermimic/data/lafan1.npz
```

The downloader pins dataset revision
`ce1572906efe6157840e8474d5a0d7aa87481e74`. Converted libraries carry source file
SHA-256 hashes, the revision, the joint ordering and sampling conventions. Dataset
files remain ignored; their original licenses still apply. LAFAN1 source data is
available for noncommercial research; consult the original dataset's terms before
other uses.

## Readiness and training

Run the checks using the **same learner arguments** intended for the full run:

```bash
uv run python -u -m projects.supermimic.train --mode sanity \
  --motions projects/supermimic/data/lafan1.npz \
  --out projects/supermimic/runs/readiness

uv run python -u -m projects.supermimic.train --mode train \
  --motions projects/supermimic/data/lafan1.npz \
  --readiness projects/supermimic/runs/readiness/readiness.json \
  --out projects/supermimic/runs/lafan1 --iterations 320000
```

Full training refuses a failed or mismatched report. The report is tied to the
Python source, resolved learner config, dataset hash, selected clips, PyTorch
version and GPU model. Training starts from the validated `sanity.pt`, including
its dynamics prefit and policy updates; the report verifies that checkpoint's hash.
Run sanity again after changing these. Development runs can
use `--num-envs 40 --policy-batch 64 --model-batch 128 --model-batches 2`.
`--small-networks` is a debugging ablation, not a paper architecture.
`--compile` opts into PyTorch compilation of model fits and imagined steps; validate
it separately because the first compilation is expensive.

Readiness checks actual simulator resets, body poses and link-origin velocities
against differentiable FK, perfect-state rewards, held-out eight-step dynamics
versus persistence, repeatable evaluation, finite nonzero BPTT gradients, short
actor/critic training with at least 10% simulator-return improvement, checkpoint
reload plus a resumed optimizer update, and GPU
memory headroom. Full training stops on nonfinite losses, gradients or states.

Training writes `config.json`, flushed `metrics.jsonl`, and atomic `last.pt`
checkpoints. SIGINT/SIGTERM request a checkpoint at the next iteration boundary.
`--max-hours` provides a wall-clock limit; zero runs to the iteration budget.

## Training plots and status

The checkout's `status.md` contains training progress, ETA, evaluation return and
failures, reward/survival curves, losses, gradient norms and per-motion scores.
Its plots are generated from the run's JSONL metrics using CPU-only Matplotlib.

```bash
# Refresh once; add --watch to refresh every minute until the training process ends.
uv run --no-sync python projects/supermimic/tools/plot_status.py \
  --run projects/supermimic/runs/lafan1_full --output status.md
```

The current background plotter's PID and log are stored under the run's `monitor/`
directory. Reload your Markdown preview to see refreshed images. The plotter only
reads training outputs; it is independent of the learner and its readiness fingerprint.
Generated status files, plots and run data should not be committed.

## Policy videos

Record actual simulator rollouts from an immutable copy of the current checkpoint:

```bash
uv run --no-sync --with imageio-ffmpeg python -m projects.supermimic.tools.record_videos \
  --checkpoint projects/supermimic/runs/lafan1_full/last.pt \
  --motions projects/supermimic/data/lafan1.npz \
  --out projects/supermimic/runs/lafan1_full/videos/current --seconds 12
```

This uses headless Newton GL, needs no display server, and writes H.264 MP4s,
JPEG previews, checkpoint/data hashes, rollout metrics and an HTML gallery.
The default selection covers walking, dancing, jumping, and getting up; use
`--clips` to select other library names, or `--all-clips --seconds 20` to record
20 seconds of every library motion. The gallery and metadata update after each
completed recording. Actions, observations and resets are
noise-free. Tracking failures are marked on screen and do not trigger a reset.
Only the actor is loaded onto the GPU; recording is independent of training.

## Resume and evaluate

```bash
# Resume with the same learner arguments as the original run.
uv run python -u -m projects.supermimic.train --mode train \
  --motions projects/supermimic/data/lafan1.npz \
  --readiness projects/supermimic/runs/readiness/readiness.json \
  --resume projects/supermimic/runs/lafan1/last.pt \
  --out projects/supermimic/runs/lafan1

# Complete-clip evaluation; use --evaluation-steps 250 for five-second diagnostics.
uv run python -u -m projects.supermimic.train --mode evaluate \
  --motions projects/supermimic/data/lafan1.npz \
  --resume projects/supermimic/runs/lafan1/last.pt \
  --out projects/supermimic/runs/lafan1-eval --evaluation-steps 0
```

Checkpoints include networks, optimizer states, running normalizers, iteration
counters, RNG states and data provenance. Resuming starts new simulator episodes
and refills replay; it is not a bit-for-bit restart of physics/warm-start buffers.
Only load trusted checkpoints (`torch.load` includes optimizer and NumPy RNG data).

## Design and intentional differences

- The shipped G1 USD has 43 joints. An isolated temporary USD variant welds its 14
  finger joints at zero before physics import. No cached source asset is edited.
  All 29 moving joints are represented in the model; hands retain their rigid mass.
  This robot's collision geometry/inertias are not asserted identical to the paper.
  Robot and floor friction are explicitly 1.0 with zero restitution. MJWarp uses
  ten solver iterations and a 50-iteration line-search cap.
- State: root-link world position, XYZW quaternion, root-link world linear/angular
  velocity, then 29 named joint positions and velocities (71 scalars). Environment
  origins are removed. Dynamics predict 70 local residual coordinates and integrate
  orientations on the rotation manifold. Inputs are invariant to global XY/yaw.
- Differentiable kinematics comes from the finalized Newton joint tree and includes
  child joint-frame offsets and link-origin velocities. It processes tree levels in
  batches, and is validated independently against simulator body data.
- Actor: ten 93-component proprioceptive observations; ten future frames, spaced
  five control steps apart, with joint positions/velocities and relative root
  orientations; a tanh-bounded 32-dimensional latent. No privileged root position
  or linear velocity enters the actor. The critic additionally sees physical state.
- Tracking follows Appendix A's six positive terms, action-rate and joint-limit
  penalties, at 50 Hz. Body positions use the simulation/reference world frames.
  Hard termination uses reference-relative height/tilt; imagined termination uses
  smooth sigmoid approximations at the same thresholds. No contact-force labels
  or learned reward/observation heads are required.
- Simulator collection and model fitting store executed noisy actions. Replay
  stores pre-reset successors and reconstructs episode-local observation history.
  Imagined rollouts stop at motion ends, retaining state/action gradients through
  frozen dynamics and critic parameters. Critic targets use detached lambda returns.
- Local defaults reduce replay/batch sizes from the H100 setup. Network widths,
  eight-step fitting and 32-step imagination retain the paper's structure. The
  critic bootstraps are clamped to [-12, 12] for stability; the negative limit is
  a numerical choice rather than a lower bound on accumulated penalties.
- Domain randomization and return-based adaptive reset sampling are deferred.
  Clips are sampled uniformly and reset times uniformly within each selected clip.
  No hardware-transfer claim is justified by this fixed-physics experiment.
- Periodic evaluation uses five-second first-frame windows across all selected
  clips. The logged failure counts refer to these windows unless `complete_clips`
  is true. Full-clip evaluation is a separate command. Training and evaluation
  motions overlap; results measure coverage rather than held-out generalization.

Robot PD parameters and tracked-body choices reference the MIT-licensed
[BeyondMimic](https://github.com/HybridRobotics/whole_body_tracking) configuration
at `9db9158`; its license is retained in `LICENSE-BeyondMimic.txt`. The paper leaves
state encoding, smoothing and numerical details partly unspecified; the choices
above are explicit implementation decisions rather than claims about author code.

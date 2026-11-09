# SuperMimic training status

**COMPLETED** · updated 2026-11-05 18:58:23 PST · refresh every 60 seconds

| Metric | Current value |
|---|---|
| Experiment | LAFAN1, 40 G1 clips, fixed physics, Newton/MJWarp |
| Progress | 320,000 / 320,000 iterations (100.00%) |
| Actor updates, including readiness | 128,190 |
| Training transitions after warm start | 81,920,000 |
| Time since launch | 21h 26m |
| Recent throughput | 4.22 simulator iterations/s |
| Estimated remaining time | 0m 00s |
| Training log freshness | 0m 07s |
| Process | PID 1294213 · not running |
| Latest checkpoint modified | 2026-11-05 18:58:11 PST (203.4 MiB) |
| Peak PyTorch GPU allocation | 2.34 GiB |

Latest evaluation: **return 0.9349**, **0 / 40 failed windows**, at iteration **320,000** (2026-11-05 18:58:16 PST).

Evaluation uses 5-second first-frame windows. These are training-set coverage diagnostics. Iteration zero already includes readiness pretraining.

## Policy videos

**40 / 40 videos available**, from checkpoint **299,000**. Each requested rollout is 20 seconds. Tracking thresholds exceeded in **0 / 40** completed recordings; failures are marked on screen and do not trigger resets.

[Open video gallery](projects/supermimic/runs/lafan1_full/videos/all_clips/index.html) · [Browse individual videos](projects/supermimic/runs/lafan1_full/videos/all_clips/README.md)

[Download all videos](projects/supermimic/runs/lafan1_full/videos/all_clips/supermimic_all_clips.zip)

## Learning progress

![Learning progress](projects/supermimic/runs/lafan1_full/monitor/learning.png)

Raw training samples are faint; solid lines show an 11-sample moving mean where available. Simulator failure fraction is measured per control step, not per episode. Simulator and imagined rewards sample different state distributions; their gap is not a model-error estimate.

## Optimization diagnostics

![Optimization diagnostics](projects/supermimic/runs/lafan1_full/monitor/optimization.png)

Model loss is the online autoregressive training loss, not held-out dynamics error. Actor loss includes a critic bootstrap and survival weighting, so it is not simulator tracking return. Gradient norms are logged before clipping.

## Motion coverage

![Per-motion evaluation](projects/supermimic/runs/lafan1_full/monitor/motions.png)

Motions are sorted by their latest return. Orange points mark the checkpoint at the start of the full run.

## Monitoring

The background plotter rewrites this file and its figures once per minute. Reload the Markdown preview to see the latest images. ETA uses the last five minutes of logged progress and can change with evaluation or compilation costs.

[Metrics JSONL](projects/supermimic/runs/lafan1_full/metrics.jsonl) · [Training log](projects/supermimic/runs/lafan1_full/train.log) · [Readiness report](projects/supermimic/runs/lafan1_full/readiness.json) · [Launch metadata](projects/supermimic/runs/lafan1_full/launch.json)

Refresh once from the IsaacLab checkout:

```bash
uv run --no-sync python projects/supermimic/tools/plot_status.py --run projects/supermimic/runs/lafan1_full --output status.md
```

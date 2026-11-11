# Go2 RFT/MPM report and results

This is a static snapshot of the completed Go2 report on branch `milad/rft-mpm`. It contains the eight final recordings, plots, their numerical inputs, final policy checkpoints and training records. The [study summary](../../RFT_MPM_STUDY.md) describes the implementation, settings, findings and remaining validation limits.

## View the report

Retrieve the binary artifacts through Git LFS after cloning:

```bash
git lfs install --local
git lfs pull --include="results/go2-rft-mpm/**" --exclude=""
uv run --no-project python -m http.server 8000 --bind 127.0.0.1 --directory results/go2-rft-mpm
```

Open **http://localhost:8000/report/**. GitHub's HTML file view shows source; use the local server to view the page and videos together. Serving this archive needs Python only, without installing Isaac Lab, Newton or GPU dependencies. To clone only source and then retrieve the report artifacts above, prefix the clone command with `GIT_LFS_SKIP_SMUDGE=1`.

When the server runs on another machine, forward its port:

```bash
ssh -N -L 18000:127.0.0.1:8000 USER@HOST
```

Then open http://localhost:18000/report/ on your browser machine. The archived page uses local relative links and has a static training-records page in place of the original live-service link. Superseded report/video navigation is omitted.

## Contents

| Location | Contents |
|---|---|
| `report/index.html` | Four clearly labeled terrain panels, each with 0.5 and 1.0 m/s videos; comparison subsections at both speeds; shared rigid/RFT learning curves |
| `report/*.png` | Combined learning curves, both tracking/height comparisons, and the MPM startup check |
| `report/results.json`, `report/gait_diagnostics.json` | Training metadata, recording/acceptance summaries and measured rear-foot separation |
| `report/verification/` | CPU and CUDA standalone RFT benchmark records |
| `max1/*max_forward*/` | Eight final MP4 files, posters, full state traces, summaries and available recording metadata |
| `max1/*validation/`, `max1/*commands_*/` | Current-policy held-out acceptance data, including randomized-command banks |
| `max1/materials_cmd_0p5/` | Three-material comparisons at 0.5 m/s: six MPM trials, three RFT reference trials, support/kinematic traces and plot |
| `max1/mpm-recheck/` | Five controlled 2 s MPM numerical-sensitivity trials at 1.0 m/s |
| `investigation/` | Referenced force-convergence, calibration, rate/history and earlier-policy investigations, clearly historical where appropriate |
| `measurements/foot-lengths.png` | The knee-to-sole measurement used to select the 118 mm depth cap |
| `rsl_rl/` | Final `model_1499.pt` checkpoints, TensorBoard events, run/curriculum metadata, and original configuration YAML files inside `configuration.zip` |
| `training/` | Static links to training artifacts and the final acceptance/status snapshot |
| `current-study.json` | Original study/run configuration and evaluation case names |
| `manifest.json` | SHA-256 checksums, byte sizes and source-revision provenance |

Videos are H.264 MP4, 960 × 640, 500 frames at 50 FPS. The state files are NumPy archives readable with `numpy.load`; numerical summaries are JSON. Files larger than routine source artifacts, including state traces and checkpoints, use Git LFS. The eight MP4 files are the published versions; intermediate recordings, original-banner backup movies and the rejected narrow-bed trial are not included.

For `rollout.npz`, `state` has shape `(control steps, environments, 10)`: time in seconds, world position XYZ in meters, body-frame linear velocity XYZ in m/s, height above the initial surface in meters, planar tracking error in m/s, and a failure flag. `commands` stores forward/lateral velocity and yaw rate. `metrics` stores absolute yaw-rate error, summed absolute joint power, motor-saturation fraction and torso tilt. `body_pose` uses world positions and XYZW quaternions; body ordering and centers of mass are in `summary.json`.

`body_wrench` contains world-frame force followed by torque about each body's center of mass. RFT values average the applied impulse over the control interval; MPM values sample the last relaxed proxy feedback. These are different sampling conventions, recorded in each summary. When present, `support_wrenches` stores interval-averaged external force XYZ followed by the rigid-contact resultant XYZ inferred from whole-robot momentum balance. That resultant includes all rigid contacts, including contacts after collapse.

The `investigation` folder retains the earlier 0.5 m/s policy study used by the linked material-analysis page. Those policy records are historical; the main report's eight recordings and current acceptance data use the newer policies trained up to 1.0 m/s. Standalone material-force checks do not depend on the policy checkpoint.

## Revisions and interpretation

The computational Newton pin is **`32f455bc4430f0954ae564fb40681fbaca95e7f3`**. The report rendering source is Isaac Lab **`d4667026b7`**; archive publication adds summaries, dependency-remote relocation and this result snapshot. The dependency-remote change preserves the exact Newton commit. Original metadata retains the runtime paths and source-fork names used by the experiments; those are provenance, not requirements for opening the archive.

Training used 4,096 environments, 1,500 iterations and seed 42 for each policy. Recordings use evaluation seed 123, fixed nominal material and the same two frozen checkpoints at both commands. MPM is inference-only.

At 0.5 m/s, the RFT policy remains above the failure thresholds in nominal MPM for 10 s, while the rigid policy reaches a threshold at 0.60 s. Both fail in the baseline 1.0 m/s MPM recordings. The early high-speed failure is sensitive to coupling numerics. MPM force convergence/material equivalence and RFT standing/slip acceptance remain unresolved; short sensitivity checks are not proof of converged MPM or reliable transfer.

The main MPM beds are 12 × 1.6 m at 1.0 m/s and 12 × 3.2 m at 0.5 m/s, both initially 118 mm deep. Policies share a bed at each speed; this width difference limits comparisons across speeds. Support plots include all robot rigid contacts and time after failure. The 1.0 m/s material-comparison plot was removed from the main report and is not part of this snapshot.

## Verify archive integrity

The manifest hashes expanded binary contents, so run `git lfs pull` first. From the repository root:

```bash
uv run --no-project python - <<'PY'
import hashlib
import json
from pathlib import Path

root = Path("results/go2-rft-mpm")
manifest = json.loads((root / "manifest.json").read_text())
for entry in manifest["files"]:
    data = (root / entry["path"]).read_bytes()
    assert len(data) == entry["bytes"], entry["path"]
    assert hashlib.sha256(data).hexdigest() == entry["sha256"], entry["path"]
print(f"Verified {len(manifest['files'])} archived files")
PY
```

The manifest excludes itself. JSON/HTML files have a final newline for repository formatting; numerical values and binary trajectories are preserved.

## Regenerate the report from the archived inputs

With Isaac Lab's dependencies installed, restore the original input layout in a new checkout that has no existing experiment logs:

```bash
mkdir -p logs/go2_granular logs/rsl_rl
cp -a results/go2-rft-mpm/rsl_rl/. logs/rsl_rl/
cp -a results/go2-rft-mpm/max1 results/go2-rft-mpm/investigation results/go2-rft-mpm/measurements logs/go2_granular/
cp -a results/go2-rft-mpm/training results/go2-rft-mpm/current-study.json logs/go2_granular/
cp -a results/go2-rft-mpm/report logs/go2_granular/
uv run --no-sync python scripts/benchmarks/report_go2_granular.py
```

This redraws the shared training curves and both tracking figures from saved events and trajectories, without rerunning physics or training. The live-workspace generator reinstates its usual service/archive navigation; the committed static snapshot retains the portable navigation above. Material and coupling plots can be regenerated from the archived JSON/NPZ records using the existing functions in `report_go2_materials.py` and `run_go2_mpm_recheck.py`.

To run inference rather than redraw saved data, use `scripts/benchmarks/evaluate_go2_granular.py --help` and the archived checkpoint paths. Original evaluation arguments and numerical parameters are recorded in the case manifests and summaries. G1 has no trained checkpoints, recordings or results in this archive.

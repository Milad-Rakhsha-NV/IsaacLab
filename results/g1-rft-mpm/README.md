# G1 RFT/MPM report and results

This snapshot contains the completed G1 study on branch `milad/rft-mpm`: eight final videos, shared rigid/RFT learning curves, tracking plots at both commands, a 0.5 m/s material comparison and controlled MPM coupling checks. The paired Go2 study is in `../go2-rft-mpm/`.

## View both reports

From the repository root, retrieve the results with Git LFS and serve their common parent:

```bash
git lfs install --local
git lfs pull --include="results/**" --exclude=""
uv run --no-project python -m http.server 8000 --bind 127.0.0.1 --directory results
```

Open **http://localhost:8000/g1-rft-mpm/report/** for G1 or **http://localhost:8000/go2-rft-mpm/report/** for Go2. Serving the static pages needs Python only. GitHub displays HTML source rather than executing the report. For a remote server, forward the port with `ssh -N -L 18000:127.0.0.1:8000 USER@HOST`, then use port 18000 in your local browser.

## Contents and provenance

| Location | Contents |
|---|---|
| `report/` | HTML, shared training figure, both tracking figures, coupling sensitivity and numerical summaries |
| `evaluations/` | Final videos/posters, held-out acceptance checks and six MPM startup trials; JSON summaries and NPZ state/force traces |
| `materials_0.5/` | Three material settings, both frozen policies in MPM and the RFT references, support/tracking plot and raw records |
| `diagnostics/` | Thin-layer frozen-policy checks, the failed standing-only initialization probe and timestep sensitivity traces; no intermediate videos |
| `rsl_rl/` | Final checkpoints, TensorBoard events, run/curriculum metadata and saved configuration YAML files in `configuration.zip` |
| `status.json`, `STATUS.md` | Final run paths, validation outcomes and published recording names |
| `material-profile.json` | Shared MPM numerical controls for the material bank |
| `manifest.json` | SHA-256 hashes and byte sizes of expanded artifact contents, with source revisions |

The rigid policy trained from scratch with 4,096 environments, seed 42 and 1,500 PPO iterations. RFT fine-tunes that accepted rigid checkpoint for 1,500 additional iterations with a reset optimizer, the same environment count and seed. This is a two-stage adaptation study, not an equal-total-cost comparison. Shared learning plots use iteration within each stage; RFT starts with the earlier rigid training already completed. Forward commands cover 0.1–1.0 m/s without a command curriculum. RFT randomizes material parameters at reset with a material curriculum; depths start at 0–10 mm and widen toward 0–118 mm with performance. MPM is excluded from training.

The robot uses 37 actions and 123 actor/critic inputs, shared thigh/calf capsules, self-collision and implicit-fast integration. The final matched reward uses foot-slip weight 1.0. Development trials with self-collision disabled or weaker slip cost are excluded from this snapshot.

The computational Newton pin is `32f455bc4430f0954ae564fb40681fbaca95e7f3`. Original metadata keeps the actual runtime paths. Movies are H.264 MP4, 960 × 640, 500 frames at 50 FPS. Arrays are readable with `numpy.load`; training configurations are archived verbatim in ZIP files. Large artifacts use Git LFS.

## Recorded outcomes

Both commands use the same two frozen checkpoints and seed 123. Metrics include initial acceleration and time after a failure threshold.

| Policy training / evaluation | Forward command | Planar error | First failure threshold |
|---|---:|---:|---|
| RIGID / RIGID | 1.00 m/s | 0.058 m/s | None in 10 s |
| RIGID / RIGID | 0.50 m/s | 0.029 m/s | None in 10 s |
| RFT / RFT | 1.00 m/s | 0.154 m/s | None in 10 s |
| RFT / RFT | 0.50 m/s | 0.097 m/s | None in 10 s |
| RFT / MPM | 1.00 m/s | 0.940 m/s | 1.14 s |
| RFT / MPM | 0.50 m/s | 0.493 m/s | 1.74 s |
| RIGID / MPM | 1.00 m/s | 0.943 m/s | 0.90 s |
| RIGID / MPM | 0.50 m/s | 0.487 m/s | 1.18 s |

The rigid policy passes all seven held-out gates. RFT passes five of eight, including full material coverage and no falls; tracking, loaded-foot velocity and one foot-placement violation remain failed. On the 118 mm reference MPM bed, both policies reach a body failure threshold at both commands. Completing the study does not establish stable transfer or full locomotion validation.

The separate three-material bank uses 3 s trials at 0.5 m/s. In the thinner, lower-density/friction 60 mm bed, the RFT policy stays above body failure thresholds for that short window, while the rigid policy fails at 1.50 s. Tracking errors are 0.131 and 0.464 m/s, respectively. The RFT trial receives substantial rigid backing-floor support. Density, friction and depth vary together; this result does not establish 10 s stability or explain the effect of any one parameter. See `materials_0.5/policy-analysis.json` and the report's material-trial table for every case.

See `status.json` for every held-out gate; completed training does not imply that every acceptance check passed. `state` in `rollout.npz` stores time, world position XYZ, body-frame velocity XYZ, surface-relative height, planar tracking error and a failure flag. `body_wrench` stores world force followed by torque; RFT uses control-interval impulse averages, while MPM samples the last relaxed proxy feedback. `support_wrenches`, when present, stores averaged external force XYZ followed by the inferred rigid-contact resultant XYZ, including contacts after collapse.

## MPM numerical limits

Main G1 videos use a prepared 12 × 3.2 × 0.118 m bed at both commands: density 1,600 kg/m³, internal friction 0.4, surface/internal ratio 1.0, PIC, 20 mm grid, four substeps per 5 ms physics frame, 50 stress iterations and one coupling pass with feedback factor 1.0. Preparation advances material history for 0.5 s with the robot fixed. Earlier Go2 main videos use factor 0.1 and different bed widths; cross-robot results are not a controlled numerical comparison.

Fixed MPM constitutive inputs are Young's modulus 10¹⁵ Pa, Poisson ratio 0.3, pressure-yield limit 10¹⁵ Pa, zero tensile-yield ratio, viscosity, dilatancy and hardening. These have no direct one-to-one RFT counterpart. Saved summaries contain the actual floating-point material ranges. Matching density, friction and layer depth is input matching rather than force-response calibration.

Controlled 2 s G1 trials use the same prepared 5 × 3.2 × 0.118 m bed and seed at each command. They compare one-pass feedback factors 0.1 and 1.0 and four-pass refinement at 1.0. These are sensitivity checks, not proof of 10 s stability, MPM convergence or material equivalence. The earlier standalone MPM force-convergence gate remains failed. The RFT bearing extension and finite-layer approximation retain the limitations documented in the study summary.

## Verify the archive

After fetching LFS contents, run from the repository root:

```bash
uv run --no-project python - <<'PY'
import hashlib
import json
from pathlib import Path
root = Path("results/g1-rft-mpm")
data = json.loads((root / "manifest.json").read_text())
for entry in data["files"]:
    blob = (root / entry["path"]).read_bytes()
    assert len(blob) == entry["bytes"], entry["path"]
    assert hashlib.sha256(blob).hexdigest() == entry["sha256"], entry["path"]
print(f"Verified {len(data['files'])} files")
PY
```

The manifest excludes itself. The static report removes live auto-refresh and uses local relative navigation. Numerical trajectories and original configurations remain unchanged.

# Go2 contact-resolution follow-up

Selected foot integration: 0.51% pooled force and 0.94% pooled torque difference from the reference. Individual outliers remain; these are numerical differences, not material-model accuracy.

The profile is a numerical candidate, with 128 triangles over each spherical foot and selective lower-calf
refinement. It is not cleared for training: prolonged support fails. G1 remains on hold. The original Go2
policy recordings retain their original implementation and geometry.

Open `index.html` alongside its PNGs. JSON and NPZ files retain the measurements, common motion samples,
material settings, source hashes and geometry. `manifest.json` verifies every archived artifact.

Reproduce from the repository root using the Newton revision pinned in `pyproject.toml`:

```bash
export UV_PROJECT_ENVIRONMENT="$HOME/.cache/rft-study/go2-contact128-venv"
uv sync --locked
uv run --no-sync python scripts/benchmarks/benchmark_go2_contact_profile.py \
    --mode performance --legacy --performance_motion results/go2-contact-refinement/motion-rigid-1.npz \
    --output /tmp/legacy.json
uv run --no-sync python scripts/benchmarks/benchmark_go2_contact_profile.py \
    --mode performance --performance_motion results/go2-contact-refinement/motion-rigid-1.npz \
    --output /tmp/refined.json
uv run --no-sync python scripts/benchmarks/benchmark_go2_contact_profile.py \
    --samples 32 --orders 16 24 \
    --trajectories results/go2-contact-refinement/motion-rigid-0.5.npz \
                   results/go2-contact-refinement/motion-rigid-1.npz \
    --output /tmp/motion.json
uv run --no-sync python scripts/benchmarks/benchmark_go2_rft_support.py --output /tmp/support.json
```

The archive includes both complete ten-second motion recordings and their summaries (including body COM
locations), plus the 64 selected exact poses and twists. Source paths and SHA-256 hashes are in the JSON.
The checkpoint is not required to reproduce the prescribed-motion audit. The robot asset and numerical
profile remain fixed during that audit; no policy inference occurs in the force comparison.

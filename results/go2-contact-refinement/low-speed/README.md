# Go2 contact-model validation — 13 November 2026

Open [index.html](index.html) for the fixed foot/calf boundary correction, actual collider-section PNG,
matched MPM load probes, insertion-force fits, scalar reaction diagnostic, and 4096-environment cost.
The [study notes](../../../GO2_RFT_LOW_SPEED.md) describe interpretation and outstanding gates.

**No replacement near-rest articulated coupling or new policy training is accepted by these records.**
All earlier policy videos retain their original solver and training surfaces.

- `compound-comparison.json` and `compound-{before,after}.*`: identical prescribed Go2 geometry/motion,
  comparing detached body boundaries with enabled fixed-link compounds. This fixture uses the earlier
  128-triangle profile so that the force change isolates containment.
- `fixed-union-motion.*`: 64 exact recorded body poses/twists, 16 field cases, planar/coarse variants
  and curved quadrature through order 24. Fixed-link connectivity is retained when batching snapshots.
- `fixed-union-performance.json`: 4096 native environments, zero-action stepping and common-state RFT cost.
- `profile.json`: the final 128-triangle-foot profile; distal-calf quadrature is refined at intersections.
- `preloaded-mpm-*`: raw dynamic-load histories and settings. Trace columns are documented in each
  summary; later records include the granular force actually delivered to the rigid solver.
- `rft-preload-comparison.npz` within each 18 N case: measured preload force, nominal published RFT
  evaluation on the same trajectory, and the insertion-fitting mask.
- `scalar-{hold,load-cycle}`: explicit and scalar implicit vertical traces. The directional-limit columns
  are undefined for the explicit variants. The scalar solver does not integrate an articulation.
- `summary.json` and `manifest.json`: provenance, fit metrics and SHA256 hashes of expanded artifacts.

The original MPM cases use Newton `6cf5b879d576c19279183ee9da0dc4de039a080f`. The 400-iteration-cap
MPM refinement, corrected geometry, integration recheck and packaged-runtime verification use
`f59677bd8c5037d68d4196c323ea0b4459c18a9c`. The first historical
four-pass MPM trace counted CUDA capture as execution; its plotted release timestamps are corrected by
one 5 ms frame. Its steady depth is unaffected. Raw source records are retained.

Example fixture reruns from the Isaac Lab repository with the separately pinned runtime:

```bash
export UV_PROJECT_ENVIRONMENT=/home/horde/.cache/rft-study/go2-fixed-union-venv
uv run --no-sync python scripts/benchmarks/benchmark_go2_contact_profile.py \
  --mode compound --output logs/go2_granular/go2/low-speed/compound-comparison.json
uv run --no-sync python scripts/benchmarks/benchmark_go2_rft_low_speed.py \
  --output logs/go2_granular/go2/low-speed/scalar-hold
uv run --no-sync python scripts/benchmarks/benchmark_go2_rft_low_speed.py --load_cycle \
  --output logs/go2_granular/go2/low-speed/scalar-load-cycle
uv run --no-sync python scripts/benchmarks/investigate_g1_loaded_foot.py \
  --geometry sphere --field mpm --mass 1 --load 18.417051 --preload_depth 0.0523165 \
  --seconds 3 --voxel_size 0.005 --domain_scale 0.75 --substeps 8 --passes 1 \
  --output logs/go2_granular/go2/low-speed/preloaded-mpm-18N-5mm-p1
uv run --no-sync python scripts/benchmarks/report_go2_rft_low_speed.py \
  --input results/go2-contact-refinement/low-speed \
  --output logs/go2_granular/go2/low-speed-rebuilt \
  --newton_revision f59677bd8c5037d68d4196c323ea0b4459c18a9c
```

The MPM reproduction uses the corrected package; the RFT-only fixed-link change leaves that isolated
MPM fixture's dynamics unchanged. Settings for the other cases are retained in their JSON summaries.

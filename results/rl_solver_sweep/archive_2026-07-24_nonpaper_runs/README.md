# Archived non-paper sweep runs — 2026-07-24

These runs are retained for reproducibility but are excluded from the clean paper figures. No files were deleted.

## Contents

- `raw/ant/initial_contact_10/`: earlier Ant Jacobi/APGD/P-SPG-FB/MJWarp pilot contact-10 set; the selected Ant Jacobi post-stabilization run remains in the active raw tree, while the pilot MJWarp log is used as the provisional native baseline.
- `raw/h1/newton_dvi/`, `newton_dvi_apgd/`, `newton_dvi_pspg/`: H1 solver and contact-budget variants not selected for the paper figure.
- `raw/go2/newton_dvi/`, `newton_dvi_apgd/`, `newton_dvi_coupling2/`, `newton_dvi_coupling2_cache_compare/`, `newton_dvi_pspg/`: Go2 solver, coupling, cache, and contact-budget variants not selected for the paper figure. Contact-15 post-stabilization and MJWarp remain active.
- `raw/g1/`: G1 prior solver/contact variants, coupling-1/2 cache comparisons, post-stabilization variants, APGD/P-SPG-FB, MJWarp contact-10, and the aborted 1000-iteration run. The selected G1 coupling-2/cache-on contact-15 run remains active.
- `raw/go2/newton_dvi_post_stabilize/contact_10/` and `raw/go2/newton_mjwarp/contact_10/`: superseded contact-10 runs moved out of the active tree.
- `raw/g1/newton_dvi_coupling2_cache_on/contact_10/`: superseded G1 contact-10 run moved out of the active tree.

Each run directory retains its raw log, completion marker where applicable, Git revisions, and hardware metadata. Only runs with `exit=0`, valid training metrics, and completion markers are eligible for plotting.

## Important interpretation

- MJWarp is a native/default-solver baseline; a directory named `contact_10` or `contact_15` does not mean MJWarp was forced to that DVI iteration count.
- A single-seed reward band represents within-run variability, not a confidence interval.
- The archive is not a replacement for raw external backup; preserve it with the complete results tree when transferring data.

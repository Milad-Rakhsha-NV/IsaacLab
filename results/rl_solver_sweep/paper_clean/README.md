# Clean paper reward-curve assets

This directory is a staging area for figures proposed for the paper. **Nothing has been copied into the technical report/book.** The figures use a TensorBoard-style EMA (span 35) and a rolling within-run standard-deviation band. A band for a single seed is variability, not a confidence interval.

## Selected plotting runs

| Environment | Jacobi run | Contact iterations | Coupling iterations | Post-stabilization | LDL cache | MJWarp run | Notes |
|---|---|---:|---:|---|---|---|---|
| Ant | `raw/ant/newton_dvi_post_stabilize/contact_10` | 10 | 1 | true | true/default | archived `raw/ant/initial_contact_10/mjwarp.log` | MJWarp native/default solver; contact-10 is a label |
| Humanoid direct | `raw/humanoid/newton_dvi_post_stabilize/contact_10` | 10 | 1 | true | true/default | **none** | No Humanoid-direct MJWarp curve is included |
| H1 | `raw/h1/newton_dvi_post_stabilize/contact_10` | 10 | 1 | true | true/default | `raw/h1/newton_mjwarp/contact_10` | MJWarp native/default solver |
| G1 | `raw/g1/newton_dvi_coupling2_cache_on/contact_15` | 15 | 2 | false | true | `raw/g1/newton_mjwarp/contact_10` | MJWarp native/default solver; contact-10 is a label |
| Go2 | `raw/go2/newton_dvi_post_stabilize/contact_15` | 15 | 1 | true | true/default | `raw/go2/newton_mjwarp/contact_15` | MJWarp native/default solver; contact-15 is a label |

`cache_factorization` is irrelevant for one coupling sweep in practice, but the selected runs record its configured value. MJWarp contact labels are not forced iteration budgets and should be described as native/default baselines.

## Figure outputs

Once Humanoid MJWarp completes, run from the Isaac Lab repository with the `dvi` environment:

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate dvi
python results/rl_solver_sweep/prepare_paper_clean_assets.py
```

Outputs are `figures/<environment>_mjwarp_vs_selected_jacobi_reward.{png,pdf,svg}`. Review these first; do not place them in the book until explicitly approved.

## Archived material

Superseded and exploratory runs are in:

`results/rl_solver_sweep/archive_2026-07-24_nonpaper_runs/`

The archive preserves raw logs and metadata. Its README and `archived_logs.txt` document the moved material. Incomplete/aborted runs remain archived and must not be plotted.

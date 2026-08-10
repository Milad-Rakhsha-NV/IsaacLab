# DR Legs DVI substep comparison

This archive records the matched 500-iteration DR Legs DVI runs used for the paper's closed-loop reinforcement-learning subsection.

## Revisions

- Isaac Lab branch: `milad/dvi-paper`
- Isaac Lab parent revision used by the runs: `cc97c7c7e4ad24691e44e42221a6e2b4d6318aad`
- Newton branch: `milad/dvi-paper`
- Newton DVI revision containing the tested D6 actuation and universal-joint constraint corrections: `862551bffa881a9e0d45b938713c16b97325054d`

The archived runs were launched while that Newton DVI delta was present as an uncommitted worktree patch on parent revision `2b944ac39ba235434403b64f9f542d9aa08a4f64`. The DVI portion of that patch was subsequently committed verbatim as the Newton revision above. Each run directory retains the original Newton worktree patch and status for provenance.

## Matched configuration

All runs used:

- seed 42
- 4096 environments
- 500 training iterations
- simulation timestep 0.004 s
- control decimation 5 (50 Hz policy frequency)
- sparse-LDL bilateral solve
- sparse-Jacobi contact solve
- 10 contact iterations
- one bilateral/contact coupling iteration
- joint post-stabilization disabled
- factorization caching enabled

Only the number of DVI simulation substeps changed.

| Substeps | Run directory | Final reward | Final success rate | Mature median steps/s |
|---:|---|---:|---:|---:|
| 1 | `raw/dr_legs/newton_dvi/restored_newton_no_post_1sub_4096_500iter_20260810T0131Z/contact_10` | 372.24 | 0.9948 | 51,283 |
| 2 | `raw/dr_legs/newton_dvi/restored_newton_no_post_2sub_4096_500iter_20260810T0024Z/contact_10` | 376.65 | 1.0000 | 29,497 |
| 4 | `raw/dr_legs/newton_dvi/restored_newton_no_post_4sub_4096_500iter_20260809T2254Z/contact_10` | 378.33 | 1.0000 | 15,866 |

The publication plot uses TensorBoard-style exponential smoothing with coefficient 0.08 and a 15-iteration rolling within-run variability band. The band is descriptive and is not a confidence interval.

## Policy rollouts

The corresponding final checkpoints were evaluated for 500 frames with 32 environments, playback seed `20260809`, and fixed commands `vx=0.3 m/s`, `vy=0`, and `wz=0`. Recording logs and videos are retained locally under `videos/dr_legs/restored_newton_no_post_forward_32env_20260810/`; videos are excluded from Git by the repository's media policy.

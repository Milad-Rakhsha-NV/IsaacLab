# Granular locomotion studies

| Robot | Study notes and data | HTML report |
|---|---|---|
| Go2 | [Archive and reproducibility](go2-rft-mpm/README.md) | [Go2 report](go2-rft-mpm/report/index.html) |
| G1 | [Archive and reproducibility](g1-rft-mpm/README.md) | [G1 report](g1-rft-mpm/report/index.html) |
| Go2 / G1 foot resolution | [Geometry, convergence and limits](rft-foot-resolution/README.md) | [Foot-resolution report](rft-foot-resolution/index.html) |

MPM supplies no training transitions. Go2 uses independently trained rigid and RFT policies. G1 fine-tunes an accepted rigid policy in RFT. The studies also use different MPM feedback settings, documented in their reports. Their cross-robot outcomes are not a controlled comparison. Material equivalence and MPM force convergence remain unvalidated.

GitHub displays HTML source. To view the reports, fetch the Git LFS artifacts and serve their common parent from the repository root:

```bash
git lfs install --local
git lfs pull --include="results/**" --exclude=""
uv run --no-project python -m http.server 8000 --bind 127.0.0.1 --directory results
```

Open [Go2](http://localhost:8000/go2-rft-mpm/report/) or [G1](http://localhost:8000/g1-rft-mpm/report/). Viewing the static reports does not require Isaac Lab or a GPU.

For a server on another machine, forward its port:

```bash
ssh -N -L 18000:127.0.0.1:8000 USER@HOST
```

Use port 18000 in the local browser. Each archive includes a manifest of expanded-file SHA-256 hashes and verification instructions.

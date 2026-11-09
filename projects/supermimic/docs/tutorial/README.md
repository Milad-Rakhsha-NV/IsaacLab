# SuperMimic implementation tutorial

This self-contained LaTeX tutorial explains goal conditioning, the task MDP,
network dimensions, reference data, actuation, rewards, replay, learned dynamics,
actor BPTT, critic lambda returns, validation, and the completed experiment.

The compiled document is `build/supermimic_tutorial.pdf`.
The portable source bundle is `build/supermimic_tutorial_latex.zip`.
Generated PDFs and build intermediates are local artifacts, not source commits.

## Compile

Run from this directory with Tectonic 0.17.0:

```bash
mkdir -p build
tectonic --keep-logs --keep-intermediates --outdir build supermimic_tutorial.tex
```

Tectonic downloads TeX resources on its first build. Installation:
https://tectonic-typesetting.github.io/en-US/install.html

A conventional installation with the required LaTeX packages also works:

```bash
latexmk -pdf -interaction=nonstopmode -outdir=build supermimic_tutorial.tex
```

The bundle contains all three figure PDFs and the generated motion table.
Neither Python nor IsaacLab is required just to compile it. Bibliography entries
are in the TeX source; no BibTeX step is needed.

## Regenerate figures

From the repository root:

```bash
uv run --no-sync python projects/supermimic/docs/tutorial/build_figures.py
```

This uses `report_data.json`, a frozen snapshot of the actual run's evaluations,
readiness evidence, motion lengths, and video validation. It writes vector PDFs
under `figures/` and `motion_table.tex`. NumPy and Matplotlib are sufficient to
regenerate them outside the repository.

To explicitly replace the snapshot from the original local experiment:

```bash
uv run --no-sync python projects/supermimic/docs/tutorial/build_figures.py --refresh
```

The snapshot includes the completed metrics file's SHA-256 hash. The PDF describes
the final 320,000-iteration checkpoint separately from the videos recorded at
299,000. Final evaluation uses five-second windows; the videos have 20 seconds of
playback. Neither is a full-clip or held-out-motion benchmark.

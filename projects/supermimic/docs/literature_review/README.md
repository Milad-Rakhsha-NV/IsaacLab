# World models, contact geometry, and policy gradients

A focused mathematical literature review for the research question:

> How much additional interaction is needed to adapt a motion-pretrained world
> model to new contact geometry, and what model structure makes its gradients
> useful for control?

Eleven primary papers are reviewed with consistent notation. The review separates
reported methods from mathematical restatements, critical interpretation, and
proposed experiments. `source_manifest.json` pins the retrieved primary PDFs by
URL/version and SHA-256. Source PDFs are not included in the bundle.

## Build

From this directory, with Tectonic 0.17.0 installed:

```bash
mkdir -p build
tectonic --keep-logs --keep-intermediates --outdir build review.tex
```

A standard TeX installation can use:

```bash
latexmk -pdf -interaction=nonstopmode -outdir=build review.tex
```

The portable source bundle contains all figures. To regenerate the synthetic
illustrations from the repository root:

```bash
uv run --no-sync python projects/supermimic/docs/literature_review/make_figures.py
```

Outside the repository, Python with NumPy and Matplotlib is sufficient. No
simulation, training, or checkpoint is used to create the illustrations. They
are explicitly labeled illustrative and are not empirical paper results.

Output: `build/review.pdf`. Portable LaTeX bundle: `build/review_latex.zip`.

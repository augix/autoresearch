# Experiment

## Purpose

To optimize a metric defined in `eval.py` by training a model on the data defined in `data.py`.

## Constraints

- Do not modify `data.py` and `eval.py`.
- `data.py` defines train dataset, val dataset, and val dataloader. `expt.py` should import these.
- `eval.py` defines the metric to be optimized, and the evaluation function. `expt.py` should import these.
- train and val can have different dataloaders and batch sizes.
- use CUDA.

## How to edit expt.py

- Summarize what learnt in previous experiments.
- Come up with a new idea. Read `suggestions.md` for inspiration.
- Implement it in `expt.py`.

## Metric

- `recon_acc` is the metric returned by `evaluation` function in `eval.py` — higher is better. 
- print the metric in the summary block.

## Time budget

- Training runs for exactly **1 minutes** of wall-clock training time (excluding startup/compilation).
- If a run exceeds **2 minutes** total wall time, kill it and treat as a failure.
- Set TIME_BUDGET in `expt.py` to the time budget.

## Summary 

When an experiment finishes, the script MUST print a summary block like this:

```
---
metric: 0.997900
training_seconds: 300.1
total_seconds: 325.9
peak_vram_mb: 45060.2
num_params_M: 50.3
```

Fields:
- `metric` — the optimization target (from `evaluation`)
- `training_seconds` — wall-clock training time (excludes startup)
- `total_seconds` — total wall-clock time including startup
- `peak_vram_mb` — peak GPU memory in MB
- `num_params_M` — total parameter count in millions

## plots
- each experiment should plot a learning curve (with fraction of trained epochs as x and train loss, val loss, train recon_acc, val recon_acc as y) in expts/learning_curve_<expt_id>.png
- each experiment should plot a side-by-side comparison of the first 10 original and reconstructed images in expts/reconstruction_<expt_id>.png

## Track experiments

Summary of each experiment should be appended to `results.tsv`.

1st Row is a header and the following rows are experiments.

```
expt_id commit	metric	memory_gb	status	description
expt0   a1b2c3d	0.997900	44.0	keep	baseline
expt1   b2c3d4e	0.993200	44.2	keep	increase LR to 0.04
expt2   c3d4e5f	1.005000	44.0	discard	switch to GeLU
expt3   d4e5f6g	NA	NA	crash	double model width (OOM)
```

Columns are separated by tabs.
- **expt_id** — unique identifier for the experiment
- **commit** — git commit hash (7 chars)
- **metric** — achieved metric (NA for crashes)
- **memory_gb** — peak VRAM in GB, rounded to .1f (NA for crashes)
- **status** — `keep` | `discard` | `crash`
- **description** — short text about what was tried

**Never commit results.tsv.** Leave it untracked by git.

## Fast-fail rules

- If `metric` is NaN or the run crashes, log it as a crash and move on.


# Auto-expt loop analysis
# 
# Analysis of `results.tsv`.
# The default objective is `metric` used as task-quality diagnostic.

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# configuration
METRIC_COLUMN = "metric"
LOWER_IS_BETTER = False

# load the TSV
df = pd.read_csv("results.tsv", sep="\t")
df.columns = df.columns.str.strip()

# Convert columns to numeric.
for col in [METRIC_COLUMN, "memory_gb"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df["status"] = df["status"].str.strip().str.upper()
df["experiment"] = np.arange(len(df))

# print summary
print(f"Total experiments: {len(df)}")
print(f"Objective metric: {METRIC_COLUMN} ({'lower' if LOWER_IS_BETTER else 'higher'} is better)")
print(f"Columns: {list(df.columns)}")
df.head(10)

# print experiment outcomes
counts = df["status"].value_counts()
print("Experiment outcomes:")
print(counts.to_string())

n_keep = counts.get("KEEP", 0)
n_discard = counts.get("DISCARD", 0)
n_crash = counts.get("CRASH", 0)
n_decided = n_keep + n_discard
if n_decided > 0:
    print(f"\nKeep rate: {n_keep}/{n_decided} = {n_keep / n_decided:.1%}")
if n_crash > 0:
    print(f"Crash rate: {n_crash}/{len(df)} = {n_crash / len(df):.1%}")

# print kept experiments
kept = df[df["status"] == "KEEP"].copy()
print(f"KEPT experiments ({len(kept)} total):\n")
for i, row in kept.iterrows():
    metric = row[METRIC_COLUMN]
    desc = row["description"]
    secondary = ""
    print(f"  #{i:3d}  {METRIC_COLUMN}={metric:.6f}{secondary}  mem={row['memory_gb']:.1f}GB  {desc}")

# plot objective over time
fig, ax = plt.subplots(figsize=(16, 8))

valid = df[df["status"] != "CRASH"].copy()
valid = valid.dropna(subset=[METRIC_COLUMN])
if valid.empty:
    raise ValueError(f"No valid rows with {METRIC_COLUMN} found")

baseline = valid.iloc[0][METRIC_COLUMN]
compare = (lambda values: values <= baseline) if LOWER_IS_BETTER else (lambda values: values >= baseline)
interesting = valid[compare(valid[METRIC_COLUMN])].copy()
if interesting.empty:
    interesting = valid.copy()

# Plot discarded experiments as faint background dots.
disc = interesting[interesting["status"] == "DISCARD"]
ax.scatter(disc["experiment"], disc[METRIC_COLUMN],
           c="#cccccc", s=30, alpha=0.8, zorder=2, label="Discarded")

# Plot kept experiments as prominent green dots.
kept_v = interesting[interesting["status"] == "KEEP"]
ax.scatter(kept_v["experiment"], kept_v[METRIC_COLUMN],
           c="#2ecc71", s=50, zorder=4, label="Kept", edgecolors="black", linewidths=0.5)

# Running best step line from kept experiments.
kept_all = valid[valid["status"] == "KEEP"].copy()
if LOWER_IS_BETTER:
    kept_all["running_best"] = kept_all[METRIC_COLUMN].cummin()
else:
    kept_all["running_best"] = kept_all[METRIC_COLUMN].cummax()
ax.step(kept_all["experiment"], kept_all["running_best"], where="post", color="#27ae60",
        linewidth=2, alpha=0.7, zorder=3, label="Running best")

# Label each kept experiment with its description.
if LOWER_IS_BETTER:
    rotation = 30
    va = "bottom"
    xytext = (6, 6)
    ha = "left"
else:
    rotation = -30
    va = "top"
    xytext = (6, -6)
    ha = "left"
for _, row in kept_all.iterrows():
    desc = str(row["description"]).strip()
    if len(desc) > 45:
        desc = desc[:42] + "..."
    ax.annotate(desc, (row["experiment"], row[METRIC_COLUMN]),
                textcoords="offset points",
                xytext=xytext, fontsize=8.0,
                color="#1a7a3a", alpha=0.9,
                rotation=rotation, ha=ha, va=va)

n_total = len(df)
n_kept = len(df[df["status"] == "KEEP"])
direction = "lower is better" if LOWER_IS_BETTER else "higher is better"
ax.set_xlabel("Experiment #", fontsize=12)
ax.set_ylabel(f"{METRIC_COLUMN} ({direction})", fontsize=12)
ax.set_title(f"Progress: {n_total} Experiments, {n_kept} Kept Improvements", fontsize=14)
ax.legend(loc="best", fontsize=9)
ax.grid(True, alpha=0.2)

best = kept_all[METRIC_COLUMN].min() if LOWER_IS_BETTER else kept_all[METRIC_COLUMN].max()
low, high = sorted([baseline, best])
margin = max((high - low) * 0.15, 1e-4)
ax.set_ylim(low - margin, high + margin)

plt.tight_layout()
plt.savefig("progress.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved to progress.png")

# print summary statistics
kept = df[df["status"] == "KEEP"].copy()
if kept.empty:
    raise ValueError("No kept experiments found")

baseline_metric = df.iloc[0][METRIC_COLUMN]
if LOWER_IS_BETTER:
    best_idx = kept[METRIC_COLUMN].idxmin()
    improvement = baseline_metric - kept.loc[best_idx, METRIC_COLUMN]
else:
    best_idx = kept[METRIC_COLUMN].idxmax()
    improvement = kept.loc[best_idx, METRIC_COLUMN] - baseline_metric
best_metric = kept.loc[best_idx, METRIC_COLUMN]
best_row = kept.loc[best_idx]

print(f"Baseline {METRIC_COLUMN}:  {baseline_metric:.6f}")
print(f"Best {METRIC_COLUMN}:      {best_metric:.6f}")
print(f"Total improvement: {improvement:.6f} ({improvement / abs(baseline_metric) * 100:.2f}%)")
print(f"Best experiment:   {best_row['description']}")
print()

print("Cumulative effort per kept experiment:")
for _, row in kept.iterrows():
    desc = str(row["description"]).strip()
    print(f"  Experiment #{row['experiment']:3.0f}: {METRIC_COLUMN}={row[METRIC_COLUMN]:.6f}  {desc}")

# print top hits
# Each kept experiment's delta is measured vs the previous kept experiment
# because experiments are cumulative and build on the last kept state.
kept = df[df["status"] == "KEEP"].copy()
kept["previous"] = kept[METRIC_COLUMN].shift(1)
if LOWER_IS_BETTER:
    kept["delta"] = kept["previous"] - kept[METRIC_COLUMN]
else:
    kept["delta"] = kept[METRIC_COLUMN] - kept["previous"]

hits = kept.iloc[1:].copy()
hits = hits.sort_values("delta", ascending=False)

print(f"{'Rank':>4}  {'Delta':>10}  {METRIC_COLUMN:>12}  Description")
print("-" * 90)
for rank, (_, row) in enumerate(hits.iterrows(), 1):
    print(f"{rank:4d}  {row['delta']:+.6f}  {row[METRIC_COLUMN]:.6f}  {row['description']}")

print(f"\n{'':>4}  {hits['delta'].sum():+.6f}  {'':>12}  TOTAL improvement over baseline")


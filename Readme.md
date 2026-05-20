An autoresearch loop. Let AI edit expt.py, run experiments by training a model on the data (defined in data.py) to optimize a metric (defined in eval.py).

# How
- prepare:
    - data.py to define and prepare train/val datasets.
    - eval.py to define the evaluation function on val dataloader, and the metric to be improved.
    - expt.py for a training experiment as baseline.
    - expt.md to constrain the experiments.
    - suggestions.md to suggest experimental ideas.
- Ask an AI to run expts according to loop.md.

# Thanks
Inspired by https://github.com/karpathy/autoresearch
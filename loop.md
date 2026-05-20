# Autoresearch Loop

Purpose: Let AI edit and run `expt.py` in a loop to optimize a metric defined in `eval.py`.

## Setup

Before starting the loop:

- **Code version control**. create a new branch with `git checkout -b <branch>` from master. For example today's date as branch name (apr20).
- **Understand the goal and constraints** by reading `expt.md`, `data.py`, `eval.py`.
- **Prepare data**: run `python data.py`.
- **Initialize results.tsv**: Create with just the header row (see format in `expt.md`).
- **Initialize history.md**: to record experiment events.
- **Confirm and start**.

## The loop

LOOP FOREVER:
```
- Read current git state (branch, commit). 
- Assign an unique <expt_id> for a new experiment.
- Edit expt.py according to `expt.md`. 
- git add expt.py && git commit -m "<description>"
- Run the experiment and print summary: `python expt.py <expt_id> > expt.log 2>&1`.
- Extract results:
   grep "^metric:\|^peak_vram_mb:" expt.log
- If grep is empty → crash. Read: tail -n 50 expt.log
   - Easy fix (typo, missing import)? Fix and re-run.
   - Fundamentally broken? Skip, log "crash", move on.
- If metric improved → keep commit (branch advances).
- If metric equal/worse → git reset --hard to previous commit.
- Append results to `results.tsv`.
- Append rationals and findings of the current experiment with details to `history.md`.
```

## After the loop

after the loop is done:
- Run `python progress.py`.
- Update `suggestions.md` with what learnt and what to try next.

## Baseline

First run must establish the baseline — run `expt.py` as-is, without any modifications. Record in `results.tsv`.

## do NOT pause

Once the loop starts, do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human may be asleep. You are autonomous. 

## Simplicity criterion

Simpler is better:
- use minimal code.
- A small improvement that adds ugly complexity → not worth it.
- Removing code and getting equal or better results → keep.
- Equal performance with much simpler code → keep.

## dependencies
- install packages if it helps improve the metric and reduce code complexity.

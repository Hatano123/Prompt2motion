# Paired pilot result snapshot

- Recorded: 2026-09-19 JST
- Last completed job: 2026-09-18 20:29:38 JST
- Experiment: `paired_pilot`
- Dataset: 10 successful paired top/side demonstrations
- Models: `single_top`, `single_side`, `multi`
- Training seeds: 0, 1, 2
- Clean evaluation: 50 shared initial states, evaluation seed 1000

This directory contains small, reviewable summaries suitable for Git. Checkpoints, videos,
dataset archives, JSONL rollouts, and PBS logs remain excluded by `.gitignore` because of their
size. The original local results are under `results/act/paired_pilot`.

Files:

- `clean_selected.csv`: clean result for the checkpoint selected separately for each run.
- `robustness.csv`: camera missing and random 25% occlusion results.

Important limitations:

- The same clean rollouts were used for checkpoint selection and reporting.
- Clean success was too low for a conclusive robustness comparison.
- Robustness success was 0% in every run and condition.
- These values are pilot evidence, not the final test described in
  `EXPERIMENT_SPEC_CLEAN50.md`.

Pilot input hashes:

- Dataset archive: `959d5d39edf22206938a8b9b9b5617fd75328547490163653c4573f23c711562`
- Evaluation manifest: `4fb8dbc60ef7f155ab7c55c383f0d676a57522b16b8b6e21d39c8bc57f54fb86`

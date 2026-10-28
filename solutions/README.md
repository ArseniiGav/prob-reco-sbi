# Reference solutions

Reference solutions of the course, one directory per stage. Each stage has its own README with the index
question ID -> script -> figures -> answer, the run instructions and the reproducibility notes.

| stage | tasks | solutions |
|---|---|---|
| 1: A Flash in the Dark (the toy detector: prior, response, physics ladder, event displays, bonus) | [TASKS.md](../TASKS.md) | [`stage1_flash/`](stage1_flash/README.md) |

Every stage keeps the same layout: scripts `task*.py` with the helpers in `common.py`, one answer file per question ID
in `answers/`, every number quoted in an answer in `results/*.json`, the figures (PDF + PNG, GIFs for animations) in
`figures/`, the answers assembled into `ANSWERS.md` by `make_answers.py` and one executed notebook per task built by
`make_notebooks.py`. The figures were made in the style of the NeuroMCT paper with the author's figure-style module,
which is not in the repository; without it the same scripts draw plain matplotlib figures. Every run uses fixed seeds on
the CPU, and the stage READMEs say what reproduces to the digit and what does not.

Everything runs from the repository root, in the environment of the README (`.[analysis,dev]`), on the CPU:

```bash
bash solutions/stage1_flash/run_all.sh            # every script in order, then the answers and the notebooks
python solutions/stage1_flash/task1_prior.py      # or any single script
python solutions/stage1_flash/make_notebooks.py   # rebuild the notebooks (Jupyter kernel prob-reco-sbi)
```

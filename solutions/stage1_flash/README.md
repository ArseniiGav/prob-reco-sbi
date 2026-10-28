# Reference solutions: A Flash in the Dark

Reference answers to the questions of [TASKS.md](../../TASKS.md), by question ID: stage 1 of the course (the index of
the stages is [../README.md](../README.md)). The cached data sets in `data/` are not in the repository: the scripts
simulate them on their first run.

| what | where |
|---|---|
| **notebooks, one per task**: figures inline and the answers | `T0_setup.ipynb`, `T1_prior.ipynb`, `T2_response.ipynb`, `T3_physics_ladder.ipynb`, `T4_event_displays.ipynb`, `T5_bonus.ipynb` (executed, outputs kept) |
| answers, one file per question ID | `answers/T*.md`, all of them in ID order in [`ANSWERS.md`](ANSWERS.md) |
| figures (PDF + PNG, one GIF) | `figures/` |
| every number quoted in an answer | `results/T0.json` ... `results/T5.json`, under the question ID |
| scripts, run from the repository root | `task*.py`, helpers `common.py`, `task3_common.py`, `event_display.py`, `photon_animation.py` |
| README figures of the public repo | `make_readme_figures.py` writes `docs/img/detector.png` and `docs/img/flash_photons.gif` |
| run everything | `run_all.sh` (see below) |
| rebuild the notebooks | `make_notebooks.py` (builds them from scripts, figures and answers, and executes them) |
| cached data sets (never committed) | `data/*.h5`, written on first use (172 files, 1.5 GB) |

## Index: question ID -> script -> figures -> answer

| ID | script | figures (`figures/<name>.pdf`, `.png`) | answer | results |
|---|---|---|---|---|
| T0.1 | `task0_setup.py` | - | `answers/T0.1.md` | `T0.json` T0.1 |
| T0.2 | `task0_timing.py` (`--threads` for the thread scan; `--gpu` optional, see below) | `t0_2_timing` | `answers/T0.2.md` | `T0.json` T0.2, T0.2_history, T0.2_threads, T0.2_gpu, T0.2_derived |
| T0.3 | `task0_inspect.py` | - | `answers/T0.3.md` | `T0.json` T0.3 |
| T1.1 | `task1_prior.py` | `t1_1_prior_marginals`, `t1_1_vertices`, `t1_1_radius` | `answers/T1.1.md` | `T1.json` T1.1 |
| T1.2 | `task1_prior.py` | `t1_2_prior_matters` | `answers/T1.2.md` | `T1.json` T1.2 |
| T2.1 | `task2_response.py` | `t2_1_sensor_charges`, `t2_1_hit_and_empty_sensors` | `answers/T2.1.md` | `T2.json` T2.1 |
| T2.2 | `task2_response.py` | `t2_2_total_charge`, `t2_2_light_collection` | `answers/T2.2.md` | `T2.json` T2.2 |
| T2.3 | `task2_response.py` | `t2_3_times`, `t2_3_relative_times` | `answers/T2.3.md` | `T2.json` T2.3 |
| T2.4 | `task2_response.py` | `t2_4_network_input` | `answers/T2.4.md` | `T2.json` T2.4 |
| T2.5 | `task2_proposals.py` | `t2_5_charge_vs_distance`, `t2_5_first_hit_time`, `t2_5_barycentre`, `t2_5_naive_energy`, `t2_5_correlations`, `t2_5_photon_fates`, `t2_5_composition`, `t2_5_azimuthal_modulation`, `t2_5_energy_radius_degeneracy`, `t2_5_pca_network_input` | `answers/T2.5.md` | `T2.json` T2.5 |
| T3.1 | `task3_1_ladder.py` (+ `task3_common.py`) | `t3_1_light_collection`, `t3_1_charge_vs_distance`, `t3_1_delays`, `t3_1_photon_fates` | `answers/T3.1.md` | `T3.json` T3.1 |
| T3.2 | `task3_2_expected_charge.py` | `t3_2_subtended_angle`, `t3_2_expected_charge`, `t3_2_pull_distribution` | `answers/T3.2.md` | `T3.json` T3.2 |
| T3.3 | `task3_3_processes.py` | `t3_3_process_effects`, `t3_3_collection_by_category`, `t3_3_timing_residuals`, `t3_3_charge_pattern` | `answers/T3.3.md` | `T3.json` T3.3 |
| T3.4 | `task3_4_naive_estimators.py` | `t3_4_energy_estimator`, `t3_4_barycentre`, `t3_4_inverse_problem` | `answers/T3.4.md` | `T3.json` T3.4 |
| T4.1 | `task4_event_displays.py` (+ `event_display.py`) | `t4_1_event_display`, `t4_1_gallery` | `answers/T4.1.md` | `T4.json` T4.1 |
| T4.2 | `task4_event_displays.py` | `t4_2_timing` | `answers/T4.2.md` | `T4.json` T4.2 |
| T4.3 | `task4_event_displays.py` | `t4_3_photon_paths` | `answers/T4.3.md` | `T4.json` T4.3 |
| T4.4 | `task4_event_displays.py` | `t4_4_presets`, `t4_4_profiles` | `answers/T4.4.md` | `T4.json` T4.4 |
| T4.5 | `task4_animation.py` (+ `photon_animation.py`) | `t4_5_photons.gif`, `t4_5_photons_frame` | `answers/T4.5.md` | `T4.json` T4.5 |
| T5.1 | `task5_fluctuations.py` | `t5_1_charge_distributions`, `t5_1_correlations`, `t5_1_time_vs_charge` | `answers/T5.1.md` | `T5.json` T5.1 |
| T5.2 | `task5_degeneracy.py` | `t5_2_degeneracy`, `t5_2_separation` | `answers/T5.2.md` | `T5.json` T5.2 |
| T5.3 | `task5_detector_design.py` | `t5_3_detector_design`, `t5_3_layouts` | `answers/T5.3.md` | `T5.json` T5.3 |
| T5.4 | `task5_detector_effects.py` | `t5_4_detector_effects`, `t5_4_feature_distributions` | `answers/T5.4.md` | `T5.json` T5.4 |
| README | `make_readme_figures.py` | `docs/img/detector.png`, `docs/img/flash_photons.gif` | - | - |

## How to run

Everything runs on the CPU, from the repository root, in the environment of the README (`.[analysis,dev]`):

```bash
bash solutions/stage1_flash/run_all.sh                 # every script in order, logs per step, stops at the first error
python solutions/stage1_flash/task2_response.py        # or any single script; each one generates the data it needs
python solutions/stage1_flash/make_answers.py          # rebuild ANSWERS.md from answers/*.md
python solutions/stage1_flash/make_notebooks.py        # rebuild and execute the six notebooks (last steps of run_all.sh)
python solutions/stage1_flash/make_notebooks.py --keep-timing   # the same without re-measuring T0.2 (figure from results/T0.json)
```

The notebooks are generated: edit a script or an answer, then rebuild them. Each notebook runs the scripts of its
task (seconds with cached data), shows the figures of every question inline and quotes the answer; they
need the Jupyter kernel `prob-reco-sbi` of the README installation.

- `run_all.sh` writes one log per step to `LOG_DIR` (default `solutions/stage1_flash/logs`, ignored by git; set
  `LOG_DIR=...` to change it) and prints the time per step and the total.
- Data sets are generated once through the public API (`common.dataset`, CPU, fixed seeds) and cached in
  `solutions/stage1_flash/data/`; a cached file is reused only if its configuration, seed, size and options match the request.
  From an empty cache the whole of `run_all.sh` took 1244 s (20.7 min) on 16 CPU threads of a shared server on
  2026-10-21 (T0 timing 4 min, T1-T2 2.5 min, T3 5 min, T4 20 s, T5 8.5 min, README figures 11 s); with the cache
  filled, most scripts take 5-30 s and the whole run is dominated by the T0.2 timing and the T5 analyses.
- The optional GPU timing of T0.2 is not part of `run_all.sh` (it needs a CUDA build of PyTorch and a visible GPU):
  `CUDA_VISIBLE_DEVICES=0 python solutions/stage1_flash/task0_timing.py --gpu` stores `T0.2_gpu`, and
  `python solutions/stage1_flash/task0_timing.py --plot`
  redraws the figure. The numbers in `answers/T0.2.md` are those of the last run of `run_all.sh` on this machine,
  with the machine load that `results/T0.json` records; the T0.2 timing is the only part of the reference that is
  not reproducible to the digit, by nature.
- T0.1 runs the fast test suite; its summary line in `answers/T0.1.md` changes when tests are added.

## Reproducibility of the reference numbers

The package fixes the seed of every random number (see "Random numbers and reproducibility" in
[docs/physics.md](../../docs/physics.md)), and since the CPU warm-up of the first simulation in a process was added to
the package, the first simulation of a process is identical to any later one. On 2026-10-21 the whole cache was
deleted and `run_all.sh` was run from scratch with the fixed package: of the 9739 stored numbers of T1-T5, all
reproduced exactly except the ones that were recomputed by design (T3.4, new estimator; T3.2, new keys) and 130
values that trace back to three fixed-theta samples (the T2.5 centre sample, seed 25; the T5.2 centre sample, whose
light collection sets E_B and so moves every scan point; the T5.4 10 MHz dark-noise sample at r0 = 0.1 m), whose
cached files had been produced by the non-reproducible first simulation of a process before the fix; the answers
were updated to the regenerated values, and the differences were in the last quoted digit (for example the largest
sensor-sensor correlation at the centre, 0.011 -> 0.012). T0.2 (timing) is re-measured on every run. The reference numbers are therefore reproducible from an empty cache on the same type of hardware and
PyTorch build; on another CPU model the vector math library of PyTorch may choose another code path and change the
photon-level realisation, which changes numbers in the last digit but no conclusion.

## Conventions

- Every answer names its figures and quotes only numbers that are in `results/*.json`; `make_answers.py` fails if an
  answer names a figure that does not exist.
- Statistical uncertainties are given where the sample is small enough for them to matter (T1.2, T2.1, T3.2, T5.1,
  T5.4); numbers from 10^5 events are quoted with the precision the statistics allow.
- Figures: the style of the NeuroMCT paper (applied by `common.style` with the author's figure-style module, which is
  not in the repository; without it the same scripts draw plain matplotlib figures): white background, no grid, ticks in
  on all four sides; readable on their own, sentence-case labels with units; framed legends inside the axes in an empty
  corner (or one legend above or below the panels of a multi-panel figure), never over the data; samples as unfilled
  step histograms and models as dashed lines of the same colour; the paper's named colours (`common.C`), one colour per
  preset across tasks (`common.PRESET_COLOURS`: straight dark green, absorb royal blue, absorb-scatter firebrick, full
  indigo; the single processes of T3 peru for scattering and teal for reflection, `common.PROCESS_COLOURS`; in T3
  absorb-scatter is dash-dotted and single processes dashed); standard log axes with fixed floors; event displays with
  identical sensors coloured black -> grey -> green by charge (`common.CHARGE_CMAP`), scattering (indigo dots) and wall
  reflection (teal squares) as separate markers, the charge barycentre (firebrick cross) as the naive vertex, the true
  vertex a gold star.
- Two public-file conventions matter for the answers: the "69 / 21 / 10 %" direct / scattered / reflected composition
  of the seminar is the disc average (the centre is 60.8 / 29.7 / 9.6 %), and the T3.4 estimators follow the TASKS.md
  hint (k = median of Q/E, median bias, half of the central 68 % interval), with the mean-based variants stored next
  to them.

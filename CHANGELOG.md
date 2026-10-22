# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-10-22

First release: the toy detector "A Flash in the Dark" and its data.

### Added

- `prob_reco_sbi.flash`: photon-transport simulation of a 2-D disc detector (radius 1 m, 32 sensor arcs of 0.10 m,
  photocoverage 0.509) written in PyTorch. It runs on the CPU and uses CUDA automatically when it is available.
  - Switchable bulk absorption, Rayleigh scattering (2-D phase function 1 + cos^2) and wall reflection (diffuse or
    specular); the presets `straight`, `absorb`, `absorb-scatter` and `full` form a cumulative physics ladder, and
    every parameter can be changed on top of a preset.
  - Detection efficiency, Gaussian time jitter, dark noise, Poisson or fixed photon statistics, and fixed vertex or
    energy (to simulate the same event parameters many times).
  - Exact per-event photon bookkeeping (detected, absorbed in the bulk, by the wall or by a sensor, lost; direct,
    scattered and reflected light), optional per-hit records and photon paths.
  - Reproducible output: explicit random generators derived from one seed. Event parameters are identical on every
    device and for every chunking; all stored arrays are identical for the same seed, device type, chunking, PyTorch
    build and hardware (the CPU math kernels are warmed up single-threaded before the first transport, which keeps
    the first run of a process identical to later ones).
- HDF5 data format (format version 1) with the configuration and provenance in every file; `save`, `load`, `merge`.
- `network_input`: log charges and charge-weighted relative mean times, the input of the neural networks to come.
- Command line interface `flashsim` (`generate`, `info`, `merge`, `presets`), also as `python -m prob_reco_sbi.flash`.
- Test suite: fast unit and physics tests, and slow tests (`pytest -m slow`): a fresh-process reproducibility test and
  statistical regression tests against the original seminar implementation and its reference numbers. The original
  code and the reference numbers are not distributed, so those comparisons run only where they are available
  (`FLASH_CORE_DIR`, `FLASH_REFERENCE_FILE`).
- Continuous integration (lint, type checks, pre-commit hooks; tests on Linux with Python 3.10 to 3.14 and on macOS and
  Windows with Python 3.12), Dependabot.
- Documentation: the README (course roadmap, installation, quick start, presets and options), `docs/install.md`
  (Windows, PyTorch builds for CPU, GPU and Intel Macs, uv, clusters, common problems), `docs/physics.md` (the model
  with its equations, what every switch does, the network input) and `docs/data_format.md` (the HDF5 layout of
  format version 1, provenance, merging).
- `TASKS.md`: the tasks of the first session, T0 to T4 and the bonus T5, with fixed question IDs.
- Examples: `examples/quickstart.py`; `examples/make_task_data.py` (the data of the tasks on any operating system);
  `examples/cluster/` with a shard script, a SLURM array job and an HTCondor submit file to adapt, the merge workflow
  and measured resources per job.
- `notebooks/00_quickstart.ipynb`: a guided first look at the data.

[Unreleased]: https://github.com/ArseniiGav/prob-reco-sbi/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/ArseniiGav/prob-reco-sbi/releases/tag/v0.1.0

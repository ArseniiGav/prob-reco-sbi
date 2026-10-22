# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Stage 2a of the course, normalizing flows in one dimension (to be released as 0.2.0).

### Added

- `prob_reco_sbi.flows1d`: normalizing flows in one dimension, written from scratch in PyTorch (no new dependencies).
  - Targets with an exact density, CDF, quantile function, sampling and entropy: `Mixture1D` (any Gaussian mixture,
    in float64; the CDF is computed with `erfc`, accurate far in both tails) and the target of the tasks, the "camel
    with a shoulder" `camel()`, with its parameters in `CAMEL`.
  - The oracle map `oracle_map(target, base)`, the exact flow of a target (`z = Phi^-1(F(x))` for the Gaussian base,
    `u = F(x)` for the uniform one) as a transform.
  - Transforms with `forward(x) -> (z, log|dz/dx|)` and `inverse(z)`: `Affine`; `Planar` and `Radial` (Rezende and
    Mohamed 2015) with the reparametrisation that keeps them invertible, and `constrained=False` to show what happens
    without it; the monotone rational-quadratic spline `RQSpline` (Durkan et al. 2019; `3 K - 1` parameters, the
    identity outside `[-B, B]`, exact inverse); `Sigmoid`, the last layer for the uniform base; `invert_monotone`
    (bisection).
  - `Flow`: a stack of transforms on a `StandardNormal` (default) or `Uniform` base, with `log_prob`, `sample`,
    `inverse`, `transform_points` (the positions of samples after every layer) and `n_params`.
  - `fit`: maximum-likelihood training (the forward KL divergence) with Adam, full batch or mini-batches. Its
    `History` records the loss of every update and, at regular snapshots, the training and validation losses, the
    parameters, the model density and the map on a grid (the frames of a training animation); `restore` loads a
    snapshot back. `split` divides samples into a training and a validation part.
  - Diagnostics: the exact `kl_to_truth` by quadrature, `qq_points`, and the one-sample Kolmogorov-Smirnov test
    `ks_statistic`.
- `tests/test_flows1d.py`: round trips and log-derivatives (against finite differences and autograd) of every
  transform, monotonicity under the constraints, the mixture and the oracle map, the quadrature KL against closed
  forms, a spline fit against the truth, flow samples against the flow density, the uniform base, the training history
  and the diagnostics.
- `docs/flows.md`: the maths of the stage: the change of variables, maximum likelihood as the forward KL divergence,
  the transforms with their invertibility conditions, the Gaussian and the uniform base, the oracle map and the
  diagnostics; reference figures and animations (`docs/img/flows/`), and a section on a one-dimensional flow in a
  real experiment, the NFDE likelihood model of NeuroMCT (Commun. Phys. 9, 63 (2026)), with three of its figures
  (`docs/img/neuromct/`).
- `notebooks/01_flows_1d.ipynb`: a guided first look at flows in one dimension with the package.
- `examples/flows_quickstart.py`: one spline fit from the command line, with the exact KL divergence to the target,
  the Kolmogorov-Smirnov test of the mapped samples and a figure.
- `TASKS_flows.md`: the tasks of stage 2a, F1 to F6 and the bonus B1 and B2, with fixed question IDs.

### Changed

- README: stage 2a in the course roadmap, and a short section with its documentation, notebook, tasks and a quick
  start.

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

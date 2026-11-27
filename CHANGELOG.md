# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-11-27

Stage 2b of the course: neural posterior estimation (NPE) of the vertex and the energy.

### Added

- `prob_reco_sbi.npe`: neural posterior estimation for the toy detector, written from scratch in PyTorch.
  - `rq_spline` and `rq_spline_inverse`: the rational-quadratic spline of `flows1d.RQSpline` with one set of knots per
    value, so that a network can compute the parameters of the spline; zero parameters give the identity.
  - The fixed maps of the event parameters to unbounded coordinates (artanh of the radius, logit of the energy) with
    their inverses and log-Jacobians: `to_unbounded`, `from_unbounded`, `energy_to_unbounded`,
    `vertex_to_unbounded`, `inside_prior`.
  - The posterior models, with `log_prob(theta, x)` and `sample(x, n)` in physical units: `ConditionalFlow1D` (one
    parameter, one conditional spline), `AutoregressiveFlow` (the chain of conditional splines by the product rule,
    with an input embedding block and one parameters head per factor) and `GaussianChain` (a multivariate normal with
    a full covariance, the comparison model); their base class `PosteriorEstimator`, which a new model subclasses.
  - `load_pairs` and `to_pairs`: the pairs `(theta, x)` of simulated events in the representations `"charge+time"`,
    `"charge"`, `"q"` and `"ganged"`; `observations`: the same `x` from arrays of charges and times.
  - `fit`: minibatch maximum likelihood with Adam, a cosine decay of the learning rate and a validation set; its
    `History` keeps the losses, the state with the lowest validation loss and snapshots for animations.
    `save_model` and `load_model`.
  - The exact references of the toy where it has a likelihood formula: `detection_probability` (the probability that
    a photon from a vertex reaches a sensor, for straight light), `energy_posterior` (a truncated Gamma
    distribution), `energy_evidence`, `grid_posterior` (the exact posterior of one event of `straight` or `absorb` on
    an adaptive grid, the energy integrated analytically) and `credible_levels`.
- `prob_reco_sbi.flash.ganged_input` and `ganged_channels`: the network input of a ganged readout, in which a sensor
  and its mirror image share one channel.
- `tests/test_npe.py`: the batched spline against `RQSpline`, inverses and log-derivatives against autograd, the
  coordinate maps, densities that integrate to 1, samples sent back to the base, training, saving and loading, the
  ganged input, the exact references against closed forms; slow tests of `detection_probability` against the
  simulation, of trained models against the exact posteriors, of `examples/npe_quickstart.py` and of a zuko flow.
- `docs/npe.md`: the maths of the stage: the loss, the conditional flow, the chain, the coordinates, the exact
  references with their derivations, two modes and the library, with the given figures and animations
  (`docs/img/npe/`).
- `notebooks/02_npe.ipynb`: a guided first look at neural posterior estimation with the package.
- `examples/make_npe_data.py`: the data of the tasks on any operating system; `examples/npe_quickstart.py`: train the
  chain and draw the posterior of a new event from the command line.
- `tasks/stage2b_npe/README.md`: the tasks of stage 2b, N1 to N5 and the bonus NB1 and NB2, with fixed question IDs.

### Changed

- zuko is in the `analysis` extra (the library flow of task N5).
- README: stage 2b in the course roadmap, with stage 3 (the reliability of the posteriors) and stage 4 (NuBench) next;
  a short section on stage 2b; its posterior animation next to the photon animation at the top.
- `tasks/README.md`: the row of stage 2b.
- `CITATION.cff`: neural posterior estimation in the abstract.
- American spelling in the text of the README, docs, task sheets, notebooks and docstrings of stages 1 and 2a; no names changed.

### Removed

- `solutions/`: the reference solutions are no longer part of the repository.

## [0.2.0] - 2026-11-05

The reference solutions of stage 1, and stage 2a of the course: normalizing flows in one dimension.

### Added

- `solutions/stage1_flash/`: the reference solutions of stage 1, one executed notebook per task (T0 to T5) with the
  scripts that make its figures and numbers, the answers by question ID (`ANSWERS.md`) and the numbers in
  `results/*.json`; `solutions/README.md`, the index of the solutions by stage.
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
- `tasks/stage2a_flows/README.md`: the tasks of stage 2a, F1 to F6 and the bonus B1 and B2, with fixed question IDs.
- `tasks/README.md`: the index of the tasks, with the documentation, the guided notebook and the reference solutions
  of every stage.
- `tests/test_examples.py`: a slow smoke test of `examples/flows_quickstart.py` with a small training budget.

### Changed

- The tasks moved from the repository root to `tasks/<stage>/`: `TASKS.md` is now
  `tasks/stage1_flash/README.md`, and every link to it is updated.
- README: stage 2a in the course roadmap, a short section with its documentation, notebook, tasks and a quick start,
  and the tasks of both stages with the public solutions of stage 1.
- `CITATION.cff` and the package description and keywords: the normalizing flows.
- `notebooks/00_quickstart.ipynb`: its next steps point to the tasks of stage 1 and to the notebook of stage 2a.

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

[Unreleased]: https://github.com/ArseniiGav/prob-reco-sbi/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/ArseniiGav/prob-reco-sbi/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/ArseniiGav/prob-reco-sbi/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/ArseniiGav/prob-reco-sbi/releases/tag/v0.1.0

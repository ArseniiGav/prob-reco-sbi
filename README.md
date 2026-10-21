# prob-reco-sbi

[![CI](https://github.com/ArseniiGav/prob-reco-sbi/actions/workflows/ci.yml/badge.svg)](https://github.com/ArseniiGav/prob-reco-sbi/actions/workflows/ci.yml)
[![Python 3.10 to 3.14](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

**Probabilistic event reconstruction with simulation-based inference** is a course for bachelor students. To
reconstruct an event is to infer where it happened and how much energy it deposited from the response of a detector.
Here the aim is the full posterior distribution of these parameters, not only a best estimate. The course starts with
**A Flash in the Dark**, a photon-transport simulation of a toy two-dimensional scintillation detector with switchable
optical processes. The later stages train neural posterior estimators on its data and then apply them to an open
neutrino-telescope benchmark. This repository holds the simulation package `prob_reco_sbi.flash`, its documentation
and the tasks for the students.

<p align="center">
  <img src="docs/img/flash_photons.gif" width="480" alt="Animation of the photons of one event spreading through the toy detector">
  <br>
  <em>The photons of one event fly out from the vertex at the speed of light in the medium. Scatterings and wall
  reflections appear as markers, absorbed photons turn grey, and detected photons light up their sensor (black: no
  light, green: most light). The event has 0.3 MeV, below the energy range of the default prior, so that its few
  hundred photons can be followed one by one.</em>
</p>

## Course roadmap

| stage | content                                                                                                  | status               |
|-------|----------------------------------------------------------------------------------------------------------|----------------------|
| 1     | **A Flash in the Dark**: the toy detector, its data and the tasks about the data ([TASKS.md](TASKS.md))  | this release (0.1.0) |
| 2     | Neural posterior estimation (NPE) of the vertex and the energy from the simulated detector response     | next                 |
| 3     | NPE on NuBench, an open benchmark for event reconstruction in neutrino telescopes ([arXiv:2511.13111](https://arxiv.org/abs/2511.13111)) | later |

For background on simulation-based inference, see K. Cranmer, J. Brehmer and G. Louppe,
[The frontier of simulation-based inference](https://doi.org/10.1073/pnas.1912789117),
*Proc. Natl. Acad. Sci. USA* **117**, 30055 (2020).

## The toy detector

<p align="center">
  <img src="docs/img/detector.png" width="640" alt="Sketch of the toy detector: a disc with 32 photosensor arcs on its wall">
</p>

A disc of radius 1 m carries 32 photosensors on its wall. An event at the vertex $(x_0, y_0)$ with the energy $E$ emits
on average 1000 photons per MeV, in random directions, at an unknown time $t_0$. Every photon is tracked until its fate.
On the way it can undergo four processes:

1. **absorption** in the bulk (absorption length 1.5 m);
2. **Rayleigh scattering** in the bulk (scattering length 2 m, phase function $\propto 1 + \cos^2\theta$);
3. **reflection** on the wall between the sensors (probability 0.5, diffuse or specular);
4. **detection** by a sensor, with a Gaussian time jitter of 1 ns.

The first three can be switched off, one by one. For every sensor the detector records the number of hits (the charge),
their mean time and the first time. Reconstructing $(x_0, y_0, E)$ from this response is the inverse problem of the
course. The model, with all equations, is in [docs/physics.md](docs/physics.md).

| parameter                         | default                                              |
|-----------------------------------|------------------------------------------------------|
| detector radius $R$               | 1.0 m                                                |
| sensors                           | 32 arcs of 0.10 m (photocoverage 0.509)              |
| light yield $Y$                   | 1000 photons/MeV, Poisson-distributed photon number  |
| speed of light $c_n$              | 0.2 m/ns                                             |
| absorption length $\lambda_a$     | 1.5 m                                                |
| scattering length $\lambda_s$     | 2.0 m                                                |
| wall reflectivity $\rho$          | 0.5, diffuse (Lambertian)                            |
| detection efficiency, time jitter | 1.0, 1 ns                                            |
| dark noise                        | off (rate per sensor; readout window 100 ns)         |
| prior                             | vertex uniform in the disc; $E$ uniform in [0.5, 5] MeV; $t_0$ uniform in [0, 10] ns |

## Installation

Python 3.10 or newer. Create a virtual environment and install PyTorch first, then the package:

```bash
git clone https://github.com/ArseniiGav/prob-reco-sbi.git
cd prob-reco-sbi
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU-only PyTorch
python -m pip install -e ".[analysis,dev]"
python -m pytest                                                               # fast tests (the "deselected" ones are the slow tests)
```

The extra `analysis` installs matplotlib, pandas, JupyterLab and the Jupyter kernel; `dev` installs the test and lint
tools. The environment takes about 1.5 GB of disk with the CPU build of PyTorch, and the data of all task commands
about 0.4 GB; on a cluster with a small home quota, put both on a work or scratch file system. In every new terminal
or job, activate the environment again: `cd prob-reco-sbi && source .venv/bin/activate`.

**Notebooks.** Start JupyterLab from the activated environment with `python -m jupyter lab` (a bare `jupyter` may
belong to another Python installation, whose kernel does not have the package). On JupyterHub, in VS Code or with
any other Jupyter that was not installed into this environment, register the environment as a kernel once and choose
it in the notebook:

```bash
python -m ipykernel install --user --name prob-reco-sbi --display-name "Python (prob-reco-sbi)"
```

**CPU or GPU.** The simulation runs on any CPU and uses an NVIDIA GPU automatically when PyTorch sees one (`--device
auto`, the default). Which PyTorch build you need depends on the machine:

- CPU only (laptops, CPU nodes of a cluster): the CPU build above, a download of about 200 MB.
- NVIDIA GPU: run `nvidia-smi` on a GPU node. The `CUDA Version` in its header is the newest CUDA that the driver
  supports. Install the build for that CUDA major version, for example the CUDA 12 build:

  ```bash
  python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
  ```

  The `cu126` build also runs with older CUDA 12 drivers (we tested a driver that reports CUDA 12.2). For a driver that
  reports CUDA 13.0 or newer, use `cu130`. The [PyTorch selector](https://pytorch.org/get-started/locally/) lists the
  current builds. On Linux, a plain `pip install torch` from PyPI installs a CUDA 13 build together with the NVIDIA
  libraries it needs.
- Check what you got: `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`.

On a cluster, load the cluster's Python module (if there is one) before you create the environment. Put the environment
on a file system with enough space, because a CUDA build of PyTorch takes several GB.

**With [uv](https://docs.astral.sh/uv/)** instead of pip:

```bash
uv venv
source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install -e ".[analysis,dev]"
```

An environment made by `uv venv` has no pip of its own: use `uv pip install ...` for every later install (or create it
with `uv venv --seed`), since `python -m pip` fails there and a bare `pip` would install into another environment.

## Quick start

On the command line (`python -m prob_reco_sbi.flash` works the same way):

```bash
flashsim presets                                        # the physics presets and the default parameters
flashsim generate -n 10000 --seed 1                     # 10 000 events -> data/flash_full_n10000_seed1.h5
flashsim generate --preset straight -n 2000 --seed 2 --save-hits   # straight light, every hit stored
flashsim info data/flash_full_n10000_seed1.h5           # datasets, configuration, provenance
flashsim generate --help                                # every option
```

In Python:

```python
from prob_reco_sbi.flash import generate, load, network_input, preset, save

data = generate(10_000, config=preset("full"), seed=1)  # on the GPU if there is one, else on the CPU
save(data, "data/flash_full.h5")
data = load("data/flash_full.h5")

data.truth.x, data.truth.y, data.truth.energy  # true event parameters, shape [10000]
data.response.charge  # hits per sensor, [10000, 32]
data.response.time_mean  # mean hit time per sensor [ns], NaN for sensors without hits
data.photons.n_detected  # photon bookkeeping per event
x = network_input(data.response.charge, data.response.time_mean)  # network input, [10000, 64]
```

[examples/quickstart.py](examples/quickstart.py) is a complete short script, and
[notebooks/00_quickstart.ipynb](notebooks/00_quickstart.ipynb) a guided first look at the data. A CPU simulates
thousands of events of the default model per second.

## Physics presets and options

The presets form a cumulative physics ladder: each one switches on one more optical process.

| preset           | absorption | scattering | reflection |
|------------------|------------|------------|------------|
| `straight`       | off        | off        | off        |
| `absorb`         | on         | off        | off        |
| `absorb-scatter` | on         | on         | off        |
| `full` (default) | on         | on         | on         |

Every parameter can be changed on top of a preset: on the command line (`--preset straight --reflection --n-sensors
16`) or in Python with the same names (`preset("straight", reflection=True, n_sensors=16)`).

| group     | options of `flashsim generate`                                                                              |
|-----------|-------------------------------------------------------------------------------------------------------------|
| geometry  | `--radius`, `--n-sensors`, `--sensor-width`                                                                 |
| light     | `--light-yield`, `--photon-statistics {poisson,fixed}`, `--speed-of-light`                                  |
| bulk      | `--absorption/--no-absorption`, `--absorption-length`, `--scattering/--no-scattering`, `--scattering-length` |
| wall      | `--reflection/--no-reflection`, `--reflectivity`, `--reflection-model {diffuse,specular}`                   |
| sensors   | `--detection-efficiency`, `--time-jitter`, `--dark-rate` (Hz per sensor), `--readout-window`                |
| prior     | `--r-max`, `--energy-min`, `--energy-max`, `--t0-min`, `--t0-max`, `--fixed-vertex X Y`, `--fixed-energy E` |
| transport | `--max-steps`                                                                                               |
| run       | `--preset`, `-n`, `--seed`, `--device {auto,cpu,cuda}`, `-o`, `--overwrite`, `--save-hits`, `--save-tracks K`, `--track-photons P`, `--first-event-id`, `--photons-per-chunk`, `--threads`, `--no-progress`, `-v`/`-q` |

What each switch does in the simulation is described in [docs/physics.md](docs/physics.md#what-the-switches-do).

## Data format

One HDF5 file per run. The configuration and the provenance (seed, device, versions, creation time, command line) are
stored as attributes of the file.

| group       | content                                                                                     |
|-------------|---------------------------------------------------------------------------------------------|
| `/truth`    | `event_id`, `x`, `y`, `energy`, `t0`                                                        |
| `/response` | `charge`, `time_mean`, `time_first` per event and sensor                                    |
| `/photons`  | photon bookkeeping per event: emitted, detected, absorbed (bulk, wall, sensor), lost; direct, scattered and reflected light; dark noise |
| `/hits`     | optional (`--save-hits`): every hit with its sensor, time, path length and history          |
| `/tracks`   | optional (`--save-tracks K`): photon paths of the first K events                            |
| `/detector` | sensor positions                                                                            |

Types, shapes, units and reading examples are in [docs/data_format.md](docs/data_format.md).

## Reproducibility

The same seed gives identical data (every stored array) on the same type of device and hardware, with the same chunking
(`--photons-per-chunk`) and the same PyTorch build, whatever the number of threads. Between two such runs only the
provenance differs: the creation time and the wall time, and the recorded thread count and command line where those
differ. The event parameters, the photon numbers and the dark noise do not depend on the device or on
the chunking. Every file records how it was made. Details, including the CPU warm-up that keeps the first simulation of
a process identical to later ones, are in [docs/physics.md](docs/physics.md#random-numbers-and-reproducibility).

## Running on a cluster

Large samples are generated as shards, one job per shard with its own seed and its own range of event ids, and
combined with `flashsim merge`. [examples/cluster](examples/cluster/README.md) describes the workflow and the
resources a job needs, with a SLURM array job and an HTCondor submit file to adapt to your cluster.

## Tasks

[TASKS.md](TASKS.md) holds the tasks for the next session: set-up (T0), the prior (T1), the detector response (T2), the
physics ladder (T3) and event displays (T4), plus a bonus on the way to inference (T5).

## Development

```bash
python -m pip install -e ".[dev]"
pre-commit install          # ruff (lint and format), standard checks and nbstripout before every commit
pre-commit run --all-files  # the same checks on the whole repository
python -m pytest            # fast unit and physics tests (under a minute on a CPU; the slow tests are deselected)
python -m pytest -m slow    # slow tests: statistical regression and fresh-process reproducibility (tens of seconds)
```

Most slow tests compare with the original seminar implementation or with its reference numbers. Both are kept by the
course team and are not part of this repository, so those tests are skipped unless `FLASH_CORE_DIR` points to the
original code and `FLASH_REFERENCE_FILE` to the reference numbers. Continuous integration (GitHub Actions) runs ruff,
mypy and the pre-commit hooks, and the fast and slow tests on Python 3.10 to 3.14 with the CPU build of PyTorch. How to contribute:
[CONTRIBUTING.md](CONTRIBUTING.md). Changes between versions: [CHANGELOG.md](CHANGELOG.md).

## Citation

If you use this material, please cite it as described in [CITATION.cff](CITATION.cff) (GitHub shows it under "Cite
this repository").

## License

[MIT](LICENSE)

# Tasks: A Flash in the Dark

These tasks are about the data of the toy detector: what we simulate, what the detector sees, how every optical
process changes it, and how hard it is to go back from the detector response to the event. They prepare the next stage
of the course, in which a neural network learns the posterior distribution of the event parameters from these data.

- **Core tasks:** T0 to T4. **Bonus:** T5.
- The question IDs (T0.1, T2.3, ...) are fixed: we go through the answers by ID in the next session.
- Background: the model is in [docs/physics.md](docs/physics.md), the file layout in
  [docs/data_format.md](docs/data_format.md), and [notebooks/00_quickstart.ipynb](notebooks/00_quickstart.ipynb) is a
  guided first look at the data.

## What to bring to the next session

1. **Figures** (PDF or PNG) with the question ID in the file name, e.g. `T2.2_light_collection.pdf`.
2. **Short answers**: a few sentences per question ID, with the numbers you measured, in one Markdown file or notebook.
3. **Your code** (scripts or notebooks) and the seeds of your data sets, so that every figure can be made again.

Make your own plots too: whatever you find telling, surprising or useful. T2.5 asks for at least three of your own
proposals, but every task welcomes more.

**Figure checklist.** A figure should be readable on its own: axis labels with units, a legend that does not hide the
data, and a title or caption that says what is shown (preset, number of events, energy range). Show statistical
uncertainties where they matter (e.g. the error of the mean in each bin), use colours that colour-blind readers can
tell apart, and use a log axis when values span several orders of magnitude.

**Where to work.** Keep your scripts on a branch of your fork (see [CONTRIBUTING.md](CONTRIBUTING.md)) or in a folder
outside the repository. Data files are not committed: `data/` and `*.h5` are ignored by git.

## Notation

| symbol | meaning |
|--------|---------|
| $\theta = (x_0, y_0, E)$ | event parameters: vertex and energy |
| $r_0 = \sqrt{x_0^2 + y_0^2}$ | radius of the vertex |
| $t_0$ | event time (unknown to the reconstruction) |
| $n_j$, $\tau_j$ | charge (number of hits) and mean hit time of sensor $j$ |
| $Q = \sum_j n_j$ | total charge of an event |
| $Y$ | light yield, 1000 photons/MeV (`data.config.optics.light_yield`) |
| $Q/(Y E)$ | light collection: detected fraction of the mean number of emitted photons |
| $(x_j, y_j)$ | centre of sensor $j$ (`data.detector.sensor_x`, `data.detector.sensor_y`) |
| $d_j$ | distance from the vertex to the centre of sensor $j$ |
| $t - t_0 - d_j / c_n$ | delay of a hit with respect to straight flight to the sensor centre |
| $\left(\sum_j n_j x_j, \sum_j n_j y_j\right) / Q$ | charge-weighted barycentre of the sensor positions (a naive vertex) |

---

## T0 Set-up (about 1 hour)

**Goal:** a working installation, a first data set and a feeling for the cost of a simulation.

**T0.1** Clone the repository, create a virtual environment, install the package and run the tests.

```bash
git clone https://github.com/ArseniiGav/prob-reco-sbi.git
cd prob-reco-sbi
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu   # or a CUDA build, see README.md
python -m pip install -e ".[analysis,dev]"
python -m pytest                 # ends with "... passed, ... deselected": the deselected ones are the slow tests
```

For notebooks, start JupyterLab from the activated environment with `python -m jupyter lab`; on JupyterHub or in
VS Code, register the environment as a kernel first (see "Notebooks" in [README.md](README.md#installation)).

**T0.2** Generate 10 000 events with the default physics and time it on a CPU node (and on a GPU node, if your cluster
has one). How does the time scale with the number of events $N$ and with the preset?

```bash
time flashsim generate -n 10000 --seed 1 --overwrite        # default device: the GPU if there is one, else the CPU
time flashsim generate -n 10000 --seed 1 --device cpu --threads 4 -o data/t0_cpu_threads4.h5 --overwrite
time flashsim generate -n 10000 --seed 1 --device cuda -o data/t0_gpu.h5 --overwrite   # on a GPU node only

for n in 1000 10000 100000; do
  for p in straight absorb absorb-scatter full; do
    flashsim generate --preset "$p" -n "$n" --seed 3 -o "data/t0_${p}_n$n.h5" -q --overwrite
  done
done
```

**T0.3** Inspect the file: which datasets does it contain, with which shapes, and which configuration is stored?

```bash
flashsim info data/flash_full_n10000_seed1.h5
```

```python
from prob_reco_sbi.flash import load

data = load("data/flash_full_n10000_seed1.h5")
print(data)  # SimResult(10000 events, 32 sensors)
print(data.config.summary())
print(data.provenance)
```

**Deliverables:** the summary line of the tests (T0.1); a plot or table of the simulation time against $N$ for every
preset and every device you tried (T0.2).

**Hints**
- The time printed by `time` includes starting Python and importing PyTorch. The simulation alone is stored in every
  file as `data.provenance["wall_time_s"]` (also shown by `flashsim info`).
- The first use of a GPU costs extra start-up time, so time more than one $N$.
- On a shared node, use as many threads (`--threads`) as you requested from the scheduler.
- For the scaling, a log-log plot of the time against $N$ shows a power law as a straight line.

## Data for T1 to T4

With the installation working, make the data of the next tasks. All files go to `data/` in the repository root, and
every path in this document is relative to the repository root (a notebook in `notebooks/` reads the same file as
`../data/t2_full.h5`). Each command takes seconds to a minute on a CPU (on a cluster, run it in a job or an
interactive session, not on the login node; see [examples/cluster](examples/cluster/README.md)). Using the same seed
for every preset gives the same events (vertices, energies, $t_0$ and photon numbers), which makes comparisons between
presets sharper; such paired samples are made for differences, not for pooling (two presets with the same seed are not
independent samples).

```bash
# T1 and T2: the default model
flashsim generate -n 50000 --seed 5 -o data/t2_full.h5

# T3: the physics ladder, the same events for every preset; small files with every hit for the timing
for p in straight absorb absorb-scatter full; do
  flashsim generate --preset "$p" -n 20000 --seed 6 -o "data/t3_$p.h5"
  flashsim generate --preset "$p" -n 2000 --seed 7 --save-hits -o "data/t3_${p}_hits.h5"
done
# T3: one process alone on top of straight light (absorption alone is the preset absorb)
for process in scattering reflection; do
  flashsim generate --preset straight --"$process" -n 20000 --seed 6 -o "data/t3_straight_$process.h5"
  flashsim generate --preset straight --"$process" -n 2000 --seed 7 --save-hits -o "data/t3_straight_${process}_hits.h5"
done

# T4: a few events with every hit and the paths of 200 photons of each of the first five events
flashsim generate -n 20 --seed 9 --save-hits --save-tracks 5 --track-photons 200 -o data/t4_events.h5
```

`flashsim generate` refuses to overwrite an existing file; add `--overwrite` to replace it. Every file is read with
`load`:

```python
from prob_reco_sbi.flash import load

data = load("data/t2_full.h5")
data.truth.x, data.truth.y, data.truth.energy, data.truth.t0  # [N]
data.response.charge, data.response.time_mean, data.response.time_first  # [N, 32]
data.photons.n_emitted, data.photons.n_detected  # photon bookkeeping per event
```

## T1 What did we simulate? The prior (about 1 hour)

**Goal:** know the distribution of the event parameters. It is the prior that the inference of the next stage will
use.

**T1.1** Plot the distributions of $x_0$, $y_0$, $E$ and $t_0$, the vertices in two dimensions, and the distributions
of $r_0$ and $r_0^2$. Why is $r_0$ not uniform while $r_0^2$ is?

**T1.2** These parameter ranges are the prior that neural posterior estimation (NPE) will use later. Why does the prior
matter?

**Deliverables:** histograms of $x_0$, $y_0$, $E$, $t_0$, $r_0$ and $r_0^2$; a scatter plot of the vertices.

**Hints**
- `data.truth` of `data/t2_full.h5` has everything. The parameters can also be drawn without any simulation:
  `sample_prior(100_000, preset("full"), rng=1)` (both from `prob_reco_sbi.flash`).
- For T1.1, compare how many vertices fall into a thin ring near the centre and into one of the same width near the
  wall.
- For T1.2, recall Bayes' theorem, $p(\theta \mid x) \propto p(x \mid \theta)\, p(\theta)$. What happens to the
  posterior of an event close to the edge of the prior, for example at $E = 0.6$ MeV or $r_0 = 0.95$ m? What could a
  network trained on this prior say about an event of 8 MeV?

## T2 The detector response: what the network will see (2 to 3 hours)

**Goal:** understand the charges and times that the detector records, and the input that the network will get.

**T2.1** Distribution of the sensor charges (all sensors of all events pooled), the number of hit sensors per event,
and the fraction of empty sensors as a function of $E$ and of $r_0$.

**T2.2** The total charge $Q$: its distribution, $Q$ against $E$, and the light collection $Q/(Y E)$ against $r_0$.

**T2.3** Times: the mean and the first arrival times of the sensors, and the relative times. Why do we use relative
times ($t_0$ is unknown)?

**T2.4** The network input (`network_input`): the distributions of both halves (log charges and relative times). What
does an empty sensor look like?

**T2.5** At least three proposals of your own. For example: the charge of a sensor against its distance $d_j$; the
first-hit time against $d_j$ (time of flight, slope $1/c_n$); the charge-weighted barycentre against the true vertex
(a naive vertex); a naive energy from $Q$ with one calibration constant (its bias against $r_0$); correlations
between sensors; photon-fate fractions; the direct / scattered / reflected composition of the detected light.

**Deliverables:** one figure per question (T2.5: at least three), each with a short interpretation.

**Hints**
- Pool the charges with `data.response.charge.ravel()`; count hit sensors with `(charge > 0).sum(axis=1)`.
- For quantities against $E$ or $r_0$, bin the events (`np.digitize`) and plot the mean in every bin with its
  statistical error (standard deviation divided by the square root of the number of events).
- Empty sensors have `time_mean` and `time_first` equal to NaN: select hit sensors with `charge > 0` or use
  `np.nanmean`.
- Do not be surprised that first hits mostly arrive *before* the straight time of flight $t_0 + d_j / c_n$, and some
  even before $t_0$: the first hit is the earliest of $n_j$ photons, each with a 1 ns Gaussian jitter, and $d_j$ is
  measured to the sensor centre while the arc has a width. This is not a bug; think about what the earliest of many
  jittered times does.
- `network_input(charge, time_mean)` returns an array `[N, 64]`: columns 0 to 31 are the log charges, 32 to 63 the
  relative times. Its definition is in [docs/physics.md](docs/physics.md#network-input).
- The photon bookkeeping (`data.photons`) gives the fates and the direct / scattered / reflected counts per event.

## T3 Switch the physics on step by step (5 to 6 hours; the parts marked optional can wait)

**Goal:** see what every optical process does to the data. Go up the ladder `straight` -> `absorb` ->
`absorb-scatter` -> `full`, and also look at each process alone on top of `straight`.

**T3.1** For every preset: the light collection against $r_0$, the charge of a sensor against its distance $d_j$, the
photon arrival delays $t - t_0 - d_j/c_n$ (from the files with `--save-hits`), and the photon-fate fractions.
Optional: split the delays by the history of the photon (direct, scattered, reflected; `data.hits.n_scatter` and
`data.hits.n_reflect`).

**T3.2** Straight light: derive the expected charge of sensor $j$ for a vertex $(x_0, y_0)$ (think of the angle under
which the sensor arc is seen from the vertex) and compare it with the simulation. Optional: then add absorption.

```bash
flashsim generate --preset straight --fixed-vertex 0.5 0.2 --fixed-energy 2 --photon-statistics fixed -n 1000 --seed 8 -o data/t3_straight_vertex.h5
flashsim generate --preset absorb --fixed-vertex 0.5 0.2 --fixed-energy 2 --photon-statistics fixed -n 1000 --seed 8 -o data/t3_absorb_vertex.h5
```

**T3.3** Which process changes the collected light most? Which one destroys the timing information? Why does the
collection grow toward the wall in the full model?

**T3.4** How hard is the inverse problem? For every preset, estimate the energy from $Q$ with one calibration constant
and the vertex from the charge barycentre, and show the bias and the resolution of both against $r_0$. What will a
learned posterior need beyond $Q$ (the charge pattern, the timing)?

**Deliverables:** for T3.1 the four quantities with all presets on shared axes; for T3.2 the expected and the
simulated charges against the sensor index, with the residuals; for T3.4 the bias and the resolution against $r_0$.
Answer T3.3 in a few sentences and refer to your figures.

**Hints**
- Hits carry `event_id`, `sensor` and `time`. Get the $t_0$ and the vertex of each hit from `data.truth` (with
  `event_id` as the index, since the first event has id 0), and $d_j$ from the vertex and the sensor centre.
- T3.2: photons are emitted uniformly in angle. The two ends of every sensor arc are in `sensor_edges(geometry)`
  (angles on the wall). Check your result first for a vertex at the centre. A fixed vertex and fixed photon numbers, as
  in the commands above, make the statistical errors of the mean charges small. For absorption, the probability that a
  photon survives a straight path follows from the free-path distribution in [docs/physics.md](docs/physics.md); which
  path length do you use for a sensor arc?
- T3.3: put the curves of all presets (and of the single processes) into one figure. Look at how the delay
  distributions change, and at the direct / scattered / reflected composition of the light against $r_0$.
- T3.4: a single calibration constant $k$ for the whole data set, e.g. the median of $Q/E$, gives
  $E_\mathrm{rec} = Q/k$. Use robust measures in every bin of $r_0$: the median of $(E_\mathrm{rec} - E)/E$ for the
  bias and half the width of its central 68 % interval for the resolution. For the vertex, look at the distance between
  the barycentre and the true vertex, and at the component of their difference along the direction of the vertex.

## T4 Event displays (about 2 hours)

**Goal:** draw single events so that the eye sees what the network will see.

**T4.1** The disc with the sensors coloured by their charge, the true vertex and the charge barycentre, with $E$,
$r_0$ and $Q$ in the title.

**T4.2** Add the timing, in a second panel or as a polar plot.

**T4.3** The photon paths of one event (from `--save-tracks`): scatterings, reflections, detections and absorptions.

**T4.4** The same $\theta$ under the four presets, side by side.

```python
from prob_reco_sbi.flash import preset, simulate

presets = ("straight", "absorb", "absorb-scatter", "full")
events = {
    name: simulate(0.6, 0.3, 2.0, t0=0.0, config=preset(name), seed=10, device="cpu", save_tracks=1) for name in presets
}
```

(`device="cpu"` makes the individual photons the same for everybody; a GPU gives a different random realisation of
the same event.)

**T4.5** Bonus: an animation of the photons of one event.

**Deliverables:** one display per question; for T4.5 a GIF or MP4.

**Hints**
- `sensor_arcs(data.config.geometry)` returns points along every sensor arc. Draw all sensors with the same size, so that
  only their colour carries the charge (dark for no light, bright for much light), and add a colour bar with its label.
- Use different markers for the true vertex and the barycentre, and a legend outside the disc.
- T4.2: relative times (the second half of `network_input`) or first times as colour, with empty sensors in a neutral
  colour; or $n_j$ and $\tau_j$ against the sensor angle.
- T4.3: `data.tracks.of_event(event_id)` gives the tracks of one event, `data.tracks.polyline(k)` the points and their
  types. Give scatterings and wall reflections their own markers. A hundred photons are usually enough to see the
  structure.
- T4.4: use the same colour scale in all four panels.
- T4.5: use `matplotlib.animation.FuncAnimation`. A photon has travelled the path length $s = c_n (t - t_0)$ at the
  time $t$; interpolate its position along the polyline with `np.interp` on `path_length`. Saving the animation with
  `writer="pillow"` gives a GIF, but a large one (a palette per frame; about 9 MB for 200 frames at 800 px): keep the
  number of frames and the dpi modest, or save an MP4 with `writer="ffmpeg"` if ffmpeg is installed.

## T5 Bonus: towards inference (2 to 4 hours)

**Goal:** first steps toward the questions that the inference will have to answer.

**T5.1** The simulator is stochastic. Simulate the same $\theta$ 20 000 times and look at the distributions of a few
sensor charges and of $Q$. Are they Poisson? Are they correlated? Repeat with fixed photon statistics.

```bash
flashsim generate --fixed-vertex 0.3 0.4 --fixed-energy 2 -n 20000 --seed 11 -o data/t5_same_theta.h5
flashsim generate --fixed-vertex 0.3 0.4 --fixed-energy 2 --photon-statistics fixed -n 20000 --seed 11 -o data/t5_same_theta_fixed.h5
```

**T5.2** Degeneracy: find two different pairs $(E, r_0)$ with the same mean $Q$. Can you tell them apart with the full
charge pattern? With the times?

**T5.3** Detector design: 8, 16, 32 and 64 sensors, at a fixed sensor width or at a fixed photocoverage. At the default
width of 0.10 m at most 62 sensors fit on the wall (the generator rejects overlapping sensors), so 64 sensors exist
only at a fixed photocoverage.

```bash
for m in 8 16 32; do                                   # fixed width 0.10 m
  flashsim generate --n-sensors "$m" -n 20000 --seed 12 -o "data/t5_width_m$m.h5"
done
for spec in 8:0.40 16:0.20 32:0.10 64:0.05; do          # fixed photocoverage: sensors times width constant
  m=${spec%:*}; a=${spec#*:}
  flashsim generate --n-sensors "$m" --sensor-width "$a" -n 20000 --seed 12 -o "data/t5_coverage_m$m.h5"
done
```

**T5.4** Detector effects: time jitter, detection efficiency and dark noise. Which features suffer most?

```bash
for jitter in 0 1 3; do flashsim generate --time-jitter "$jitter" -n 20000 --seed 13 -o "data/t5_jitter$jitter.h5"; done
for eff in 1.0 0.5 0.25; do flashsim generate --detection-efficiency "$eff" -n 20000 --seed 13 -o "data/t5_eff$eff.h5"; done
for rate in 0 1e6 1e7; do flashsim generate --dark-rate "$rate" -n 20000 --seed 13 -o "data/t5_dark$rate.h5"; done
```

**Deliverables:** one or two figures per question with a short answer.

**Hints**
- T5.1: compare the variance of a charge with its mean, and compute the correlation matrix of the charges
  (`np.corrcoef`) and their correlations with $Q$. With fixed photon statistics, which source of fluctuations is gone?
  A single pair correlation has a statistical noise of $1/\sqrt{N}$ (0.007 for 20 000 events), which is about the size
  of the effect you are looking for: look at the average over all pairs and at the variance of $Q$ over its mean.
- T5.2: read candidate pairs off your light-collection curve (T2.2 or T3.1), simulate each pair with `--fixed-vertex`
  and `--fixed-energy`, and compare the distributions of a few features other than $Q$.
- T5.3: compare, for example, the fraction of empty sensors, the barycentre and the energy from $Q$ (T3.4) for the
  different designs.
- T5.4: change one effect at a time, with the same seed. The dark rate is per sensor: 1 MHz gives on average 0.1 noise
  hits per sensor and event in the default readout window of 100 ns. Measure "suffer" with the estimators of T3.4 or
  with how much the distributions of the network input change. Note that noise hits have times uniform in the fixed
  readout window, which do not move with $t_0$: with dark noise the relative times are no longer independent of $t_0$
  (compare the same events simulated with `--t0-min 0 --t0-max 0` and with `--t0-min 50 --t0-max 50`).

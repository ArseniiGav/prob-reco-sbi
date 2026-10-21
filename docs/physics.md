# Physics of the toy detector

"A Flash in the Dark" is a two-dimensional toy model of a scintillation detector. A point-like event inside a disc
emits photons, every photon is tracked through the detector until it is detected, absorbed or lost, and the
photosensors on the wall record what arrives. This page defines the model as implemented in `prob_reco_sbi.flash`
(version 0.1.0). The defaults are those of the `full` preset. The data files are described in
[data_format.md](data_format.md).

## Units and coordinates

| quantity                    | unit                    |
|-----------------------------|-------------------------|
| lengths, positions, paths   | m                       |
| times                       | ns                      |
| energies                    | MeV                     |
| speed of light in the medium| m/ns                    |
| dark-noise rate             | Hz (per sensor)         |
| angles                      | rad                     |

The centre of the detector is the origin. Angles are measured counter-clockwise from the $+x$ axis.

## Detector

The detector is a disc of radius $R$ whose wall carries $M$ identical photosensors. Sensor $j = 0, \dots, M-1$ is the
wall arc of length $a$ centred at the angle

```math
\phi_j = \frac{2\pi j}{M},
```

so it covers the angles $|\phi - \phi_j| \le a/(2R)$ (angular distances wrapped into $[-\pi, \pi)$). The fraction of
the wall covered by sensors, the photocoverage, is

```math
f_\mathrm{cov} = \frac{M a}{2 \pi R}.
```

| parameter            | symbol | option         | default | unit |
|----------------------|--------|----------------|---------|------|
| radius               | $R$    | `radius`       | 1.0     | m    |
| number of sensors    | $M$    | `n_sensors`    | 32      |      |
| sensor arc length    | $a$    | `sensor_width` | 0.10    | m    |
| photocoverage        | $f_\mathrm{cov}$ | (derived) | 0.509 |  |

Sensors may not overlap: configurations with $M a > 2\pi R$ are rejected. With $a = 0.10$ m and $R = 1$ m at most
62 sensors fit on the wall. For drawing, `sensor_positions`, `sensor_edges` and `sensor_arcs` return the sensor
centres, the angles of the arc ends and points along every arc.

## Event parameters and prior

An event has the parameters $\theta = (x_0, y_0, E)$: the vertex and the deposited energy. Its emission time $t_0$ is a
nuisance parameter: it is simulated, but a reconstruction does not know it. The prior is

```math
(x_0, y_0) \sim \text{uniform in the disc } r_0 = \sqrt{x_0^2 + y_0^2} \le r_\mathrm{max}, \qquad
E \sim \mathcal{U}(E_\mathrm{min}, E_\mathrm{max}), \qquad
t_0 \sim \mathcal{U}(t_{0,\mathrm{min}}, t_{0,\mathrm{max}}),
```

where "uniform in the disc" means the same density per unit area everywhere inside $r_0 \le r_\mathrm{max}$.

| parameter                    | option                  | default    | unit |
|------------------------------|-------------------------|------------|------|
| maximum vertex radius        | `r_max`                 | 1.0        | m    |
| energy range                 | `energy_min`, `energy_max` | 0.5, 5.0 | MeV |
| range of the event time      | `t0_min`, `t0_max`      | 0.0, 10.0  | ns   |
| fixed vertex                 | `fixed_vertex`          | none       | m    |
| fixed energy                 | `fixed_energy`          | none       | MeV  |

`fixed_vertex` and `fixed_energy` replace the corresponding distribution by a single value, to simulate the same
$\theta$ many times; setting `t0_min = t0_max` fixes $t_0$. `simulate(x, y, energy, t0)` takes the event parameters
directly instead of drawing them from the prior.

## Light emission

An event emits $N_\gamma$ photons at its vertex at the time $t_0$, with

```math
N_\gamma \sim \mathrm{Poisson}(Y E)
```

by default, or exactly $N_\gamma = \mathrm{round}(Y E)$ photons with `photon_statistics="fixed"`. Every photon starts
in a direction whose angle is uniform in $[0, 2\pi)$ (isotropic emission in the plane).

| parameter          | symbol | option              | default   | unit        |
|--------------------|--------|---------------------|-----------|-------------|
| light yield        | $Y$    | `light_yield`       | 1000      | photons/MeV |
| photon statistics  |        | `photon_statistics` | `poisson` |             |

## Photon propagation

Photons fly along straight segments with the speed $c_n$ in a single medium (no refraction). Absorption and Rayleigh
scattering in the bulk are described by the coefficients

```math
\mu_a = 1/\lambda_a, \qquad \mu_s = 1/\lambda_s, \qquad \mu = \mu_a + \mu_s,
```

where a process that is switched off contributes 0. The distance to the next bulk interaction is exponential,

```math
p(s) = \mu \, e^{-\mu s}, \qquad s \ge 0,
```

sampled as $s = -\ln(1-u)/\mu$ with $u \sim \mathcal{U}(0, 1)$. The distance to the wall from the position
$\mathbf{p}$ along the unit direction $\mathbf{u}$ is the positive root of $|\mathbf{p} + s \mathbf{u}| = R$,

```math
s_\mathrm{wall} = -b + \sqrt{b^2 - c}, \qquad b = \mathbf{p} \cdot \mathbf{u}, \qquad c = |\mathbf{p}|^2 - R^2 .
```

In every step the photon moves by $\min(s, s_\mathrm{wall})$, its path length $L$ grows by the same amount, and then it
interacts in the bulk ($s < s_\mathrm{wall}$) or at the wall.

| parameter             | symbol      | option              | default | unit |
|-----------------------|-------------|---------------------|---------|------|
| speed of light        | $c_n$       | `speed_of_light`    | 0.2     | m/ns |
| absorption length     | $\lambda_a$ | `absorption_length` | 1.5     | m    |
| scattering length     | $\lambda_s$ | `scattering_length` | 2.0     | m    |
| maximum interactions  |             | `max_steps`         | 80      |      |

## Interactions

One uniform random number $u$ per photon and step decides what happens at the interaction point.

**In the bulk.** The photon is absorbed if $u < \mu_a / \mu$ and Rayleigh-scattered otherwise. A scattering turns the
direction counter-clockwise by the angle $\vartheta \in [0, 2\pi)$ drawn from the two-dimensional phase function

```math
p(\vartheta) = \frac{1 + \cos^2\vartheta}{3\pi}.
```

**At the wall.** If the photon lands on a sensor arc, it is detected if $u < \varepsilon$ and absorbed by the sensor
otherwise. Between the sensors it is reflected if $u < \rho$ and absorbed by the wall otherwise. A reflected photon gets
a new direction:

- diffuse (Lambertian, default): the angle $\alpha$ between the new direction and the inward normal
  $\mathbf{n} = -\mathbf{p}/|\mathbf{p}|$ follows
  ```math
  p(\alpha) = \frac{\cos\alpha}{2}, \qquad \alpha \in [-\pi/2, \pi/2];
  ```
- specular (mirror): $\mathbf{u}' = \mathbf{u} - 2 (\mathbf{u} \cdot \mathbf{n}) \, \mathbf{n}$.

After a reflection the photon restarts 1 µm inside the wall.

**End of the path.** A photon still in flight after `max_steps` interactions is counted as lost. With the defaults this
does not happen in practice (0 of 2.8 × 10<sup>7</sup> photons in 10 000 events of the `full` preset).

| parameter              | symbol        | option                 | default   |
|------------------------|---------------|------------------------|-----------|
| detection efficiency   | $\varepsilon$ | `detection_efficiency` | 1.0       |
| wall reflectivity      | $\rho$        | `reflectivity`         | 0.5       |
| reflection model       |               | `reflection_model`     | `diffuse` |

## Hit times

A detected photon with the total path length $L$ (all segments, including those after scatterings and reflections)
gives a hit at the time

```math
t = t_0 + \frac{L}{c_n} + \delta, \qquad \delta \sim \mathcal{N}(0, \sigma_t^2),
```

with the time jitter $\sigma_t$ (`time_jitter`, default 1 ns; 0 disables it).

## Dark noise

Dark noise is off by default (`dark_rate = 0`). Otherwise every sensor of every event gets

```math
k \sim \mathrm{Poisson}\!\left(r_\mathrm{dark} \, T\right)
```

noise hits, with the rate $r_\mathrm{dark}$ per sensor (`dark_rate`, in Hz) and the readout window $T$
(`readout_window`, default 100 ns), and noise-hit times uniform in $[0, T]$. The mean $r_\mathrm{dark} T$ is a pure
number once both are in the same time unit: 1 MHz and 100 ns give 0.1 noise hits per sensor and event. Noise hits
count in the charges and times like photon hits; in the hit list they are flagged with `is_noise`. The readout window
only sets the noise exposure: photon hits are never cut by it.

## Readout

For every event and sensor the detector records

| quantity      | symbol             | definition                                              |
|---------------|--------------------|---------------------------------------------------------|
| charge        | $n_j$              | number of hits (detected photons and noise hits)        |
| mean time     | $\tau_j$           | mean time of the hits of sensor $j$ [ns]                |
| first time    | $t_{\mathrm{first},j}$ | earliest hit time of sensor $j$ [ns]                |

Both times are NaN for sensors without hits ($n_j = 0$). The charge is an integer count of photo-electrons: the toy has
no charge resolution, saturation or dead time.

## Photon bookkeeping

Every emitted photon ends in exactly one way: detected, absorbed in the bulk, absorbed by the wall, absorbed by a sensor
(only if $\varepsilon < 1$) or lost. For every event, exactly,

```math
n_\mathrm{emitted} = n_\mathrm{detected} + n_\mathrm{absorbed,bulk} + n_\mathrm{absorbed,wall}
+ n_\mathrm{absorbed,sensor} + n_\mathrm{lost}, \qquad \sum_j n_j = n_\mathrm{detected} + n_\mathrm{noise}.
```

Detected photons are split into three categories: *reflected* if the photon was reflected at least once, otherwise
*scattered* if it was scattered at least once, otherwise *direct*. These counts are stored per event; with
`--save-hits` the numbers of scatterings and reflections and the path length are stored for every hit.

## What the switches do

| option                         | effect in the simulation                                                                 |
|--------------------------------|------------------------------------------------------------------------------------------|
| `absorption = False`           | $\mu_a = 0$ ($\lambda_a = \infty$): no photon is absorbed in the bulk                    |
| `scattering = False`           | $\mu_s = 0$ ($\lambda_s = \infty$): photons never change direction in the bulk           |
| both off                       | $\mu = 0$: no free path is drawn and photons fly straight to the wall ("straight light") |
| `reflection = False`           | $\rho = 0$: the wall between the sensors is black                                        |
| `reflection_model = "specular"`| mirror reflection instead of diffuse reflection                                          |
| `detection_efficiency` below 1 | photons that reach a sensor are detected with probability $\varepsilon$                  |
| `time_jitter = 0`              | exact hit times $t_0 + L/c_n$                                                            |
| `dark_rate` above 0            | noise hits in every sensor, see above                                                    |
| `photon_statistics = "fixed"`  | $N_\gamma = \mathrm{round}(Y E)$ instead of a Poisson number                             |
| `fixed_vertex`, `fixed_energy` | the same vertex or energy for every event                                                |

Specular reflection on a perfect circle keeps the angle between the photon and the wall. A photon emitted exactly at the
centre therefore bounces back and forth along one diameter; if that diameter misses the sensors, the photon is absorbed
by the wall at one of its bounces (with $\rho = 1$ it bounces until `max_steps` and is counted as lost).

## Presets: the physics ladder

| preset           | absorption | scattering | reflection |
|------------------|------------|------------|------------|
| `straight`       | off        | off        | off        |
| `absorb`         | on         | off        | off        |
| `absorb-scatter` | on         | on         | off        |
| `full` (default) | on         | on         | on         |

All other parameters are the same in the four presets, and every parameter can be changed on top of a preset
(`preset("straight", reflection=True)` in Python, `--preset straight --reflection` on the command line).

## Network input

The neural networks of the next stage of the course see a fixed representation of every event, computed by
`network_input(charge, time_mean)`:

```math
\mathbf{x} = \bigl(\ln(1 + n_1), \dots, \ln(1 + n_M),\; \Delta\tau_1, \dots, \Delta\tau_M\bigr), \qquad
\Delta\tau_j = \begin{cases} \tau_j - \bar\tau & n_j > 0 \\ 0 & n_j = 0 \end{cases}, \qquad
\bar\tau = \frac{\sum_{j:\, n_j > 0} n_j \tau_j}{\sum_j n_j}.
```

The result has the shape `[N, 2M]` (float32); `use_time=False` returns only the $M$ log charges. Empty sensors get
$\ln(1 + 0) = 0$ and a relative time of 0, and an event without any hit gets zeros everywhere. Adding the same constant
to every hit time leaves the relative times unchanged, so the unknown event time $t_0$ drops out of them. With dark
noise switched on, the noise hits enter $n_j$ and $\tau_j$ like photon hits; since their times are uniform in the
fixed readout window and do not move with $t_0$, the relative times of noisy sensors then depend on $t_0$, which
becomes a real nuisance parameter.

## Random numbers and reproducibility

All random numbers derive from one seed. Independent NumPy streams give the event-level quantities (prior, $t_0$, photon
numbers, dark noise), and a `torch.Generator` on the compute device gives the photon-level ones (directions, free paths,
interactions, time jitter). Consequently:

- the event parameters, photon numbers and noise counts do not depend on the device or on the chunking;
- all stored arrays are identical for the same seed, device type, chunking (`photons_per_chunk`), PyTorch build and
  hardware, whatever the number of threads (the realisation can change between CPU models, because the vector math
  library of PyTorch chooses its code path by the processor's instruction set; a different GPU model can change it as
  well). Only the provenance differs between two such runs: the creation time and the wall time, and the recorded
  thread count and command line where those differ.

Photons are transported in chunks of about `photons_per_chunk` photons (default $2^{21}$ on a CPU and $2^{23}$ on a
GPU); the value used is stored in every file. Before the first transport in a process, the package evaluates the CPU
math kernels once single-threaded (`warm_up_cpu_kernels`): without this, the first multi-threaded call of `sin`,
`cos`, `sqrt` or `log` in a process could compute one thread's block of elements with errors of order $10^{-4}$ (an
initialisation race in the vector math library of the MKL-backed CPU builds of PyTorch), which flips the fate of a few
photons and, through the random draws of the time jitter, changes the whole realisation of that run. On our test
machine (torch 2.14.1+cpu, 16 threads) the first large kernel call was affected in 13 of 40 fresh processes without
the warm-up and in 0 of 40 with it, and 105 fresh simulations and command-line runs with the warm-up all reproduced
the canonical data. The warm-up relies on this behaviour of the library; a test of the package starts fresh
subprocesses to catch a regression (`tests/test_simulate.py`). If you ever see two runs with the same seed disagree,
please report it with the output of `flashsim info` of both files.

Two runs with the same seed but different configurations (for example two presets) share the event parameters, the
photon numbers and the dark noise, so they are paired at the event level. The photon-level random numbers (directions,
free paths, interactions, time jitter) are drawn from one stream per run in the order of the chunks, so they are only
shared as long as the two configurations consume the same number of random numbers; a change of the number of
detected photons in a chunk, for instance, shifts the stream of all later chunks. Paired samples give precise
differences between configurations, but they must not be pooled as if they were independent.

## What the toy leaves out

The model is deliberately simple: photons move in a plane, the medium is uniform, and absorption length, scattering
length and reflectivity do not depend on the wavelength. There is no refraction or total internal reflection, no
re-emission, no sensor charge resolution, saturation or dead time, and an event is a single point that emits all its
light at $t_0$.

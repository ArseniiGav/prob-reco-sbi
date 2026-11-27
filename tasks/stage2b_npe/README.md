# Tasks: Neural posterior estimation

In stage 1 you saw how hard it is to go back from the detector response to the event, and in stage 2a you learned
normalizing flows in one dimension. Now a network learns the posterior of the event parameters
$\theta = (x_0, y_0, E)$ from simulated events: neural posterior estimation (NPE). The flow of 2a becomes conditional
on the observations, then a chain of three such flows, then a flow from a library. Wherever the toy allows it, every
model is compared with an exact posterior.

- **Core tasks:** N1 to N5 (about 9.5 hours). **Bonus:** NB1 and NB2.
- The question IDs (N1.1, N2.4, ...) are fixed: we go through the answers by ID together.
- **Two meetings:** the first after N1 and N2, the second after N3 to N5. For every section, be ready to explain its
  figure in your own words and to answer its "why" question (both are named under the section title).
- **Levels.** *easy*: one call of the package and one figure; the work is the interpretation. *medium*: a short
  piece of your own code or a short derivation, with a skeleton or a helper in the hints.
- Background: the maths is in [docs/npe.md](../../docs/npe.md), with the given figures and animations, and
  [notebooks/02_npe.ipynb](../../notebooks/02_npe.ipynb) is a guided first look at the package `prob_reco_sbi.npe`.
- **Training is maximum likelihood on simulated pairs** $\lbrace \theta_i, x_i \rbrace$: the forward KL divergence of
  2a, averaged over the observations ([docs/npe.md](../../docs/npe.md#the-loss-in-three-lines)). The simulator gives
  pairs, never the posterior.
- **Where to run.** A laptop is enough (CPU only). A network of the short budget trains in about a minute, one of
  the default budget in a few minutes; `torch.set_num_threads(4)` is plenty.

## Set-up

Update your clone with `git pull` and install again, because the library zuko is new (it is in the `analysis`
extra). Then check the new subpackage with its tests:

```bash
python -m pip install -e ".[analysis,dev]"
python -m pytest tests/test_npe.py               # ends with "... passed"
```

## Data

All files go to `data/` in the repository root. One script makes all of them with fixed seeds, on any operating
system (a few minutes and about 0.2 GB):

```bash
python examples/make_npe_data.py
```

| file | preset | events | used in |
|---|---|---|---|
| `data/n1_center.h5` | `full`, vertex fixed at $(0, 0)$ | 60 000 | N1 |
| `data/n2_absorb_train.h5`, `data/n2_absorb_test.h5` | `absorb` | 310 000 and 2000 | N2, N5.3 |
| `data/n3_full_train.h5`, `data/n3_full_test.h5` | `full` | 310 000 and 5000 | N3, N4, N5 |

The same files with `flashsim`:

```bash
flashsim generate --fixed-vertex 0 0 -n 60000 --seed 21 -o data/n1_center.h5
flashsim generate --preset absorb -n 310000 --seed 22 -o data/n2_absorb_train.h5
flashsim generate --preset absorb -n 2000 --seed 23 -o data/n2_absorb_test.h5
flashsim generate -n 310000 --seed 24 -o data/n3_full_train.h5
flashsim generate -n 5000 --seed 25 -o data/n3_full_test.h5
```

`load_pairs` reads the pairs of a file in the representation that the network is given:

```python
from prob_reco_sbi.npe import load_pairs

theta, x = load_pairs("data/n3_full_train.h5", "charge+time")  # [310000, 3] (x0, y0 in m, E in MeV), [310000, 64]
theta, x = load_pairs("data/n2_absorb_train.h5", "charge")  # 32 log charges
theta, x = load_pairs("data/n3_full_train.h5", "ganged")  # 17 + 17 values of the ganged readout (N4)
theta, x = load_pairs("data/n1_center.h5", "q")  # theta = (E,), x = ln Q: the vertex is fixed
```

**The budgets.** Every task names one. The validation pairs are always the last 10 000 of the file.

```python
import torch
from prob_reco_sbi.npe import AutoregressiveFlow, fit, load_pairs

theta, x = load_pairs("data/n3_full_train.h5", "charge+time")
val = (theta[-10_000:], x[-10_000:])  # the validation pairs of every budget
torch.manual_seed(1)  # the initial weights
flow = AutoregressiveFlow(obs_dim=64)
history = fit(flow, theta[:100_000], x[:100_000], epochs=20, validation=val, seed=1)  # short budget
# default budget: fit(flow, theta[:300_000], x[:300_000], epochs=40, validation=val, seed=1)
```

In N1 the budget is the first 50 000 pairs of `n1_center.h5` for 30 epochs, with the last 10 000 for validation.
`fit` uses minibatches of 512 pairs and Adam with the learning rate $10^{-3}$, decaying to 0 as a cosine; use the
same in your own loops. The initial weights come from torch's random state: call `torch.manual_seed(...)` before you
build a network, and note the seed. The `seed` of `fit` fixes the order of the minibatches.

## What to hand in

1. **Figures** (PDF or PNG) and **animations** (GIF or MP4) with the question ID in the file name, e.g.
   `N2.5_exact_vs_flow_wall.pdf`.
2. **Short answers**: a few sentences per question ID and for the "why" question of every section, with the numbers
   you measured and the budget of every network (simulations, epochs, wall time, seed).
3. **Your code** (scripts or notebooks) with the seeds, so that every figure can be made again. Save your trained
   networks (`save_model`; your own models with `torch.save(model.state_dict(), path)`), so that later tasks do not
   retrain them.

Make your own plots too: whatever you find telling, surprising or useful.

**Figure checklist.** As in 2a: a figure should be readable on its own, with axis labels and units (m, MeV, nats), a
legend for every mark, and a caption with the network and its budget. Draw the truth wherever it exists: the true
$\theta$, the exact posterior, the prior range. Posterior samples are best shown as small transparent dots or as
contours.

**Where to work.** As in [stage 1](../stage1_flash/README.md): on your own branch (`git switch -c <your-name>`) or in
a folder outside the repository.

## Notation

| symbol | meaning |
|---|---|
| $\theta = (x_0, y_0, E)$ | the event parameters: the vertex [m] and the energy [MeV]; in N1 only $E$ |
| $x$ | the observations: what the network is given for one event |
| $x_o$ | one observed event, whose posterior is asked |
| $\pi(\theta)$ | the prior of stage 1: the vertex uniform in the disc, $E$ uniform in [0.5, 5] MeV |
| $p(x \mid \theta)$ | the likelihood: the density of the observations, known in general only through the simulator |
| $\mathcal{L}(x \mid \theta)$ | the likelihood as a function of $\theta$, where it has a formula (N1, N2) |
| $\kappa_j(x_0, y_0)$, $\kappa = \sum_j \kappa_j$ | probability that a photon from the vertex is detected by sensor $j$; the light collection $\langle Q \rangle / (Y E)$ |
| $\mu_j = Y E\, \kappa_j$ | the expected charge of sensor $j$ |
| $q(\theta \mid x)$ | the posterior, $q(\theta \mid x) \propto \mathcal{L}(x \mid \theta)\, \pi(\theta)$ |
| $\hat q(\theta \mid x)$ | the posterior estimated by the network; the hat marks a learned density |
| $\phi$ | the parameters of the network (all its weights) |
| $f_\phi$ | the transformation of the flow, in the normalizing direction: $z = f_\phi(\theta \mid x)$ |
| $z$, $p_Z$ | the base variable and the base density, the standard normal |
| normalizing, generative direction | from $\theta$ to $z$ (the density), from $z$ to $\theta$ (samples) |
| $\tilde\theta$ | the parameters after the fixed transforms of the package to unbounded, standardized coordinates |
| $h(x)$ | the output of the input embedding block, a network that summarizes the observations |
| $\psi_i$ | the conditioning variables of factor $i$ of the chain, $\psi_i = (h(x), \tilde\theta_1, \dots, \tilde\theta_{i-1})$ |
| parameters head | the network that turns $\psi_i$ into the $3K - 1$ parameters of the spline of factor $i$ |
| $\mathcal{L}(\phi)$ | the loss, $-\sum_{i=1}^{N} \log \hat q(\theta_i \mid x_i)$ over $N$ training pairs (the code takes the mean over a minibatch) |
| $D_\mathrm{KL}(q \,\Vert\, \hat q)$ | the forward Kullback-Leibler divergence, in nats |
| $K$, $B$ | the number of bins and the bound of a spline, as in 2a |
| $N_\mathrm{flows}$ | the number of flows in a sequence: 1 in every factor of the chain, 3 in zuko's flow |
| $r_0$, $Q$, $Y$, $n_j$, $\tau_j$, $t_0$ | as in stage 1: the radius of the vertex, the total charge, the light yield, the charge and the mean time of sensor $j$, the event time |

Two changes from 2a. The flow now acts on the parameters $\theta$ and is conditioned on the observations $x$; in 2a
it acted on $x$. The target is the posterior $q$ and the model is $\hat q$; in 2a the target was $p$ and the model
$q$. 2a also wrote $\varphi$ for the standard normal density; from now on the base density is $p_Z$, because $\phi$
are the weights of the network.

---

## N1 One parameter with an exact posterior: the energy at a known vertex (about 2 hours)

**Goal:** how well does the total charge fix the energy of an event at the center? A conditional flow, the NPE loss
and amortized inference, on a problem whose posterior is known exactly for every $Q$.

**Data:** `data/n1_center.h5`, 60 000 events of the preset `full` at the center: $\theta = E$, $x = \ln Q$. The N1
budget.

**From the package:** `energy_posterior` only to check your exact posterior; `rq_spline` (N1.5) and
`energy_to_unbounded` (N1.6). Write the networks, the loss, the training loop and the KL divergence yourself;
`GaussianChain` and `ConditionalFlow1D` are the package versions to compare with.

**At the meeting.** *Figure:* the KL divergence against $Q$ of three models: the Gaussian (N1.4), the flow without
the transform (N1.5) and the flow with it (N1.6). *Why:* the network was trained on pairs drawn from the prior and the
simulator. Why does it return the posterior, and where did the prior enter?

**N1.1** *(easy)* The pairs $\lbrace \theta_i, x_i \rbrace$: plot $Q$ against $E$ for the events at the center.
Measure the light collection $\kappa = \langle Q \rangle / (Y E)$. Is $Q$ Poisson-distributed at a given $E$
(variance over mean)?

**N1.2** *(medium)* The exact posterior. With $Q \sim \mathrm{Poisson}(Y \kappa E)$ and the flat prior,
$q(E \mid Q)$ is a Gamma distribution cut to the prior range. Write its density and draw it for $Q$ = 150, 650 and
1250, on top of the histogram of the energies of the simulated events with that $Q$. How wide may you make the window
of $Q$ to get more events?

**N1.3** *(medium)* The simplest conditional density, a Gaussian: a network maps $\ln Q$ to a mean $m$ and a width
$s$, $\hat q(E \mid Q) = \mathcal{N}(E;\, m, s^2)$. Train it on the training pairs with minibatches and the loss
$\mathcal{L}(\phi)$. Show the training and validation loss against the epoch, and $\hat q$ against $q$ for the three
$Q$ of N1.2.

**N1.4** *(easy)* The exact $D_\mathrm{KL}(q \,\Vert\, \hat q)$ by quadrature, against $Q$, for the Gaussian.

**N1.5** *(medium)* The conditional flow: the network now returns the $3K - 1$ parameters of a spline with $K = 16$
bins, on the standardized energy; same loop. Show $\hat q$ against $q$, and the KL against $Q$ for both models.
Does the flow beat the Gaussian anywhere? Why is the Gaussian already this good at the center?

**N1.6** *(medium)* The edges of the prior. For the $Q$ of events near the energy limits (e.g. $E$ = 0.52 and
4.95 MeV), how much of $\hat q$ lies outside the prior, for the Gaussian and for the flow? Switch on the logit
transform of the package and train the flow again: where does it now beat the Gaussian? Why does the transform work,
and what is its relative in F5? What does the network say about an event of 8 MeV (T1.2)?

**Deliverables:** $Q$ against $E$ with $\kappa$ and the variance over the mean (N1.1); three exact posteriors on top
of their histograms (N1.2); the loss curves and $\hat q$ against $q$ for the Gaussian (N1.3) and for the flow (N1.5);
the KL against $Q$ for the three models in one figure (N1.4 to N1.6); $\hat q$ at an edge without and with the
transform, and the posterior of the 8 MeV event (N1.6).

**Hints**
- [`pairs_to_posterior.gif`](../../docs/npe.md#from-the-density-of-the-observations-to-the-posterior) shows why a narrow
  window in $Q$ is a posterior. $Y$ = 1000 photons/MeV. $Q$ itself is
  `load("data/n1_center.h5").response.charge.sum(axis=1)` (`load` from `prob_reco_sbi.flash`).
- N1.1: at a given $E$ the mean of $Q$ is $Y \kappa E$; average $(Q - Y \kappa E)^2 / (Y \kappa E)$ over the events.
- N1.2: the derivation, step by step, is in [docs/npe.md](../../docs/npe.md#one-parameter-the-energy-at-a-known-vertex).
  `energy_posterior(Q, kappa)` returns the exact posterior with `pdf`, `mean` and `std`, for the check.
- N1.3: standardize $\ln Q$ and $E$ with the means and standard deviations of the training pairs. A density of the
  standardized energy $\tilde E$ becomes one of $E$ by the change of variables,
  $\log \hat q(E) = \log \hat q(\tilde E) - \log \mathrm{std}(E)$. Let the network return $\log s$, so that
  $s > 0$. A skeleton:
  ```python
  import torch
  from torch import nn


  class Gaussian(nn.Module):
      def __init__(self):
          super().__init__()
          self.net = nn.Sequential(nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2))

      def log_prob(self, e, lnq):  # standardized energies [N] and log charges [N, 1]
          m, log_s = self.net(lnq).unbind(-1)
          return ...  # log N(e; m, s^2)


  torch.manual_seed(1)
  model = Gaussian()
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
  for epoch in range(30):
      for batch in torch.randperm(len(e_train)).split(512):
          loss = -model.log_prob(e_train[batch], lnq_train[batch]).mean()
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
      # the validation loss of this epoch, under torch.no_grad()
  ```
  The package version is `GaussianChain(obs_dim=1, theta_dim=1, unbounded=False)`, trained with `fit`; it is a
  larger network (27 298 parameters against 4418 in the skeleton).
- N1.4: the exact posterior is narrow: integrate with `torch.trapezoid` on a fine grid of $E$ over its mean $\pm$ 10
  standard deviations, clipped to the prior range (the quadrature of F2).
- N1.5: `rq_spline(theta, params, bound)` is the spline of 2a with one set of knots per value: `params` has the shape
  `[N, 3K - 1]`, one row per pair, and it returns $z$ and $\log \lvert \mathrm{d}z/\mathrm{d}\theta \rvert$. Zero
  parameters give the identity: start the last layer of the network at zero (`torch.nn.init.zeros_`), so that
  $\hat q$ starts as $\mathcal{N}(0, 1)$ in $\tilde E$. Use `bound=5`. See
  [`conditional_flow.gif` and `energy_training.gif`](../../docs/npe.md#the-conditional-flow), drawn for the flow with
  the logit of N1.6. The package version is `ConditionalFlow1D(bins=16, unbounded=False)`.
- N1.6: `energy_to_unbounded(E)` returns $u = \mathrm{logit}[(E - E_\mathrm{min}) / (E_\mathrm{max} - E_\mathrm{min})]$
  and $\log \lvert \mathrm{d}u/\mathrm{d}E \rvert$: standardize $u$, put the flow on it, and add the log-Jacobian to
  $\log \hat q$. See [`coordinates_map.png`](../../docs/npe.md#the-coordinates); the package version is
  `ConditionalFlow1D(bins=16)`. The mass outside the prior: integrate $\hat q$ beyond the limits, or count samples
  outside. An event of 8 MeV (a fixed energy is not limited to the prior range; its $x$ is
  `observations(data.response.charge, None, "q")` with `data = load("data/n1_8mev.h5")`):
  ```bash
  flashsim generate --fixed-vertex 0 0 --fixed-energy 8 -n 1 --seed 26 -o data/n1_8mev.h5
  ```

## N2 Three parameters: the chain and the exact posterior (about 2.5 hours)

**Goal:** where was the event, and with which energy, from the charges of the 32 sensors? The product rule, an input
embedding block, and an exact posterior to measure the chain against.

**Data:** preset `absorb`, charges only: $x$ = 32 log charges (`"charge"`). Without scattering and reflection every
photon flies straight, and the likelihood has a formula.

**From the package:** `rq_spline` and `rq_spline_inverse`, the coordinate map and the standardization (through
`PosteriorEstimator`), `fit`, `AutoregressiveFlow` (the package chain, to compare with), `detection_probability`,
`energy_evidence`, `grid_posterior` (the check of N2.4, the tool of N2.6) and `credible_levels`.

**At the meeting.** *Figure:* the exact posterior against the flow for the event near the wall (N2.5). *Why:* the
loss is one number for all events. Where is the flow of the short budget farthest from the exact posterior, and why
there?

The chain is the product rule, every factor a conditional spline as in N1.5:

```math
\hat q(\theta \mid x) = \hat q(x_0 \mid \psi_1)\, \hat q(y_0 \mid \psi_2)\, \hat q(E \mid \psi_3), \qquad
\psi_i = \big(h(x), \tilde\theta_1, \dots, \tilde\theta_{i-1}\big).
```

**N2.1** *(medium)* The normalizing direction: write the log-density of the chain, with $K = 8$ bins per factor.
Show the number of parameters, and the loss of the untrained chain against its expected value.

**N2.2** *(medium)* The generative direction: sample coordinate by coordinate, each with the inverse of its factor.
Check it: the samples, sent back through the normalizing direction, are standard normal.

**N2.3** *(easy)* Train your chain with the short budget, and the package's `AutoregressiveFlow(obs_dim=32)` with the
same budget and seed. Show the training and validation loss against the epoch, yours and the package's.

**N2.4** *(medium)* The exact posterior of one central event, the first event of `data/n2_absorb_test.h5` with
$r_0$ < 0.2 m. Write the Poisson likelihood of the 32 charges,

```math
\mathcal{L}(x \mid \theta) = \prod_{j=1}^{32} \frac{\mu_j^{n_j}\, e^{-\mu_j}}{n_j!}, \qquad
\mu_j = Y E\, \kappa_j(x_0, y_0),
```

on a grid of $(x_0, y_0)$, with $E$ integrated out. Show the exact 68 % and 95 % contours with the samples of your
chain of N2.3 in $(x_0, y_0)$, and the exact and the flow posterior of $E$.

**N2.5** *(easy)* The same for an event near the wall, the first test event with $r_0$ > 0.9 m.

**N2.6** *(easy)* Many events: the width of $\hat q$ over the width of $q$ against $r_0$, for the first 200 test
events, with the short and with the default budget.

**Deliverables:** the number of parameters and the loss of the untrained chain (N2.1); the sampling check (N2.2); the
loss curves of both chains (N2.3); exact against flow for the central and for the wall event (N2.4, N2.5); the width
ratio against $r_0$ for both budgets (N2.6).

**Hints**
- N2.1: write your chain as a subclass of `PosteriorEstimator`. The base class maps $\theta$ to $\tilde\theta$ (the
  coordinate map and the standardization, with their Jacobian) and standardizes $x$; you write the learned part.
  Then `log_prob`, `sample` and `fit` work as for the package's models. See
  [`chain_diagram.png`](../../docs/npe.md#the-chain).
  ```python
  from torch import nn
  from prob_reco_sbi.npe import PosteriorEstimator, mlp, n_spline_params, rq_spline, rq_spline_inverse


  class MyChain(PosteriorEstimator):
      def __init__(self, obs_dim, bins=8, bound=5.0):
          super().__init__(obs_dim, theta_dim=3)
          self.bound = bound
          self.embedding = mlp(obs_dim, [128, 128], 32)  # the input embedding block h(x)
          # one parameters head per factor; zero_last=True: every factor starts as the identity
          heads = [mlp(32 + i, [64, 64], n_spline_params(bins), zero_last=True) for i in range(3)]
          self.heads = nn.ModuleList(heads)

      def context(self, x_t):
          return self.embedding(x_t)  # h(x), once per observation

      def _forward_tilde(self, theta_t, h):
          """Normalizing direction: theta~ [N, 3] and h [N, 32] -> z [N, 3] and log|det dz/dtheta~| [N]."""
          ...  # N2.1

      def _inverse_tilde(self, z, h):
          """Generative direction: z [N, 3] and h [N, 32] -> theta~ [N, 3]."""
          ...  # N2.2
  ```
- N2.1: an untrained chain is the identity in $\tilde\theta$. What is $\hat q$ in $\tilde\theta$ then, and what is
  the mean of $-\log \hat q$ over the training pairs? Remember the Jacobian of $\theta \to \tilde\theta$
  (`chain.theta_to_tilde(theta)` returns it, after `chain.set_standardization(theta, x)`). `chain.n_params` counts
  the parameters.
- N2.2: `rq_spline_inverse(z, params, bound)`. Factor $i$ needs $\tilde\theta_1, \dots, \tilde\theta_{i-1}$, which
  are known when you invert the factors in order. `chain(samples, x_o)` returns $z$ and the log-determinant; compare
  every column of $z$ with $\mathcal{N}(0, 1)$, e.g. with `ks_statistic` of 2a. See
  [`chain_sampling.gif`](../../docs/npe.md#the-chain).
- N2.3: the loss is $-\log \hat q$ with $\hat q$ a density per $\mathrm{m}^2\,\mathrm{MeV}$, so it can be negative.
  `history.epoch`, `history.train_loss`, `history.val_loss` and `history.seconds` hold the curves and the wall time.
  See [`posterior_training.gif`](../../docs/npe.md#the-chain). `save_model` stores only the models of the package:
  save your chain with `torch.save(model.state_dict(), path)`. With the same seeds, and its layers built in the order
  of the package's, your chain gives the package's losses to the last digit: a check of your code, nothing more.
- N2.4: the derivation is in
  [docs/npe.md](../../docs/npe.md#three-parameters-the-vertex-and-the-energy-with-straight-light).
  `detection_probability(X, Y, "absorb")` returns $\kappa_j$ on a grid, shape `[..., 32]`, and
  `energy_evidence(Q, kappa)` the logarithm of the integral over $E$ (with the prior and $Y$ of stage 1). Center a
  grid of about 161 by 161 points on the flow samples, a few of their standard deviations wide, and check that the
  exact posterior falls off well before the edges. At a fixed vertex the energy posterior is that of N1.2 with
  $\kappa(x_0, y_0)$; the marginal of $E$ is their mixture, weighted with the vertex posterior.
  `energy_posterior(Q, kappa)` takes an array of $\kappa$, one per grid point, and its `pdf(E[:, None])` gives the
  energy posteriors of all grid points at once. See [`likelihood_product.gif` and
  `exact_vs_flow.png`](../../docs/npe.md#three-parameters-the-vertex-and-the-energy-with-straight-light).
- N2.4: contours: `credible_levels(density, [0.68, 0.95])` gives the levels, and matplotlib's `contour` wants them
  increasing. The check: `grid_posterior(charge, "absorb")`; its `x` and `y` are 2D arrays on a grid turned to the
  radial and tangential directions of the vertex.
- N2.6: `grid_posterior` takes about a second per event; save its results. The width of the vertex posterior is
  $\sqrt{\mathrm{var}\, x_0 + \mathrm{var}\, y_0}$ (`vertex_width` of the exact one), the same from the samples of
  $\hat q$. For the default budget, train the package chain (or yours).

## N3 The full detector (about 2 hours)

**Goal:** what do scattering, reflection and the times do to the answer, across the detector? Posteriors that have no
exact reference: read them, and compare them with the naive estimators of stage 1.

**Data:** preset `full`, charges and times: $x$ = 32 log charges and 32 relative times (`"charge+time"`).

**From the package:** everything.

**At the meeting.** *Figure:* the three displays of N3.3. *Why:* how and why do the size and the shape of the
posterior change from the center to the wall, and why does the energy follow the band?

**N3.1** *(easy)* Train the package chain `AutoregressiveFlow(obs_dim=64)` with the default budget: the loss curves
and the time it took.

**N3.2** *(medium)* One observation $x_o$, the first event of `data/n3_full_test.h5`: its posterior samples on your
event display of T4.1, with the true vertex and the charge barycenter.

**N3.3** *(easy)* Three events: the first test events with $r_0$ < 0.2 m and between 0.45 and 0.55 m, and the event
of N3.2, the first above 0.9 m. The 68 % and 95 % contours on the display: how do the size and the shape change?

**N3.4** *(medium)* The energy: the samples of the event of N3.3 between 0.45 and 0.55 m in the $(r_0, E)$ plane,
with the naive energy $Q/k$ of T3.4 and the band $E\, \kappa(r_0) = Q/Y$; the correlation of $E$ and $r_0$.

**N3.5** *(medium)* Against T3.4: bias and resolution of the posterior mean against $r_0$, with the barycenter and
$Q/k$ on the same test events.

**N3.6** *(easy)* The width across the detector: the median posterior width of the vertex and $\sigma_E / E$
against $r_0$. Compare $\sigma_E / E$ with the photon statistics $1/\sqrt{Q}$.

**Deliverables:** the loss curves (N3.1); the display (N3.2); three displays with contours (N3.3); the $(r_0, E)$
plane with the band and the correlation (N3.4); the figure of T3.4 with the NPE curves added (N3.5); the two widths
against $r_0$ with the line $1/\sqrt{Q}$ (N3.6).

**Hints**
- N3.1: save the network, `save_model("data/n3_chain_default.pt", flow, history, seed=...)`, and load it with
  `load_model` in N3.2 to N3.6 and in N5.2.
- N3.2: `flow.sample(x_o, 2000)` returns `[2000, 3]` samples in m and MeV. See
  [`posterior_gallery.png`](../../docs/npe.md#reading-a-posterior-without-an-exact-reference).
- N3.3: contours from samples: histogram 20 000 samples on a grid of equal cells (`np.histogram2d`) and take the levels
  from `credible_levels`. See [`posterior_walk.gif` and
  `posterior_energy.gif`](../../docs/npe.md#reading-a-posterior-without-an-exact-reference).
- N3.4: $\kappa(r_0)$ of the preset `full` is your light-collection curve of T2.2, averaged over the azimuth;
  `np.corrcoef` gives the correlation. Near the wall this average fails for most events: there $\kappa$ depends on
  whether the vertex faces a sensor or a gap (see
  [`posterior_walk.gif`](../../docs/npe.md#reading-a-posterior-without-an-exact-reference)), and the band needs
  $\kappa(x_0, y_0)$, e.g. from simulations at fixed vertices along the azimuth of the event.
- N3.5: `flow.sample(x_test, 500)` samples all test events at once, shape `[500, 5000, 3]`; the posterior mean is the
  mean over the first axis. Use the robust measures of T3.4.
- N3.6: $\sigma_E / E$ of one event is the standard deviation of its energy samples over their mean. What would it
  mean if the curve lay on $1/\sqrt{Q}$?

## N4 Two modes (about 1.5 hours)

**Goal:** what if the readout cannot tell a sensor from its mirror image? A posterior with two modes, which a
Gaussian cannot describe, and an exact test of the weights of the modes.

**Data:** the files of N3 with the ganged readout: sensors $j$ and $32 - j$ share one channel, which gives 17 channels
and $x$ = 17 log charges and 17 relative times (`"ganged"`). Only N4.1 simulates.

**From the package:** everything, in particular `ganged_input`, `ganged_channels` (from `prob_reco_sbi.flash`) and
`GaussianChain`.

**At the meeting.** *Figure:* the display with two modes (N4.2). *Why:* why can no network, however long it is
trained, choose between the two modes, and why must each mode hold exactly one half?

**N4.1** *(easy)* The ganged readout. Simulate the mirror events $(0.5, 0.3)$ m and $(0.5, -0.3)$ m at 2 MeV, 1000
times each, and compare their mean ganged charges, and their mean charges of the 32 sensors.

```bash
flashsim generate --fixed-vertex 0.5 0.3 --fixed-energy 2 -n 1000 --seed 41 -o data/n4_mirror_up.h5
flashsim generate --fixed-vertex 0.5 -0.3 --fixed-energy 2 -n 1000 --seed 42 -o data/n4_mirror_down.h5
```

**N4.2** *(easy)* Train the package chain `AutoregressiveFlow(obs_dim=34)` on the ganged observations with the short
budget, and draw the posterior of the first event of `data/n4_mirror_up.h5` on the display.

**N4.3** *(easy)* The exact test: the posterior is symmetric under $y_0 \to -y_0$, so
$P(y_0 > 0 \mid x_o) = 1/2$. Measure the fraction of samples with $y_0 > 0$ for the first 2000 test events: its
distribution, its mean and its rms around 1/2.

**N4.4** *(medium)* Between the modes. For every event, the fraction of samples with
$\lvert y \rvert \lt \lvert y_0 \rvert / 2$, where $y$ is the $y_0$ of a sample and $y_0$ the true one; its median
against $\lvert y_0 \rvert$. How does this fraction change toward the mirror axis, and why? Train the chain again
with 16 bins: do more bins change anything?

**N4.5** *(easy)* The Gaussian comparison model `GaussianChain(obs_dim=34)` on the same data and budget: what does
it do with the two modes?

**Deliverables:** the mean patterns of the mirror events and their difference (N4.1); the display with two modes
(N4.2); the distribution of $P(y_0 > 0)$ with its mean and rms (N4.3); the median fraction between the modes against
$\lvert y_0 \rvert$ for 8 and 16 bins (N4.4); the display of the Gaussian model and the validation losses of both
models (N4.5).

**Hints**
- N4.1: `ganged_channels(32)` gives the channel of every sensor. The charge of a channel is the sum over its sensors:
  with `one_hot = ganged_channels(32)[:, None] == np.arange(17)`, it is `charge @ one_hot`. See
  [`ganged_readout.png`](../../docs/npe.md#two-modes).
- N4.2: the observations of a file whose vertex and energy are fixed:
  `observations(data.response.charge, data.response.time_mean, "ganged")` with `data = load("data/n4_mirror_up.h5")`
  (`load_pairs` refuses such a file: it has no parameter to infer). Mark the true vertex and its mirror image. See
  [`two_modes_training.gif`](../../docs/npe.md#two-modes).
- N4.3: a fraction measured with $n$ samples has a statistical error $\sqrt{p (1 - p) / n}$: compare the rms with it.
- N4.4: see [`two_modes_merge.gif`](../../docs/npe.md#two-modes). `AutoregressiveFlow(obs_dim=34, bins=16)` has 16
  bins. For the why, think of the likelihood as a function of $y_0^2$.
- N4.5: `history.best_val_loss` is the lowest validation loss of a training.

## N5 The same with a library (about 1.5 hours)

**Goal:** a standard flow from a library in your own training, and how to compare two models.

**From the package:** everything; from zuko: `zuko.flows.NSF`, a neural spline flow with $N_\mathrm{flows} = 3$
flows in alternating order of the parameters, each a spline whose parameters come from one masked conditioning
network (see [`masked_flow_diagram.png`](../../docs/npe.md#the-library)). zuko calls the observations the context:

```python
import zuko

flow = zuko.flows.NSF(features=3, context=64, transforms=3, hidden_features=[128, 128])
log_q = flow(x).log_prob(theta)  # x [N, 64], theta [N, 3] -> [N]
samples = flow(x[0]).sample((500,))  # [500, 3]
```

**At the meeting.** *Figure:* the width ratio of the chain and of zuko (N5.3). *Why:* N5.5.

**N5.1** *(medium)* Build zuko's NSF and evaluate the two calls. Write the adapter that puts it behind the coordinate
map and the standardization of the package, as your chain of N2.1, and train it with `fit` on the data of N3.1 with
the default budget. Show the number of parameters and the loss curves.

**N5.2** *(easy)* Chain (N3.1) against zuko (N5.1) on the full detector: the losses, the median widths of the vertex
and of the energy, and the training time, in one table.

**N5.3** *(easy)* Against the exact posterior: zuko on the data of N2 with the short budget, and the width ratio of
N2.6 against $r_0$, for the chain and for zuko.

**N5.4** *(easy)* Two modes: zuko on the ganged observations with the short budget, and the tests of N4.3 and N4.4
next to the numbers of the chain.

**N5.5** *(medium)* Which model is better, and how do you know?

**Deliverables:** the number of parameters and the loss curves (N5.1); the table (N5.2); the width ratio against
$r_0$ for the chain and for zuko (N5.3); the numbers of N4.3 and N4.4 for both models (N5.4); a few sentences that
refer to N5.2 to N5.4 (N5.5).

**Hints**
- N5.1: a skeleton of the adapter. The base class passes the standardized observations $\tilde x$ to the learned
  part as the context:
  ```python
  import zuko
  from prob_reco_sbi.npe import PosteriorEstimator


  class ZukoFlow(PosteriorEstimator):
      def __init__(self, obs_dim, theta_dim=3):
          super().__init__(obs_dim, theta_dim)
          self.flow = zuko.flows.NSF(features=theta_dim, context=obs_dim, transforms=3, hidden_features=[128, 128])

      def _forward_tilde(self, theta_t, x_t):
          """Normalizing direction: theta~ [N, 3] and x~ [N, obs_dim] -> z [N, 3] and log|det dz/dtheta~| [N]."""
          ...  # N5.1: zuko's transformation for the context x~

      def _inverse_tilde(self, z, x_t):
          """Generative direction: z [N, 3] and x~ [N, obs_dim] -> theta~ [N, 3]."""
          ...  # N5.1
  ```
  `self.flow(x_t)` is a distribution, and its `.transform` is the transformation from $\tilde\theta$ to $z$: look at
  its methods `call_and_ladj` and `inv`. `save_model` stores only the models of the package: save yours with
  `torch.save(model.state_dict(), path)`.
- N5.2: time both trainings with the same number of threads.
- N5.5: put the numbers of N5.2 to N5.4 side by side: validation losses, widths, $P(y_0 > 0)$, training time.

## NB Bonus (2 to 3 hours)

**From the package:** everything.

**NB1** How much simulation and training is enough? In `absorb`: from 3000 to 300 000 simulations and from 20 to 60
epochs. Show the losses, the width ratio and the shift of the mean against the exact posterior, as functions of the
number of simulations and of the number of epochs.

**NB2** Your own animation, one of: the posterior of your event during training; an event that walks to the wall or
to the mirror axis; your chain in the generative direction, coordinate by coordinate.

**Deliverables:** the quantities of NB1 against the number of simulations and of epochs; the GIF of NB2.

**Hints**
- NB1: reuse the exact posteriors of N2.6. The shift of the mean: the distance between the means of $\hat q$ and $q$
  in units of the width of $q$. Keep the validation set fixed for every budget.
- NB2: `fit(..., every=1)` keeps a snapshot after every epoch and `extra_steps=range(0, 200, 10)` more within the
  first epoch; `history.restore(flow, i)` loads snapshot `i` and `history.snapshot_epoch` says when it was taken. A
  GIF grows with the number of frames and the dpi: keep both modest.

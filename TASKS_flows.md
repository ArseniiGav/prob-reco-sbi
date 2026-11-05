# Tasks: Normalizing flows in one dimension

In the next stage a neural network learns the posterior distribution of the event parameters from simulated events,
and it describes this distribution with a normalizing flow. These tasks introduce flows where everything can be drawn
and compared with the exact answer: one dimension. You learn a known distribution from its samples, first with flows
that you write yourself and then with the spline flow of the package, and you measure how close every flow comes to
the truth.

- **Core tasks:** F1 to F6 (about 8 hours). **Bonus:** B1 and B2.
- The question IDs (F1.1, F2.3, ...) are fixed: we go through the answers by ID together.
- Background: the maths is in [docs/flows.md](docs/flows.md), and
  [notebooks/01_flows_1d.ipynb](notebooks/01_flows_1d.ipynb) is a guided first look at the package
  `prob_reco_sbi.flows1d`.
- **Training is always maximum likelihood on samples**, which minimises the forward KL divergence
  ([docs/flows.md](docs/flows.md#training-maximum-likelihood-is-the-forward-kl-divergence)). This is how neural
  posterior estimation (NPE) trains its flows: the simulator gives samples, never the value of a density.
- **Where to run.** A laptop is enough (CPU only): a fit takes from a few seconds to a few tens of seconds. More
  threads hardly help, because the tensors are small; on a shared machine `torch.set_num_threads(4)` is plenty.

## Set-up

No new installation is needed. Update your clone with `git pull` (if you work on your own branch, merge the update
into it); the editable installation of T0.1 finds the new subpackage by itself. Check it with its tests:

```bash
python -m pytest tests/test_flows1d.py           # ends with "... passed"
```

**The samples.** Every task learns the same target, the "camel with a shoulder" of
[docs/flows.md](docs/flows.md#the-target-a-camel-with-a-shoulder),

```math
p(x) = 0.30\,\mathcal{N}(x; -2.2, 0.25^2) + 0.50\,\mathcal{N}(x; 0.5, 0.9^2) + 0.20\,\mathcal{N}(x; 3.5, 1.4^2),
```

from the same 20 000 samples:

```python
from prob_reco_sbi.flows1d import CAMEL, camel

print(CAMEL)  # {'weights': (0.3, 0.5, 0.2), 'means': (-2.2, 0.5, 3.5), 'stds': (0.25, 0.9, 1.4)}
target = camel()
x = target.sample(20_000, seed=11)  # float64 tensor of shape [20000]: the training samples of every task
```

When you need further independent samples (held-out data, the bonus tasks), draw them with other seeds and write the
seeds down.

**The training budget.** Start with 2000 Adam steps at the learning rate $10^{-2}$, with all 20 000 samples in every
step (full batch). This is a starting point, not a rule: if the loss is still falling at the end, train longer and say
so, and give two flows the same budget whenever you compare them.

**The initial parameters.** Planar and radial layers start from random parameters: call `torch.manual_seed(...)`
before you create them and note the seed. A spline (`RQSpline`) starts as the identity and needs no seed. For your
own planar layer, the start of `flows1d.Planar` is one choice: $w, b \sim \mathcal{N}(0, 1)$ and
$u \sim \mathcal{N}(0, 0.1^2)$. Two traps: at $u = w = b = 0$ every gradient is exactly zero, so the layer never
trains; and a small $u$ does not give a small $\hat u$ (see F2.2): for $u \to 0$,
$\hat u \approx (\log 2 - 1)/w \approx -0.31/w$, a large shift when $w$ starts close to 0.

## What to hand in

1. **Figures** (PDF or PNG) and **animations** (GIF or MP4) with the question ID in the file name, e.g.
   `F2.3_kl_vs_depth.pdf` or `F6.1_training_spline.gif`.
2. **Short answers**: a few sentences per question ID, with the numbers you measured (KL divergences in nats,
   parameter counts, training times), in one Markdown file or notebook.
3. **Your code** (scripts or notebooks) with the seeds of the samples and of the initial parameters, so that every
   figure can be made again.

Make your own plots too: whatever you find telling, surprising or useful. Some ideas: the KL divergence against the
number of training samples (500 to 50 000); a fit to a few hundred samples, with the loss on the training samples
against the loss on held-out samples (`split` and `fit(..., x_val=...)`); the training time against the number of
parameters.

**Figure checklist.** A figure should be readable on its own: axis labels with units (KL divergences in nats), a
legend that does not hide the data, and a title or caption that says what is shown (the flow, its number of layers or
bins, the number of samples and of training steps). Draw the truth wherever it exists (the target density, the oracle
map, the base), so that the eye can compare with it. KL divergences span orders of magnitude: plot them on a log axis.

**Where to work.** As in [TASKS.md](TASKS.md): on your own branch (`git switch -c <your-name>`) or in a folder outside
the repository.

## Notation

| symbol | meaning |
|--------|---------|
| $p(x)$, $F(x)$ | the target: its density and its cumulative distribution function (CDF) |
| $q(x)$ | the density of the flow (the model) |
| $f$ | the map of the flow, strictly increasing: $z = f(x)$; a stack of layers is $f = f_L \circ \dots \circ f_1$ |
| $z$ | the variable of a Gaussian base, $z \sim \mathcal{N}(0, 1)$, with the density $\varphi$ and the CDF $\Phi$ |
| $u$ | the variable of a uniform base, $u \sim \mathcal{U}(0, 1)$ |
| $p_Z$, $p_U$ | the density of the base: Gaussian ($p_Z = \varphi$) or uniform ($p_U$) |
| $\mathrm{KL}(p \,\Vert\, q)$ | $\int p \log(p/q)\,\mathrm{d}x \ge 0$, the forward Kullback-Leibler divergence in nats |
| $H(p)$ | $-\int p(x) \log p(x)\,\mathrm{d}x$, the entropy of the target |
| oracle map | the exact flow of the target: $z = \Phi^{-1}(F(x))$ (Gaussian base) or $u = F(x)$ (uniform base) |
| $N$ | the number of training samples (20 000) |
| $K$, $B$ | the number of bins of a spline and its bound: the spline acts on $[-B, B]$ |
| softplus | $\mathrm{softplus}(a) = \log(1 + e^{a}) > 0$ (`torch.nn.functional.softplus`) |

---

## F1 The target and the oracle map (about 1 hour)

**Goal:** know the target exactly, and see that in one dimension the best possible flow is known in closed form.

**From the package:** the parameters and the samples (`CAMEL`, `camel()`); `target.pdf`, `target.cdf`,
`target.quantile` and `oracle_map` only to check your own functions; the diagnostics `qq_points` and `ks_statistic`.
Write the density, the CDF, the oracle map and the inverse map yourself. No network in this task.

**F1.1** Implement the density and the CDF of the mixture,

```math
p(x) = \sum_k \frac{w_k}{\sqrt{2\pi}\,\sigma_k} \exp\!\left[-\frac{(x - \mu_k)^2}{2\sigma_k^2}\right], \qquad
F(x) = \sum_k \frac{w_k}{2} \left[1 + \operatorname{erf}\!\left(\frac{x - \mu_k}{\sqrt{2}\,\sigma_k}\right)\right],
```

and plot the density, the CDF and the histogram of the 20 000 samples.

**F1.2** The oracle map $z = \Phi^{-1}(F(x))$: Gaussianize the samples, compare the histogram of $z$ with
$\mathcal{N}(0, 1)$, make a QQ plot, and draw the map $x \mapsto z$ together with the density. Test $z$ against
$\mathcal{N}(0, 1)$ with the Kolmogorov-Smirnov test. Why is its distance $D$ the same as that of the samples $x$
against $F$?

**F1.3** The uniform base: show that $u = F(x)$ is flat. Then run the map backwards: turn uniform random numbers into
samples of $p$ with $x = F^{-1}(u)$ (inverse-transform sampling) and compare them with the target. This is what
"sampling from a flow" means.

**Deliverables:** the density, the CDF and the histogram (F1.1); the histogram of $z$ with $\mathcal{N}(0, 1)$, the
QQ plot and the map (F1.2); the histogram of $u$, and the samples made from uniform numbers against the target (F1.3).

**Hints**
- Work in float64 (`torch.float64` or NumPy). `torch.erf` and `torch.special.ndtri` (that is $\Phi^{-1}$) work on
  tensors, `math.erf` only on numbers.
- Check your functions against `target.pdf` and `target.cdf` on a grid; the density should integrate to 1
  (`torch.trapezoid`).
- On a wide grid, $1 + \operatorname{erf}(t)$ runs out of digits far in the left tail, and $\Phi^{-1}(F(x))$ becomes
  $-\infty$. The same function written as $\tfrac{1}{2}\operatorname{erfc}(-t)$ (`torch.erfc`) keeps its precision.
- `qq_points(z)` returns the quantiles of $\mathcal{N}(0, 1)$ and of the samples at the same probabilities: plot one
  against the other, with the diagonal. `ks_statistic(z)` returns the distance $D$ and its p-value. Both take any
  reference with a `quantile` or `cdf` method, e.g. `ks_statistic(u, Uniform())` or `ks_statistic(x, target)`.
- A p-value is a random number too: repeat the test with samples of a few other seeds and see how much it varies, even
  for the exact map.
- $F^{-1}$ has no closed form: solve $F(x) = u$ by bisection ($F$ is increasing), or interpolate on a fine grid with
  `np.interp(u, F(grid), grid)`. `target.quantile(u)` is the check.

## F2 Affine and planar flows, by hand (about 2 hours)

**Goal:** write the maximum-likelihood training of a flow yourself, and see what a simple layer can and cannot do.

**From the package:** the samples, `oracle_map`, the diagnostics and `invert_monotone`. Write the transforms, the
log-density, the loss and the training loop yourself (not with `fit`, `Affine` or `Planar`). To use `kl_to_truth` with
your own layers, make them subclasses of `Transform` and put them into a `Flow` (see the hints). In F2.3, the package's
`Planar`, `Flow` and `fit` are the comparison for your own results.

**F2.1** Write the training loop. A flow maps $x$ to $z = f(x)$ with a strictly increasing $f$; with a standard
normal base its density is

```math
\log q(x) = \log \varphi\big(f(x)\big) + \log f'(x), \qquad
\log \varphi(z) = -\tfrac{1}{2} z^2 - \tfrac{1}{2}\log(2\pi).
```

For a stack $f = f_L \circ \dots \circ f_1$, with $z_0 = x$ and $z_l = f_l(z_{l-1})$, the log-derivatives add up:
$\log f'(x) = \sum_l \log f_l'(z_{l-1})$. Training minimises the mean negative log-likelihood of the samples with Adam
(`torch.optim.Adam`),

```math
\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \log q(x_i) \;\to\; H(p) + \mathrm{KL}(p \,\Vert\, q) \quad (N \to \infty).
```

Fit an affine flow, $z = e^{s} x + t$ with $\log f'(x) = s$ (the scale $e^{s}$ is positive for every $s$, so the map is
always invertible): plot the loss against the step and $q$ against $p$, and compute $\mathrm{KL}(p \,\Vert\, q)$. An
affine flow cannot learn the camel: where does its KL floor come from?

**F2.2** Implement the planar transform (Rezende and Mohamed, 2015) in one dimension,

```math
z = x + \hat u \tanh(w x + b), \qquad f'(x) = 1 + \hat u w \left[1 - \tanh^2(w x + b)\right].
```

Since $1 - \tanh^2$ takes every value in $(0, 1]$, the derivative is positive everywhere (the map is invertible, with a
finite $\log f'$) if and only if $\hat u w > -1$. Enforce this as in the paper: train the unconstrained parameters
$u$, $w$, $b$ and use

```math
\hat u = u + \big[m(w u) - w u\big] \frac{w}{w^2}, \qquad m(a) = -1 + \mathrm{softplus}(a),
```

so that $\hat u w = m(w u) > -1$ for every $u$ and $w$. Check your layer before you train it: its log-derivative
against a finite difference, and its inverse (bisection, see the hints) against the forward map. Fit one layer: what
does it do to the base density? Show the map, $q$ against $p$, and the KL.

**F2.3** Stacks of 2, 4, 8 and 16 planar layers: the KL against the depth (with the single layer of F2.2), and the
learned map against the oracle map. Then train stacks of `flows1d.Planar` of the same depths with `fit` and the same
budget, and compare them with yours.

**Deliverables:** the loss curve and $q$ against $p$ for the affine flow (F2.1) and for one planar layer (F2.2); the
KL against the depth, yours and the package's in one figure, and the learned maps against the oracle (F2.3).

**Hints**
- A layer written as a subclass of `Transform` (a `torch.nn.Module`) can be put into a `Flow`, and then `kl_to_truth`
  gives its exact KL divergence. A template with a plain shift, $z = x + t$:
  ```python
  import torch
  from prob_reco_sbi.flows1d import Flow, Transform, camel, kl_to_truth


  class Shift(Transform):
      def __init__(self):
          super().__init__()
          self.t = torch.nn.Parameter(torch.zeros(()))

      def forward(self, x):
          return x + self.t, torch.zeros_like(x)  # (z, log f'(x))


  layer = Shift()
  # ... train the parameters of `layer` with your own loop ...
  print(kl_to_truth(Flow([layer]), camel()))  # in nats
  ```
  You can also compute the KL yourself by the trapezoidal rule (`torch.trapezoid`) on a fine grid that covers the
  target, e.g. from -6 to 15; the two should agree.
- F2.1: what are the mean and the standard deviation of your fitted Gaussian? Compare them with those of the target
  (`target.mean`, `target.std`).
- The reparametrisation divides by $w^2$: keep it finite at $w = 0$, e.g. with `(w * w).clamp_min(1e-12)`.
- The inverse: since $|z - x| < |\hat u|$, the solution of $f(x) = z$ lies in $[z - |\hat u|, z + |\hat u|]$. Halve
  this bracket about 60 times, or use `invert_monotone(f, z, lo, hi)` of the package. A stack is inverted layer by
  layer, in reverse order.
- The initialisation matters: try a few seeds of the initial parameters and report the spread of the KL, not one value.
  For the deepest stacks, look at the largest $|\hat u|$ of the initial layers and at the loss curve: a stack that
  starts far from the data may need more steps.
- F2.3: `fit` trains in float32 (it converts the samples to the type of the flow), while a loop of your own on the
  float64 samples runs in float64. From the same initial parameters the two agree for a few layers, but deep stacks
  can end in different optima: compare the spread over seeds, not single values.
- In one dimension the increasing map that turns $p$ into the base is unique (see
  [docs/flows.md](docs/flows.md#the-oracle-map-the-exact-answer-in-one-dimension)): every good flow approximates the
  same curve $\Phi^{-1}(F(x))$. Draw both maps on a grid, and their difference.

## F3 Radial flows, by hand (about 1.5 hours)

**Goal:** a second classic layer, and what each family is good at.

**From the package:** as in F2: your own transform and your own training loop; in F3.2, the package's `Radial`, `Flow`
and `fit` for the comparison.

**F3.1** Implement the radial transform (Rezende and Mohamed, 2015) in one dimension,

```math
z = x + \hat\beta\, \frac{x - x_0}{\hat\alpha + |x - x_0|}, \qquad
f'(x) = 1 + \frac{\hat\alpha \hat\beta}{(\hat\alpha + |x - x_0|)^2} .
```

For $\hat\beta < 0$ the derivative is smallest at $x = x_0$, where it equals $1 + \hat\beta / \hat\alpha$ (for
$\hat\beta > 0$ it is above 1 everywhere): the derivative is positive everywhere if and only if $\hat\alpha > 0$ and
$\hat\beta > -\hat\alpha$. Train the unconstrained parameters $x_0$, $\alpha$, $\beta$ and use

```math
\hat\alpha = \mathrm{softplus}(\alpha), \qquad \hat\beta = -\hat\alpha + \mathrm{softplus}(\beta).
```

Check the layer as in F2.2 and fit one layer: what does it do to the base density? Show the map, $q$ against $p$, and
the KL.

**F3.2** The KL against the depth (1, 2, 4, 8 and 16 layers) as in F2.3, with stacks of `flows1d.Radial` as the
comparison.

**F3.3** Planar against radial flows with the same number of parameters (3 per layer for both): which family handles
the narrow spike, which the heavy right tail, and why?

**Deliverables:** the map and $q$ against $p$ for one radial layer (F3.1); the KL against the depth for both families
in one figure (F3.2); a figure that shows where each family fails, with a few sentences (F3.3).

**Hints**
- The inverse: $|z - x| < |\hat\beta|$, so the bracket is $[z - |\hat\beta|, z + |\hat\beta|]$.
- F3.3: look at $\log q - \log p$ and at the map residual (learned minus oracle) on a grid, separately around the
  spike ($x$ near $-2.2$) and far in the right tail. Compare the widths of the components ($\sigma_k$ from 0.25 to
  1.4) with the region over which one layer changes the map, and look at what a layer does far away from it.

## F4 Splines, from the package (about 1.5 hours)

**Goal:** the workhorse of modern flows, the monotone rational-quadratic spline (Durkan et al., 2019), and its two
choices: the number of bins and the bound.

**From the package:** everything: `RQSpline`, `Flow`, `fit`, `kl_to_truth`, `oracle_map` and the diagnostics, next to
your planar and radial results of F2 and F3.

A spline with $K$ bins on $[-B, B]$ has $3K - 1$ parameters (the widths and the heights of the bins and the
derivatives at the inner knots); outside $[-B, B]$ it is the identity. Its formulas are in
[docs/flows.md](docs/flows.md#rational-quadratic-spline-durkan-et-al-2019).

```python
from prob_reco_sbi.flows1d import Flow, RQSpline, camel, fit, kl_to_truth

target = camel()
x = target.sample(20_000, seed=11)
flow = Flow([RQSpline(bins=16, bound=10.0)])  # standard normal base
history = fit(flow, x, steps=2000, lr=1e-2)  # maximum likelihood with Adam, full batch
print(flow.n_params, kl_to_truth(flow, target))  # number of parameters; KL in nats
```

**F4.1** Fit splines with 4, 8, 16 and 32 bins: the KL against the number of bins and against the number of
parameters, the latter in one figure with the planar and radial stacks of F2.3 and F3.2.

**F4.2** The learned map against the oracle: where is the residual (in the tails, between the modes, at the spike)?

**F4.3** The bound $B$: fit 16-bin splines with a few bounds (e.g. 2, 3, 5 and 10) and compare samples drawn from the
flow with the target, in particular beyond the bound. What does a too small $B$ do, and why?

**Deliverables:** the KL against the number of parameters for the three families (F4.1); the map residual, with the
density for reference (F4.2); histograms of flow samples against the target for several bounds, and the KL against $B$
(F4.3).

**Hints**
- `flow.n_params` counts the parameters of any flow, and `history.seconds` is the wall time of the fit. `flow(x)`
  returns $z$ and $\log f'(x)$ (call `.detach()` before plotting); `oracle_map(target)(x)` returns the exact pair.
- `flow.transforms[0].knots()` returns the knots $x_k$, $y_k$ and the derivatives $d_k$ of the spline: draw the knots
  on the map (they carry the gradient graph of the parameters: call `.detach()` before plotting).
- F4.3: what is $f(x)$ for $|x| > B$, and therefore $q(x)$ there? `flow.sample(20_000, seed=...)` draws samples from
  the flow; histograms on a log axis show the tails. Optional: is a very large bound (e.g. 50 with 16 bins) better?

## F5 The uniform base (about 1 hour)

**Goal:** the base is a choice too: the same flow with a uniform base instead of a Gaussian one.

**From the package:** everything, in particular `Uniform`, `Sigmoid` and `Flow`.

With the base $\mathcal{U}(0, 1)$, $\log p_U(u) = 0$ for $0 \le u \le 1$ and $-\infty$ outside. The package's flow for
this base ends with a logistic layer,

```math
u = \sigma(y) = \frac{1}{1 + e^{-y}}, \qquad
\log \frac{\mathrm{d}u}{\mathrm{d}y} = \log \sigma(y) + \log\big(1 - \sigma(y)\big),
```

```python
from prob_reco_sbi.flows1d import Flow, RQSpline, Sigmoid, Uniform

flow = Flow([RQSpline(bins=16), Sigmoid()], Uniform())
```

**F5.1** Fit this flow to the same samples with the same budget as the Gaussian-base spline of F4: the histogram of
$u = f(x)$, the learned map against the oracle $u = F(x)$, and the KL.

**F5.2** The boundaries: what happens near $u = 0$ and $u = 1$? Try the flow without the `Sigmoid` and explain what
you see: why does the flow need a final squashing layer?

**F5.3** Gaussian against uniform base, with the same spline: the KL, the training curves, which one is easier to
train, and why.

**Deliverables:** the histogram of $u$ and the map (F5.1); a figure of the region near the boundaries (F5.2); the
training curves of both bases in one figure, and their KL (F5.3).

**Hints**
- Look at the histogram of $u$ with fine bins near 0 and 1, and at the smallest and the largest samples drawn from the
  flow. When sampling, $u$ is clipped to $[\epsilon, 1 - \epsilon]$, with $\epsilon$ the machine precision of the
  flow's floating-point type (`torch.finfo(torch.float32).eps` by default).
- Without the `Sigmoid`, `fit` stops with an error: read its message, and evaluate `flow.log_prob(x)` of the untrained
  flow.
- Training curves: `history.loss` is the loss of every update; a reference line is the loss of the exact density on
  the same samples, `-target.log_pdf(x).mean()`. For the exact KL during training, take snapshots
  (`fit(..., every=50)`), load them one by one with `history.restore(flow, i)` and call `kl_to_truth`.
- F5.3: the spline is followed by the `Sigmoid`, so the spline itself maps $x$ to $y = \log[u / (1 - u)]$. If $u$ is
  uniform, what is the distribution of $y$, and how do its tails compare with those of $\mathcal{N}(0, 1)$? What does
  the spline then have to learn, and what does it do outside $[-B, B]$?

## F6 Animations (about 1 hour)

**Goal:** watch a flow learn, and watch it carry the samples to the base.

**From the package:** everything: `fit` records the frames of the training, `Flow.transform_points` the transport.

**F6.1** The training animation of one flow of your choice: in every frame, $q$ against $p$, the learned map against
the oracle, the histogram of $z = f(x)$ against $\mathcal{N}(0, 1)$, and the loss up to that step; a frame every 10
steps.

**F6.2** The particle picture: samples of $p$ moving layer by layer to the base, for a planar stack and for a spline.

**Deliverables:** the two animations (GIF or MP4), and one still frame of each as a figure.

**Hints**
- `fit(flow, x, every=10, grid=grid)` keeps a snapshot every 10 steps: `history.step` holds the step of every
  snapshot, `history.q_grid` ($q$ on the grid) and `history.z_grid` ($f$ on the grid) one row per snapshot, and
  `history.loss` the loss of every step. For the histogram of $z$ in frame `i`, load the snapshot with
  `history.restore(flow, i)` and evaluate `flow(x)[0]`; load the last one again (`history.restore(flow)`) when you are
  done.
- `flow.transform_points(x[:500])` returns the positions after every layer, of shape $[L + 1, 500]$: row 0 is $x$,
  the last row $z$. A single spline is one layer and gives only two rows: interpolate between consecutive rows for a
  smooth motion. Draw every sample as a dot at its position, with a random vertical offset so that the dots do not
  overlap, or as a line from layer to layer.
- `matplotlib.animation.FuncAnimation` saved with `writer="pillow"` gives a GIF, with one palette per frame, so the
  file grows quickly with the number of frames and the dpi: keep both modest, or save an MP4 with `writer="ffmpeg"` if
  ffmpeg is installed.

## B Bonus (2 to 5 hours)

**From the package:** everything.

**B1** Your own distribution: define a mixture of your choice and find out which flow needs how many parameters to
learn it.

```python
from prob_reco_sbi.flows1d import Mixture1D

mine = Mixture1D(weights=[0.5, 0.5], means=[-1.0, 1.0], stds=[0.3, 0.3])
```

**B2** A conditional flow, the teaser for NPE. Let the weight of the spike be a parameter $\theta$, with the other two
components keeping their ratio 5 : 2,

```math
p(x \mid \theta) = \theta\,\mathcal{N}(x; -2.2, 0.25^2)
  + (1 - \theta)\left[\tfrac{5}{7}\,\mathcal{N}(x; 0.5, 0.9^2) + \tfrac{2}{7}\,\mathcal{N}(x; 3.5, 1.4^2)\right]
```

(the camel is $\theta = 0.3$). Draw pairs: $\theta_i$ uniform in $[0.1, 0.5]$ and one $x_i \sim p(x \mid \theta_i)$
for each. Train a conditional flow $q(x \mid \theta)$ on the pairs by maximum likelihood,
$\mathcal{L} = -\frac{1}{N}\sum_i \log q(x_i \mid \theta_i)$, and compare $q(x \mid \theta)$ with $p(x \mid \theta)$
for a few values of $\theta$, inside and outside the training range. NPE works the same way, with the roles
swapped: its flow models the event parameters given the detector response, $q(\theta \mid x)$, and is trained on
simulated pairs. And a conditional flow in the direction of this task, $q(x \mid \theta)$ with $x$ one-dimensional,
is what a real analysis uses as a learned likelihood: see "Where this is used" in [docs/flows.md](docs/flows.md).

**Deliverables:** for B1, the KL against the number of parameters for a few flows on your mixture; for B2,
$q(x \mid \theta)$ against $p(x \mid \theta)$ for a few $\theta$, and the KL against $\theta$.

**Hints**
- B1: e.g. two narrow spikes far apart, a strongly skewed mixture, or a thin spike on top of a broad hump. Every method
  of the camel works for your mixture (`pdf`, `cdf`, `sample`, `entropy`), and so does `kl_to_truth(flow, mine)`.
- B2: "conditional" means that the parameters of the layers are functions of $\theta$. A small network (e.g. a
  `torch.nn.Sequential` with two hidden layers of 64 units) takes $\theta$ and returns the $3L$ parameters of $L$
  planar layers, one set per sample: write your planar layer of F2.2 as a function of $x$, $u$, $w$ and $b$, with
  tensors $u$, $w$, $b$ of the same shape as $x$. A new network returns values close to 0, and a $w$ close to 0 gives
  a huge $\hat u$ (see the F2 hints): add a constant to the $w$ output, e.g. $w = 1 + \text{output}$, so that the
  layers start as gentle bends.
- B2: draw the pairs without a loop: a component index for every sample from `torch.rand` compared with the weights,
  then $x = \mu_k + \sigma_k \varepsilon$ with $\varepsilon \sim \mathcal{N}(0, 1)$. With 50 000 pairs and 8 planar
  layers, 2000 full-batch steps take a minute or two on a CPU.
- B2: for a fixed $\theta$, compute the KL by the trapezoidal rule on a grid (as in F2.1) against the
  `Mixture1D` of that $\theta$. Recall T1.2: what can a model trained on $\theta \in [0.1, 0.5]$ say about
  $\theta = 0.7$?

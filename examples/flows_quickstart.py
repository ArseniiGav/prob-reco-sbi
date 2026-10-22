"""Quick start of the one-dimensional normalizing flows: fit a spline flow to the camel target and check it.

Run from the repository root; it takes about ten seconds on a CPU::

    python examples/flows_quickstart.py
    python examples/flows_quickstart.py --bins 8 --steps 1000 --samples 5000

It trains a rational-quadratic spline flow by maximum likelihood on samples of the target (the forward KL divergence),
prints the exact KL divergence to the target, the Kolmogorov-Smirnov test of the mapped samples and the moments of
samples drawn from the flow, and, if matplotlib is installed, saves ``data/flows_quickstart.png``: the model density
next to the target and the learned map next to the exact (oracle) map. The maths is in ``docs/flows.md``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from prob_reco_sbi.flows1d import Flow, History, RQSpline, camel, fit, kl_to_truth, ks_statistic, oracle_map

#: The figure style of the course (the owner's paper style): white, no grid, ticks in on all four sides, minor ticks.
PAPER_STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": ".15",
    "axes.labelcolor": ".15",
    "axes.linewidth": 1.25,
    "axes.grid": False,
    "axes.axisbelow": True,
    "axes.spines.left": True,
    "axes.spines.bottom": True,
    "axes.spines.right": True,
    "axes.spines.top": True,
    "text.color": ".15",
    "xtick.color": ".15",
    "ytick.color": ".15",
    "xtick.major.width": 1.25,
    "ytick.major.width": 1.25,
    "xtick.minor.width": 1.0,
    "ytick.minor.width": 1.0,
    "xtick.major.size": 6.0,
    "ytick.major.size": 6.0,
    "xtick.minor.size": 4.0,
    "ytick.minor.size": 4.0,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans", "Bitstream Vera Sans", "sans-serif"],
    "lines.linewidth": 1.5,
    "lines.markersize": 6.0,
    "patch.linewidth": 1.0,
    "patch.edgecolor": "w",
    "patch.force_edgecolor": True,
    "lines.solid_capstyle": "round",
    "legend.frameon": True,
    "image.cmap": "viridis",
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.direction": "in",
    "xtick.top": True,
    "xtick.bottom": True,
    "xtick.labelsize": 12,
    "ytick.direction": "in",
    "ytick.right": True,
    "ytick.left": True,
    "ytick.labelsize": 12,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "pdf.fonttype": 42,
}
#: One colour per meaning, as in every figure of the course: the target, the spline flow, maps.
COLOURS = {"target": "royalblue", "spline": "indigo", "map": "black"}


def parse_args() -> argparse.Namespace:
    """Read the command-line options."""
    parser = argparse.ArgumentParser(description="Fit a spline flow to the camel target and compare it with the truth.")
    parser.add_argument("--samples", type=int, default=20_000, help="number of training samples (default: 20000)")
    parser.add_argument("--bins", type=int, default=16, help="number of spline bins (default: 16)")
    parser.add_argument("--steps", type=int, default=2000, help="number of Adam steps (default: 2000)")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate (default: 0.01)")
    parser.add_argument(
        "--seed", type=int, default=11, help="seed of the training samples (default: 11, as in the tasks)"
    )
    parser.add_argument("--out", type=Path, default=Path("data"), help="output directory (default: data)")
    return parser.parse_args()


def plot_fit(flow: Flow, history: History, path: Path, kl: float) -> None:
    """Save a figure of the fit: model density vs target, learned map vs oracle map, training loss vs entropy."""
    import matplotlib.pyplot as plt

    target = camel()
    x = torch.linspace(-5.0, 9.0, 701, dtype=torch.float64)
    with torch.no_grad():
        q = torch.exp(flow.log_prob(x)).double().numpy()
        z = flow(x)[0].double().numpy()
    z_exact = oracle_map(target)(x)[0].numpy()
    p, x = target.pdf(x).numpy(), x.numpy()

    with plt.rc_context(PAPER_STYLE):
        fig, (left, middle, right) = plt.subplots(1, 3, figsize=(10.0, 3.4))
        left.fill_between(x, p, color=COLOURS["target"], alpha=0.25, lw=0)
        left.plot(x, p, color=COLOURS["target"], label="target $p(x)$")
        left.plot(x, q, color=COLOURS["spline"], label="flow $q(x)$")
        left.set(xlabel="$x$", ylabel="Density", xlim=(-5, 9), ylim=(0, 0.6), title=f"Density (KL {kl:.4f} nats)")
        left.legend(loc="upper right")
        middle.plot(x, z, color=COLOURS["map"], label="learned map $f(x)$")
        middle.plot(x, z_exact, color=COLOURS["map"], ls="--", lw=1.25, label=r"oracle map $\Phi^{-1}(F(x))$")
        middle.set(xlabel="$x$", ylabel="$z$", xlim=(-5, 9), ylim=(-4.5, 4.5), title="Map to the Gaussian base")
        middle.legend(loc="upper left")
        right.plot(history.step, history.train_loss, color=COLOURS["spline"], label="training loss")
        right.axhline(target.entropy(), color="black", ls="--", lw=1.0, label="entropy $H(p)$")
        right.set(xlabel="Adam step", ylabel="Loss [nats]", ylim=(target.entropy() - 0.1, None), title="Training")
        right.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    """Fit the flow, print how close it is to the truth and save a figure."""
    args = parse_args()
    target = camel()
    print(f"target: {target}")
    print(f"  mean {target.mean:.3f}, standard deviation {target.std:.3f}, entropy H(p) = {target.entropy():.4f} nats")

    # Training samples: the flow learns from these only, never from the density of the target.
    x = target.sample(args.samples, seed=args.seed)

    # A spline flow with a standard normal base, trained by maximum likelihood (Adam, full batch).
    flow = Flow([RQSpline(bins=args.bins, bound=10.0)])
    history = fit(flow, x, steps=args.steps, lr=args.lr, every=10)
    loss = history.train_loss[-1]
    print(f"\n{args.bins}-bin spline, {flow.n_params} parameters: {args.steps} steps in {history.seconds:.1f} s")
    # The loss of the exact density on the same samples is the reference: their difference estimates the KL divergence
    # (H(p) is not, since the mean over N samples fluctuates by about std(log p) / sqrt(N) around its expectation).
    # On the training samples the estimate is biased low, because the flow was fitted to them.
    reference = float(-target.log_pdf(x).mean())
    print(f"  final training loss {loss:.4f} nats; exact density on the same samples {reference:.4f} nats")
    print(f"  difference {loss - reference:+.4f} nats (a sample estimate of the KL divergence, biased low)")

    # The truth metric of the course: the exact KL divergence by quadrature.
    kl = kl_to_truth(flow, target)
    print(f"  KL(p || q) = {kl:.4f} nats (exact, by quadrature)")

    # If the flow is right, z = f(x) of new samples of the target follows the base N(0, 1).
    x_new = target.sample(args.samples, seed=args.seed + 1000)
    with torch.no_grad():
        z = flow(x_new)[0]
    ks = ks_statistic(z)
    print(f"  Kolmogorov-Smirnov test of z = f(x) of new samples: D = {ks.statistic:.4f}, p = {ks.pvalue:.2f}")

    # Sampling runs the flow backwards: z from the base, x = f^-1(z).
    samples = flow.sample(args.samples, seed=2).double()
    print(f"  {args.samples} samples of the flow: mean {samples.mean():.3f}, standard deviation {samples.std():.3f}")

    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print("matplotlib is not installed: no figure (pip install -e '.[analysis]')")
        return
    args.out.mkdir(parents=True, exist_ok=True)
    figure = args.out / "flows_quickstart.png"
    plot_fit(flow, history, figure, kl)
    print(f"figure: {figure}")


if __name__ == "__main__":
    main()

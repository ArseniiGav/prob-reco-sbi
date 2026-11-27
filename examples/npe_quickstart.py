"""Quick start of neural posterior estimation: train the chain on simulated pairs and draw the posterior of one event.

Run from the repository root; it takes well under a minute on a CPU (20 000 simulations, 20 epochs)::

    python examples/npe_quickstart.py
    python examples/npe_quickstart.py --data data/n3_full_train.h5 --n-train 100000 --epochs 20

It simulates ``--n-train`` events of the full detector (or reads them from ``--data``), trains the chain of
conditional spline flows ``q^(x0, y0, E | x)`` on the charges and times of the 32 sensors, simulates a new event and
prints its posterior next to the truth and the charge barycenter. If matplotlib is installed it saves
``data/npe_quickstart.png``: the posterior samples on the event display, the energy posterior and the loss curves.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from prob_reco_sbi.flash import SimResult, generate, preset, sensor_positions, simulate
from prob_reco_sbi.npe import AutoregressiveFlow, History, fit, load_pairs, to_pairs

#: One color per meaning: the posterior samples, the truth, the naive estimate, the training curves.
COLORS = {"samples": "indigo", "truth": "black", "barycenter": "darkorange", "train": "indigo", "val": "royalblue"}


def parse_args() -> argparse.Namespace:
    """Read the command-line options."""
    parser = argparse.ArgumentParser(description="Train the NPE chain and draw the posterior of one event.")
    parser.add_argument("--data", type=Path, default=None, help="training file (default: simulate the events)")
    parser.add_argument("--n-train", type=int, default=20_000, help="training pairs (default: 20000)")
    parser.add_argument("--epochs", type=int, default=20, help="passes over the training pairs (default: 20)")
    parser.add_argument("--seed", type=int, default=1, help="seed of the simulation and of the network (default: 1)")
    parser.add_argument("--vertex", type=float, nargs=2, default=(0.5, 0.3), metavar=("X0", "Y0"), help="test vertex")
    parser.add_argument("--energy", type=float, default=2.0, help="test energy [MeV] (default: 2)")
    parser.add_argument("--out", type=Path, default=Path("data"), help="output directory (default: data)")
    return parser.parse_args()


def barycenter(event: SimResult) -> np.ndarray:
    """Return the charge-weighted mean of the sensor positions, the naive vertex estimate of stage 1."""
    sx, sy = sensor_positions(event.config.geometry)
    charge = event.response.charge[0].astype(np.float64)
    return np.array([charge @ sx, charge @ sy]) / charge.sum()


def plot(event: SimResult, samples: np.ndarray, history: History, path: Path) -> None:
    """Save the event display with the posterior samples, the energy posterior and the loss curves."""
    import matplotlib.pyplot as plt

    geometry = event.config.geometry
    sx, sy = sensor_positions(geometry)
    truth = np.array([event.truth.x[0], event.truth.y[0], event.truth.energy[0]])
    bary = barycenter(event)
    fig, (display, energy, loss) = plt.subplots(1, 3, figsize=(13.0, 4.2))

    phi = np.linspace(0, 2 * np.pi, 400)
    display.plot(geometry.radius * np.cos(phi), geometry.radius * np.sin(phi), color="0.6", lw=1.0)
    hits = display.scatter(sx, sy, c=event.response.charge[0], cmap="viridis", s=60, marker="s", zorder=3)
    fig.colorbar(hits, ax=display, label="Sensor charge $n_j$", fraction=0.046, pad=0.04)
    display.scatter(samples[:, 0], samples[:, 1], s=2, alpha=0.3, color=COLORS["samples"], label=r"samples of $\hat q$")
    display.plot(*truth[:2], "*", ms=14, color=COLORS["truth"], label="true vertex")
    display.plot(*bary, "X", ms=10, color=COLORS["barycenter"], label="barycenter")
    display.set(xlabel="$x_0$ [m]", ylabel="$y_0$ [m]", aspect="equal", xlim=(-1.15, 1.15), ylim=(-1.15, 1.15))
    display.legend(loc="lower left", fontsize=8)

    energy.hist(samples[:, 2], bins=50, density=True, color=COLORS["samples"], alpha=0.6, label=r"samples of $\hat q$")
    energy.axvline(truth[2], color=COLORS["truth"], ls="--", label="true energy")
    energy.set(xlabel="$E$ [MeV]", ylabel="Posterior density [1/MeV]")
    energy.legend(loc="upper right", fontsize=8)

    loss.plot(history.epoch[1:], history.train_loss[1:], color=COLORS["train"], label="training")
    if history.val_loss is not None:
        loss.plot(history.epoch[1:], history.val_loss[1:], color=COLORS["val"], ls="--", label="validation")
    loss.set(xlabel="Epoch", ylabel="Loss [nats]")
    loss.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Train the chain, sample the posterior of one new event, print it and save a figure."""
    args = parse_args()
    torch.manual_seed(args.seed)
    n_val = max(args.n_train // 10, 1)
    config = preset("full")

    # Pairs {theta_i, x_i}: theta from the prior, x from the simulator (charges and relative times, 64 values).
    start = time.perf_counter()
    if args.data is not None:
        theta, x = load_pairs(args.data, "charge+time")
        theta, x = theta[: args.n_train + n_val], x[: args.n_train + n_val]
    else:
        theta, x = to_pairs(generate(args.n_train + n_val, config=config, seed=args.seed), "charge+time")
    print(
        f"{len(theta)} pairs (theta: {tuple(theta.shape)}, x: {tuple(x.shape)}) in {time.perf_counter() - start:.0f} s"
    )

    # The chain q^(x0 | h(x)) q^(y0 | x0, h(x)) q^(E | x0, y0, h(x)), trained by maximum likelihood.
    flow = AutoregressiveFlow(obs_dim=x.shape[1])
    history = fit(flow, theta, x, epochs=args.epochs, val_fraction=n_val / len(theta), seed=args.seed)
    print(
        f"chain with {flow.n_params} parameters: {args.epochs} epochs in {history.seconds:.0f} s, "
        f"validation loss {history.best_val_loss:.3f} nats (epoch {history.best_epoch})"
    )

    # A new event; its posterior needs no new training (amortized inference).
    event = simulate(*args.vertex, args.energy, config=config, seed=args.seed + 1000)
    samples = flow.sample(to_pairs(event, "charge+time")[1][0], 2000, seed=args.seed).numpy()
    truth = (event.truth.x[0], event.truth.y[0], event.truth.energy[0])
    bary = barycenter(event)
    print(f"\nnew event: vertex ({truth[0]:.3f}, {truth[1]:.3f}) m, E = {truth[2]:.2f} MeV")
    for i, (name, unit) in enumerate([("x0", "m"), ("y0", "m"), ("E", "MeV")]):
        print(f"  {name}: posterior mean {samples[:, i].mean():7.3f} +- {samples[:, i].std():.3f} {unit}")
    print(f"  barycenter ({bary[0]:.3f}, {bary[1]:.3f}) m")

    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print("matplotlib is not installed: no figure (pip install -e '.[analysis]')")
        return
    args.out.mkdir(parents=True, exist_ok=True)
    figure = args.out / "npe_quickstart.png"
    plot(event, samples, history, figure)
    print(f"figure: {figure}")


if __name__ == "__main__":
    main()

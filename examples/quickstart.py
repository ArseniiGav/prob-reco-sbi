"""Quick start: simulate events of the toy detector, save them, read them back and look at what was stored.

Run from the repository root; it takes seconds on a CPU::

    python examples/quickstart.py
    python examples/quickstart.py --events 5000 --preset straight --device cpu

It writes ``data/quickstart_<preset>.h5`` and, if matplotlib is installed, ``data/quickstart_<preset>.png`` with the
network input of the first event.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from prob_reco_sbi.flash import PRESETS, SimResult, generate, load, network_input, preset, save


def parse_args() -> argparse.Namespace:
    """Read the command-line options."""
    parser = argparse.ArgumentParser(description="Simulate, save and read back a small data set of the toy detector.")
    parser.add_argument("--events", type=int, default=2000, help="number of events (default: 2000)")
    parser.add_argument("--preset", choices=list(PRESETS), default="full", help="physics preset (default: full)")
    parser.add_argument("--seed", type=int, default=1, help="random seed (default: 1)")
    parser.add_argument("--device", default="auto", help="'auto' (GPU if available), 'cpu' or 'cuda' (default: auto)")
    parser.add_argument("--out", type=Path, default=Path("data"), help="output directory (default: data)")
    return parser.parse_args()


def summarise(data: SimResult) -> None:
    """Print the shapes of the stored arrays, the photon bookkeeping and the first event."""
    truth, response, photons = data.truth, data.response, data.photons
    print(f"truth:     x, y, energy, t0 of shape {truth.x.shape}")
    print(f"response:  charge, time_mean, time_first of shape {response.charge.shape}")

    emitted = photons.n_emitted.sum()
    fates = {
        "detected": photons.n_detected,
        "absorbed in the bulk": photons.n_absorbed_bulk,
        "absorbed by the wall": photons.n_absorbed_wall,
        "absorbed by a sensor": photons.n_absorbed_sensor,
        "lost": photons.n_lost,
    }
    print(f"\nfate of the {emitted:.3g} emitted photons:")
    for name, count in fates.items():
        print(f"  {name:<22s} {count.sum() / emitted:7.2%}")
    ended = sum(fates.values())
    print(f"  every photon accounted for in every event: {bool(np.all(ended == photons.n_emitted))}")

    charge = response.charge[0]
    r0 = np.hypot(truth.x[0], truth.y[0])
    print(
        f"\nfirst event: vertex ({truth.x[0]:.3f}, {truth.y[0]:.3f}) m, r0 = {r0:.3f} m, E = {truth.energy[0]:.2f} MeV,"
        f" t0 = {truth.t0[0]:.2f} ns"
    )
    print(f"  total charge Q = {charge.sum()} hits, {np.count_nonzero(charge)} of {charge.size} sensors hit")


def plot_first_event(data: SimResult, x: np.ndarray, path: Path, label: str) -> None:
    """Save a figure of the network input of the first event: log charges and relative times per sensor."""
    import matplotlib.pyplot as plt

    m = data.n_sensors
    sensor = np.arange(m)
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(6.4, 4.4), sharex=True, constrained_layout=True)
    top.bar(sensor, x[0, :m], color="#2a78d6")
    top.set_ylabel("Log charge, ln(1 + n)")
    bottom.bar(sensor, x[0, m:], color="#eb6834")
    bottom.axhline(0.0, color="#52514e", linewidth=0.8)
    bottom.set_ylabel("Relative time [ns]")
    bottom.set_xlabel("Sensor index j")
    top.set_title(f"Network input of the first event (preset {label}, {len(data)} events simulated)")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    """Simulate a small data set, save it, read it back, print a summary and plot the first event."""
    args = parse_args()
    config = preset(args.preset)
    print(f"preset {args.preset}:\n{config.summary()}\n")

    # Simulate: the event parameters are drawn from the prior and every photon is tracked.
    data = generate(args.events, config=config, seed=args.seed, device=args.device)
    provenance = data.provenance
    print(f"{data} simulated on {provenance['device']} in {provenance['wall_time_s']:.2f} s")

    # Save and read back: the file also holds the configuration and how the data were made.
    path = save(data, args.out / f"quickstart_{args.preset}.h5")
    data = load(path)
    print(f"saved to {path} ({path.stat().st_size / 1e6:.2f} MB) and read back: {data}\n")
    summarise(data)

    # The fixed input of the neural networks of the next stage: log charges and relative times.
    x = network_input(data.response.charge, data.response.time_mean)
    print(f"\nnetwork input of shape {x.shape}")

    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print("matplotlib is not installed: no figure (pip install -e '.[analysis]')")
        return
    figure = path.with_suffix(".png")
    plot_first_event(data, x, figure, args.preset)
    print(f"figure: {figure}")


if __name__ == "__main__":
    main()

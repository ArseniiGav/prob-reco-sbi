"""T0.2 How long does the simulation take, and how does the time scale with N and with the preset.

Run from the repository root::

    python solutions/stage1_flash/task0_timing.py            # CPU timings (16 torch threads), CLI run and figure
    python solutions/stage1_flash/task0_timing.py --gpu      # optional short GPU timing (CUDA torch build and a GPU)
    python solutions/stage1_flash/task0_timing.py --threads  # CPU thread scan: 10 000 full-model events, 1-16 threads
    python solutions/stage1_flash/task0_timing.py --plot     # redraw the figure from results/T0.json

CPU: ``generate`` with the default prior for every preset and N = 1k ... 50k events, three repeats (median; the repeat
loop is outermost so that a change of the machine load affects every point alike). The time is fitted with
``t(N) = t_fixed + N / rate``. To explain the differences between the presets, the mean number of transport steps per
photon (one step = flight to the next interaction) is counted on fully recorded photon paths. Finally the student
command ``flashsim generate -n 10000`` is timed end to end (Python start-up, import, simulation, HDF5 writing).

GPU (optional, ``--gpu``): one timing per preset for N = 1k, 10k, 50k (plus 100k for the full model), under a minute of
GPU time. Thread scan (``--threads``): 10 000 events of the full model with 1, 2, 4, 8 and 16 torch threads, two
repeats. Results go to ``results/T0.json`` under ``T0.2`` (CPU), ``T0.2_gpu`` and ``T0.2_threads``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np
import torch
from common import DATA, PRESET_COLOURS, REPO, RES, C, figure_legend, save_results, savefig, style

from prob_reco_sbi.flash import Interaction, generate, load, preset

PRESETS = ["straight", "absorb", "absorb-scatter", "full"]
COLOURS = PRESET_COLOURS  # the same colour per preset as in the T3 and T4.4 figures
N_CPU = [1000, 2000, 5000, 10_000, 20_000, 50_000]
N_GPU = [1000, 10_000, 50_000]
REPEATS = 3
THREADS = 16
THREAD_SCAN = [1, 2, 4, 8, 16]
CLI_FILE = DATA / "t0_cli_full_n10000_seed1.h5"


def timed_generate(n: int, name: str, seed: int, device: str) -> tuple[float, int]:
    """Wall time of ``generate`` (prior sampling + transport + readout) and the number of emitted photons."""
    config = preset(name)
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    result = generate(n, config=config, seed=seed, device=device)
    if device == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - start, int(result.photons.n_emitted.sum())


def fit_linear(n: np.ndarray, t: np.ndarray) -> tuple[float, float]:
    """Fit ``t = t_fixed + n / rate`` with relative weights (least squares of (t_i - model) / t_i)."""
    design = np.stack([1 / t, n / t], axis=1)
    (t_fixed, per_event), *_ = np.linalg.lstsq(design, np.ones_like(t), rcond=None)
    return float(t_fixed), float(1 / per_event)


def steps_per_photon(name: str, n_events: int = 300, seed: int = 77) -> dict[str, float]:
    """Mean numbers of transport steps, scatterings and reflections per photon, from fully recorded paths."""
    result = generate(n_events, config=preset(name), seed=seed, device="cpu", save_tracks=n_events, track_photons=None)
    tracks = result.tracks
    assert tracks is not None
    points = np.diff(tracks.offsets)
    return {
        "photons": len(tracks),
        "steps": float(np.mean(points - 1)),
        "scatterings": float(np.sum(tracks.kind == Interaction.SCATTER) / len(tracks)),
        "reflections": float(np.sum(tracks.kind == Interaction.REFLECT) / len(tracks)),
    }


def time_cli(repeats: int = REPEATS) -> dict[str, Any]:
    """End-to-end time of the student command ``flashsim generate -n 10000`` and of the import alone."""
    exe = shutil.which("flashsim")
    command = [exe] if exe else [sys.executable, "-m", "prob_reco_sbi.flash"]
    output = str(CLI_FILE.relative_to(REPO))
    args = ["generate", "-n", "10000", "--seed", "1", "--threads", str(THREADS), "-o", output, "--overwrite"]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    walls, sims = [], []
    for _ in range(repeats):
        start = time.perf_counter()
        subprocess.run([*command, *args, "-q"], check=True, env=env, cwd=REPO)
        walls.append(time.perf_counter() - start)
        sims.append(load(CLI_FILE, hits=False, tracks=False).provenance["wall_time_s"])
    imports = []
    for _ in range(repeats):
        start = time.perf_counter()
        subprocess.run([sys.executable, "-c", "import prob_reco_sbi.flash"], check=True, env=env)
        imports.append(time.perf_counter() - start)
    return {
        "command": "flashsim " + " ".join(args),
        "wall_s": [round(v, 2) for v in walls],
        "wall_median_s": round(float(np.median(walls)), 2),
        "simulation_median_s": round(float(np.median(sims)), 2),
        "import_median_s": round(float(np.median(imports)), 2),
        "file": CLI_FILE.name,
        "file_size_MB": round(CLI_FILE.stat().st_size / 1e6, 2),
    }


def run_cpu() -> dict[str, Any]:
    """CPU timings of every preset and N, the transport steps per photon and the command-line run."""
    torch.set_num_threads(THREADS)
    load_before = os.getloadavg()
    for name in PRESETS:  # warm-up: thread pool, allocator
        timed_generate(2000, name, seed=999, device="cpu")
    times: dict[str, dict[int, list[float]]] = {name: {n: [] for n in N_CPU} for name in PRESETS}
    photons: dict[str, dict[int, list[int]]] = {name: {n: [] for n in N_CPU} for name in PRESETS}
    for rep in range(REPEATS):
        for name in PRESETS:
            for n in N_CPU:
                seconds, emitted = timed_generate(n, name, seed=rep, device="cpu")
                times[name][n].append(seconds)
                photons[name][n].append(emitted)
                print(f"  cpu repeat {rep}: {name:<15s} N = {n:>6d}: {seconds:6.2f} s")
    load_after = os.getloadavg()

    out: dict[str, Any] = {
        "device": "cpu",
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "torch_threads": THREADS,
        "torch_version": torch.__version__,
        "repeats": REPEATS,
        "n_events": N_CPU,
        "load_average_1min_before_after": [round(load_before[0], 2), round(load_after[0], 2)],
        "presets": {},
    }
    for name in PRESETS:
        t = np.array([np.median(times[name][n]) for n in N_CPU])
        t_fixed, rate = fit_linear(np.array(N_CPU, float), t)
        big = N_CPU.index(50_000)
        slope = np.polyfit(np.log(N_CPU[2:]), np.log(t[2:]), 1)[0]  # N >= 5000
        n_photons = np.median(photons[name][50_000])
        out["presets"][name] = {
            "median_s": t.round(4).tolist(),
            "min_s": [round(min(times[name][n]), 4) for n in N_CPU],
            "max_s": [round(max(times[name][n]), 4) for n in N_CPU],
            "fit_fixed_s": round(t_fixed, 4),
            "fit_events_per_s": round(rate),
            "loglog_slope_N_ge_5000": round(float(slope), 3),
            "events_per_s_at_50k": round(50_000 / t[big]),
            "photons_per_s_at_50k": float(f"{n_photons / t[big]:.3g}"),
            "ns_per_photon_at_50k": round(1e9 * t[big] / n_photons, 2),
            "mean_photons_per_event": round(n_photons / 50_000, 1),
            "time_10k_s": round(float(t[N_CPU.index(10_000)]), 3),
        }
    print("  counting transport steps per photon ...")
    for name in PRESETS:
        out["presets"][name]["transport"] = steps_per_photon(name)
    rows = out["presets"]
    full = rows["full"]
    out["relative_time_per_photon_vs_straight"] = {
        name: round(rows[name]["ns_per_photon_at_50k"] / rows["straight"]["ns_per_photon_at_50k"], 2)
        for name in PRESETS
    }
    steps = np.array([rows[name]["transport"]["steps"] for name in PRESETS])
    per_photon = np.array([rows[name]["ns_per_photon_at_50k"] for name in PRESETS])
    out["ns_per_transport_step_fit"] = round(float(steps @ per_photon / (steps @ steps)), 1)  # fit through the origin
    out["ns_per_transport_step_by_preset"] = {
        name: round(float(t / s), 1) for name, s, t in zip(PRESETS, steps, per_photon, strict=True)
    }
    print("  timing the command line ...")
    out["cli_10k_full"] = time_cli()
    out["full_10k_from_fit_s"] = round(full["fit_fixed_s"] + 10_000 / full["fit_events_per_s"], 2)
    return out


def run_gpu() -> dict[str, Any]:
    """One short GPU timing per preset and N (under a minute of GPU time)."""
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device visible: run with CUDA_VISIBLE_DEVICES=0 and a CUDA build of torch")
    timed_generate(2000, "full", seed=999, device="cuda")  # warm-up: CUDA context and kernels
    out: dict[str, Any] = {
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "note": "single timing per point; the GPU was shared with another job",
        "presets": {},
    }
    total = 0.0
    for name in PRESETS:
        sizes = N_GPU + ([100_000] if name == "full" else [])
        rows = {}
        for n in sizes:
            seconds, emitted = timed_generate(n, name, seed=0, device="cuda")
            total += seconds
            rows[str(n)] = {"s": round(seconds, 3), "events_per_s": round(n / seconds), "photons": emitted}
            print(f"  gpu: {name:<15s} N = {n:>6d}: {seconds:6.2f} s ({n / seconds:,.0f} events/s)")
        out["presets"][name] = rows
    out["total_gpu_seconds"] = round(total, 1)
    return out


def summary(run: dict[str, Any]) -> dict[str, Any]:
    """Compact record of one CPU timing run (kept in T0.2_history, since the machine load changes between runs)."""
    return {
        "created": run.get("created"),
        "load_average_1min_before_after": run["load_average_1min_before_after"],
        "fit_events_per_s": {name: run["presets"][name]["fit_events_per_s"] for name in PRESETS},
        "ns_per_photon_at_50k": {name: run["presets"][name]["ns_per_photon_at_50k"] for name in PRESETS},
        "cli_10k_full_wall_median_s": run["cli_10k_full"]["wall_median_s"],
    }


def store_cpu(run: dict[str, Any]) -> None:
    """Save the run as T0.2 and append it to T0.2_history (the earlier T0.2 seeds an empty history)."""
    path = RES / "T0.json"
    current = json.loads(path.read_text()) if path.exists() else {}
    history = list(current.get("T0.2_history", []))
    if not history and "T0.2" in current:
        history.append(summary(current["T0.2"]))
    history.append(summary(run))
    rates = {name: [h["fit_events_per_s"][name] for h in history] for name in PRESETS}
    spread = {name: [min(v), max(v)] for name, v in rates.items()}
    save_results("T0", {"T0.2": run, "T0.2_history": history, "T0.2_history_rate_range": spread})


def derived() -> dict[str, Any]:
    """Numbers derived from the stored CPU and GPU timings (no new measurement): times at 50k and GPU speed-ups."""
    results = json.loads((RES / "T0.json").read_text())
    cpu, gpu = results["T0.2"], results.get("T0.2_gpu")
    big = N_CPU.index(50_000)
    out: dict[str, Any] = {"cpu_median_time_50k_s": {name: cpu["presets"][name]["median_s"][big] for name in PRESETS}}
    if gpu is not None:
        gpu_rate = {name: gpu["presets"][name]["50000"]["events_per_s"] for name in PRESETS}
        out["gpu_over_cpu_rate_50k"] = {
            name: round(gpu_rate[name] / cpu["presets"][name]["events_per_s_at_50k"], 2) for name in PRESETS
        }
        out["gpu_time_per_event_relative_to_straight_50k"] = {
            name: round(gpu_rate["straight"] / gpu_rate[name], 2) for name in PRESETS
        }
    save_results("T0", {"T0.2_derived": out})
    return out


def run_threads(n_events: int = 10_000, repeats: int = 2) -> dict[str, Any]:
    """Events per second of the full model on the CPU for several numbers of torch threads."""
    load_before = os.getloadavg()
    rates: dict[str, list[int]] = {}
    for threads in THREAD_SCAN:
        torch.set_num_threads(threads)
        timed_generate(1000, "full", seed=999, device="cpu")  # warm-up of the thread pool
        seconds = [timed_generate(n_events, "full", seed=rep, device="cpu")[0] for rep in range(repeats)]
        rates[str(threads)] = [round(n_events / t) for t in seconds]
        print(f"  {threads:>2d} threads: {', '.join(f'{v:,}' for v in rates[str(threads)])} events/s")
    best = {k: max(v) for k, v in rates.items()}
    return {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "preset": "full",
        "n_events": n_events,
        "repeats": repeats,
        "events_per_s": rates,
        "best_events_per_s": best,
        "speed_up_vs_1_thread": {k: round(v / best["1"], 2) for k, v in best.items()},
        "load_average_1min_before_after": [round(load_before[0], 2), round(os.getloadavg()[0], 2)],
    }


def plot() -> None:
    """Figure t0_2_timing from results/T0.json: time versus N per preset, cost per photon, thread scan."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    results = json.loads((RES / "T0.json").read_text())
    cpu = results["T0.2"]
    gpu = results.get("T0.2_gpu")
    scan = results.get("T0.2_threads")
    style()
    if scan is None:
        fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.6, 3.9), gridspec_kw={"width_ratios": [1.25, 1]})
    else:
        fig, (ax, bx, cx) = plt.subplots(1, 3, figsize=(13.0, 3.9), gridspec_kw={"width_ratios": [1.25, 1, 0.9]})
        pairs = sorted((int(k), v) for k, v in scan["best_events_per_s"].items())  # JSON keys are sorted as text
        threads = np.array([k for k, _ in pairs])
        best = np.array([v for _, v in pairs], dtype=float)
        cx.plot(threads, best, "o-", color=COLOURS["full"], ms=6, lw=1.3)
        cx.plot(threads, best[0] * threads, color=C["ink2"], ls=":", lw=1.2)
        cx.set_xscale("log", base=2)
        cx.set_xticks(threads, [str(t) for t in threads])
        cx.minorticks_off()
        cx.set_ylim(0, 1.25 * best.max())
        cx.set_xlabel("Torch threads")
        cx.set_ylabel("Events per second")
        cx.set_title(f"Full model, {scan['n_events']:,} events".replace(",", " "))
    n_fine = np.logspace(np.log10(800), np.log10(6e4), 100)
    for name in PRESETS:
        row = cpu["presets"][name]
        n = np.array(cpu["n_events"])
        med = np.array(row["median_s"])
        err = [med - np.array(row["min_s"]), np.array(row["max_s"]) - med]
        ax.errorbar(n, med, yerr=err, fmt="o", color=COLOURS[name], ms=5, capsize=2, lw=1)
        ax.plot(n_fine, row["fit_fixed_s"] + n_fine / row["fit_events_per_s"], color=COLOURS[name], lw=1.2)
        if gpu is not None:
            rows = gpu["presets"][name]
            ng = np.array([int(k) for k in rows])
            ax.plot(ng, [rows[k]["s"] for k in rows], "D", color=COLOURS[name], ms=5, mfc="white", mew=1.4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of events $N$")
    ax.set_ylabel("Wall time [s]")
    ax.set_title("Time to simulate $N$ events")
    handles = [Line2D([], [], color=COLOURS[k], marker="s", ls="none", ms=7, label=k) for k in PRESETS]
    handles += [
        Line2D([], [], color=C["ink2"], marker="o", ls="none", label=f"CPU, {cpu['torch_threads']} threads"),
        Line2D([], [], color=C["ink2"], lw=1.2, label="fit $t_\\mathrm{fixed} + N/\\mathrm{rate}$"),
    ]
    if gpu is not None:
        handles.append(Line2D([], [], color=C["ink2"], marker="D", mfc="white", ls="none", label="GPU (A30)"))
    if scan is not None:
        handles.append(Line2D([], [], color=C["ink2"], ls=":", lw=1.2, label="right: ideal scaling from 1 thread"))
    per_step = cpu["ns_per_transport_step_fit"]
    handles.append(
        Line2D(
            [],
            [],
            color=C["ink2"],
            ls="--",
            lw=1.1,
            label=f"{'middle' if scan is not None else 'right'}: time $\\propto$ steps, {per_step:.0f} ns per step",
        )
    )
    figure_legend(fig, handles, ncol=4)

    steps = [cpu["presets"][k]["transport"]["steps"] for k in PRESETS]
    per_photon = [cpu["presets"][k]["ns_per_photon_at_50k"] for k in PRESETS]
    grid = np.array([0.0, max(steps) * 1.2])
    bx.plot(grid, per_step * grid, color=C["ink2"], ls="--", lw=1.1)
    for k, s, t in zip(PRESETS, steps, per_photon, strict=True):
        bx.plot(s, t, "o", color=COLOURS[k], ms=7)
        bx.annotate(k, (s, t), xytext=(8, -14 if k == "straight" else 5), textcoords="offset points", fontsize=9)
    bx.set_xlabel("Mean transport steps per photon")
    bx.set_ylabel("CPU time per photon [ns]")
    bx.set_title("Cost per photon (CPU, $N$ = 50 000)")
    bx.set_xlim(0.8, max(steps) * 1.2)
    bx.set_ylim(0, max(per_photon) * 1.25)
    savefig(fig, "t0_2_timing")


def main() -> None:
    """Command line: CPU timings (default), --gpu, --threads or --plot."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gpu", action="store_true", help="time on the GPU instead (short)")
    parser.add_argument("--plot", action="store_true", help="only redraw the figure from results/T0.json")
    parser.add_argument("--threads", action="store_true", help="CPU thread scan of the full model")
    args = parser.parse_args()
    if args.threads:
        save_results("T0", {"T0.2_threads": run_threads()})
        plot()
        return
    if args.gpu:
        out = run_gpu()
        save_results("T0", {"T0.2_gpu": out})
        print(f"GPU total {out['total_gpu_seconds']} s")
        return
    if not args.plot:
        out = run_cpu()
        store_cpu(out)
        print("\npreset            events/s (fit)  fixed [s]  slope  ns/photon  steps/photon")
        for name in PRESETS:
            row = out["presets"][name]
            print(
                f"{name:<17s} {row['fit_events_per_s']:>14,d} {row['fit_fixed_s']:>10.3f} "
                f"{row['loglog_slope_N_ge_5000']:>6.3f}"
                f" {row['ns_per_photon_at_50k']:>10.1f} {row['transport']['steps']:>13.2f}"
            )
        cli = out["cli_10k_full"]
        print(f"\n{cli['command']}: {cli['wall_median_s']} s end to end ({cli['simulation_median_s']} s simulation)")
    print(f"derived: {derived()}")
    plot()


if __name__ == "__main__":
    main()

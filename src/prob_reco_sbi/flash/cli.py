"""Command line interface ``flashsim`` (also available as ``python -m prob_reco_sbi.flash``).

::

    flashsim generate [--preset NAME] [-n N] [--seed S] [--device auto|cpu|cuda] [-o FILE] [physics options]
    flashsim info FILE
    flashsim merge -o OUT FILE [FILE ...]
    flashsim presets

Every configuration parameter has an option of the same name (``n_sensors`` -> ``--n-sensors``); options given on
the command line change the chosen preset, e.g. ``flashsim generate --preset straight --reflection``.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import shlex
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .config import PRESETS, SimConfig, config_fields, preset

__all__ = ["build_parser", "default_output", "main"]

log = logging.getLogger("prob_reco_sbi.flash")

_PART_TITLES = {
    "geometry": "detector geometry",
    "optics": "light and optics (toggles switch a process on or off on top of the preset)",
    "readout": "sensor readout",
    "prior": "prior of the event parameters",
}


class _HelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """Show the default of an option, unless it is empty (``None`` or ``False``)."""

    def _get_help_string(self, action: argparse.Action) -> str:
        if action.default is None or action.default is False or action.default is argparse.SUPPRESS:
            return action.help or ""
        return super()._get_help_string(action) or ""


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser of ``flashsim``."""
    parser = argparse.ArgumentParser(
        prog="flashsim",
        description='Photon-transport simulation of the toy detector "A Flash in the Dark".',
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    gen = commands.add_parser(
        "generate",
        help="simulate events and write them to an HDF5 file",
        description="Simulate events with parameters drawn from the prior and write them to an HDF5 file.",
        formatter_class=_HelpFormatter,
    )
    gen.add_argument("--preset", choices=list(PRESETS), default="full", help="physics preset")
    gen.add_argument("-n", "--n-events", type=int, default=10_000, metavar="N", help="number of events")
    gen.add_argument("--seed", type=int, default=0, help="random seed (non-negative integer)")
    gen.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        metavar="FILE",
        help="output file (default: data/flash_<preset>_n<N>_seed<S>.h5)",
    )
    gen.add_argument("--overwrite", action="store_true", help="replace an existing output file")
    gen.add_argument("--save-hits", action="store_true", help="store every hit (/hits)")
    gen.add_argument(
        "--save-tracks", type=int, default=0, metavar="K", help="store the photon paths of the first K events"
    )
    adv = gen.add_argument_group("advanced options")
    adv.add_argument("--device", default="auto", help="'auto' (CUDA if available), 'cpu', 'cuda' or 'cuda:N'")
    adv.add_argument("--threads", type=int, default=None, help="CPU threads used by torch (default: torch's choice)")
    adv.add_argument("--track-photons", type=int, default=100, metavar="P", help="photons per stored event (0: all)")
    adv.add_argument("--first-event-id", type=int, default=0, metavar="ID", help="event id of the first event")
    adv.add_argument(
        "--photons-per-chunk",
        type=int,
        default=None,
        metavar="P",
        help="photons transported at once (memory control; default 2^21 on CPU, 2^23 on CUDA)",
    )
    adv.add_argument(
        "--progress", action=argparse.BooleanOptionalAction, default=True, help="show a progress indicator"
    )
    _add_logging_options(gen)
    _add_config_options(gen)

    info = commands.add_parser("info", help="describe a data file", description="Describe a data file.")
    info.add_argument("file", type=Path, help="HDF5 file written by flashsim")
    _add_logging_options(info)

    mrg = commands.add_parser(
        "merge",
        help="concatenate data files (e.g. cluster shards)",
        description="Concatenate data files of the same configuration, e.g. shards generated with different seeds "
        "and disjoint --first-event-id ranges.",
    )
    mrg.add_argument("-o", "--output", type=Path, required=True, metavar="OUT", help="output file")
    mrg.add_argument("files", type=Path, nargs="+", metavar="FILE", help="input files")
    mrg.add_argument("--overwrite", action="store_true", help="replace an existing output file")
    _add_logging_options(mrg)

    pre = commands.add_parser("presets", help="list the physics presets", description="List the physics presets.")
    _add_logging_options(pre)
    return parser


def default_output(preset_name: str, config: SimConfig, n_events: int, seed: int) -> Path:
    """Default file name ``data/flash_<preset>_n<N>_seed<S>.h5`` (with a tag if the preset was modified)."""
    tag = preset_name
    if config != preset(preset_name):
        tag += "-custom-" + hashlib.sha1(config.to_json().encode()).hexdigest()[:6]
    return Path("data") / f"flash_{tag}_n{n_events}_seed{seed}.h5"


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point of ``flashsim``; returns the exit status."""
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(argv)
    level = logging.WARNING if args.quiet else logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("prob_reco_sbi").setLevel(level)
    try:
        if args.command == "generate":
            return _generate(args, parser, argv)
        if args.command == "info":
            from .io import describe

            print(describe(args.file))
        elif args.command == "merge":
            from .io import merge

            out = merge(args.files, args.output, overwrite=args.overwrite)
            log.info("merged %d files into %s", len(args.files), out)
        elif args.command == "presets":
            print(_presets_table())
    except (ValueError, OSError, RuntimeError) as error:
        print(f"flashsim {args.command}: error: {error}", file=sys.stderr)
        return 1
    return 0


# ----------------------------------------------------------------------------------------------------------------------


def _add_logging_options(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-v", "--verbose", action="store_true", help="more log messages")
    group.add_argument("-q", "--quiet", action="store_true", help="only warnings and errors")


def _add_config_options(parser: argparse.ArgumentParser) -> None:
    """One option per configuration parameter; the default ``None`` keeps the preset's value."""
    defaults = SimConfig()
    groups = {name: parser.add_argument_group(title) for name, title in _PART_TITLES.items()}
    for part, f in config_fields():
        flag = "--" + f.name.replace("_", "-")
        meta = f.metadata
        unit = f" [{meta['unit']}]" if meta["unit"] else ""
        value = getattr(getattr(defaults, part), f.name)
        kwargs: dict[str, Any] = {"default": None, "dest": f.name}
        if isinstance(f.default, bool):
            on = [name for name, cfg in PRESETS.items() if getattr(getattr(cfg, part), f.name)]
            kwargs["action"] = argparse.BooleanOptionalAction
            kwargs["help"] = f"{meta['help']} (on in: {', '.join(on) or 'no preset'})"
        else:
            kwargs["help"] = f"{meta['help']}{unit}" + ("" if value is None else f" (full preset: {value})")
            if "choices" in meta:
                kwargs["choices"] = meta["choices"]
            elif meta.get("nargs") == 2:
                kwargs |= {"nargs": 2, "type": float, "metavar": meta["metavar"]}
            else:
                kwargs |= {"type": int if isinstance(f.default, int) else float}
                kwargs["metavar"] = meta.get("metavar", "VALUE")
        groups[part].add_argument(flag, **kwargs)


def _overrides(args: argparse.Namespace) -> dict[str, Any]:
    out = {}
    for _, f in config_fields():
        value = getattr(args, f.name)
        if value is not None:
            out[f.name] = tuple(value) if f.metadata.get("nargs") == 2 else value
    return out


def _generate(args: argparse.Namespace, parser: argparse.ArgumentParser, argv: list[str]) -> int:
    import torch

    from .io import save
    from .simulate import generate, resolve_device

    if args.threads is not None:
        if args.threads < 1:
            parser.exit(1, f"flashsim generate: error: --threads must be at least 1, got {args.threads}\n")
        torch.set_num_threads(args.threads)
    overrides = _overrides(args)
    config = preset(args.preset, **overrides)
    output = args.output or default_output(args.preset, config, args.n_events, args.seed)
    if output.exists() and not args.overwrite:
        parser.exit(1, f"flashsim generate: error: {output} exists (use --overwrite to replace it)\n")
    # fail before the simulation, not after it: the device and the output directory
    resolve_device(args.device)
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        if not os.access(output.parent, os.W_OK):
            raise PermissionError(f"no write permission in {output.parent.resolve()}")
    except OSError as error:
        parser.exit(1, f"flashsim generate: error: cannot write {output} ({error})\n")
    log.info("preset %s%s", args.preset, f" with {overrides}" if overrides else "")
    result = generate(
        args.n_events,
        config=config,
        seed=args.seed,
        device=args.device,
        save_hits=args.save_hits,
        save_tracks=args.save_tracks,
        track_photons=args.track_photons or None,
        photons_per_chunk=args.photons_per_chunk,
        first_event_id=args.first_event_id,
        progress=args.progress and not args.quiet,
    )
    result.provenance |= {
        "preset": args.preset,
        "overrides": overrides,
        "command": "flashsim " + shlex.join(argv),
    }
    save(result, output, overwrite=args.overwrite)
    seconds = result.provenance["wall_time_s"]
    log.info("wrote %s: %d events in %.1f s (%.0f events/s)", output, len(result), seconds, len(result) / seconds)
    return 0


def _presets_table() -> str:
    def state(value: bool) -> str:
        return "on" if value else "off"

    lines = [f"{'preset':<16s} {'absorption':<11s} {'scattering':<11s} reflection"]
    for name, cfg in PRESETS.items():
        o = cfg.optics
        label = name + (" (default)" if name == "full" else "")
        lines.append(f"{label:<16s} {state(o.absorption):<11s} {state(o.scattering):<11s} {state(o.reflection)}")
    lines.append("")
    lines.append("parameters of the full preset (the other presets differ only in the three switches):")
    lines += ["  " + line for line in SimConfig().summary().splitlines()]
    lines.append("")
    lines.append("example: flashsim generate --preset straight --reflection -n 1000")
    return "\n".join(lines)

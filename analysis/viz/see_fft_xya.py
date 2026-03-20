#!/usr/bin/env python3
from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import sys
from pathlib import Path

import numpy as np

from plotting.common import render_figure
from plotting.frequency import plot_component_pair_frequency_grid
from tools.cli import add_colormap_arg, add_output_args, add_signal_processing_args, add_track2_input_args
from tools.derived import derive_spacing_dataset
from tools.io import load_track2_dataset
from tools.spectral import analyze_spacing_dataset_for_display, analyze_spacing_dataset_with_welch_for_display

COMPONENT_SUFFIXES = ("x", "y", "a")


def _parse_bool_arg(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Overlay FFT or Welch spectra and show per-component sliding FFTs for Track2-derived block spacing.",
    )
    add_track2_input_args(parser)
    add_signal_processing_args(parser)
    add_colormap_arg(parser)
    add_output_args(parser, include_title=True)

    parser.add_argument(
        "--fft-log",
        nargs="?",
        const=True,
        default=True,
        type=_parse_bool_arg,
        metavar="BOOL",
        help="Use a log y-axis for the FFT curve panel. Pass --fft-log false for a linear axis.",
    )

    sliding_group = parser.add_mutually_exclusive_group()
    sliding_group.add_argument(
        "--sliding-log",
        dest="sliding_plot_scale",
        action="store_const",
        const="log",
        help="Use dB display for the sliding panel (default).",
    )
    sliding_group.add_argument(
        "--sliding-linear",
        dest="sliding_plot_scale",
        action="store_const",
        const="linear",
        help="Use linear power/amplitude display for the sliding panel.",
    )
    parser.set_defaults(sliding_plot_scale="log")

    parser.add_argument("--sliding-len-s", type=float, default=20.0)
    parser.add_argument(
        "--welch",
        action="store_true",
        help="Replace the left FFT panel with a Welch spectrum panel.",
    )
    parser.add_argument(
        "--welch-log",
        nargs="?",
        const=True,
        default=True,
        type=_parse_bool_arg,
        metavar="BOOL",
        help="Use a log y-axis for the Welch curve panel. Pass --welch-log false for a linear axis.",
    )
    parser.add_argument("--welch-len-s", type=float, default=20.0)
    parser.add_argument("--welch-overlap", type=float, default=0.5)
    parser.add_argument("--fft-min-hz", type=float, default=None)
    parser.add_argument("--fft-max-hz", type=float, default=None)
    parser.add_argument("--welch-min-hz", type=float, default=None)
    parser.add_argument("--welch-max-hz", type=float, default=None)
    parser.add_argument("--sliding-min-hz", type=float, default=None)
    parser.add_argument("--sliding-max-hz", type=float, default=None)
    parser.add_argument("--time-interval-s", nargs=2, type=float, metavar=("START", "STOP"))
    parser.add_argument("--only", choices=["fft", "sliding"], default=None)
    parser.add_argument(
        "--full-image",
        action="store_true",
        help="Replace each sliding spectrogram panel with a 2D image of the full FFT spectrum.",
    )
    parser.add_argument(
        "--full-couple",
        nargs="?",
        const=True,
        default=True,
        type=_parse_bool_arg,
        metavar="BOOL",
        help="Couple FFT-panel frequency zoom with the image-panel frequency axis. Pass --full-couple false to disable.",
    )
    parser.add_argument(
        "--disable",
        type=int,
        action="append",
        default=[],
        help="Disable pair index (0-based). Can be used multiple times.",
    )
    parser.add_argument(
        "--disable-component",
        choices=COMPONENT_SUFFIXES,
        action="append",
        default=[],
        help="Disable one of the suffixed component datasets: x, y, or a. Can be used multiple times.",
    )
    return parser


def _strip_component_suffix(name: str) -> str:
    for suffix in COMPONENT_SUFFIXES:
        token = f"_{suffix}"
        if name.endswith(token):
            return name[: -len(token)]
    return name


def _base_dataset_context(
    dataset: str | None,
    track2_path: str | None,
    track_data_root: str,
) -> tuple[str, str]:
    if track2_path is not None:
        path = Path(track2_path).resolve()
        if path.name != "track2_permanence.msgpack":
            raise ValueError("--track2 must point to a track2_permanence.msgpack file")
        return _strip_component_suffix(path.parent.name), str(path.parent.parent)

    if dataset is None:
        raise ValueError("Provide either DATASET or --track2")

    return _strip_component_suffix(dataset), track_data_root


def _load_component_results(args) -> dict[str, list]:
    base_dataset, data_root = _base_dataset_context(args.dataset, args.track2, args.track_data_root)
    disabled_components = set(args.disable_component)

    component_results: dict[str, list] = {}
    missing_components: list[str] = []

    for component in COMPONENT_SUFFIXES:
        if component in disabled_components:
            continue

        dataset_name = f"{base_dataset}_{component}"
        try:
            track2 = load_track2_dataset(
                dataset=dataset_name,
                track_data_root=data_root,
            )
        except FileNotFoundError:
            missing_components.append(component)
            continue

        spacing = derive_spacing_dataset(track2)
        if args.welch:
            results = analyze_spacing_dataset_with_welch_for_display(
                spacing,
                disabled_indices=args.disable,
                longest=args.longest,
                handlenan=args.handlenan,
                welch_len_s=args.welch_len_s,
                welch_overlap_fraction=args.welch_overlap,
                sliding_len_s=args.sliding_len_s,
            )
        else:
            results = analyze_spacing_dataset_for_display(
                spacing,
                disabled_indices=args.disable,
                longest=args.longest,
                handlenan=args.handlenan,
                sliding_len_s=args.sliding_len_s,
            )
        component_results[component] = results

        dt = np.diff(track2.frame_times_s)
        dt = dt[np.isfinite(dt) & (dt > 0)]
        print(f"Track2 ({component}): {track2.track2_path}")
        if dt.size > 0:
            approx_fs = 1.0 / float(np.median(dt))
            approx_nyquist = 0.5 * approx_fs
            print(f"Approx sampling rate ({component}): {approx_fs:.4f} Hz | Approx Nyquist: {approx_nyquist:.4f} Hz")
        else:
            print(f"Approx sampling rate ({component}): unavailable")

    if len(component_results) == 0:
        searched = ", ".join(f"{base_dataset}_{component}" for component in COMPONENT_SUFFIXES if component not in disabled_components)
        raise FileNotFoundError(f"None of the requested component datasets were found. Tried: {searched}")

    if missing_components:
        print(
            "Missing component datasets: "
            + ", ".join(f"{base_dataset}_{component}" for component in missing_components),
            file=sys.stderr,
        )

    return component_results


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        component_results = _load_component_results(args)
        fig = plot_component_pair_frequency_grid(
            component_results,
            fft_log=args.fft_log,
            welch_log=args.welch_log,
            sliding_plot_scale=args.sliding_plot_scale,
            only=args.only,
            full_image=args.full_image,
            full_couple=args.full_couple,
            use_welch=args.welch,
            fft_min_hz=args.fft_min_hz,
            fft_max_hz=args.fft_max_hz,
            welch_min_hz=args.welch_min_hz,
            welch_max_hz=args.welch_max_hz,
            sliding_min_hz=args.sliding_min_hz,
            sliding_max_hz=args.sliding_max_hz,
            time_interval=tuple(args.time_interval_s) if args.time_interval_s is not None else None,
            cmap_index=args.cm,
            title=args.title,
        )
        render_figure(fig, save=args.save)
        return 0

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse

from .io import get_default_track_data_root


def add_track2_input_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "dataset",
        nargs="?",
        help="Dataset stem, e.g. IMG_0584. Ignored if --track2 is given.",
    )
    parser.add_argument(
        "--track2",
        default=None,
        help="Explicit path to track2_permanence.msgpack.",
    )
    parser.add_argument(
        "--track-data-root",
        default=str(get_default_track_data_root()),
        help="Root directory containing track datasets. Default: sibling ../track/data/",
    )


def add_track_data_root_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--track-data-root",
        default=str(get_default_track_data_root()),
        help="Root directory containing track datasets. Default: sibling ../track/data/",
    )


def add_signal_processing_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--longest",
        action="store_true",
        help="Use the longest contiguous valid segment instead of interpolating through gaps.",
    )
    parser.add_argument(
        "--handlenan",
        action="store_true",
        help="Discard invalid time/value samples before uniform resampling.",
    )


def add_output_args(parser: argparse.ArgumentParser, *, include_title: bool = False) -> None:
    parser.add_argument(
        "--save",
        default=None,
        help="Optional path to save the figure.",
    )
    if include_title:
        parser.add_argument(
            "--title",
            default=None,
            help="Optional title override.",
        )


def add_colormap_arg(parser: argparse.ArgumentParser, *, default: int = 6) -> None:
    parser.add_argument(
        "--cm",
        type=int,
        choices=range(1, 11),
        default=default,
        help=f"Colormap index. Default: {default}",
    )


def add_normalization_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--normalize",
        required=True,
        choices=["absolute", "relative"],
        help="Normalization mode.",
    )
    parser.add_argument(
        "--relative-range",
        nargs=2,
        type=float,
        metavar=("START_HZ", "STOP_HZ"),
        default=(2.0, 8.0),
        help="Relative-normalization band in Hz. Default: 2 8",
    )


def add_average_domain_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--average-linear",
        dest="average_domain",
        action="store_const",
        const="linear",
        help="Average spectra in linear amplitude space (default).",
    )
    group.add_argument(
        "--average-log",
        dest="average_domain",
        action="store_const",
        const="log",
        help="Average spectra in dB amplitude space.",
    )
    parser.set_defaults(average_domain="linear")


def add_plot_scale_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--plot-linear",
        dest="plot_scale",
        action="store_const",
        const="linear",
        help="Use a linear display scale (default).",
    )
    group.add_argument(
        "--plot-log",
        dest="plot_scale",
        action="store_const",
        const="log",
        help="Use a log / dB display scale.",
    )
    parser.set_defaults(plot_scale="linear")

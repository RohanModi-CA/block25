#!/usr/bin/env python3
from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import sys

from plotting.common import render_figure
from plotting.trajectory import plot_spacing_timeseries
from tools.cli import add_output_args, add_track2_input_args
from tools.derived import derive_spacing_dataset
from tools.io import load_track2_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot stacked block-spacing time series derived from Track2 permanence.",
    )
    add_track2_input_args(parser)
    add_output_args(parser, include_title=True)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        track2 = load_track2_dataset(
            dataset=args.dataset,
            track2_path=args.track2,
            track_data_root=args.track_data_root,
        )
        spacing = derive_spacing_dataset(track2)

        print(f"Track2: {track2.track2_path}")
        print(f"Pairs: {spacing.spacing_matrix.shape[1]}")
        print(f"Frames: {spacing.spacing_matrix.shape[0]}")

        fig = plot_spacing_timeseries(
            track2.frame_times_s,
            spacing.spacing_matrix,
            spacing.pair_labels,
            title=args.title,
        )
        render_figure(fig, save=args.save)
        return 0

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

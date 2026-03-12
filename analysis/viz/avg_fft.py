#!/usr/bin/env python3
from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import sys

from plotting.common import render_figure
from plotting.frequency import plot_average_spectrum
from tools.cli import (
    add_average_domain_args,
    add_colormap_arg,
    add_normalization_args,
    add_output_args,
    add_plot_scale_args,
    add_signal_processing_args,
    add_track_data_root_arg,
)
from tools.selection import build_configured_bond_signals, load_dataset_selection
from tools.spectral import (
    compute_average_spectrum,
    compute_fft_contributions,
    compute_reference_average_spectrum,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Average normalized FFT spectra across datasets using Track2 permanence data.",
    )
    parser.add_argument("config_json", help="Dataset-selection JSON file.")
    add_track_data_root_arg(parser)
    add_normalization_args(parser)
    add_average_domain_args(parser)
    add_plot_scale_args(parser)
    add_signal_processing_args(parser)
    add_colormap_arg(parser)
    add_output_args(parser, include_title=True)

    parser.add_argument("--freq-min-hz", type=float, default=None)
    parser.add_argument("--freq-max-hz", type=float, default=None)
    parser.add_argument(
        "--allow-duplicate-bonds",
        action="store_true",
        help="Allow the same configured bond id to contribute multiple times.",
    )
    parser.add_argument(
        "--full-image",
        action="store_true",
        help="Render the averaged spectrum as a 2D frequency image instead of a 1D curve.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.freq_min_hz is not None and args.freq_max_hz is not None and args.freq_max_hz <= args.freq_min_hz:
        print("Error: --freq-max-hz must be greater than --freq-min-hz", file=sys.stderr)
        return 1

    rel_low, rel_high = map(float, args.relative_range)
    if rel_high <= rel_low:
        print("Error: --relative-range STOP_HZ must be greater than START_HZ", file=sys.stderr)
        return 1

    try:
        config = load_dataset_selection(args.config_json)
        records = build_configured_bond_signals(
            config,
            track_data_root=args.track_data_root,
            allow_duplicate_ids=args.allow_duplicate_bonds,
        )
        contributions = compute_fft_contributions(
            records,
            longest=args.longest,
            handlenan=args.handlenan,
        )
        if len(contributions) == 0:
            raise ValueError("No spectra were accepted from the provided config")

        result = compute_average_spectrum(
            contributions,
            normalize_mode=args.normalize,
            relative_range=tuple(args.relative_range),
            average_domain=args.average_domain,
            lowest_freq=args.freq_min_hz,
            highest_freq=args.freq_max_hz,
        )

        reference_amp = None
        if args.full_image and args.plot_scale == "linear":
            reference = compute_reference_average_spectrum(
                contributions,
                normalize_mode=args.normalize,
                relative_range=tuple(args.relative_range),
                average_domain=args.average_domain,
            )
            reference_amp = reference.avg_amp

        n_contributors = len(result.contributors)
        n_datasets = len({contrib.record.dataset_name for contrib in result.contributors})

        if args.title:
            title = args.title
        else:
            norm_desc = args.normalize
            if args.normalize == "relative":
                title_range = tuple(args.relative_range)
                norm_desc = f"relative [{title_range[0]}, {title_range[1]}] Hz"
            title = (
                f"Average FFT | contributors={n_contributors} | datasets={n_datasets} | "
                f"avg={args.average_domain} | norm={norm_desc}"
            )

        print(f"Accepted contributors: {n_contributors}")
        print(f"Unique datasets: {n_datasets}")
        print(f"Frequency window: [{result.freq_low:.6f}, {result.freq_high:.6f}] Hz")
        print(f"Normalization window: [{result.norm_low:.6f}, {result.norm_high:.6f}] Hz")
        print(f"Common grid points: {len(result.freq_grid)}")
        print(f"Display mode: {'full image' if args.full_image else 'curve'}")
        if args.full_image and args.plot_scale == "linear":
            print("Image color scale reference: full implicit frequency window")

        fig = plot_average_spectrum(
            result,
            full_image=args.full_image,
            plot_scale=args.plot_scale,
            cmap_index=args.cm,
            title=title,
            reference_amp_for_norm=reference_amp,
        )
        render_figure(fig, save=args.save)
        return 0

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

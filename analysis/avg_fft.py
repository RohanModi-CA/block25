#!/usr/bin/env python3
"""
avg_fft.py

Average normalized FFT spectra across many datasets.

This tool is intended to sit beside the existing analysis scripts and reuse the
current Track2/Track3 loading and FFT preprocessing pipeline from:

- io_tracks.py
- fft_tools.py

JSON schema
-----------
The input JSON must be an object keyed by dataset stem. Each dataset object must
contain all required fields:

{
  "IMG_943": {
    "include": true,
    "discards": [0, 2],
    "pair_ids": [101, 203, 305]
  },
  "IMG_944": {
    "include": false,
    "discards": [],
    "pair_ids": [101]
  }
}

Selection semantics
-------------------
1. Dataset iteration order is the JSON object order.
2. For each dataset, local 0-based pair indices in `discards` are removed first.
3. The remaining local pairs are then labeled in order by the config-supplied
   `pair_ids`. Their lengths must match exactly.
4. Without --allowduplicatebonds, a config pair ID may contribute only once
   across the entire run. The first accepted occurrence wins; later ones are
   warned and skipped.

Normalization
-------------
Normalization is mandatory and is applied after interpolation onto the common
frequency grid, so the denominator is computed from the actual spectrum being
averaged/plotted.

- --normalize absolute:
    divide by the integral over the selected frequency window.
- --normalize relative:
    divide by the integral over the intersection of the selected frequency
    window and --relativerange.

Averaging domain
----------------
- --linear: arithmetic mean in linear amplitude space.
- --log: arithmetic mean in dB amplitude space, returned to linear amplitude
         as a geometric mean so plotting can still be linear or log.

Frequency grid
--------------
Accepted spectra are interpolated onto a common overlap grid. To avoid creating
artificial high-frequency detail, the common grid uses the coarsest native FFT
spacing among contributors inside the overlap window.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from fft_tools import compute_one_sided_fft, preprocess_signal
from io_tracks import load_fft_dataset, resolve_track2_track3_paths


COLORMAPS = {
    1: "viridis",
    2: "plasma",
    3: "inferno",
    4: "magma",
    5: "cividis",
    6: "turbo",
    7: "jet",
    8: "nipy_spectral",
    9: "ocean",
    10: "cubehelix",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Average normalized FFT spectra across many datasets using the "
            "existing Track2/Track3 FFT pipeline."
        )
    )

    parser.add_argument(
        "config_json",
        help="Dataset-selection JSON file.",
    )
    parser.add_argument(
        "--normalize",
        required=True,
        choices=["absolute", "relative"],
        help="Mandatory normalization mode.",
    )
    parser.add_argument(
        "--relativerange",
        nargs=2,
        type=float,
        metavar=("START_HZ", "STOP_HZ"),
        default=(2.0, 8.0),
        help="Relative-normalization band in Hz. Default: 2 8",
    )

    avg_group = parser.add_mutually_exclusive_group()
    avg_group.add_argument(
        "--linear",
        dest="average_domain",
        action="store_const",
        const="linear",
        help="Average normalized spectra in linear amplitude space (default).",
    )
    avg_group.add_argument(
        "--log",
        dest="average_domain",
        action="store_const",
        const="log",
        help="Average normalized spectra in dB amplitude space.",
    )
    parser.set_defaults(average_domain="linear")

    plot_group = parser.add_mutually_exclusive_group()
    plot_group.add_argument(
        "--plotlinear",
        dest="plot_scale",
        action="store_const",
        const="linear",
        help="Plot the final averaged spectrum on a linear y-axis (default).",
    )
    plot_group.add_argument(
        "--plotlog",
        dest="plot_scale",
        action="store_const",
        const="log",
        help="Plot the final averaged spectrum on a log y-axis.",
    )
    parser.set_defaults(plot_scale="linear")

    parser.add_argument("--longest", action="store_true")
    parser.add_argument(
        "--handlenan",
        action="store_true",
        help=(
            "Use finite-sample handling before uniform-time resampling. By "
            "default the legacy NaN interpolation behavior is preserved."
        ),
    )

    parser.add_argument("--lowest_freq", type=float, default=None)
    parser.add_argument("--highest_freq", type=float, default=None)
    parser.add_argument(
        "--allowduplicatebonds",
        action="store_true",
        help="Allow the same config pair ID to contribute multiple times.",
    )
    parser.add_argument(
        "--full_image",
        action="store_true",
        help=(
            "Replace the 1D average FFT curve with a 2D color image built from "
            "the final averaged FFT spectrum. X axis is arbitrary, Y axis is frequency."
        ),
    )
    parser.add_argument(
        "--cm",
        type=int,
        choices=range(1, 11),
        default=6,
        help="Colormap index for --full_image. Default: 6 (turbo)",
    )
    parser.add_argument(
        "--save",
        default=None,
        help="Optional path to save the figure.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional plot title override.",
    )

    return parser


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f, object_pairs_hook=OrderedDict)

    if not isinstance(cfg, dict) or len(cfg) == 0:
        raise ValueError("Top-level JSON must be a non-empty object keyed by dataset stem")

    validated = OrderedDict()
    required = {"include", "discards", "pair_ids"}

    for dataset_name, entry in cfg.items():
        if not isinstance(dataset_name, str) or not dataset_name:
            raise ValueError("Each top-level JSON key must be a non-empty dataset string")
        if not isinstance(entry, dict):
            raise ValueError(f"Dataset '{dataset_name}' entry must be an object")

        missing = required.difference(entry.keys())
        if missing:
            raise ValueError(
                f"Dataset '{dataset_name}' is missing required key(s): {sorted(missing)}"
            )

        include = entry["include"]
        discards = entry["discards"]
        pair_ids = entry["pair_ids"]

        if not isinstance(include, bool):
            raise ValueError(f"Dataset '{dataset_name}' field 'include' must be boolean")
        if not isinstance(discards, list):
            raise ValueError(f"Dataset '{dataset_name}' field 'discards' must be a list")
        if not isinstance(pair_ids, list):
            raise ValueError(f"Dataset '{dataset_name}' field 'pair_ids' must be a list")

        discards_out = []
        for idx in discards:
            if not isinstance(idx, int) or idx < 0:
                raise ValueError(
                    f"Dataset '{dataset_name}' has invalid discard index {idx!r}; expected non-negative int"
                )
            discards_out.append(int(idx))

        pair_ids_out = []
        for pair_id in pair_ids:
            if not isinstance(pair_id, int):
                raise ValueError(
                    f"Dataset '{dataset_name}' has invalid pair_id {pair_id!r}; expected int"
                )
            pair_ids_out.append(int(pair_id))

        validated[dataset_name] = {
            "include": include,
            "discards": discards_out,
            "pair_ids": pair_ids_out,
        }

    return validated


def warn(msg: str) -> None:
    print(f"Warning: {msg}", file=sys.stderr)


def median_positive_step(x: np.ndarray) -> float | None:
    if x.size < 2:
        return None
    dx = np.diff(x)
    dx = dx[np.isfinite(dx) & (dx > 0)]
    if dx.size == 0:
        return None
    return float(np.median(dx))


def build_common_grid(spectra, freq_low: float, freq_high: float) -> np.ndarray:
    dfs = []
    for spec in spectra:
        mask = (spec["freq"] >= freq_low) & (spec["freq"] <= freq_high)
        f_local = spec["freq"][mask]
        df = median_positive_step(f_local)
        if df is not None:
            dfs.append(df)

    if not dfs:
        raise ValueError("Could not determine a common positive FFT spacing in the overlap window")

    df_target = max(dfs)

    grid = np.arange(freq_low, freq_high + 0.5 * df_target, df_target, dtype=float)
    grid = grid[(grid >= freq_low - 1e-12) & (grid <= freq_high + 1e-12)]

    if grid.size < 2:
        grid = np.array([freq_low, freq_high], dtype=float)

    if grid.size < 2 or not np.all(np.isfinite(grid)) or grid[-1] <= grid[0]:
        raise ValueError("Failed to build a valid common frequency grid")

    return grid


def interp_amplitude(freq_src: np.ndarray, amp_src: np.ndarray, freq_dst: np.ndarray) -> np.ndarray:
    return np.interp(freq_dst, freq_src, amp_src)


def integral_over_window(freq: np.ndarray, amp: np.ndarray, low: float, high: float) -> float:
    mask = (freq >= low) & (freq <= high)
    if np.count_nonzero(mask) < 2:
        return 0.0
    return float(np.trapezoid(amp[mask], freq[mask]))


def denominator_too_small(denom: float, amp: np.ndarray, low: float, high: float) -> bool:
    width = max(high - low, 1e-12)
    amp_scale = float(np.nanmax(np.abs(amp))) if amp.size > 0 else 0.0
    tol = 1e4 * np.finfo(float).eps * max(1.0, amp_scale * width)
    return (not np.isfinite(denom)) or (abs(denom) <= tol)


def normalize_spectrum(freq, amp, *, norm_low: float, norm_high: float) -> np.ndarray | None:
    denom = integral_over_window(freq, amp, norm_low, norm_high)
    if denominator_too_small(denom, amp, norm_low, norm_high):
        return None
    return amp / denom


def choose_frequency_window(spectra, args):
    min_supported = max(float(spec["freq"][0]) for spec in spectra)
    max_supported = min(float(spec["freq"][-1]) for spec in spectra)

    freq_low = min_supported if args.lowest_freq is None else max(min_supported, args.lowest_freq)
    freq_high = max_supported if args.highest_freq is None else min(max_supported, args.highest_freq)

    if not np.isfinite(freq_low) or not np.isfinite(freq_high) or freq_high <= freq_low:
        raise ValueError("No overlapping frequency window across accepted spectra")

    return float(freq_low), float(freq_high)


def gather_requested_spectra(config, args):
    gathered = []
    seen_pair_ids = set()

    for dataset_name, entry in config.items():
        include = entry["include"]
        discards = set(entry["discards"])
        config_pair_ids = entry["pair_ids"]

        if not include:
            print(f"Skipping dataset '{dataset_name}' (include=false)")
            continue

        resolved_track2, resolved_track3 = resolve_track2_track3_paths(dataset=dataset_name)
        dataset = load_fft_dataset(dataset=dataset_name)

        T = dataset["frame_times_s"]
        spacing = dataset["spacing_matrix"]
        pair_labels = dataset["pair_labels"]
        n_pairs = int(spacing.shape[1])

        print(f"Dataset: {dataset_name}")
        print(f"  Track2: {resolved_track2}")
        print(f"  Track3: {resolved_track3}")

        remaining_local_indices = [local_idx for local_idx in range(n_pairs) if local_idx not in discards]

        if len(remaining_local_indices) != len(config_pair_ids):
            raise ValueError(
                f"Dataset '{dataset_name}' has {len(remaining_local_indices)} remaining local pairs after discards "
                f"but {len(config_pair_ids)} config pair_ids were provided; these lengths must match exactly"
            )

        for local_idx, requested_pair_id in zip(remaining_local_indices, config_pair_ids):
            if (not args.allowduplicatebonds) and (requested_pair_id in seen_pair_ids):
                warn(
                    f"Skipping duplicate pair_id {requested_pair_id} from dataset '{dataset_name}' "
                    "because an earlier occurrence was already accepted"
                )
                continue

            y = spacing[:, local_idx]
            processed, error_msg = preprocess_signal(
                T,
                y,
                longest=args.longest,
                handlenan=args.handlenan,
            )
            if processed is None:
                warn(
                    f"Skipping dataset '{dataset_name}' pair_id {requested_pair_id} "
                    f"(local pair {local_idx}) because preprocessing failed: {error_msg}"
                )
                continue

            fft_result = compute_one_sided_fft(processed.y, processed.dt)
            freq = np.asarray(fft_result.freq, dtype=float)
            amp = np.asarray(fft_result.amplitude, dtype=float)

            if freq.size < 2 or amp.size != freq.size:
                warn(
                    f"Skipping dataset '{dataset_name}' pair_id {requested_pair_id} "
                    "because FFT output was invalid"
                )
                continue

            label = str(pair_labels[local_idx]).upper() if local_idx < len(pair_labels) else "?"
            gathered.append(
                {
                    "dataset": dataset_name,
                    "local_idx": int(local_idx),
                    "pair_id": int(requested_pair_id),
                    "label": label,
                    "freq": freq,
                    "amp": amp,
                }
            )
            seen_pair_ids.add(int(requested_pair_id))

    return gathered


def average_spectra(normalized_stack: np.ndarray, domain: str) -> np.ndarray:
    eps = np.finfo(float).tiny

    if domain == "linear":
        return np.mean(normalized_stack, axis=0)

    if domain == "log":
        db_stack = 20.0 * np.log10(np.maximum(normalized_stack, eps))
        mean_db = np.mean(db_stack, axis=0)
        return 10.0 ** (mean_db / 20.0)

    raise ValueError(f"Unsupported averaging domain: {domain}")


def centers_to_edges(vals: np.ndarray, fallback_step: float = 1.0) -> np.ndarray:
    vals = np.asarray(vals, dtype=float)

    if vals.size == 0:
        return np.array([0.0, 1.0], dtype=float)

    if vals.size == 1:
        step = fallback_step if np.isfinite(fallback_step) and fallback_step > 0 else 1.0
        half = 0.5 * step
        return np.array([vals[0] - half, vals[0] + half], dtype=float)

    diffs = np.diff(vals)
    finite_diffs = diffs[np.isfinite(diffs) & (diffs != 0)]
    step = np.median(np.abs(finite_diffs)) if finite_diffs.size > 0 else fallback_step
    if not np.isfinite(step) or step <= 0:
        step = 1.0

    edges = np.empty(vals.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (vals[:-1] + vals[1:])
    edges[0] = vals[0] - 0.5 * (vals[1] - vals[0])
    edges[-1] = vals[-1] + 0.5 * (vals[-1] - vals[-2])

    if not np.isfinite(edges[0]):
        edges[0] = vals[0] - 0.5 * step
    if not np.isfinite(edges[-1]):
        edges[-1] = vals[-1] + 0.5 * step

    return edges


def robust_nonnegative_norm(data: np.ndarray, percentile: float = 99.0):
    finite = np.asarray(data, dtype=float)
    finite = finite[np.isfinite(finite)]

    if finite.size == 0:
        return None

    vmax = np.percentile(finite, percentile)
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = np.max(finite)

    if not np.isfinite(vmax) or vmax <= 0:
        return None

    return Normalize(vmin=0.0, vmax=vmax)


def compute_average_on_window(gathered, *, freq_low: float, freq_high: float, args):
    freq_grid = build_common_grid(gathered, freq_low, freq_high)

    rel_low, rel_high = map(float, args.relativerange)
    if args.normalize == "absolute":
        norm_low = freq_low
        norm_high = freq_high
    else:
        norm_low = max(freq_low, rel_low)
        norm_high = min(freq_high, rel_high)
        if norm_high <= norm_low:
            raise ValueError(
                "Relative normalization range does not overlap the selected frequency window"
            )

    normalized_rows = []
    accepted_rows = []
    for spec in gathered:
        amp_interp = interp_amplitude(spec["freq"], spec["amp"], freq_grid)
        amp_norm = normalize_spectrum(
            freq_grid,
            amp_interp,
            norm_low=norm_low,
            norm_high=norm_high,
        )
        if amp_norm is None:
            warn(
                f"Skipping dataset '{spec['dataset']}' pair_id {spec['pair_id']} "
                f"because normalization denominator in [{norm_low:.6g}, {norm_high:.6g}] Hz was zero or near-zero"
            )
            continue
        normalized_rows.append(amp_norm)
        accepted_rows.append(spec)

    if len(normalized_rows) == 0:
        raise ValueError("All contributors were rejected during normalization")

    normalized_stack = np.vstack(normalized_rows)
    avg_amp = average_spectra(normalized_stack, args.average_domain)

    return {
        "freq_grid": freq_grid,
        "avg_amp": avg_amp,
        "norm_low": float(norm_low),
        "norm_high": float(norm_high),
        "accepted_rows": accepted_rows,
        "normalized_stack": normalized_stack,
    }


def compute_reference_image_norm(gathered, args):
    min_supported = max(float(spec["freq"][0]) for spec in gathered)
    max_supported = min(float(spec["freq"][-1]) for spec in gathered)

    reference = compute_average_on_window(
        gathered,
        freq_low=min_supported,
        freq_high=max_supported,
        args=args,
    )
    return robust_nonnegative_norm(reference["avg_amp"])


def plot_average(freq_grid, avg_amp, args, n_contributors, n_datasets, image_norm=None):
    fig, ax = plt.subplots(figsize=(12, 5))

    if args.full_image:
        cmap_str = COLORMAPS[args.cm]
        x_cols = 64
        image_2d = np.tile(avg_amp[:, None], (1, x_cols))

        if args.plot_scale == "log":
            image_plot = 20.0 * np.log10(image_2d + np.finfo(float).eps)
            z_label = "Amplitude (dB)"
            pcm_norm = None
        else:
            image_plot = image_2d
            z_label = "Normalized Amplitude"
            pcm_norm = image_norm if image_norm is not None else robust_nonnegative_norm(image_plot)

        x_edges = np.linspace(0.0, 1.0, x_cols + 1)
        fallback_step = float(np.median(np.diff(freq_grid))) if freq_grid.size > 1 else 1.0
        y_edges = centers_to_edges(freq_grid, fallback_step=fallback_step)

        pcm = ax.pcolormesh(
            x_edges,
            y_edges,
            image_plot,
            shading="flat",
            cmap=cmap_str,
            norm=pcm_norm,
        )
        fig.colorbar(pcm, ax=ax, label=z_label)
        ax.set_xlabel("Arbitrary X")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(freq_grid[0], freq_grid[-1])
        ax.set_xticks([])
    else:
        if args.plot_scale == "log":
            ax.semilogy(freq_grid, avg_amp, linewidth=1.5)
        else:
            ax.plot(freq_grid, avg_amp, linewidth=1.5)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Normalized Amplitude")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(freq_grid[0], freq_grid[-1])

    if args.title:
        title = args.title
    else:
        norm_desc = args.normalize
        if args.normalize == "relative":
            norm_desc = f"relative [{args.relativerange[0]}, {args.relativerange[1]}] Hz"
        title = (
            f"Average FFT | contributors={n_contributors} | datasets={n_datasets} | "
            f"avg={args.average_domain} | norm={norm_desc}"
        )
    ax.set_title(title)

    plt.tight_layout()

    if args.save is not None:
        save_dir = os.path.dirname(os.path.abspath(args.save))
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(args.save, dpi=300)
        print(f"Plot saved to: {args.save}")

    plt.show()


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.highest_freq is not None and args.lowest_freq is not None:
        if args.highest_freq <= args.lowest_freq:
            print("Error: --highest_freq must be greater than --lowest_freq", file=sys.stderr)
            return 1

    rel_low, rel_high = map(float, args.relativerange)
    if rel_high <= rel_low:
        print("Error: --relativerange STOP_HZ must be greater than START_HZ", file=sys.stderr)
        return 1

    try:
        config = load_config(args.config_json)
        gathered = gather_requested_spectra(config, args)

        if len(gathered) == 0:
            raise ValueError("No spectra were accepted from the provided config")

        freq_low, freq_high = choose_frequency_window(gathered, args)
        result = compute_average_on_window(
            gathered,
            freq_low=freq_low,
            freq_high=freq_high,
            args=args,
        )

        image_norm = None
        if args.full_image and args.plot_scale == "linear":
            image_norm = compute_reference_image_norm(gathered, args)

        accepted_rows = result["accepted_rows"]
        freq_grid = result["freq_grid"]
        avg_amp = result["avg_amp"]
        norm_low = result["norm_low"]
        norm_high = result["norm_high"]

        print(f"Accepted contributors: {len(accepted_rows)}")
        print(f"Unique datasets: {len({row['dataset'] for row in accepted_rows})}")
        print(f"Frequency window: [{freq_low:.6f}, {freq_high:.6f}] Hz")
        print(f"Normalization window: [{norm_low:.6f}, {norm_high:.6f}] Hz")
        print(f"Common grid points: {len(freq_grid)}")
        print(f"Display mode: {'full image' if args.full_image else 'curve'}")
        if args.full_image and args.plot_scale == "linear":
            print("Image color scale reference: full implicit frequency window")

        plot_average(
            freq_grid,
            avg_amp,
            args,
            n_contributors=len(accepted_rows),
            n_datasets=len({row['dataset'] for row in accepted_rows}),
            image_norm=image_norm,
        )
        return 0

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


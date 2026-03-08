#!/usr/bin/env python3
"""
localize_peaks.py

Analyze the spatial distribution of specific frequency peaks across the chain.

This tool loads a list of global peak frequencies and a dataset configuration.
For every bond (pair) in the configuration, it calculates the normalized amplitude
of each target peak. It then generates a multi-panel plot showing Amplitude vs.
Bond Index for each peak frequency to visualize localization.

Usage:
    python3 localize_peaks.py datasets.json peaks.csv --normalize relative
"""

import argparse
import csv
import json
import sys
import os
from collections import defaultdict, OrderedDict

import matplotlib.pyplot as plt
import numpy as np

# Import shared tools
from fft_tools import compute_one_sided_fft, preprocess_signal
from io_tracks import load_fft_dataset, resolve_track2_track3_paths


def load_config(path: str):
    """
    Load and validate the dataset configuration JSON.
    Same logic as avg_fft.py.
    """
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f, object_pairs_hook=OrderedDict)

    if not isinstance(cfg, dict) or len(cfg) == 0:
        raise ValueError("Top-level JSON must be a non-empty object keyed by dataset stem")

    return cfg


def load_peaks(path: str) -> list[float]:
    """
    Load target peak frequencies from a CSV file.
    Expected format: 1.2, 4.5, 6.7 (one line or multiple lines)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Peaks file not found: {path}")

    peaks = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            for cell in row:
                try:
                    val = float(cell.strip())
                    if val > 0:
                        peaks.append(val)
                except ValueError:
                    continue

    if not peaks:
        raise ValueError("No valid floating point peaks found in CSV.")

    return peaks


def get_peak_amplitude(
    freqs: np.ndarray,
    amps: np.ndarray,
    target: float,
    width: float
) -> tuple[float, bool]:
    """
    Find the maximum amplitude in the window [target - width, target + width].
    Returns (amplitude, found_boolean).
    """
    # Define window
    f_min = target - width
    f_max = target + width

    # Check if window is completely outside the frequency range
    if f_max < freqs[0] or f_min > freqs[-1]:
        return 0.0, False

    mask = (freqs >= f_min) & (freqs <= f_max)

    # If no points in window (too sparse or window too small), try nearest neighbor
    if not np.any(mask):
        # Determine if we are "close enough" to a point
        idx = (np.abs(freqs - target)).argmin()
        nearest_f = freqs[idx]
        if abs(nearest_f - target) <= width:
            return float(amps[idx]), True
        else:
            return 0.0, False

    return float(np.max(amps[mask])), True


def compute_normalization_factor(
    freq: np.ndarray,
    amp: np.ndarray,
    mode: str,
    rel_range: tuple[float, float]
) -> float:
    """
    Compute the denominator for normalization.
    """
    if mode == 'absolute':
        # Integral over entire range
        val = np.trapezoid(amp, freq)
    elif mode == 'relative':
        start, stop = rel_range
        mask = (freq >= start) & (freq <= stop)
        if np.sum(mask) < 2:
            return 0.0  # Invalid
        val = np.trapezoid(amp[mask], freq[mask])
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return float(val)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Verify peak localization by plotting amplitude vs bond index."
    )

    parser.add_argument("config_json", help="Dataset-selection JSON file.")
    parser.add_argument("peaks_csv", help="CSV file containing peak frequencies.")

    # Normalization args (same as avg_fft)
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

    # Peak search args
    parser.add_argument(
        "--search_width",
        type=float,
        default=0.25,
        help="Frequency window (+/- Hz) to search for local peak max. Default: 0.25"
    )

    # Standard processing args
    parser.add_argument("--longest", action="store_true", help="Use longest valid segment.")
    parser.add_argument(
        "--handlenan",
        action="store_true",
        help="Use finite-sample handling before resampling."
    )
    parser.add_argument("--allowduplicatebonds", action="store_true", help="Average duplicate bond IDs.")

    # Plot filtering
    parser.add_argument(
        "--disableplot",
        type=int,
        nargs='+',
        default=[],
        help="List of peak indices (0-based) to skip plotting."
    )
    parser.add_argument(
        "--onlyenableplots",
        type=int,
        nargs='+',
        default=None,
        help="List of peak indices (0-based) to exclusively plot."
    )

    # Plot output
    parser.add_argument("--save", default=None, help="Save figure to path.")

    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Load inputs
    try:
        config = load_config(args.config_json)
        peaks = load_peaks(args.peaks_csv)
    except Exception as e:
        print(f"Error loading inputs: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(peaks)} peaks: {peaks}")

    # 2. Determine active peaks for visualization
    # Indices correspond to the order in the CSV
    all_indices = set(range(len(peaks)))

    if args.onlyenableplots is not None:
        active_indices = set(args.onlyenableplots)
    else:
        active_indices = all_indices

    # Remove disabled
    active_indices = active_indices - set(args.disableplot)

    # Filter valid range, then sort by peak frequency greatest -> least
    active_indices = sorted(
        [i for i in active_indices if i in all_indices],
        key=lambda i: peaks[i],
        reverse=True,
    )

    if not active_indices:
        print("No peaks selected for plotting.", file=sys.stderr)
        sys.exit(0)

    print(f"Active peak indices (sorted high->low): {active_indices}")

    # 3. Data Gathering
    # Structure: data_store[peak_idx][bond_id] = [amp1, amp2, ...]
    data_store = defaultdict(lambda: defaultdict(list))

    # To track duplicates if not allowed
    seen_pairs_global = set()

    for dataset_name, entry in config.items():
        if not entry.get("include", False):
            continue

        discards = set(entry.get("discards", []))
        config_pair_ids = entry.get("pair_ids", [])

        try:
            dataset = load_fft_dataset(dataset=dataset_name)
        except Exception as e:
            print(f"Skipping {dataset_name}: {e}", file=sys.stderr)
            continue

        T = dataset["frame_times_s"]
        spacing = dataset["spacing_matrix"]
        n_pairs = spacing.shape[1]

        # Calculate logical remaining indices
        remaining_local_indices = [i for i in range(n_pairs) if i not in discards]

        if len(remaining_local_indices) != len(config_pair_ids):
            print(f"Warning: {dataset_name} has length mismatch (Local valid: {len(remaining_local_indices)}, Config: {len(config_pair_ids)}). Skipping.", file=sys.stderr)
            continue

        print(f"Processing {dataset_name}...")

        for local_idx, bond_id in zip(remaining_local_indices, config_pair_ids):

            # Check for duplicates
            if not args.allowduplicatebonds and bond_id in seen_pairs_global:
                continue

            seen_pairs_global.add(bond_id)

            # Process Signal
            y = spacing[:, local_idx]
            processed, msg = preprocess_signal(T, y, longest=args.longest, handlenan=args.handlenan)

            if processed is None:
                print(f"  Warning: Bond {bond_id} in {dataset_name} invalid signal ({msg}). Recording 0s.")
                for p_idx in active_indices:
                    data_store[p_idx][bond_id].append(0.0)
                continue

            # Compute FFT
            fft_res = compute_one_sided_fft(processed.y, processed.dt)
            freqs = fft_res.freq
            amps = fft_res.amplitude

            # Normalize
            norm_factor = compute_normalization_factor(
                freqs, amps,
                args.normalize,
                tuple(args.relativerange)
            )

            if norm_factor <= 1e-12:
                print(f"  Warning: Bond {bond_id} in {dataset_name} has zero norm factor. Recording 0s.")
                for p_idx in active_indices:
                    data_store[p_idx][bond_id].append(0.0)
                continue

            normalized_amps = amps / norm_factor

            # Find Peaks
            for p_idx in active_indices:
                target_freq = peaks[p_idx]
                val, found = get_peak_amplitude(freqs, normalized_amps, target_freq, args.search_width)

                if not found:
                    print(f"  Warning: Could not find peak {p_idx} ({target_freq} Hz) for Bond {bond_id}. Recording 0.")

                data_store[p_idx][bond_id].append(val)

    # 4. Plotting
    num_plots = len(active_indices)
    if num_plots == 0:
        print("No data collected.")
        return

    ncols = 1
    nrows = num_plots
    figsize = (10, 3 * num_plots)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=figsize,
        sharex=True,
        constrained_layout=True
    )
    if num_plots == 1:
        axes = [axes]

    # Collect global bond range so every subplot uses the same x ticks every 1 bond
    all_bond_ids = sorted({
        bond_id
        for p_idx in active_indices
        for bond_id in data_store[p_idx].keys()
    })

    if all_bond_ids:
        xmin = min(all_bond_ids)
        xmax = max(all_bond_ids)
        xticks = np.arange(xmin, xmax + 1, 1)
    else:
        xticks = np.array([])

    for i, p_idx in enumerate(active_indices):
        ax = axes[i]
        freq = peaks[p_idx]
        bond_data = data_store[p_idx]

        if not bond_data:
            ax.text(0.5, 0.5, "No Data", transform=ax.transAxes, ha='center')
            ax.set_title(f"Peak {p_idx}: {freq} Hz")
            if xticks.size > 0:
                ax.set_xticks(xticks)
                ax.tick_params(axis='x', labelbottom=True)
            continue

        sorted_bonds = sorted(bond_data.keys())
        x_vals = sorted_bonds
        y_vals = []
        y_errs = []

        for b_id in sorted_bonds:
            vals = bond_data[b_id]
            mean_val = np.mean(vals)
            y_vals.append(mean_val)
            if len(vals) > 1:
                y_errs.append(np.std(vals))
            else:
                y_errs.append(0.0)

        ax.plot(
            x_vals, y_vals,
            marker='o',
            linestyle='-',
            linewidth=1.5,
            markersize=5,
            label='Mean Amp'
        )

        if any(e > 0 for e in y_errs):
            ax.fill_between(
                x_vals,
                np.array(y_vals) - np.array(y_errs),
                np.array(y_vals) + np.array(y_errs),
                alpha=0.2
            )

        ax.set_title(f"Peak {p_idx}: {freq} Hz")
        ax.set_ylabel("Norm. Amplitude")
        ax.grid(True, alpha=0.3)

        # Force every subplot to show x ticks and labels every 1 bond
        if xticks.size > 0:
            ax.set_xticks(xticks)
            ax.set_xlim(xticks[0], xticks[-1])
            ax.tick_params(axis='x', labelbottom=True)

    for ax in axes:
        ax.set_xlabel("Bond Index")

    norm_desc = args.normalize
    if args.normalize == 'relative':
        norm_desc += f" [{args.relativerange[0]}-{args.relativerange[1]} Hz]"

    fig.suptitle(f"Peak Localization Analysis | Norm: {norm_desc}", fontsize=14)

    if args.save:
        plt.savefig(args.save, dpi=300)
        print(f"Saved plot to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()

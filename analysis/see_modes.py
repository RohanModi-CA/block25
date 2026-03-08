#!/usr/bin/env python3
"""
mode_shapes.py

Extract phase-resolved mode shapes from Track2 site trajectories and optionally
compare the derived bond-space mode to the directly measured Track3 spacing mode.

This version uses a CSV file of peak frequencies, matching the workflow used by
localize_peaks.py.

Examples:
    python3 mode_shapes.py IMG_0545 peaks.csv --compare-track3

    python3 mode_shapes.py --config datasets.json peaks.csv --compare-track3

    python3 mode_shapes.py IMG_0545 peaks.csv --onlyenableplots 0 2 4
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import msgpack

from fft_tools import (
    align_complex_mode_shape,
    complex_mode_to_bonds,
    compute_one_sided_fft_complex,
    get_complex_at_frequency,
    preprocess_signal,
)
from io_tracks import load_fft_dataset, resolve_track2_track3_paths


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f, object_pairs_hook=OrderedDict)

    if not isinstance(cfg, dict) or len(cfg) == 0:
        raise ValueError("Top-level JSON must be a non-empty object keyed by dataset stem")
    return cfg


def load_peaks(path: str) -> list[float]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Peaks file not found: {path}")

    peaks = []
    with open(path, "r", encoding="utf-8") as f:
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract phase-resolved mode shapes from Track2 site trajectories. "
            "Optionally compare bond-space modes derived from Track2 with directly "
            "measured Track3 spacing modes."
        )
    )

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "dataset",
        nargs="?",
        help="Single dataset stem, e.g. IMG_0545",
    )
    src.add_argument(
        "--config",
        help="Dataset-selection JSON file. Each included dataset is processed separately.",
    )

    parser.add_argument(
        "peaks_csv",
        help="CSV file containing peak frequencies.",
    )

    parser.add_argument(
        "--search_width",
        type=float,
        default=0.25,
        help="Window (+/- Hz) for peak picking. Default: 0.25",
    )
    parser.add_argument(
        "--pick_strategy",
        choices=["max_amplitude", "nearest"],
        default="max_amplitude",
        help="How to choose the FFT bin inside the search window.",
    )

    parser.add_argument("--longest", action="store_true", help="Use longest valid segment.")
    parser.add_argument(
        "--handlenan",
        action="store_true",
        help="Use finite-sample handling before resampling.",
    )

    parser.add_argument(
        "--compare-track3",
        action="store_true",
        help="Also extract bond-space coefficients directly from Track3 spacing signals.",
    )

    parser.add_argument(
        "--normalize",
        choices=["none", "max"],
        default="max",
        help="Normalize each mode shape before plotting. Default: max",
    )

    parser.add_argument(
        "--plot",
        choices=["real", "magnitude", "both"],
        default="both",
        help="Which site/bond representation to plot. Default: both",
    )

    parser.add_argument(
        "--disableplot",
        type=int,
        nargs="+",
        default=[],
        help="List of peak indices (0-based) to skip plotting.",
    )
    parser.add_argument(
        "--onlyenableplots",
        type=int,
        nargs="+",
        default=None,
        help="List of peak indices (0-based) to exclusively plot.",
    )

    parser.add_argument(
        "--save_dir",
        default=None,
        help="Optional directory to save figures instead of only displaying them.",
    )

    return parser


def normalize_mode(mode: np.ndarray, how: str) -> np.ndarray:
    mode = np.asarray(mode)
    if how == "none":
        return mode
    if how == "max":
        scale = np.max(np.abs(mode))
        if scale > 0:
            return mode / scale
        return mode
    raise ValueError(f"Unknown normalization mode: {how}")


def gather_datasets_from_config(path: str) -> list[str]:
    cfg = load_config(path)
    out = []
    for dataset_name, entry in cfg.items():
        if entry.get("include", False):
            out.append(dataset_name)
    if not out:
        raise ValueError("No datasets with include=true in config.")
    return out


def select_active_peak_indices(peaks: list[float], args) -> list[int]:
    all_indices = set(range(len(peaks)))

    if args.onlyenableplots is not None:
        active_indices = set(args.onlyenableplots)
    else:
        active_indices = all_indices

    active_indices = active_indices - set(args.disableplot)

    active_indices = sorted(
        [i for i in active_indices if i in all_indices],
        key=lambda i: peaks[i],
        reverse=True,
    )

    if not active_indices:
        raise ValueError("No peaks selected for plotting.")

    return active_indices


def load_track2_xpositions(track2_path: str) -> np.ndarray:
    if not os.path.exists(track2_path):
        raise FileNotFoundError(f"Track2 file not found: {track2_path}")

    with open(track2_path, "rb") as f:
        t2 = msgpack.unpackb(f.read(), raw=False)

    try:
        x_positions = np.asarray(t2["xPositions"], dtype=float)
    except KeyError as exc:
        raise KeyError("Track2 is missing key 'xPositions'") from exc

    if x_positions.ndim != 2:
        raise ValueError(f"xPositions must be 2D. Got shape {x_positions.shape}")

    return x_positions


def extract_site_mode(dataset_name: str, target_freq: float, args):
    dataset = load_fft_dataset(dataset=dataset_name)
    track2_path, track3_path = resolve_track2_track3_paths(dataset=dataset_name)

    T = dataset["frame_times_s"]
    x_positions = load_track2_xpositions(track2_path)

    _, n_sites = x_positions.shape
    site_mode = np.zeros(n_sites, dtype=complex)
    selected_freqs = []
    failures = []

    for site_idx in range(n_sites):
        y = x_positions[:, site_idx]
        processed, _ = preprocess_signal(
            T,
            y,
            longest=args.longest,
            handlenan=args.handlenan,
        )

        if processed is None:
            failures.append(site_idx)
            site_mode[site_idx] = 0.0 + 0.0j
            continue

        fft_res = compute_one_sided_fft_complex(processed.y, processed.dt)
        picked = get_complex_at_frequency(
            fft_res.freq,
            fft_res.spectrum,
            target_freq,
            width=args.search_width,
            strategy=args.pick_strategy,
        )

        if picked.found:
            site_mode[site_idx] = picked.complex_value
            selected_freqs.append(picked.selected_freq)
        else:
            failures.append(site_idx)
            site_mode[site_idx] = 0.0 + 0.0j

    median_selected = float(np.median(selected_freqs)) if selected_freqs else float("nan")

    return {
        "dataset_name": dataset_name,
        "track2_path": track2_path,
        "track3_path": track3_path,
        "site_mode": site_mode,
        "selected_freq": median_selected,
        "n_sites": n_sites,
        "failures": failures,
    }


def extract_track3_bond_mode(dataset_name: str, target_freq: float, args):
    dataset = load_fft_dataset(dataset=dataset_name)
    T = dataset["frame_times_s"]
    spacing = dataset["spacing_matrix"]

    _, n_bonds = spacing.shape
    bond_mode = np.zeros(n_bonds, dtype=complex)
    selected_freqs = []
    failures = []

    for bond_idx in range(n_bonds):
        y = spacing[:, bond_idx]
        processed, _ = preprocess_signal(
            T,
            y,
            longest=args.longest,
            handlenan=args.handlenan,
        )

        if processed is None:
            failures.append(bond_idx)
            bond_mode[bond_idx] = 0.0 + 0.0j
            continue

        fft_res = compute_one_sided_fft_complex(processed.y, processed.dt)
        picked = get_complex_at_frequency(
            fft_res.freq,
            fft_res.spectrum,
            target_freq,
            width=args.search_width,
            strategy=args.pick_strategy,
        )

        if picked.found:
            bond_mode[bond_idx] = picked.complex_value
            selected_freqs.append(picked.selected_freq)
        else:
            failures.append(bond_idx)
            bond_mode[bond_idx] = 0.0 + 0.0j

    median_selected = float(np.median(selected_freqs)) if selected_freqs else float("nan")

    return {
        "bond_mode": bond_mode,
        "selected_freq": median_selected,
        "n_bonds": n_bonds,
        "failures": failures,
    }


def phase_align_pair(site_mode: np.ndarray, bond_mode: np.ndarray | None):
    site_aligned = align_complex_mode_shape(site_mode)

    if bond_mode is None:
        return site_aligned, None

    mags = np.abs(site_mode)
    if not np.any(mags > 0):
        return site_aligned, bond_mode

    ref_idx = int(np.argmax(mags))
    ref_phase = np.angle(site_mode[ref_idx])
    bond_aligned = bond_mode * np.exp(-1j * ref_phase)
    return site_aligned, bond_aligned


def summarize_mode_difference(a: np.ndarray, b: np.ndarray) -> str:
    if a.shape != b.shape:
        return "shape mismatch"

    a_norm = normalize_mode(a, "max")
    b_norm = normalize_mode(b, "max")

    diff = np.linalg.norm(a_norm - b_norm)
    denom = max(np.linalg.norm(a_norm), 1e-12)
    rel = diff / denom
    return f"relative L2 diff = {rel:.4f}"


def plot_one_mode(
    dataset_name: str,
    peak_idx: int,
    target_freq: float,
    selected_site_freq: float,
    site_mode: np.ndarray,
    derived_bond_mode: np.ndarray,
    *,
    selected_track3_freq: float | None,
    track3_bond_mode: np.ndarray | None,
    args,
):
    site_mode = normalize_mode(site_mode, args.normalize)
    derived_bond_mode = normalize_mode(derived_bond_mode, args.normalize)
    if track3_bond_mode is not None:
        track3_bond_mode = normalize_mode(track3_bond_mode, args.normalize)

    ncols = 2 if args.plot == "both" else 1
    fig, axes = plt.subplots(
        nrows=2,
        ncols=ncols,
        figsize=(6 * ncols, 7.0),
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)

    site_x = np.arange(1, len(site_mode) + 1)
    bond_x = np.arange(1, len(derived_bond_mode) + 1)

    def plot_component(ax, x, vals, title, ylabel, style):
        if style == "real":
            ax.plot(x, np.real(vals), "o-", linewidth=1.5, label="Derived from Track2")
        elif style == "magnitude":
            ax.plot(x, np.abs(vals), "o-", linewidth=1.5, label="Derived from Track2")
        else:
            raise ValueError(style)
        ax.set_title(title)
        ax.set_xlabel("Index")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    if args.plot in ("real", "both"):
        col = 0
        plot_component(
            axes[0, col],
            site_x,
            site_mode,
            f"Site mode (real)\nPeak {peak_idx}: target {target_freq:.4f} Hz | picked {selected_site_freq:.4f} Hz",
            "Amplitude",
            "real",
        )
        plot_component(
            axes[1, col],
            bond_x,
            derived_bond_mode,
            "Bond mode (real)",
            "Amplitude",
            "real",
        )
        if track3_bond_mode is not None:
            axes[1, col].plot(
                bond_x,
                np.real(track3_bond_mode),
                "x--",
                linewidth=1.2,
                label=f"Track3 direct | picked {selected_track3_freq:.4f} Hz",
            )
            axes[1, col].legend()

    if args.plot in ("magnitude", "both"):
        col = 1 if args.plot == "both" else 0
        plot_component(
            axes[0, col],
            site_x,
            site_mode,
            f"Site mode (|.|)\nPeak {peak_idx}: target {target_freq:.4f} Hz | picked {selected_site_freq:.4f} Hz",
            "Magnitude",
            "magnitude",
        )
        plot_component(
            axes[1, col],
            bond_x,
            derived_bond_mode,
            "Bond mode (|.|)",
            "Magnitude",
            "magnitude",
        )
        if track3_bond_mode is not None:
            axes[1, col].plot(
                bond_x,
                np.abs(track3_bond_mode),
                "x--",
                linewidth=1.2,
                label=f"Track3 direct | picked {selected_track3_freq:.4f} Hz",
            )
            axes[1, col].legend()

    fig.suptitle(f"{dataset_name} | Phase-resolved mode shape")

    if args.save_dir is not None:
        os.makedirs(args.save_dir, exist_ok=True)
        safe_freq = f"{target_freq:.4f}".replace(".", "p")
        save_path = os.path.join(args.save_dir, f"{dataset_name}_peak{peak_idx}_{safe_freq}Hz.png")
        fig.savefig(save_path, dpi=300)
        print(f"Saved: {save_path}")

    plt.show()


def main():
    parser = build_parser()
    args = parser.parse_args()

    try:
        peaks = load_peaks(args.peaks_csv)
        active_peak_indices = select_active_peak_indices(peaks, args)
        dataset_names = [args.dataset] if args.dataset is not None else gather_datasets_from_config(args.config)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(peaks)} peaks: {peaks}")
    print(f"Active peak indices (sorted high->low): {active_peak_indices}")

    for dataset_name in dataset_names:
        print(f"\n=== Dataset: {dataset_name} ===")

        for peak_idx in active_peak_indices:
            target_freq = peaks[peak_idx]

            try:
                site_res = extract_site_mode(dataset_name, target_freq, args)
            except Exception as exc:
                print(f"  Error extracting site mode for peak {peak_idx} ({target_freq:.6f} Hz): {exc}", file=sys.stderr)
                continue

            site_mode = site_res["site_mode"]
            derived_bond_mode = complex_mode_to_bonds(site_mode)

            track3_mode = None
            track3_freq = None

            if args.compare_track3:
                try:
                    track3_res = extract_track3_bond_mode(dataset_name, target_freq, args)
                    track3_mode = track3_res["bond_mode"]
                    track3_freq = track3_res["selected_freq"]
                except Exception as exc:
                    print(f"  Warning: Track3 comparison failed for peak {peak_idx} ({target_freq:.6f} Hz): {exc}", file=sys.stderr)

            site_mode, track3_mode = phase_align_pair(site_mode, track3_mode)
            derived_bond_mode = complex_mode_to_bonds(site_mode)

            print(f"  Peak {peak_idx}: target {target_freq:.6f} Hz")
            print(f"    Site picked freq (median across sites): {site_res['selected_freq']:.6f} Hz")
            print(f"    Site failures: {len(site_res['failures'])} / {site_res['n_sites']}")

            if track3_mode is not None:
                print(f"    Track3 picked freq (median across bonds): {track3_freq:.6f} Hz")
                print(f"    Bond comparison: {summarize_mode_difference(derived_bond_mode, track3_mode)}")

            plot_one_mode(
                dataset_name=dataset_name,
                peak_idx=peak_idx,
                target_freq=target_freq,
                selected_site_freq=site_res["selected_freq"],
                site_mode=site_mode,
                derived_bond_mode=derived_bond_mode,
                selected_track3_freq=track3_freq,
                track3_bond_mode=track3_mode,
                args=args,
            )


if __name__ == "__main__":
    main()

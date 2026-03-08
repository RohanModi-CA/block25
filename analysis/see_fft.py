#!/usr/bin/env python3
"""
see_fft.py

Visualization CLI for full FFT and sliding FFT of Track3 block-spacing data.
Signal processing lives in fft_tools.py. Track loading / path resolution lives in
io_tracks.py.
"""

from __future__ import annotations

import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from fft_tools import compute_complex_spectrogram, compute_one_sided_fft, preprocess_signal
from io_tracks import load_fft_dataset


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


def centers_to_edges(vals, fallback_step=1.0):
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


def robust_nonnegative_norm(data, percentile=99.0):
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


def set_panel_message(args, pair_idx, label, ax_fft=None, ax_spec=None, message=""):
    title = f"Pair {pair_idx} ({label}) - {message}"
    if args.only != "sliding" and ax_fft is not None:
        ax_fft.set_title(title)
        ax_fft.set_xlabel("Frequency (Hz)")
        ax_fft.set_ylabel("Amplitude")
    if args.only != "fft" and ax_spec is not None:
        ax_spec.set_title(title)
        ax_spec.set_xlabel("Time (s)" if not args.full_image else "Arbitrary X")
        ax_spec.set_ylabel("Frequency (Hz)")


def plot_fft_panel(ax, fft_result, *, pair_idx, label, proc_msg, nyquist, args):
    freq = fft_result.freq
    amp = fft_result.amplitude

    if args.log:
        ax.semilogy(freq, amp, linewidth=1)
        positive_vals = amp[amp > 0]
        if positive_vals.size > 0:
            ymin = np.percentile(positive_vals, 0.1)
            ymax = np.max(positive_vals)
            ymin *= 0.7
            ymax *= 1.3
            if np.isfinite(ymin) and np.isfinite(ymax) and ymin > 0 and ymax > ymin:
                ax.set_ylim(ymin, ymax)
    else:
        ax.plot(freq, amp, linewidth=1)

    fft_max = args.highest_freq if args.highest_freq is not None else nyquist
    fft_min = args.lowest_freq if args.lowest_freq is not None else 0.0

    ax.set_title(f"Pair {pair_idx} ({label}) FFT | {proc_msg}")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")
    ax.set_xlim(max(0.0, fft_min), min(fft_max, nyquist))
    ax.grid(True, alpha=0.3)


def plot_full_image_panel(fig, ax, fft_result, *, pair_idx, label, y_min_plot, y_max_plot, args, cmap_str, eps):
    freq_mask = (fft_result.freq >= y_min_plot) & (fft_result.freq <= y_max_plot)
    if not np.any(freq_mask):
        ax.set_title(f"Pair {pair_idx} ({label}) - no frequencies in range")
        ax.set_xlabel("Arbitrary X")
        ax.set_ylabel("Frequency (Hz)")
        return

    fft_freqs_img = fft_result.freq[freq_mask]
    fft_vals_img = fft_result.amplitude[freq_mask]

    x_cols = 64
    image_2d = np.tile(fft_vals_img[:, None], (1, x_cols))

    if args.slidinglog:
        S_plot = 20.0 * np.log10(image_2d + eps)
        z_unit = "dB"
        pcm_norm = None
    else:
        S_plot = image_2d
        z_unit = "Amplitude"
        pcm_norm = robust_nonnegative_norm(S_plot)

    x_edges = np.linspace(0.0, 1.0, x_cols + 1)
    if fft_freqs_img.size > 1:
        df_med = float(np.median(np.diff(fft_freqs_img)))
    else:
        df_med = max(1e-6, y_max_plot - y_min_plot)
    y_edges = centers_to_edges(fft_freqs_img, fallback_step=df_med)

    pcm = ax.pcolormesh(
        x_edges,
        y_edges,
        S_plot,
        shading="flat",
        cmap=cmap_str,
        norm=pcm_norm,
    )

    ax.set_ylim(y_min_plot, y_max_plot)
    ax.set_xlim(0.0, 1.0)
    ax.set_title(f"Pair {pair_idx} ({label}) Full FFT Image")
    ax.set_xlabel("Arbitrary X")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xticks([])

    fig.colorbar(pcm, ax=ax, label=z_unit)


def plot_spectrogram_panel(
    fig,
    ax,
    spec_result,
    *,
    pair_idx,
    label,
    t_start,
    y_min_plot,
    y_max_plot,
    args,
    cmap_str,
    eps,
):
    f_spec = spec_result.f
    t_spec_global = spec_result.t + t_start
    mag = np.abs(spec_result.S_complex)

    if args.slidinglog:
        S_plot = 20.0 * np.log10(mag + eps)
        z_unit = "dB"
        pcm_norm = None
    else:
        S_plot = mag ** 2
        z_unit = "Power"
        pcm_norm = robust_nonnegative_norm(S_plot)

    t_step = (t_spec_global[1] - t_spec_global[0]) if t_spec_global.size > 1 else args.sliding_len_s
    f_step = (f_spec[1] - f_spec[0]) if f_spec.size > 1 else max(1e-6, y_max_plot - y_min_plot)

    t_edges = centers_to_edges(t_spec_global, fallback_step=t_step)
    f_edges = centers_to_edges(f_spec, fallback_step=f_step)

    pcm = ax.pcolormesh(
        t_edges,
        f_edges,
        S_plot,
        shading="flat",
        cmap=cmap_str,
        norm=pcm_norm,
    )

    ax.set_ylim(y_min_plot, y_max_plot)
    ax.set_title(f"Pair {pair_idx} ({label}) Sliding FFT")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")

    if args.timeinterval_s is not None:
        start, stop = args.timeinterval_s
        ax.set_xlim(start, stop)

    fig.colorbar(pcm, ax=ax, label=z_unit)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "FFT and sliding FFT of block spacing. Normally use a dataset name "
            "corresponding to ../track/data/<dataset>/."
        )
    )

    parser.add_argument(
        "dataset",
        nargs="?",
        help=(
            "Dataset stem, e.g. 'foo', which resolves to "
            "../track/data/foo/track2_x_permanence.msgpack and "
            "../track/data/foo/track3_analysis.msgpack."
        ),
    )
    parser.add_argument(
        "-i",
        "--input",
        default=None,
        help="Explicit Track2 path. Overrides the dataset-derived Track2 path.",
    )
    parser.add_argument(
        "--track3",
        default=None,
        help="Explicit Track3 path. Overrides the dataset-derived Track3 path.",
    )

    parser.add_argument("--log", action="store_true")

    parser.add_argument(
        "--slidinglog",
        dest="slidinglog",
        action="store_true",
        help="Use dB display for the sliding panel (default).",
    )
    parser.add_argument(
        "--slidinglinear",
        dest="slidinglog",
        action="store_false",
        help="Use linear power display for the sliding panel.",
    )
    parser.set_defaults(slidinglog=True)

    parser.add_argument("--sliding_len_s", type=float, default=20.0)
    parser.add_argument("--longest", action="store_true")
    parser.add_argument(
        "--handlenan",
        action="store_true",
        help=(
            "Use finite-sample handling: discard invalid time/value samples first, "
            "then resample uniformly in time. By default the legacy NaN behavior "
            "is preserved."
        ),
    )

    parser.add_argument("--cm", type=int, choices=range(1, 11), default=6)

    parser.add_argument("--highest_freq", type=float, default=None)
    parser.add_argument("--lowest_freq", type=float, default=None)
    parser.add_argument("--highest_freq_sliding", type=float, default=None)
    parser.add_argument("--lowest_freq_sliding", type=float, default=None)

    parser.add_argument("--timeinterval_s", nargs=2, type=float, metavar=("START", "STOP"))
    parser.add_argument("--only", choices=["fft", "sliding"], default=None)

    parser.add_argument(
        "--full_image",
        action="store_true",
        help=(
            "Replace the sliding FFT panel with a 2D color image built from the "
            "full FFT spectrum. X axis is arbitrary, Y axis is frequency."
        ),
    )

    parser.add_argument(
        "--disable",
        type=int,
        action="append",
        default=[],
        help="Disable pair index (0-based). Can be used multiple times.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    try:
        dataset = load_fft_dataset(
            dataset=args.dataset,
            track2_path=args.input,
            track3_path=args.track3,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    track2_path = dataset["track2_path"]
    track3_path = dataset["track3_path"]
    T = dataset["frame_times_s"]
    spacing = dataset["spacing_matrix"]
    pair_labels = dataset["pair_labels"]

    print(f"Track2: {track2_path}")
    print(f"Track3: {track3_path}")

    n_frames, n_pairs = spacing.shape
    disabled = set(args.disable)
    active_pairs = [i for i in range(n_pairs) if i not in disabled]

    if len(active_pairs) == 0:
        print("All pairs disabled.")
        return

    dt_global = np.diff(T)
    dt_global = dt_global[np.isfinite(dt_global) & (dt_global > 0)]
    if dt_global.size == 0:
        print("Error: could not compute valid sampling interval from frameTimes_s.", file=sys.stderr)
        sys.exit(1)

    median_dt_global = float(np.median(dt_global))
    Fs_global = 1.0 / median_dt_global
    nyquist_global = 0.5 * Fs_global

    print(f"Approx global sampling rate: {Fs_global:.4f} Hz | Approx Nyquist: {nyquist_global:.4f} Hz")
    print(f"Sliding mode: {'log (dB)' if args.slidinglog else 'linear power'}")

    ncols = 1 if args.only in ("fft", "sliding") else 2
    fig, axes = plt.subplots(
        nrows=len(active_pairs),
        ncols=ncols,
        figsize=(14, 3.5 * len(active_pairs)),
        constrained_layout=True,
    )

    if ncols == 1:
        axes = np.atleast_1d(axes)
    else:
        axes = np.atleast_2d(axes)

    cmap_str = COLORMAPS[args.cm]
    eps = np.finfo(float).eps

    for row_idx, pair_idx in enumerate(active_pairs):
        if args.only == "fft":
            ax_fft = axes[row_idx]
            ax_spec = None
        elif args.only == "sliding":
            ax_fft = None
            ax_spec = axes[row_idx]
        else:
            ax_fft = axes[row_idx][0]
            ax_spec = axes[row_idx][1]

        label = str(pair_labels[pair_idx]).upper()
        y = spacing[:, pair_idx]

        processed, error_msg = preprocess_signal(
            T,
            y,
            longest=args.longest,
            handlenan=args.handlenan,
        )
        if processed is None:
            set_panel_message(args, pair_idx, label, ax_fft=ax_fft, ax_spec=ax_spec, message=error_msg)
            continue

        fft_result = compute_one_sided_fft(processed.y, processed.dt)

        if args.only != "sliding":
            plot_fft_panel(
                ax_fft,
                fft_result,
                pair_idx=pair_idx,
                label=label,
                proc_msg=processed.proc_msg,
                nyquist=processed.nyquist,
                args=args,
            )

        if args.only == "fft":
            continue

        spec_max = args.highest_freq_sliding if args.highest_freq_sliding is not None else processed.nyquist
        spec_min = args.lowest_freq_sliding if args.lowest_freq_sliding is not None else 0.01
        y_min_plot = max(0.01, spec_min)
        y_max_plot = min(spec_max, processed.nyquist)

        if y_max_plot <= y_min_plot:
            set_panel_message(args, pair_idx, label, ax_fft=ax_fft, ax_spec=ax_spec, message="invalid frequency range")
            continue

        if args.full_image:
            plot_full_image_panel(
                fig,
                ax_spec,
                fft_result,
                pair_idx=pair_idx,
                label=label,
                y_min_plot=y_min_plot,
                y_max_plot=y_max_plot,
                args=args,
                cmap_str=cmap_str,
                eps=eps,
            )
            continue

        spec_result = compute_complex_spectrogram(processed.y, processed.Fs, args.sliding_len_s)
        if spec_result is None:
            set_panel_message(args, pair_idx, label, ax_fft=ax_fft, ax_spec=ax_spec, message="window too short")
            continue

        plot_spectrogram_panel(
            fig,
            ax_spec,
            spec_result,
            pair_idx=pair_idx,
            label=label,
            t_start=processed.t[0],
            y_min_plot=y_min_plot,
            y_max_plot=y_max_plot,
            args=args,
            cmap_str=cmap_str,
            eps=eps,
        )

    plt.show()


if __name__ == "__main__":
    main()

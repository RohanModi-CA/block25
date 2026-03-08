import os
import sys
import argparse
import msgpack
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from matplotlib.colors import TwoSlopeNorm


# =============================================================================
# DATASET / PATH CONVENTION
# =============================================================================
#
# This script lives in:
#
#     analysis/see_fft.py
#
# while the tracking pipeline outputs live under:
#
#     ../track/data/
#
# The convention is that the tracking pipeline is anchored by the original
# Track1 output file:
#
#     ../track/data/<dataset>.msgpack
#
# and all downstream outputs for that dataset are stored in:
#
#     ../track/data/<dataset>/
#
# with standard filenames:
#
#     ../track/data/<dataset>/track2_verified.msgpack
#     ../track/data/<dataset>/track2_x_permanence.msgpack
#     ../track/data/<dataset>/track3_analysis.msgpack
#
# Therefore, from the analysis directory, the normal usage is:
#
#     python3 see_fft.py foo
#
# which resolves to:
#
#     track2 -> ../track/data/foo/track2_x_permanence.msgpack
#     track3 -> ../track/data/foo/track3_analysis.msgpack
#
# You can also override either path explicitly:
#
#     python3 see_fft.py foo -i ../track/data/foo/track2_x_permanence.msgpack
#     python3 see_fft.py foo --track3 ../track/data/foo/track3_analysis.msgpack
#
# or bypass the dataset convention entirely:
#
#     python3 see_fft.py -i /path/to/track2_x_permanence.msgpack \
#                        --track3 /path/to/track3_analysis.msgpack
#
# Resolution rules:
#
# 1. Explicit flags win.
# 2. Otherwise, if DATASET is given, paths are derived from:
#        ../track/data/<dataset>/
# 3. If neither is enough to determine both files, the script errors out.
#
# Note:
# - DATASET should be the dataset stem, e.g. "foo", not "foo.msgpack".
# - This script uses Track2 for frame times and Track3 for spacing data.
# =============================================================================


STANDARD_TRACK_ROOT = os.path.join("..", "track", "data")
STANDARD_TRACK2_NAME = "track2_x_permanence.msgpack"
STANDARD_TRACK3_NAME = "track3_analysis.msgpack"

COLORMAPS = {
    1: 'viridis', 2: 'plasma', 3: 'inferno', 4: 'magma',
    5: 'cividis', 6: 'turbo', 7: 'jet', 8: 'nipy_spectral',
    9: 'ocean', 10: 'cubehelix'
}


def dataset_dir_from_name(dataset: str) -> str:
    return os.path.join(STANDARD_TRACK_ROOT, dataset)


def default_track2_path(dataset: str) -> str:
    return os.path.join(dataset_dir_from_name(dataset), STANDARD_TRACK2_NAME)


def default_track3_path(dataset: str) -> str:
    return os.path.join(dataset_dir_from_name(dataset), STANDARD_TRACK3_NAME)


def resolve_paths(args):
    if args.dataset is None and args.input is None and args.track3 is None:
        raise ValueError("Provide either DATASET or both -i/--input and --track3")

    track2_path = args.input
    track3_path = args.track3

    if args.dataset is not None:
        if track2_path is None:
            track2_path = default_track2_path(args.dataset)
        if track3_path is None:
            track3_path = default_track3_path(args.dataset)

    if track2_path is None or track3_path is None:
        raise ValueError("Could not determine both Track2 and Track3 paths")

    return track2_path, track3_path


def get_longest_valid_segment(arr):
    mask = ~np.isnan(arr)
    bounded = np.concatenate(([False], mask, [False]))
    diffs = np.diff(bounded.astype(int))

    starts = np.where(diffs == 1)[0]
    stops = np.where(diffs == -1)[0]

    if len(starts) == 0:
        return 0, 0

    lengths = stops - starts
    best_idx = np.argmax(lengths)

    return starts[best_idx], stops[best_idx]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "FFT and sliding FFT of block spacing. "
            "Normally use a dataset name corresponding to ../track/data/<dataset>/."
        )
    )

    parser.add_argument(
        "dataset",
        nargs="?",
        help=(
            "Dataset stem, e.g. 'foo', which resolves to "
            "../track/data/foo/track2_x_permanence.msgpack and "
            "../track/data/foo/track3_analysis.msgpack."
        )
    )
    parser.add_argument(
        "-i", "--input",
        default=None,
        help="Explicit Track2 path. Overrides the dataset-derived Track2 path."
    )
    parser.add_argument(
        "--track3",
        default=None,
        help="Explicit Track3 path. Overrides the dataset-derived Track3 path."
    )

    parser.add_argument('--log', action='store_true')
    parser.add_argument('--slidinglog', action='store_true')

    parser.add_argument('--sliding_len_s', type=float, default=20.0)

    parser.add_argument('--longest', action='store_true')

    parser.add_argument('--cm', type=int, choices=range(1, 11), default=6)

    parser.add_argument('--highest_freq', type=float, default=None)
    parser.add_argument('--lowest_freq', type=float, default=None)
    parser.add_argument('--highest_freq_sliding', type=float, default=None)
    parser.add_argument('--lowest_freq_sliding', type=float, default=None)

    parser.add_argument('--timeinterval_s', nargs=2, type=float,
                        metavar=('START', 'STOP'))

    parser.add_argument('--only', choices=['fft', 'sliding'], default=None)

    parser.add_argument(
        '--full_image',
        action='store_true',
        help=(
            "Replace the sliding FFT panel with a 2D color image built from the "
            "full FFT spectrum. X axis is arbitrary, Y axis is frequency."
        )
    )

    parser.add_argument('--disable', type=int, action='append', default=[],
                        help="Disable pair index (0-based). Can be used multiple times.")

    return parser.parse_args()


def main():
    args = parse_args()

    try:
        track2_path, track3_path = resolve_paths(args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(track2_path):
        print(f"Error: {track2_path} not found.", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(track3_path):
        print(f"Error: {track3_path} not found. Run track3_differences.py first.", file=sys.stderr)
        sys.exit(1)

    print(f"Track2: {track2_path}")
    print(f"Track3: {track3_path}")

    with open(track2_path, 'rb') as f:
        t2 = msgpack.unpackb(f.read())

    with open(track3_path, 'rb') as f:
        t3 = msgpack.unpackb(f.read())

    T = np.array(t2['frameTimes_s'], dtype=float)
    spacing = np.array(t3['spacing_matrix'], dtype=float)
    pair_labels = t3['pair_colors']

    if spacing.ndim != 2:
        print(f"Error: spacing_matrix must be 2D. Got shape {spacing.shape}", file=sys.stderr)
        sys.exit(1)

    n_frames, n_pairs = spacing.shape

    if T.shape[0] != n_frames:
        print(
            f"Error: time vector length ({T.shape[0]}) does not match spacing rows ({n_frames}).",
            file=sys.stderr
        )
        sys.exit(1)

    disabled = set(args.disable)
    active_pairs = [i for i in range(n_pairs) if i not in disabled]

    if len(active_pairs) == 0:
        print("All pairs disabled.")
        return

    dt = np.diff(T)
    dt = dt[np.isfinite(dt)]

    if dt.size == 0:
        print("Error: could not compute valid sampling interval from frameTimes_s.", file=sys.stderr)
        sys.exit(1)

    median_dt = np.nanmedian(dt)
    Fs = 1.0 / median_dt
    nyquist = Fs / 2.0

    print(f"Sampling rate: {Fs:.2f} Hz | Nyquist: {nyquist:.2f} Hz")

    if args.only in ('fft', 'sliding'):
        ncols = 1
    else:
        ncols = 2

    fig, axes = plt.subplots(
        nrows=len(active_pairs),
        ncols=ncols,
        figsize=(14, 3.5 * len(active_pairs)),
        constrained_layout=True
    )

    if len(active_pairs) == 1:
        axes = np.array([axes])

    cmap_str = COLORMAPS[args.cm]

    for row_idx, pair_idx in enumerate(active_pairs):

        if args.only == 'fft':
            ax_fft = axes[row_idx]
        elif args.only == 'sliding':
            ax_spec = axes[row_idx]
        else:
            ax_fft = axes[row_idx][0]
            ax_spec = axes[row_idx][1]

        label = str(pair_labels[pair_idx]).upper()
        y = spacing[:, pair_idx]

        if args.longest:

            start, stop = get_longest_valid_segment(y)

            if stop - start < 10:
                if args.only != 'sliding':
                    ax_fft.set_title(f"Pair {pair_idx} ({label}) - no valid data")
                if args.only != 'fft':
                    ax_spec.set_title(f"Pair {pair_idx} ({label}) - no valid data")
                continue

            y_proc = y[start:stop]
            t_proc = T[start:stop]
            proc_msg = f"Longest Seq: {stop - start}"

        else:

            nan_mask = np.isnan(y)

            if np.any(nan_mask):

                valid_idx = np.where(~nan_mask)[0]

                if len(valid_idx) < 2:
                    if args.only != 'sliding':
                        ax_fft.set_title(f"Pair {pair_idx} ({label}) - insufficient valid data")
                    if args.only != 'fft':
                        ax_spec.set_title(f"Pair {pair_idx} ({label}) - insufficient valid data")
                    continue

                y_proc = np.interp(np.arange(n_frames), valid_idx, y[valid_idx])

            else:
                y_proc = y.copy()

            t_proc = T
            proc_msg = "Interpolated"

        y_proc = y_proc - np.mean(y_proc)
        N = len(y_proc)

        fft_vals = np.abs(np.fft.rfft(y_proc))
        fft_freqs = np.fft.rfftfreq(N, d=median_dt)

        if args.only != 'sliding':

            if args.log:
                ax_fft.semilogy(fft_freqs, fft_vals, linewidth=1)

                positive_vals = fft_vals[fft_vals > 0]
                if len(positive_vals) > 0:
                    ymin = np.percentile(positive_vals, 0.1)
                    ymax = np.max(positive_vals)

                    ymin *= 0.7
                    ymax *= 1.3
                    ax_fft.set_ylim(ymin, ymax)

            else:
                ax_fft.plot(fft_freqs, fft_vals, linewidth=1)

            ax_fft.set_title(f"Pair {pair_idx} ({label}) FFT | {proc_msg}")
            ax_fft.set_xlabel("Frequency (Hz)")
            ax_fft.set_ylabel("Amplitude")
            ax_fft.grid(True, alpha=0.3)

            fft_max = args.highest_freq if args.highest_freq else nyquist
            fft_min = args.lowest_freq if args.lowest_freq else 0
            ax_fft.set_xlim(max(0, fft_min), min(fft_max, nyquist))

        if args.only != 'fft':

            spec_max = args.highest_freq_sliding if args.highest_freq_sliding else nyquist
            spec_min = args.lowest_freq_sliding if args.lowest_freq_sliding else 0.01
            y_min_plot = max(0.01, spec_min)
            y_max_plot = min(spec_max, nyquist)

            if args.full_image:
                freq_mask = (fft_freqs >= y_min_plot) & (fft_freqs <= y_max_plot)

                if not np.any(freq_mask):
                    ax_spec.set_title(f"Pair {pair_idx} ({label}) - no frequencies in range")
                    ax_spec.set_xlabel("Arbitrary X")
                    ax_spec.set_ylabel("Frequency (Hz)")
                    continue

                fft_freqs_img = fft_freqs[freq_mask]
                fft_vals_img = fft_vals[freq_mask]

                # Replicate the 1D FFT spectrum into a narrow 2D image.
                x_cols = 64
                image_2d = np.tile(fft_vals_img[:, None], (1, x_cols))

                if args.slidinglog:
                    S_plot = 10 * np.log10(image_2d + 1e-12)
                    z_unit = "dB"
                    pcm_norm = None
                else:
                    S_plot = image_2d
                    z_unit = "Amplitude"
                    vmax = np.percentile(S_plot, 98)
                    if np.isfinite(vmax) and vmax > 0:
                        pcm_norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
                    else:
                        pcm_norm = None

                x_edges = np.linspace(0.0, 1.0, x_cols + 1)

                if len(fft_freqs_img) > 1:
                    df = np.diff(fft_freqs_img)
                    df_med = np.median(df)
                else:
                    df_med = max(1e-6, y_max_plot - y_min_plot)

                y_edges = np.empty(len(fft_freqs_img) + 1, dtype=float)
                y_edges[1:-1] = 0.5 * (fft_freqs_img[:-1] + fft_freqs_img[1:])
                y_edges[0] = fft_freqs_img[0] - 0.5 * df_med
                y_edges[-1] = fft_freqs_img[-1] + 0.5 * df_med

                pcm = ax_spec.pcolormesh(
                    x_edges,
                    y_edges,
                    S_plot,
                    shading='auto',
                    cmap=cmap_str,
                    norm=pcm_norm
                )

                ax_spec.set_ylim(y_min_plot, y_max_plot)
                ax_spec.set_xlim(0.0, 1.0)
                ax_spec.set_title(f"Pair {pair_idx} ({label}) Full FFT Image")
                ax_spec.set_xlabel("Arbitrary X")
                ax_spec.set_ylabel("Frequency (Hz)")
                ax_spec.set_xticks([])

                fig.colorbar(pcm, ax=ax_spec, label=z_unit)

            else:
                nperseg = int(args.sliding_len_s * Fs)
                nperseg = min(nperseg, N)

                if nperseg < 4:
                    ax_spec.set_title(f"Pair {pair_idx} ({label}) - window too short")
                    ax_spec.set_xlabel("Time (s)")
                    ax_spec.set_ylabel("Frequency (Hz)")
                    continue

                noverlap = int(nperseg * 0.8)

                f_spec, t_spec_local, Sxx = signal.spectrogram(
                    y_proc,
                    fs=Fs,
                    nperseg=nperseg,
                    noverlap=noverlap,
                    scaling='spectrum'
                )

                t_spec_global = t_spec_local + t_proc[0]

                if args.slidinglog:
                    S_plot = 10 * np.log10(Sxx + 1e-12)
                    z_unit = "dB"
                    pcm_norm = None
                else:
                    S_plot = Sxx
                    z_unit = "Power"
                    vmax = np.percentile(S_plot, 98)
                    if np.isfinite(vmax) and vmax > 0:
                        pcm_norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
                    else:
                        pcm_norm = None

                pcm = ax_spec.pcolormesh(
                    t_spec_global,
                    f_spec,
                    S_plot,
                    shading='gouraud',
                    cmap=cmap_str,
                    norm=pcm_norm
                )

                ax_spec.set_ylim(y_min_plot, y_max_plot)

                ax_spec.set_title(f"Pair {pair_idx} ({label}) Sliding FFT")
                ax_spec.set_xlabel("Time (s)")
                ax_spec.set_ylabel("Frequency (Hz)")

                if args.timeinterval_s is not None:
                    start, stop = args.timeinterval_s
                    ax_spec.set_xlim(start, stop)

                fig.colorbar(pcm, ax=ax_spec, label=z_unit)

    plt.show()


if __name__ == '__main__':
    main()

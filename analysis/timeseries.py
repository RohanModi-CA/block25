import os
import sys
import argparse
import msgpack
import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# DATASET / PATH CONVENTION
# =============================================================================
#
# This script lives in:
#
#     analysis/timeseries.py
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
#     python3 timeseries.py foo
#
# which resolves to:
#
#     track2 -> ../track/data/foo/track2_x_permanence.msgpack
#     track3 -> ../track/data/foo/track3_analysis.msgpack
#
# You can also override either path explicitly:
#
#     python3 timeseries.py foo -i ../track/data/foo/track2_x_permanence.msgpack
#     python3 timeseries.py foo --track3 ../track/data/foo/track3_analysis.msgpack
#
# or bypass the dataset convention entirely:
#
#     python3 timeseries.py -i /path/to/track2_x_permanence.msgpack \
#                           --track3 /path/to/track3_analysis.msgpack
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot stacked block-spacing time series from Track2 and Track3 outputs. "
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
    parser.add_argument(
        "--save",
        default=None,
        help="Optional path to save the plot image instead of only displaying it."
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

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

    # -------------------------------------------------------------------------
    # LOAD DATA
    # -------------------------------------------------------------------------
    print("Loading analysis data...")
    print(f"  Track2: {track2_path}")
    print(f"  Track3: {track3_path}")

    with open(track2_path, 'rb') as f:
        t2_data = msgpack.unpackb(f.read())

    with open(track3_path, 'rb') as f:
        t3_data = msgpack.unpackb(f.read())

    time = np.array(t2_data['frameTimes_s'], dtype=float)
    spacing = np.array(t3_data['spacing_matrix'], dtype=float)
    pair_labels = t3_data['pair_colors']

    if spacing.ndim != 2:
        print(f"Error: spacing_matrix must be 2D. Got shape {spacing.shape}", file=sys.stderr)
        sys.exit(1)

    n_frames, n_pairs = spacing.shape

    if time.shape[0] != n_frames:
        print(
            f"Error: time vector length ({time.shape[0]}) does not match spacing rows ({n_frames}).",
            file=sys.stderr
        )
        sys.exit(1)

    if n_pairs == 0:
        print("No block pairs found to plot.")
        return

    # -------------------------------------------------------------------------
    # PLOTTING (STACKED SUBPLOTS)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(
        n_pairs,
        1,
        figsize=(12, 2 * n_pairs),
        sharex=True
    )

    if n_pairs == 1:
        axes = [axes]

    color_map = {'rg': '#e74c3c', 'gr': '#2ecc71'}
    global_avg = np.nanmean(spacing)

    print(f"Generating {n_pairs} stacked subplots...")

    for i in range(n_pairs):
        ax = axes[i]

        label = str(pair_labels[i]).lower()
        color = color_map.get(label, '#3498db')

        ax.plot(time, spacing[:, i], color=color, linewidth=1)

        local_avg = np.nanmean(spacing[:, i])

        ax.set_ylabel("Dist (px)")
        ax.set_xlabel("Time (seconds)")
        ax.set_title(f"Pair {i}: {label.upper()}  |  Avg: {local_avg:.1f}px")

        ax.grid(True, linestyle='--', alpha=0.3)
        ax.tick_params(labelbottom=True)

    # -------------------------------------------------------------------------
    # COMMON LABELING
    # -------------------------------------------------------------------------
    axes[-1].set_xlabel("Time (seconds)")

    fig.suptitle(
        f"Block Spacing Time-Series (Global Avg: {global_avg:.1f}px)",
        fontsize=14
    )

    plt.xlim(time[0], time[-1])
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    # -------------------------------------------------------------------------
    # SAVE / DISPLAY
    # -------------------------------------------------------------------------
    if args.save is not None:
        save_dir = os.path.dirname(os.path.abspath(args.save))
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(args.save, dpi=300)
        print(f"Plot saved to: {args.save}")

    plt.show()


if __name__ == '__main__':
    main()

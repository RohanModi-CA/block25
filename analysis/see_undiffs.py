import os
import sys
import argparse
import msgpack
import numpy as np
import matplotlib.pyplot as plt
import cv2


# =============================================================================
# DATASET / PATH CONVENTION
# =============================================================================
#
# This script lives in:
#
#     analysis/see_undiffs.py
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
#     python3 see_undiffs.py foo
#
# which resolves to:
#
#     track2 -> ../track/data/foo/track2_x_permanence.msgpack
#
# You can also override the Track2 path explicitly:
#
#     python3 see_undiffs.py foo -i ../track/data/foo/track2_x_permanence.msgpack
#
# or bypass the dataset convention entirely:
#
#     python3 see_undiffs.py -i /path/to/track2_x_permanence.msgpack
#
# Resolution rules:
#
# 1. Explicit flags win.
# 2. Otherwise, if DATASET is given, paths are derived from:
#        ../track/data/<dataset>/
# 3. If neither is enough to determine the Track2 file, the script errors out.
#
# Note:
# - DATASET should be the dataset stem, e.g. "foo", not "foo.msgpack".
# - This script visualizes the undifferenced persistent trajectories from Track2.
# - Optionally, it can also show a sparse strip of sampled video frames from the
#   original video path recorded in Track2. If the video is missing or cannot be
#   opened, the script degrades gracefully and still shows the other panels.
# =============================================================================


STANDARD_TRACK_ROOT = os.path.join("..", "track", "data")
STANDARD_TRACK2_NAME = "track2_x_permanence.msgpack"


def dataset_dir_from_name(dataset: str) -> str:
    return os.path.join(STANDARD_TRACK_ROOT, dataset)


def default_track2_path(dataset: str) -> str:
    return os.path.join(dataset_dir_from_name(dataset), STANDARD_TRACK2_NAME)


def resolve_path(args):
    if args.dataset is None and args.input is None:
        raise ValueError("Provide either DATASET or -i/--input")

    if args.input is not None:
        return args.input

    return default_track2_path(args.dataset)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize undifferenced persistent block trajectories from Track2. "
            "Normally use a dataset name corresponding to ../track/data/<dataset>/."
        )
    )

    parser.add_argument(
        "dataset",
        nargs="?",
        help=(
            "Dataset stem, e.g. 'foo', which resolves to "
            "../track/data/foo/track2_x_permanence.msgpack."
        )
    )
    parser.add_argument(
        "-i", "--input",
        default=None,
        help="Explicit Track2 path. Overrides the dataset-derived Track2 path."
    )
    parser.add_argument(
        "--save",
        default=None,
        help="Optional path to save the plot image instead of only displaying it."
    )
    parser.add_argument(
        "--framestrip",
        action="store_true",
        help="Show a third panel with sparse sampled frames from the original video."
    )
    parser.add_argument(
        "--nframestrip",
        type=int,
        default=8,
        help="Number of sampled video frames to show when --framestrip is enabled. Default: 8"
    )

    return parser


def read_msgpack(path: str):
    with open(path, "rb") as f:
        return msgpack.unpackb(f.read())


def sample_video_frames(video_path: str, sample_times: np.ndarray):
    """
    Returns:
        images: list[np.ndarray] in RGB
        valid_times: list[float]
        status_msg: str or None
    """
    if not video_path:
        return [], [], "No originalVideoPath found in Track2."

    if not os.path.exists(video_path):
        return [], [], f"Original video not found: {video_path}"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], [], f"Could not open original video: {video_path}"

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if not np.isfinite(fps) or fps <= 0:
        cap.release()
        return [], [], f"Could not determine FPS for video: {video_path}"

    if frame_count <= 0:
        cap.release()
        return [], [], f"Video appears to have zero frames: {video_path}"

    images = []
    valid_times = []

    for t in sample_times:
        frame_idx = int(round(t * fps))
        frame_idx = max(0, min(frame_idx, frame_count - 1))

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame_bgr = cap.read()
        if not ret or frame_bgr is None:
            continue

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        images.append(frame_rgb)
        valid_times.append(float(t))

    cap.release()

    if len(images) == 0:
        return [], [], f"Could not decode any sampled frames from video: {video_path}"

    return images, valid_times, None


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.nframestrip <= 0:
        print("Error: --nframestrip must be positive.", file=sys.stderr)
        sys.exit(1)

    try:
        track2_path = resolve_path(args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(track2_path):
        print(f"Error: {track2_path} not found.", file=sys.stderr)
        sys.exit(1)

    print(f"Track2: {track2_path}")

    t2 = read_msgpack(track2_path)

    time = np.array(t2["frameTimes_s"], dtype=float)
    x_positions = np.array(t2["xPositions"], dtype=float)
    block_colors = [str(c).lower() for c in t2["blockColors"]]
    original_video_path = t2.get("originalVideoPath", "")

    if x_positions.ndim != 2:
        print(f"Error: xPositions must be 2D. Got shape {x_positions.shape}", file=sys.stderr)
        sys.exit(1)

    n_frames, n_blocks = x_positions.shape

    if time.shape[0] != n_frames:
        print(
            f"Error: time vector length ({time.shape[0]}) does not match xPositions rows ({n_frames}).",
            file=sys.stderr
        )
        sys.exit(1)

    if n_frames == 0:
        print("Error: no frames found in Track2.", file=sys.stderr)
        sys.exit(1)

    if n_blocks == 0:
        print("No persistent blocks found to plot.")
        sys.exit(0)

    visible_mask = np.isfinite(x_positions)
    visible_counts = np.sum(visible_mask, axis=0)
    nan_fraction = 1.0 - np.mean(visible_mask)

    print(f"Frames: {n_frames}")
    print(f"Persistent blocks: {n_blocks}")
    print(f"Time range: [{time[0]:.6f}, {time[-1]:.6f}] s")
    print(f"Overall NaN fraction: {nan_fraction:.3f}")
    print("Visible samples per block:")
    print("  " + ", ".join(f"{i}:{int(c)}" for i, c in enumerate(visible_counts)))

    bad_order_frames = 0
    for k in range(n_frames):
        row = x_positions[k, :]
        row = row[np.isfinite(row)]
        if row.size >= 2 and np.any(np.diff(row) <= 0):
            bad_order_frames += 1

    print(f"Frames with non-increasing visible x-order: {bad_order_frames}")

    if args.framestrip:
        fig = plt.figure(figsize=(14, 3 * n_blocks / 2 + 6))
        gs = fig.add_gridspec(3, 1, height_ratios=[3.0, 1.2, 1.8], hspace=0.25)
        ax_main = fig.add_subplot(gs[0, 0])
        ax_vis = fig.add_subplot(gs[1, 0], sharex=ax_main)
        ax_strip = fig.add_subplot(gs[2, 0], sharex=ax_main)
    else:
        fig = plt.figure(figsize=(14, 3 * n_blocks / 2 + 3))
        gs = fig.add_gridspec(2, 1, height_ratios=[3.0, 1.2], hspace=0.20)
        ax_main = fig.add_subplot(gs[0, 0])
        ax_vis = fig.add_subplot(gs[1, 0], sharex=ax_main)
        ax_strip = None

    color_map = {"r": "#e74c3c", "g": "#2ecc71"}

    # -------------------------------------------------------------------------
    # TOP PANEL: persistent trajectories
    # -------------------------------------------------------------------------
    for j in range(n_blocks):
        c = block_colors[j] if j < len(block_colors) else "?"
        line_color = color_map.get(c, "#3498db")
        y = x_positions[:, j]

        ax_main.plot(time, y, color=line_color, linewidth=1.5, alpha=0.95)

        valid_idx = np.where(np.isfinite(y))[0]
        if valid_idx.size > 0:
            last_idx = valid_idx[-1]
            ax_main.text(
                time[last_idx],
                y[last_idx],
                f" {j}{c.upper()}",
                color=line_color,
                fontsize=9,
                va="center",
                ha="left",
                clip_on=True
            )

    ax_main.set_title("Persistent Block X-Position Trajectories (Undifferenced)")
    ax_main.set_ylabel("X Position (px)")
    ax_main.grid(True, linestyle="--", alpha=0.3)

    # -------------------------------------------------------------------------
    # MIDDLE PANEL: visibility mask
    # -------------------------------------------------------------------------
    vis_img = visible_mask.astype(float)
    ax_vis.imshow(
        vis_img,
        aspect="auto",
        interpolation="nearest",
        origin="upper",
        extent=[time[0], time[-1], n_blocks - 0.5, -0.5],
        cmap="gray_r",
        vmin=0,
        vmax=1
    )
    ax_vis.set_title("Visibility Mask (white = visible, black = hidden)")
    ax_vis.set_ylabel("Block Index")
    ax_vis.set_xlabel("Time (seconds)")

    # -------------------------------------------------------------------------
    # OPTIONAL BOTTOM PANEL: sparse video frame strip
    # -------------------------------------------------------------------------
    if ax_strip is not None:
        sample_times = np.linspace(time[0], time[-1], args.nframestrip)

        images, valid_times, status_msg = sample_video_frames(original_video_path, sample_times)

        if len(images) == 0:
            ax_strip.text(
                0.5, 0.5,
                "Frame strip unavailable\n" + (status_msg if status_msg else ""),
                ha="center", va="center",
                transform=ax_strip.transAxes,
                fontsize=11
            )
            ax_strip.set_xticks([])
            ax_strip.set_yticks([])
            ax_strip.set_title("Sampled Video Frames")
            print(status_msg if status_msg else "Frame strip unavailable.")
        else:
            ax_strip.set_title("Sampled Video Frames")
            ax_strip.set_yticks([])
            ax_strip.set_xlabel("Time (seconds)")

            y0, y1 = 0.0, 1.0
            for img, t in zip(images, valid_times):
                h, w = img.shape[:2]
                aspect = w / max(h, 1)

                if len(valid_times) == 1:
                    dt_half = 0.5 * (time[-1] - time[0]) if time[-1] > time[0] else 0.5
                else:
                    sample_step = np.median(np.diff(valid_times))
                    dt_half = 0.45 * sample_step

                x0 = t - dt_half
                x1 = t + dt_half

                ax_strip.imshow(
                    img,
                    extent=[x0, x1, y0, y1],
                    aspect="auto",
                    interpolation="nearest"
                )
                ax_strip.text(
                    t, y1 + 0.02, f"{t:.2f}s",
                    ha="center", va="bottom", fontsize=8
                )

                ax_main.axvline(t, color="k", alpha=0.12, linewidth=1)
                ax_vis.axvline(t, color="k", alpha=0.12, linewidth=1)

            ax_strip.set_xlim(time[0], time[-1])
            ax_strip.set_ylim(0, 1.15)

    fig.suptitle(
        f"Undifferenced Track2 Verification View | Blocks: {n_blocks} | Frames: {n_frames}",
        fontsize=14
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    if args.save is not None:
        save_dir = os.path.dirname(os.path.abspath(args.save))
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(args.save, dpi=300)
        print(f"Plot saved to: {args.save}")

    plt.show()


if __name__ == "__main__":
    main()

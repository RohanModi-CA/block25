import os
import sys
import argparse
import numpy as np
import msgpack
from dataclasses import asdict

from tracking_classes import Track2XPermanence, Track3Analysis


STANDARD_TRACK2_NAME = "track2_x_permanence.msgpack"
STANDARD_TRACK3_NAME = "track3_analysis.msgpack"


def load_track2(path: str) -> Track2XPermanence:
    with open(path, 'rb') as f:
        data = msgpack.unpackb(f.read())
    return Track2XPermanence(**data)


def save_track3(path: str, t3: Track3Analysis) -> None:
    with open(path, 'wb') as f:
        f.write(msgpack.packb(asdict(t3)))


def log(msg: str, quiet: bool = False) -> None:
    if not quiet:
        print(msg)


def dataset_dir_from_anchor(anchor_path: str) -> str:
    anchor_abs = os.path.abspath(anchor_path)
    anchor_dir = os.path.dirname(anchor_abs)
    anchor_base = os.path.splitext(os.path.basename(anchor_abs))[0]
    return os.path.join(anchor_dir, anchor_base)


def default_input_path_from_anchor(anchor_path: str) -> str:
    return os.path.join(dataset_dir_from_anchor(anchor_path), STANDARD_TRACK2_NAME)


def default_output_path_from_anchor(anchor_path: str) -> str:
    return os.path.join(dataset_dir_from_anchor(anchor_path), STANDARD_TRACK3_NAME)


def resolve_paths(args):
    if args.dataset is None and args.input is None:
        raise ValueError("Provide either DATASET or -i/--input")

    if args.dataset is not None:
        dataset_anchor = args.dataset
        default_input = default_input_path_from_anchor(dataset_anchor)
        default_output = default_output_path_from_anchor(dataset_anchor)
    else:
        dataset_anchor = None
        default_input = None
        default_output = None

    input_file_path = args.input if args.input is not None else default_input
    output_file_path = args.output if args.output is not None else default_output

    if input_file_path is None:
        raise ValueError("Could not determine input path")
    if output_file_path is None:
        raise ValueError("Could not determine output path")

    return dataset_anchor, input_file_path, output_file_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute Track3 analysis from a Track2 permanence MessagePack file. "
            "You can either pass a DATASET anchor path like data/foo.msgpack, "
            "or use -i/--input and -o/--output explicitly. "
            "This produces neighboring block spacing, per-block velocity, and time deltas. "
            "You can optionally crop the analysis to a time window."
        )
    )

    parser.add_argument(
        "dataset",
        nargs="?",
        help=(
            "Dataset anchor path, typically the original Track1 file such as data/foo.msgpack. "
            "If provided, default input becomes data/foo/track2_x_permanence.msgpack and "
            "default output becomes data/foo/track3_analysis.msgpack."
        )
    )
    parser.add_argument(
        "-i", "--input",
        default=None,
        help="Explicit Track2 MessagePack input path. Overrides the dataset-derived input path."
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Explicit Track3 MessagePack output path. Overrides the dataset-derived output path."
    )
    parser.add_argument(
        "--tmin",
        type=float,
        default=None,
        help="Minimum time in seconds to keep (inclusive)."
    )
    parser.add_argument(
        "--tmax",
        type=float,
        default=None,
        help="Maximum time in seconds to keep (inclusive)."
    )
    parser.add_argument(
        "--rezero-time",
        action="store_true",
        help="Shift cropped time so the first kept frame has time 0."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run analysis but do not write an output file."
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Reduce console output."
    )

    return parser


def main(args):
    dataset_anchor, input_file_path, output_file_path = resolve_paths(args)

    if not os.path.isfile(input_file_path):
        raise FileNotFoundError(f"Input file not found: {input_file_path}")

    if args.tmin is not None and args.tmax is not None and args.tmin > args.tmax:
        raise ValueError("--tmin must be <= --tmax")

    if not args.dry_run:
        output_dir = os.path.dirname(os.path.abspath(output_file_path))
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    if dataset_anchor is not None and not args.quiet:
        print(f"Dataset anchor: {dataset_anchor}")
        print(f"Dataset directory: {dataset_dir_from_anchor(dataset_anchor)}")

    log(f"Loading {input_file_path}...", args.quiet)
    t2 = load_track2(input_file_path)

    X_full = np.array(t2.xPositions, dtype=float)
    T_full = np.array(t2.frameTimes_s, dtype=float)
    frame_numbers_full = np.array(t2.frameNumbers, dtype=int)
    colors = list(t2.blockColors)

    if X_full.ndim != 2:
        raise ValueError(f"xPositions must be 2D. Got shape {X_full.shape}")

    n_frames_full, n_blocks = X_full.shape
    if T_full.shape[0] != n_frames_full:
        raise ValueError(
            f"frameTimes_s length mismatch. Got {T_full.shape[0]} times for {n_frames_full} rows."
        )
    if frame_numbers_full.shape[0] != n_frames_full:
        raise ValueError(
            f"frameNumbers length mismatch. Got {frame_numbers_full.shape[0]} numbers for {n_frames_full} rows."
        )

    if n_frames_full == 0:
        raise ValueError("Track2 file contains zero frames.")

    original_tmin = float(T_full[0])
    original_tmax = float(T_full[-1])

    # -------------------------------------------------------------------------
    # OPTIONAL TIME CROP
    # -------------------------------------------------------------------------
    keep_mask = np.ones(n_frames_full, dtype=bool)
    if args.tmin is not None:
        keep_mask &= (T_full >= args.tmin)
    if args.tmax is not None:
        keep_mask &= (T_full <= args.tmax)

    kept_indices = np.where(keep_mask)[0]
    if kept_indices.size == 0:
        raise ValueError("Time crop removed all frames.")

    X = X_full[keep_mask, :]
    T = T_full[keep_mask].copy()

    if args.rezero_time:
        T -= T[0]

    n_frames = X.shape[0]
    cropped_tmin = float(T[0])
    cropped_tmax = float(T[-1])

    if n_frames == 0:
        raise ValueError("No frames remain after cropping.")

    log(f"Original data shape: {n_frames_full} frames x {n_blocks} blocks", args.quiet)
    log(f"Cropped data shape:  {n_frames} frames x {n_blocks} blocks", args.quiet)
    log(
        f"Original time range: [{original_tmin:.6f}, {original_tmax:.6f}] s",
        args.quiet
    )
    log(
        f"Cropped time range:  [{cropped_tmin:.6f}, {cropped_tmax:.6f}] s",
        args.quiet
    )

    # -------------------------------------------------------------------------
    # 1. NEIGHBOR DIFFERENCES (SPACING)
    # -------------------------------------------------------------------------
    log("Calculating neighboring block differences...", args.quiet)

    if n_blocks < 2:
        spacing_matrix = np.empty((n_frames, 0), dtype=float)
        pair_strings = []
    else:
        spacing_matrix = X[:, 1:] - X[:, :-1]
        pair_strings = [f"{colors[i]}{colors[i + 1]}" for i in range(n_blocks - 1)]

    # -------------------------------------------------------------------------
    # 2. VELOCITY (FRAME-TO-FRAME DIFFERENCES)
    # -------------------------------------------------------------------------
    log("Calculating block velocities...", args.quiet)

    dt = np.diff(T)
    dt = np.insert(dt, 0, np.nan)

    dx = np.diff(X, axis=0)
    dx = np.insert(dx, 0, np.nan, axis=0)

    velocity_matrix = dx / dt[:, None]

    # -------------------------------------------------------------------------
    # SAVE OUTPUT
    # -------------------------------------------------------------------------
    t3 = Track3Analysis(
        track2_source_path=input_file_path,
        pair_colors=pair_strings,
        spacing_matrix=spacing_matrix.tolist(),
        velocity_matrix=velocity_matrix.tolist(),
        time_deltas=dt.tolist()
    )

    if args.dry_run:
        log("Dry run complete. No file written.", args.quiet)
    else:
        log(f"Saving analysis to {output_file_path}...", args.quiet)
        save_track3(output_file_path, t3)

    # -------------------------------------------------------------------------
    # REPORT
    # -------------------------------------------------------------------------
    valid_spacings = spacing_matrix[~np.isnan(spacing_matrix)]
    avg_spacing = float(np.mean(valid_spacings)) if valid_spacings.size > 0 else 0.0

    valid_vels = velocity_matrix[~np.isnan(velocity_matrix)]
    avg_vel = float(np.mean(valid_vels)) if valid_vels.size > 0 else 0.0

    if not args.quiet:
        print("track3 complete.")
        print(f"  Original frames: {n_frames_full}")
        print(f"  Cropped frames: {n_frames}")
        print(f"  Blocks: {n_blocks}")
        print(f"  Neighbor pairs: {len(pair_strings)}")
        print(f"  Original time range: [{original_tmin:.6f}, {original_tmax:.6f}] s")
        print(f"  Cropped time range: [{cropped_tmin:.6f}, {cropped_tmax:.6f}] s")
        print(f"  Rezero time: {'yes' if args.rezero_time else 'no'}")
        print(f"  Avg Block Spacing: {avg_spacing:.2f} px")
        print(f"  Avg Block Velocity: {avg_vel:.2f} px/s")
        if args.dry_run:
            print("  Output: none (dry run)")
        else:
            print(f"  Output: {output_file_path}")


if __name__ == '__main__':
    parser = build_parser()
    parsed_args = parser.parse_args()

    try:
        main(parsed_args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

import os
import sys
import argparse
import msgpack
import numpy as np
from dataclasses import asdict

from tracking_classes import VideoCentroids, Track2XPermanence


STANDARD_VERIFIED_NAME = "track2_verified.msgpack"
STANDARD_TRACK2_NAME = "track2_x_permanence.msgpack"


# =========================================================================
# HELPER FUNCTIONS
# =========================================================================

def opposite_color(c: str) -> str:
    if c == 'r':
        return 'g'
    if c == 'g':
        return 'r'
    raise ValueError(f"Expected 'r' or 'g', got '{c}'")


def color_seq_to_str(c):
    return " ".join(c)


def load_video_centroids(path: str) -> VideoCentroids:
    with open(path, 'rb') as f:
        data = msgpack.unpackb(f.read())
    return VideoCentroids.from_dict(data)


def save_track2(path: str, t2x: Track2XPermanence) -> None:
    with open(path, 'wb') as f:
        f.write(msgpack.packb(asdict(t2x)))


def log(msg: str, quiet: bool = False) -> None:
    if not quiet:
        print(msg)


def debug_log(msg: str, debug: bool = False, quiet: bool = False) -> None:
    if debug and not quiet:
        print(msg)


def dataset_dir_from_anchor(anchor_path: str) -> str:
    anchor_abs = os.path.abspath(anchor_path)
    anchor_dir = os.path.dirname(anchor_abs)
    anchor_base = os.path.splitext(os.path.basename(anchor_abs))[0]
    return os.path.join(anchor_dir, anchor_base)


def default_input_path_from_anchor(anchor_path: str) -> str:
    return os.path.join(dataset_dir_from_anchor(anchor_path), STANDARD_VERIFIED_NAME)


def default_output_path_from_anchor(anchor_path: str) -> str:
    return os.path.join(dataset_dir_from_anchor(anchor_path), STANDARD_TRACK2_NAME)


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

    input_msgpack_path = args.input if args.input is not None else default_input
    output_msgpack_path = args.output if args.output is not None else default_output

    if input_msgpack_path is None:
        raise ValueError("Could not determine input path")
    if output_msgpack_path is None:
        raise ValueError("Could not determine output path")

    return dataset_anchor, input_msgpack_path, output_msgpack_path


def decide_increase_side(this_x, this_colors, prev_x, block_colors, prev_left_idx, prev_right_idx):
    """
    Decide if +1 count change is entry from LEFT (True) or RIGHT (False).
    Returns (choose_left, left_ok, right_ok, pen_left, pen_right)
    """
    pen_left = np.sum((this_x[1:] - prev_x) ** 2)
    pen_right = np.sum((this_x[:-1] - prev_x) ** 2)

    n_cols = len(block_colors)

    if prev_left_idx > 0:
        expected_left = block_colors[prev_left_idx - 1: prev_right_idx + 1]
    else:
        new_left_col = opposite_color(block_colors[0])
        expected_left = [new_left_col] + block_colors[prev_left_idx: prev_right_idx + 1]
    left_color_ok = (list(this_colors) == list(expected_left))

    if prev_right_idx < n_cols - 1:
        expected_right = block_colors[prev_left_idx: prev_right_idx + 2]
    else:
        new_right_col = opposite_color(block_colors[-1])
        expected_right = block_colors[prev_left_idx: prev_right_idx + 1] + [new_right_col]
    right_color_ok = (list(this_colors) == list(expected_right))

    if left_color_ok and not right_color_ok:
        return True, True, False, pen_left, pen_right
    if right_color_ok and not left_color_ok:
        return False, False, True, pen_left, pen_right
    if left_color_ok and right_color_ok:
        return (pen_left <= pen_right), True, True, pen_left, pen_right
    return True, False, False, pen_left, pen_right


def decide_decrease_side(this_x, this_colors, prev_x, block_colors, prev_left_idx, prev_right_idx):
    """
    Decide if -1 count change is exit from LEFT (True) or RIGHT (False).
    Returns (choose_left, left_ok, right_ok, pen_left, pen_right)
    """
    pen_left_exit = np.sum((this_x - prev_x[1:]) ** 2)
    pen_right_exit = np.sum((this_x - prev_x[:-1]) ** 2)

    expected_left_exit = block_colors[prev_left_idx + 1: prev_right_idx + 1]
    left_color_ok = (list(this_colors) == list(expected_left_exit))

    expected_right_exit = block_colors[prev_left_idx: prev_right_idx]
    right_color_ok = (list(this_colors) == list(expected_right_exit))

    if left_color_ok and not right_color_ok:
        return True, True, False, pen_left_exit, pen_right_exit
    if right_color_ok and not left_color_ok:
        return False, False, True, pen_left_exit, pen_right_exit
    if left_color_ok and right_color_ok:
        return (pen_left_exit <= pen_right_exit), True, True, pen_left_exit, pen_right_exit
    return True, False, False, pen_left_exit, pen_right_exit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the Track2 permanence matrix from a verified Track1 MessagePack file. "
            "You can either pass a DATASET anchor path like data/foo.msgpack, "
            "or use -i/--input and -o/--output explicitly. "
            "The resolved input must have passed verification (vc.passedVerification == True)."
        )
    )

    parser.add_argument(
        'dataset',
        nargs='?',
        help=(
            "Dataset anchor path, typically the original Track1 file such as data/foo.msgpack. "
            "If provided, default input becomes data/foo/track2_verified.msgpack and "
            "default output becomes data/foo/track2_x_permanence.msgpack."
        )
    )
    parser.add_argument(
        '-i', '--input',
        default=None,
        help='Explicit verified Track1 MessagePack input path. Overrides the dataset-derived input path.'
    )
    parser.add_argument(
        '-o', '--output',
        default=None,
        help='Explicit Track2 MessagePack output path. Overrides the dataset-derived output path.'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Run the permanence construction but do not write an output file.'
    )
    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Reduce console output.'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Print per-frame decision details.'
    )

    return parser


def main(args):
    dataset_anchor, input_msgpack_path, output_msgpack_path = resolve_paths(args)

    if not os.path.isfile(input_msgpack_path):
        raise FileNotFoundError(f"Input file not found: {input_msgpack_path}")

    if not args.dry_run:
        output_dir = os.path.dirname(os.path.abspath(output_msgpack_path))
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    if dataset_anchor is not None and not args.quiet:
        print(f"Dataset anchor: {dataset_anchor}")
        print(f"Dataset directory: {dataset_dir_from_anchor(dataset_anchor)}")

    log(f"Loading {input_msgpack_path}...", args.quiet)
    vc = load_video_centroids(input_msgpack_path)

    if not vc.passedVerification:
        raise RuntimeError(
            "Refusing to run: vc.passedVerification is False. Run track2_verifystep.py first."
        )

    frames = vc.frames
    n_frames = len(frames)
    if n_frames == 0:
        raise ValueError("vc.frames is empty.")

    log(f"Loaded {n_frames} frames. Verification check passed.", args.quiet)

    # ---------------------------------------------------------------------
    # FIND FIRST NON-EMPTY FRAME
    # ---------------------------------------------------------------------
    first_non_empty_idx = -1
    for k, f in enumerate(frames):
        if len(f.detections) >= 1:
            first_non_empty_idx = k
            break

    if first_non_empty_idx == -1:
        raise RuntimeError("No frame contains detections. Cannot initialize permanence.")

    init_frame = frames[first_non_empty_idx]
    init_x = np.array([d.x for d in init_frame.detections], dtype=float)
    init_colors = [d.color for d in init_frame.detections]

    log(
        f"First non-empty frame: k={first_non_empty_idx} "
        f"(frame={init_frame.frame_number}, t={init_frame.frame_time_s:.3f}s)",
        args.quiet,
    )

    # ---------------------------------------------------------------------
    # INITIALIZE STATE
    # ---------------------------------------------------------------------
    block_colors = list(init_colors)

    prev_left_idx = 0
    prev_right_idx = len(init_x) - 1

    first_row = [float('nan')] * len(block_colors)
    for i, val in enumerate(init_x):
        first_row[i] = float(val)

    all_positions_matrix = [first_row]

    prev_x = init_x
    prev_num = len(init_x)

    count_same = 0
    count_plus_one = 0
    count_minus_one = 0
    expand_left_count = 0
    expand_right_count = 0
    max_visible_blocks = prev_num

    # ---------------------------------------------------------------------
    # MAIN LOOP
    # ---------------------------------------------------------------------
    log("Processing frames...", args.quiet)

    for k in range(first_non_empty_idx + 1, n_frames):
        curr_frame = frames[k]
        this_x = np.array([d.x for d in curr_frame.detections], dtype=float)
        this_colors = [d.color for d in curr_frame.detections]
        this_num = len(this_x)

        if this_num == 0:
            raise RuntimeError(
                f"Zero-detection frame after tracking start. Frame k={k}, #={curr_frame.frame_number}"
            )

        delta = this_num - prev_num
        max_visible_blocks = max(max_visible_blocks, this_num)

        if abs(delta) > 1:
            raise RuntimeError(f"Count jump > 1 at frame k={k}. Prev={prev_num}, Curr={this_num}")

        curr_left_idx = -1
        curr_right_idx = -1

        if delta == 0:
            count_same += 1
            curr_left_idx = prev_left_idx
            curr_right_idx = prev_right_idx
            debug_log(
                f"k={k}: delta=0 | keeping interval [{curr_left_idx}, {curr_right_idx}]",
                args.debug,
                args.quiet,
            )

        elif delta == 1:
            count_plus_one += 1
            choose_left, left_ok, right_ok, pen_l, pen_r = decide_increase_side(
                this_x, this_colors, prev_x, block_colors, prev_left_idx, prev_right_idx
            )

            debug_log(
                f"k={k}: delta=+1 | left_ok={left_ok} right_ok={right_ok} "
                f"pen_left={pen_l:.3f} pen_right={pen_r:.3f} "
                f"choice={'left' if choose_left else 'right'}",
                args.debug,
                args.quiet,
            )

            if not left_ok and not right_ok:
                raise RuntimeError(
                    f"Frame {k}: Count +1 but neither side color-consistent.\n"
                    f"Colors: {color_seq_to_str(this_colors)} vs Global: {color_seq_to_str(block_colors)}\n"
                    f"Penalties: L={pen_l:.1f}, R={pen_r:.1f}"
                )

            if choose_left:
                if prev_left_idx > 0:
                    curr_left_idx = prev_left_idx - 1
                    curr_right_idx = prev_right_idx
                else:
                    block_colors.insert(0, this_colors[0])
                    for row in all_positions_matrix:
                        row.insert(0, float('nan'))
                    prev_left_idx += 1
                    prev_right_idx += 1
                    curr_left_idx = prev_left_idx - 1
                    curr_right_idx = prev_right_idx
                    expand_left_count += 1
                    debug_log(f"k={k}: expanded matrix left", args.debug, args.quiet)
            else:
                if prev_right_idx < len(block_colors) - 1:
                    curr_left_idx = prev_left_idx
                    curr_right_idx = prev_right_idx + 1
                else:
                    block_colors.append(this_colors[-1])
                    for row in all_positions_matrix:
                        row.append(float('nan'))
                    curr_left_idx = prev_left_idx
                    curr_right_idx = prev_right_idx + 1
                    expand_right_count += 1
                    debug_log(f"k={k}: expanded matrix right", args.debug, args.quiet)

        elif delta == -1:
            count_minus_one += 1
            choose_left_exit, left_ok, right_ok, pen_l, pen_r = decide_decrease_side(
                this_x, this_colors, prev_x, block_colors, prev_left_idx, prev_right_idx
            )

            debug_log(
                f"k={k}: delta=-1 | left_ok={left_ok} right_ok={right_ok} "
                f"pen_left={pen_l:.3f} pen_right={pen_r:.3f} "
                f"choice={'left' if choose_left_exit else 'right'}",
                args.debug,
                args.quiet,
            )

            if not left_ok and not right_ok:
                raise RuntimeError(
                    f"Frame {k}: Count -1 but neither side exit valid.\n"
                    f"Colors: {color_seq_to_str(this_colors)} vs Global: {color_seq_to_str(block_colors)}"
                )

            if choose_left_exit:
                curr_left_idx = prev_left_idx + 1
                curr_right_idx = prev_right_idx
            else:
                curr_left_idx = prev_left_idx
                curr_right_idx = prev_right_idx - 1

        width = curr_right_idx - curr_left_idx + 1
        if width != this_num:
            raise RuntimeError(
                f"Interval width mismatch at frame {k}. Calc width={width}, Actual det={this_num}"
            )

        this_row = [float('nan')] * len(block_colors)
        for i, val in enumerate(this_x):
            this_row[curr_left_idx + i] = float(val)

        all_positions_matrix.append(this_row)

        prev_x = this_x
        prev_num = this_num
        prev_left_idx = curr_left_idx
        prev_right_idx = curr_right_idx

    # ---------------------------------------------------------------------
    # FINALIZE: PREPEND LEADING EMPTY FRAMES
    # ---------------------------------------------------------------------
    final_n_cols = len(block_colors)
    leading_empty_count = first_non_empty_idx

    if leading_empty_count > 0:
        empty_rows = [[float('nan')] * final_n_cols for _ in range(leading_empty_count)]
        all_positions_matrix = empty_rows + all_positions_matrix

    if len(all_positions_matrix) != n_frames:
        raise RuntimeError(
            f"Final row count mismatch. Matrix={len(all_positions_matrix)}, nFrames={n_frames}"
        )

    frame_times = [f.frame_time_s for f in frames]
    frame_numbers = [f.frame_number for f in frames]

    t2x = Track2XPermanence(
        originalVideoPath=vc.filepath,
        trackingResultsPath=input_msgpack_path,
        blockColors=block_colors,
        xPositions=all_positions_matrix,
        frameTimes_s=frame_times,
        frameNumbers=frame_numbers,
    )

    if args.dry_run:
        log("Dry run complete. No file written.", args.quiet)
    else:
        log(f"Saving to {output_msgpack_path}...", args.quiet)
        save_track2(output_msgpack_path, t2x)

    if not args.quiet:
        print("track2 complete.")
        print(f"  Frames processed: {len(all_positions_matrix)}")
        print(f"  First non-empty frame index: {first_non_empty_idx}")
        print(f"  Total unique blocks found: {len(block_colors)}")
        print(f"  Max visible blocks in one frame: {max_visible_blocks}")
        print(f"  Delta counts: same={count_same}, plus_one={count_plus_one}, minus_one={count_minus_one}")
        print(f"  Matrix expansions: left={expand_left_count}, right={expand_right_count}")
        print(f"  Block Sequence: {color_seq_to_str(block_colors)}")
        if args.dry_run:
            print("  Output: none (dry run)")
        else:
            print(f"  Output: {output_msgpack_path}")


if __name__ == '__main__':
    parser = build_parser()
    parsed_args = parser.parse_args()

    try:
        main(parsed_args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

import os
import sys
import argparse
import numpy as np
import msgpack
from dataclasses import asdict

from tracking_classes import DetectionRecord, VideoCentroids


STANDARD_VERIFIED_NAME = "track2_verified.msgpack"


def load_video_centroids(path: str) -> VideoCentroids:
    with open(path, 'rb') as f:
        data_dict = msgpack.unpackb(f.read())
    return VideoCentroids.from_dict(data_dict)


def save_video_centroids(path: str, vc: VideoCentroids) -> None:
    with open(path, 'wb') as f_out:
        f_out.write(msgpack.packb(asdict(vc)))


def log(msg: str, quiet: bool = False) -> None:
    if not quiet:
        print(msg)


def summarize_bad_runs(bad_indices: np.ndarray):
    if len(bad_indices) == 0:
        return []

    diff_bad = np.diff(bad_indices)
    run_ends = np.where(diff_bad > 1)[0]
    start_idxs = np.insert(bad_indices[run_ends + 1], 0, bad_indices[0])
    end_idxs = np.append(bad_indices[run_ends], bad_indices[-1])
    return list(zip(start_idxs.tolist(), end_idxs.tolist()))


def dataset_dir_from_anchor(anchor_path: str) -> str:
    anchor_abs = os.path.abspath(anchor_path)
    anchor_dir = os.path.dirname(anchor_abs)
    anchor_base = os.path.splitext(os.path.basename(anchor_abs))[0]
    return os.path.join(anchor_dir, anchor_base)


def default_output_path_from_anchor(anchor_path: str) -> str:
    return os.path.join(dataset_dir_from_anchor(anchor_path), STANDARD_VERIFIED_NAME)


def resolve_paths(args):
    if args.dataset is None and args.input is None:
        raise ValueError("Provide either DATASET or -i/--input")

    if args.dataset is not None:
        dataset_anchor = args.dataset
        default_input = dataset_anchor
        default_output = default_output_path_from_anchor(dataset_anchor)
    else:
        dataset_anchor = None
        default_input = None
        default_output = None

    input_file_path = args.input if args.input is not None else default_input

    if input_file_path is None:
        raise ValueError("Could not determine input path")

    if args.overwrite and args.output is not None:
        raise ValueError("Use either --overwrite or --output, not both")

    if args.overwrite:
        output_file_path = input_file_path
    elif args.output is not None:
        output_file_path = args.output
    else:
        if default_output is None:
            raise ValueError("When not using DATASET, you must provide -o/--output unless using --overwrite")
        output_file_path = default_output

    return dataset_anchor, input_file_path, output_file_path


def verify_and_optionally_sanitize(
    vc: VideoCentroids,
    ratio_min: float,
    ratio_max: float,
    repair: bool,
    quiet: bool = False,
):
    frames = vc.frames
    n_frames = len(frames)

    if n_frames == 0:
        raise ValueError("vc.frames is empty. Cannot verify.")

    log(f"track2_verifystep: loaded {n_frames} frames", quiet)

    # ---------------------------------------------------------------------
    # PASS 0: ESTABLISH blockDistanceReference
    # ---------------------------------------------------------------------
    ref_frame_idx = -1
    block_distance_reference = float('nan')

    for k, f in enumerate(frames):
        dets = f.detections
        if len(dets) < 2:
            continue

        x = np.array([d.x for d in dets], dtype=float)
        c = np.array([d.color for d in dets])

        if not np.all(np.isfinite(x)):
            continue
        if np.any(np.diff(x) < 0):
            continue
        if np.any(~np.isin(c, ['r', 'g'])):
            continue
        if np.any(c[:-1] == c[1:]):
            continue

        ref_frame_idx = k
        block_distance_reference = float(np.mean(np.diff(x)))
        break

    if np.isnan(block_distance_reference):
        raise RuntimeError("Could not find a valid reference frame to establish spacing.")

    log(
        f"Reference frame: {ref_frame_idx} | Reference spacing: {block_distance_reference:.6f} px",
        quiet,
    )

    # ---------------------------------------------------------------------
    # PASS 1: EVALUATE ALL FRAMES
    # ---------------------------------------------------------------------
    is_bad = np.zeros(n_frames, dtype=bool)
    first_non_empty = -1

    for k, f in enumerate(frames):
        dets = f.detections

        if len(dets) > 0 and first_non_empty == -1:
            first_non_empty = k

        if len(dets) == 0:
            if first_non_empty != -1:
                is_bad[k] = True
            continue

        x = np.array([d.x for d in dets], dtype=float)
        c = np.array([d.color for d in dets])

        if np.any(~np.isfinite(x)) or np.any(np.diff(x) < 0) or np.any(~np.isin(c, ['r', 'g'])):
            is_bad[k] = True
            continue

        if len(dets) >= 2:
            spacing_ratio = np.diff(x) / block_distance_reference
            if np.any(c[:-1] == c[1:]) or np.any((spacing_ratio < ratio_min) | (spacing_ratio > ratio_max)):
                is_bad[k] = True

    bad_indices = np.where(is_bad)[0]
    bad_runs = summarize_bad_runs(bad_indices)

    log(f"First non-empty frame index: {first_non_empty}", quiet)
    log(f"Bad frames detected: {len(bad_indices)}", quiet)
    log(f"Bad segments detected: {len(bad_runs)}", quiet)

    sanitized_runs = 0
    sanitized_frames = 0

    # ---------------------------------------------------------------------
    # PASS 2: SANITIZE BAD SEGMENTS
    # ---------------------------------------------------------------------
    if len(bad_indices) > 0 and repair:
        log(f"Sanitizing {len(bad_indices)} bad frames...", quiet)

        for idx_start, idx_end in bad_runs:
            idxL, idxR = idx_start - 1, idx_end + 1
            if idxL < 0 or idxR >= n_frames:
                log(
                    f"Skipping bad segment [{idx_start}, {idx_end}] because it touches a boundary.",
                    quiet,
                )
                continue

            DL, DR = frames[idxL].detections, frames[idxR].detections
            if not DL or not DR:
                log(
                    f"Skipping bad segment [{idx_start}, {idx_end}] because a neighboring anchor frame is empty.",
                    quiet,
                )
                continue

            XL = np.array([d.x for d in DL], dtype=float)
            YL = np.array([d.y for d in DL], dtype=float)
            AL = np.array([d.area for d in DL], dtype=float)
            CL = np.array([d.color for d in DL])

            XR = np.array([d.x for d in DR], dtype=float)
            YR = np.array([d.y for d in DR], dtype=float)
            AR = np.array([d.area for d in DR], dtype=float)
            CR = np.array([d.color for d in DR])

            best_offset = None
            max_overlap = 0
            min_err = float('inf')

            for offset in range(-(len(DR) - 1), len(DL)):
                iL_s = max(0, offset)
                iL_e = min(len(DL), len(DR) + offset)
                if iL_s >= iL_e:
                    continue

                if np.array_equal(CL[iL_s:iL_e], CR[iL_s - offset:iL_e - offset]):
                    err = np.mean(np.abs(XL[iL_s:iL_e] - XR[iL_s - offset:iL_e - offset]))
                    overlap = iL_e - iL_s
                    if overlap > max_overlap or (overlap == max_overlap and err < min_err):
                        max_overlap = overlap
                        min_err = err
                        best_offset = offset

            if best_offset is None:
                log(
                    f"Skipping bad segment [{idx_start}, {idx_end}] because no color-consistent overlap was found.",
                    quiet,
                )
                continue

            S_shift = -min(0, best_offset)
            N_union = max(len(DL) + S_shift, len(DR) + best_offset + S_shift)

            U_XL = np.full(N_union, np.nan)
            U_XR = np.full(N_union, np.nan)
            U_YL = np.full(N_union, np.nan)
            U_YR = np.full(N_union, np.nan)
            U_AL = np.full(N_union, np.nan)
            U_AR = np.full(N_union, np.nan)
            U_C = ['?'] * N_union

            if max_overlap > 0:
                overlap_start_L = max(0, best_offset)
                overlap_end_L = min(len(DL), len(DR) + best_offset)
                overlap_start_R = overlap_start_L - best_offset
                overlap_end_R = overlap_end_L - best_offset
                dx_avg = np.mean(XR[overlap_start_R:overlap_end_R] - XL[overlap_start_L:overlap_end_L])
                dy_avg = np.mean(YR[overlap_start_R:overlap_end_R] - YL[overlap_start_L:overlap_end_L])
            else:
                dx_avg = 0.0
                dy_avg = 0.0

            for u in range(N_union):
                iL = u - S_shift
                iR = u - best_offset - S_shift
                inL = 0 <= iL < len(DL)
                inR = 0 <= iR < len(DR)

                if inL and inR:
                    U_XL[u], U_XR[u] = XL[iL], XR[iR]
                    U_YL[u], U_YR[u] = YL[iL], YR[iR]
                    U_AL[u], U_AR[u] = AL[iL], AR[iR]
                    U_C[u] = CL[iL]
                elif inL:
                    U_XL[u], U_XR[u] = XL[iL], XL[iL] + dx_avg
                    U_YL[u], U_YR[u] = YL[iL], YL[iL] + dy_avg
                    U_AL[u], U_AR[u] = AL[iL], AL[iL]
                    U_C[u] = CL[iL]
                elif inR:
                    U_XR[u], U_XL[u] = XR[iR], XR[iR] - dx_avg
                    U_YR[u], U_YL[u] = YR[iR], YR[iR] - dy_avg
                    U_AR[u], U_AL[u] = AR[iR], AR[iR]
                    U_C[u] = CR[iR]

            tL = frames[idxL].frame_time_s
            tR = frames[idxR].frame_time_s
            if tR == tL:
                raise RuntimeError(
                    f"Anchor frames around bad segment [{idx_start}, {idx_end}] have identical timestamps."
                )

            for k in range(idx_start, idx_end + 1):
                alpha = (frames[k].frame_time_s - tL) / (tR - tL)
                sk = int(round(S_shift + alpha * best_offset))
                ek = int(round(len(DL) + S_shift - 1 + alpha * (len(DR) - len(DL))))

                frames[k].detections = [
                    DetectionRecord(
                        x=float(U_XL[u] + alpha * (U_XR[u] - U_XL[u])),
                        y=float(U_YL[u] + alpha * (U_YR[u] - U_YL[u])),
                        color=U_C[u],
                        area=float(U_AL[u] + alpha * (U_AR[u] - U_AL[u])),
                    )
                    for u in range(sk, ek + 1)
                ]

            sanitized_runs += 1
            sanitized_frames += (idx_end - idx_start + 1)

    elif len(bad_indices) > 0 and not repair:
        log("Bad frames were found, but repair is disabled (--verify-only).", quiet)

    # ---------------------------------------------------------------------
    # PASS 3: FINAL VERIFICATION
    # ---------------------------------------------------------------------
    log("Pass 3/3: Final verification...", quiet)

    all_diffs = []
    first_found = False
    for k, f in enumerate(frames):
        if len(f.detections) > 0:
            first_found = True
        elif first_found:
            raise RuntimeError(f"Gap at frame {k}")

        if len(f.detections) >= 2:
            all_diffs.extend(np.diff([d.x for d in f.detections]))

    final_bad = []
    for k, f in enumerate(frames):
        dets = f.detections
        if len(dets) == 0:
            if first_non_empty != -1 and k >= first_non_empty:
                final_bad.append(k)
            continue

        x = np.array([d.x for d in dets], dtype=float)
        c = np.array([d.color for d in dets])
        if np.any(~np.isfinite(x)) or np.any(np.diff(x) < 0) or np.any(~np.isin(c, ['r', 'g'])):
            final_bad.append(k)
            continue

        if len(dets) >= 2:
            spacing_ratio = np.diff(x) / block_distance_reference
            if np.any(c[:-1] == c[1:]) or np.any((spacing_ratio < ratio_min) | (spacing_ratio > ratio_max)):
                final_bad.append(k)

    if final_bad:
        preview = ", ".join(map(str, final_bad[:10]))
        suffix = "..." if len(final_bad) > 10 else ""
        raise RuntimeError(f"Verification failed after processing. Bad frames remain: {preview}{suffix}")

    vc.passedVerification = True
    vc.meanBlockDistance = float(np.mean(all_diffs)) if all_diffs else 0.0

    summary = {
        'n_frames': n_frames,
        'reference_frame_idx': ref_frame_idx,
        'reference_spacing_px': block_distance_reference,
        'first_non_empty_idx': first_non_empty,
        'initial_bad_frames': int(len(bad_indices)),
        'bad_segments': int(len(bad_runs)),
        'sanitized_runs': int(sanitized_runs),
        'sanitized_frames': int(sanitized_frames),
        'final_mean_block_distance': vc.meanBlockDistance,
    }

    return vc, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify and optionally sanitize Track1 centroid detections. "
            "You can either pass a DATASET anchor path like data/foo.msgpack, "
            "or use -i/--input explicitly."
        )
    )

    parser.add_argument(
        'dataset',
        nargs='?',
        help=(
            "Dataset anchor path, typically the Track1 file such as data/foo.msgpack. "
            "If provided without -o, output defaults to data/foo/track2_verified.msgpack."
        )
    )
    parser.add_argument(
        '-i', '--input',
        default=None,
        help='Explicit Track1 MessagePack input path. Overrides the dataset-derived input path.'
    )
    parser.add_argument(
        '-o', '--output',
        default=None,
        help='Explicit output MessagePack path. Overrides the dataset-derived default output path.'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite the resolved input file in place.'
    )
    parser.add_argument(
        '--ratio-min',
        type=float,
        default=0.50,
        help='Minimum allowed neighbor-spacing ratio relative to the reference spacing. Default: 0.50'
    )
    parser.add_argument(
        '--ratio-max',
        type=float,
        default=1.50,
        help='Maximum allowed neighbor-spacing ratio relative to the reference spacing. Default: 1.50'
    )
    parser.add_argument(
        '--verify-only',
        action='store_true',
        help='Check validity only. Do not sanitize bad frames.'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Run verification/sanitization logic but do not write an output file.'
    )
    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Reduce console output.'
    )

    return parser


def main(args):
    if args.ratio_min <= 0:
        raise ValueError("--ratio-min must be > 0")
    if args.ratio_max <= 0:
        raise ValueError("--ratio-max must be > 0")
    if args.ratio_min >= args.ratio_max:
        raise ValueError("--ratio-min must be less than --ratio-max")

    dataset_anchor, input_file_path, output_file_path = resolve_paths(args)

    if not os.path.isfile(input_file_path):
        raise FileNotFoundError(f"Input file does not exist: {input_file_path}")

    if not args.dry_run:
        output_dir = os.path.dirname(os.path.abspath(output_file_path))
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    if dataset_anchor is not None and not args.quiet:
        print(f"Dataset anchor: {dataset_anchor}")
        print(f"Dataset directory: {dataset_dir_from_anchor(dataset_anchor)}")

    vc = load_video_centroids(input_file_path)

    vc, summary = verify_and_optionally_sanitize(
        vc=vc,
        ratio_min=args.ratio_min,
        ratio_max=args.ratio_max,
        repair=not args.verify_only,
        quiet=args.quiet,
    )

    if args.dry_run:
        log("Dry run complete. No file written.", args.quiet)
    else:
        save_video_centroids(output_file_path, vc)
        log(f"Saved verified file to: {output_file_path}", args.quiet)

    if not args.quiet:
        print("Verification complete.")
        print(f"  Frames: {summary['n_frames']}")
        print(f"  Reference frame index: {summary['reference_frame_idx']}")
        print(f"  Reference spacing: {summary['reference_spacing_px']:.6f} px")
        print(f"  First non-empty frame index: {summary['first_non_empty_idx']}")
        print(f"  Initial bad frames: {summary['initial_bad_frames']}")
        print(f"  Bad segments: {summary['bad_segments']}")
        print(f"  Sanitized runs: {summary['sanitized_runs']}")
        print(f"  Sanitized frames: {summary['sanitized_frames']}")
        print(f"  Final mean block distance: {summary['final_mean_block_distance']:.6f} px")
        print(f"  Mode: {'verify-only' if args.verify_only else 'verify-and-sanitize'}")
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

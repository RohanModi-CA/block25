#!/usr/bin/env python3
"""
2.verify_and_process.py  —  Verify detections and build the permanence matrix.

Loads data/{name}/track1.msgpack, verifies every frame, repairs any bad
segments (with user confirmation), then builds the permanence matrix and
writes data/{name}/track2_permanence.msgpack.

Usage
-----
    python3 2.verify_and_process.py IMG_9282
    python3 2.verify_and_process.py 9282
"""

import sys
import os
import argparse
import msgpack
from dataclasses import asdict

from helper.video_io     import find_video, track1_output_path, track2_output_path, video_name
from helper.verification import scan_bad_frames, verify_and_sanitize
from helper.permanence   import build_permanence
from tracking_classes    import VideoCentroids


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_vc(path: str) -> VideoCentroids:
    with open(path, 'rb') as fh:
        return VideoCentroids.from_dict(msgpack.unpackb(fh.read()))


def _save_msgpack(path: str, obj) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'wb') as fh:
        fh.write(msgpack.packb(asdict(obj)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify Track1 detections, repair bad frames, and build the permanence matrix."
    )
    parser.add_argument(
        'name',
        help="Video name or numeric suffix, e.g. IMG_9282 or 9282",
    )
    parser.add_argument(
        '--ratio-min', type=float, default=0.50,
        help="Min spacing ratio relative to reference (default 0.50).",
    )
    parser.add_argument(
        '--ratio-max', type=float, default=1.50,
        help="Max spacing ratio relative to reference (default 1.50).",
    )
    args = parser.parse_args()

    # ---- Resolve name (video file not required for this step) ----
    video_path = find_video(args.name, "Videos")
    if video_path is not None:
        name = video_name(video_path)
    else:
        name = os.path.splitext(os.path.basename(args.name))[0]
        print(f"Note: video file not found in Videos/ — using name '{name}' for paths.")

    t1_path = track1_output_path(name)
    t2_path = track2_output_path(name)

    if not os.path.exists(t1_path):
        print(f"Error: track1 output not found: {t1_path}")
        print(f"  Run first: python3 1.track_run.py {args.name}")
        sys.exit(1)

    # ---- Load ----
    print(f"Loading {t1_path}…")
    vc = _load_vc(t1_path)
    print(f"  {len(vc.frames)} frames loaded.")

    # ---- Non-destructive scan ----
    print("\nScanning for bad frames…")
    n_bad, n_seg, ref_spacing = scan_bad_frames(
        vc, ratio_min=args.ratio_min, ratio_max=args.ratio_max
    )

    if n_bad > 0:
        print(f"\n  ⚠  {n_bad} bad frame(s) detected in {n_seg} segment(s).")
        print(f"     (reference spacing: {ref_spacing:.4f} px, "
              f"ratio bounds: [{args.ratio_min}, {args.ratio_max}])")
        ans = input("\n  Apply automatic interpolation repair and continue? [y/N] ").strip().lower()
        if ans != 'y':
            print("Aborted — no files written.")
            sys.exit(0)

        # Reload clean copy before repair (scan may have had side-effects via numpy views)
        vc = _load_vc(t1_path)
        print("\nRepairing…")
        vc, summary = verify_and_sanitize(
            vc,
            ratio_min=args.ratio_min,
            ratio_max=args.ratio_max,
            repair=True,
            quiet=False,
        )
        print(f"  Repaired {summary['sanitized_frames']} frame(s) "
              f"in {summary['sanitized_runs']} segment(s).")
    else:
        print("  ✓  All frames passed — no repair needed.")
        vc, summary = verify_and_sanitize(
            vc,
            ratio_min=args.ratio_min,
            ratio_max=args.ratio_max,
            repair=False,
            quiet=True,
        )

    print(f"  Mean block distance: {summary['final_mean_block_distance']:.4f} px")

    # ---- Build permanence matrix ----
    print("\nBuilding permanence matrix…")
    t2 = build_permanence(vc, quiet=False)
    t2.trackingResultsPath = t1_path

    # ---- Save ----
    _save_msgpack(t2_path, t2)

    print(f"\nDone.")
    print(f"  Blocks  : {len(t2.blockColors)}  ({' '.join(t2.blockColors)})")
    print(f"  Frames  : {len(t2.xPositions)}")
    print(f"  Output  : {t2_path}")


if __name__ == '__main__':
    main()

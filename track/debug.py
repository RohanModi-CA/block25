#!/usr/bin/env python3
"""
debug_reference_spacing_black.py

Inspect Step 1 detections and diagnose why reference x-spacing
cannot be established for black-track videos.

Usage
-----
python3 debug_reference_spacing_black.py IMG_0662_rot90
python3 debug_reference_spacing_black.py IMG_0662_rot90 --show 30
"""

import os
import sys
import argparse
import math
import statistics
import msgpack
import numpy as np

from tracking_classes import VideoCentroids


def track1_path(name: str) -> str:
    name = os.path.splitext(os.path.basename(name))[0]
    return os.path.join("data", name, "track1.msgpack")


def load_vc(path: str) -> VideoCentroids:
    with open(path, "rb") as fh:
        return VideoCentroids.from_dict(msgpack.unpackb(fh.read()))


def frame_summary(frame):
    dets = frame.detections
    n = len(dets)

    if n == 0:
        return {
            "n": 0,
            "finite_x": True,
            "sorted_x": True,
            "dx": [],
            "median_dx": math.nan,
            "min_dx": math.nan,
            "max_dx": math.nan,
        }

    xs = np.array([d.x for d in dets], dtype=float)
    finite_x = np.all(np.isfinite(xs))
    sorted_x = bool(np.all(np.diff(xs) >= 0)) if n >= 2 else True
    dx = np.diff(xs) if n >= 2 else np.array([], dtype=float)

    return {
        "n": n,
        "finite_x": bool(finite_x),
        "sorted_x": sorted_x,
        "dx": dx.tolist(),
        "median_dx": float(np.median(dx)) if len(dx) else math.nan,
        "min_dx": float(np.min(dx)) if len(dx) else math.nan,
        "max_dx": float(np.max(dx)) if len(dx) else math.nan,
    }


def is_reference_candidate(frame, min_spacing_px=20.0):
    dets = frame.detections
    if len(dets) < 2:
        return False, "fewer than 2 detections"

    xs = np.array([d.x for d in dets], dtype=float)

    if not np.all(np.isfinite(xs)):
        return False, "non-finite x"

    if np.any(np.diff(xs) < 0):
        return False, "x not sorted"

    dx = np.diff(xs)
    if len(dx) == 0:
        return False, "no dx"

    if np.any(dx <= 0):
        return False, "non-positive dx"

    med = float(np.median(dx))
    if med < min_spacing_px:
        return False, f"median dx too small ({med:.2f}px)"

    # Robust outlier check
    lo = 0.35 * med
    hi = 3.0 * med
    if np.any((dx < lo) | (dx > hi)):
        return False, f"dx spread too wide around median ({med:.2f}px)"

    return True, f"candidate median dx = {med:.2f}px"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name", help="Dataset name")
    parser.add_argument("--show", type=int, default=20, help="How many candidate/rejected examples to print")
    args = parser.parse_args()

    path = track1_path(args.name)
    if not os.path.exists(path):
        print(f"track1 not found: {path}")
        sys.exit(1)

    vc = load_vc(path)
    frames = vc.frames
    print(f"Loaded {len(frames)} frames from {path}")

    n_nonempty = 0
    candidate_rows = []
    rejected_rows = []

    medians = []

    for k, f in enumerate(frames):
        s = frame_summary(f)
        if s["n"] > 0:
            n_nonempty += 1
        if s["n"] >= 2 and math.isfinite(s["median_dx"]):
            medians.append(s["median_dx"])

        ok, reason = is_reference_candidate(f)
        row = (k, s["n"], s["median_dx"], s["min_dx"], s["max_dx"], reason)

        if ok:
            candidate_rows.append(row)
        elif s["n"] >= 2:
            rejected_rows.append(row)

    print(f"Non-empty frames: {n_nonempty}")
    if medians:
        print(f"All-frame median of per-frame median dx: {statistics.median(medians):.2f}px")
        print(f"Min/Max per-frame median dx: {min(medians):.2f}px / {max(medians):.2f}px")
    else:
        print("No frames with 2+ detections.")

    print()
    print(f"Reference candidates found: {len(candidate_rows)}")
    for row in candidate_rows[:args.show]:
        k, n, med, mindx, maxdx, reason = row
        print(f"  frame {k:6d} | n={n:2d} | median_dx={med:8.2f} | min={mindx:8.2f} | max={maxdx:8.2f} | {reason}")

    print()
    print(f"Rejected 2+-detection frames: {len(rejected_rows)}")
    for row in rejected_rows[:args.show]:
        k, n, med, mindx, maxdx, reason = row
        print(f"  frame {k:6d} | n={n:2d} | median_dx={med:8.2f} | min={mindx:8.2f} | max={maxdx:8.2f} | {reason}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
rotate_track1.py

Rotate a track1.msgpack dataset by 90 / 180 / 270 degrees.

This updates all detection coordinates accordingly.

IMPORTANT:
- Requires the original video to determine frame width/height.
- Rotation is applied in image coordinate space (top-left origin).

Usage
-----
python3 rotate_track1.py IMG_0662 --deg 90
python3 rotate_track1.py IMG_0662 --deg 180
python3 rotate_track1.py IMG_0662 --deg 270

Optional:
--out data/IMG_0662_rot/track1.msgpack
"""

import os
import sys
import argparse
import msgpack
import cv2

from dataclasses import asdict
from tracking_classes import VideoCentroids


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def find_video(name: str, video_dir="Videos"):
    name = os.path.splitext(os.path.basename(name))[0]

    for f in os.listdir(video_dir):
        base = os.path.splitext(f)[0]
        if base == name or base.endswith(name):
            return os.path.join(video_dir, f)

    return None


def track1_path(name: str):
    name = os.path.splitext(os.path.basename(name))[0]
    return os.path.join("data", name, "track1.msgpack")


def load_vc(path: str) -> VideoCentroids:
    with open(path, "rb") as fh:
        return VideoCentroids.from_dict(msgpack.unpackb(fh.read()))


def save_vc(path: str, vc: VideoCentroids):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(msgpack.packb(asdict(vc), use_bin_type=True))


# ---------------------------------------------------------------------
# Rotation math
# ---------------------------------------------------------------------

def rotate_point(x, y, w, h, deg):
    """
    Rotate point (x, y) within image of size (w, h).
    Coordinate system: origin at top-left.
    """

    if deg == 90:
        # clockwise
        return h - 1 - y, x

    elif deg == 180:
        return w - 1 - x, h - 1 - y

    elif deg == 270:
        # clockwise 270 = ccw 90
        return y, w - 1 - x

    else:
        raise ValueError("deg must be 90, 180, or 270")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name", help="Dataset name or suffix")
    parser.add_argument("--deg", type=int, required=True, choices=[90, 180, 270])
    parser.add_argument("--out", help="Optional output path")
    args = parser.parse_args()

    t1_path = track1_path(args.name)
    if not os.path.exists(t1_path):
        print("track1.msgpack not found.")
        sys.exit(1)

    video_path = find_video(args.name)
    if video_path is None:
        print("Video not found (needed for dimensions).")
        sys.exit(1)

    print(f"Loading {t1_path}…")
    vc = load_vc(t1_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Cannot open video.")
        sys.exit(1)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    print(f"Video size: {w} x {h}")
    print(f"Rotating by {args.deg} degrees...")

    # -----------------------------------------------------------------
    # Rotate all detections
    # -----------------------------------------------------------------

    for frame in vc.frames:
        for det in frame.detections:
            x_new, y_new = rotate_point(det.x, det.y, w, h, args.deg)
            det.x = float(x_new)
            det.y = float(y_new)

        frame.detections.sort(key=lambda d: d.x)

    # -----------------------------------------------------------------
    # Update output path
    # -----------------------------------------------------------------

    name = os.path.splitext(os.path.basename(args.name))[0]

    if args.out:
        out_path = args.out
    else:
        suffix = f"_rot{args.deg}"
        out_dir = os.path.join("data", name + suffix)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "track1.msgpack")

    save_vc(out_path, vc)

    print(f"Done.")
    print(f"Output → {out_path}")


if __name__ == "__main__":
    main()

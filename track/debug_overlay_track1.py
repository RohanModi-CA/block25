
#!/usr/bin/env python3
"""
debug_overlay_track1.py

Visualize Step 1 detections directly on the video.

Shows:
- centroids
- indices (left→right)
- spacing lines between neighbors
- optional live spacing stats

Usage
-----
python3 debug_overlay_track1.py IMG_0662
python3 debug_overlay_track1.py IMG_0662 --no-lines
python3 debug_overlay_track1.py IMG_0662 --print-spacing
"""

import os
import sys
import argparse
import msgpack
import cv2
import numpy as np

from tracking_classes import VideoCentroids


WIN_W = 1400
WIN_H = 800


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


# ---------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------

def draw_overlay(frame, detections, draw_lines=True):
    """
    Draw centroids + optional spacing lines.
    """
    h, w = frame.shape[:2]

    # sort just in case
    dets = sorted(detections, key=lambda d: d.x)

    xs = [d.x for d in dets]
    ys = [d.y for d in dets]

    # draw lines between neighbors
    if draw_lines and len(dets) >= 2:
        for i in range(len(dets) - 1):
            x1, y1 = int(xs[i]), int(ys[i])
            x2, y2 = int(xs[i+1]), int(ys[i+1])

            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

            dx = xs[i+1] - xs[i]
            mx = int((x1 + x2) / 2)
            my = int((y1 + y2) / 2)

            cv2.putText(
                frame,
                f"{dx:.1f}",
                (mx, my - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
                cv2.LINE_AA
            )

    # draw points
    for i, d in enumerate(dets):
        x = int(round(d.x))
        y = int(round(d.y))

        cv2.circle(frame, (x, y), 10, (0, 0, 255), 2)

        cv2.putText(
            frame,
            f"{i}",
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            f"({d.x:.0f},{d.y:.0f})",
            (x + 10, y + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 128, 255),
            1,
            cv2.LINE_AA
        )

    return frame


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name", help="Video name or suffix")
    parser.add_argument("--no-lines", action="store_true")
    parser.add_argument("--print-spacing", action="store_true")
    args = parser.parse_args()

    video_path = find_video(args.name)
    if video_path is None:
        print("Video not found.")
        sys.exit(1)

    t1_path = track1_path(args.name)
    if not os.path.exists(t1_path):
        print("track1.msgpack not found.")
        sys.exit(1)

    print(f"Loading {t1_path}…")
    vc = load_vc(t1_path)

    print(f"{len(vc.frames)} frames")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Cannot open video.")
        sys.exit(1)

    win = "Track1 Debug Overlay"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, WIN_W, WIN_H)

    idx = 0

    while True:
        if idx >= len(vc.frames):
            break

        frame_data = vc.frames[idx]

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_data.frame_number)
        ret, frame = cap.read()
        if not ret:
            break

        dets = frame_data.detections

        # print spacing info
        if args.print_spacing and len(dets) >= 2:
            xs = np.array([d.x for d in dets])
            dx = np.diff(xs)

            print(
                f"frame {idx:5d} | n={len(dets):2d} | "
                f"median dx={np.median(dx):.1f} | "
                f"min={dx.min():.1f} | max={dx.max():.1f}"
            )

        overlay = draw_overlay(frame.copy(), dets, draw_lines=not args.no_lines)

        cv2.putText(
            overlay,
            f"frame={idx}  n={len(dets)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 0, 255),
            2,
            cv2.LINE_AA
        )

        cv2.imshow(win, overlay)

        key = cv2.waitKey(30) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('d'):
            idx += 1
        elif key == ord('a'):
            idx = max(0, idx - 1)
        elif key == ord(' '):
            # play forward
            for _ in range(30):
                idx += 1
                if idx >= len(vc.frames):
                    break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

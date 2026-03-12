#!/usr/bin/env python3
"""
B1.batch_track_run.py  —  Run 1.track_run.py on multiple videos.

Scans Videos/ for video files and runs the tracking step on each.

Default behaviour:
  • Only processes videos that already have data/<name>/params.json.
  • Skips videos without params.json.

Flags
-----
--exclude NAME [NAME ...]
    Skip particular videos. Names can be full base names (IMG_9282)
    or numeric suffixes (9282), using the same matching logic used
    elsewhere in the pipeline.

--nojsons
    If set, videos without params.json will also be processed.

--no-preview
    Pass through to 1.track_run.py to disable the OpenCV preview.

Usage
-----
    python3 B1.batch_track_run.py
    python3 B1.batch_track_run.py --exclude 9282 IMG_0537
    python3 B1.batch_track_run.py --nojsons
    python3 B1.batch_track_run.py --no-preview
"""

import os
import argparse
import subprocess

from helper.video_io import video_name, params_path

VIDEO_EXTS = (".mov", ".avi", ".mp4")


def find_videos(video_dir="Videos"):
    """Return full paths of all videos in VIDEO_EXTS."""
    vids = []
    for f in sorted(os.listdir(video_dir)):
        if f.lower().endswith(VIDEO_EXTS):
            vids.append(os.path.join(video_dir, f))
    return vids


def _match_name(name: str, token: str) -> bool:
    """
    Return True if token refers to this video name.

    Accepts either:
      • full base name (IMG_9282)
      • numeric suffix (9282)
    """
    token = os.path.splitext(os.path.basename(token))[0]

    if token == name:
        return True

    if name.endswith(token):
        return True

    return False


def main():
    parser = argparse.ArgumentParser(
        description="Run 1.track_run.py on multiple videos."
    )

    parser.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="Video names or numeric suffixes to exclude."
    )

    parser.add_argument(
        "--nojsons",
        action="store_true",
        help="Process videos even if params.json is missing."
    )

    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Disable preview during tracking."
    )

    args = parser.parse_args()

    videos = find_videos()

    if not videos:
        print("No videos found in Videos/.")
        return

    tasks = []

    for vpath in videos:
        name = video_name(vpath)

        # Exclusion logic
        if any(_match_name(name, tok) for tok in args.exclude):
            continue

        ppath = params_path(name)

        if not args.nojsons and not os.path.exists(ppath):
            continue

        tasks.append((name, vpath))

    if not tasks:
        print("No videos selected for tracking.")
        return

    print(f"{len(tasks)} video(s) will be processed:\n")

    for name, vpath in tasks:
        ppath = params_path(name)
        status = "params.json" if os.path.exists(ppath) else "no params.json"
        print(f"  {name}  ({status})")

    print()

    for i, (name, _) in enumerate(tasks, 1):
        print(f"[{i}/{len(tasks)}] Tracking {name}...\n")

        cmd = ["python3", "1.track_run.py", name]

        if args.no_preview:
            cmd.append("--no-preview")

        subprocess.run(cmd)

        print()

    print("Batch tracking complete.")


if __name__ == "__main__":
    main()

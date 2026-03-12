
#!/usr/bin/env python3
"""
B0.batch_prepare.py  —  Prepare all unprepared videos.

Scans Videos/ for video files and checks whether each has an existing
data/<name>/params.json. For any video that does not, it runs:

    python3 0.video_prepare.py <name>

Usage
-----
    python3 X.batch_prepare.py
    python3 X.batch_prepare.py --no-preview
"""

import os
import subprocess
import argparse

from helper.video_io import video_name, params_path

VIDEO_EXTS = (".mov", ".avi", ".mp4")


def find_videos(video_dir="Videos"):
    """Return full paths of all videos in VIDEO_EXTS."""
    vids = []
    for f in sorted(os.listdir(video_dir)):
        if f.lower().endswith(VIDEO_EXTS):
            vids.append(os.path.join(video_dir, f))
    return vids


def main():
    parser = argparse.ArgumentParser(
        description="Run 0.video_prepare.py on any video missing params.json."
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Disable detection preview during tests."
    )
    args = parser.parse_args()

    videos = find_videos()

    if not videos:
        print("No videos found in Videos/.")
        return

    to_prepare = []

    for vpath in videos:
        name = video_name(vpath)
        ppath = params_path(name)
        if not os.path.exists(ppath):
            to_prepare.append((name, vpath))

    if not to_prepare:
        print("All videos already have params.json.")
        return

    print(f"{len(to_prepare)} video(s) require preparation:\n")

    for name, vpath in to_prepare:
        print(f"  {name}  ({vpath})")

    print()

    for i, (name, _) in enumerate(to_prepare, 1):
        print(f"[{i}/{len(to_prepare)}] Preparing {name}...\n")

        cmd = ["python3", "0.video_prepare.py", name]
        if args.no_preview:
            cmd.append("--no-preview")

        subprocess.run(cmd)

        print()

    print("Batch preparation complete.")


if __name__ == "__main__":
    main()

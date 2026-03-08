#!/usr/bin/env python3
import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# Utilities
# ============================================================

def die(msg: str, code: int = 1) -> None:
    print(f"Error: {msg}", file=sys.stderr)
    sys.exit(code)


def run_cmd(cmd, capture=False, text=True):
    try:
        return subprocess.run(
            cmd,
            check=True,
            capture_output=capture,
            text=text,
        )
    except FileNotFoundError:
        die(f"Required executable not found: {cmd[0]}")
    except subprocess.CalledProcessError as e:
        if capture:
            stderr = e.stderr.strip() if e.stderr else ""
            stdout = e.stdout.strip() if e.stdout else ""
            extra = "\n".join(x for x in [stdout, stderr] if x)
            die(f"Command failed:\n{' '.join(shlex.quote(c) for c in cmd)}\n{extra}")
        die(f"Command failed:\n{' '.join(shlex.quote(c) for c in cmd)}")


def ffprobe_info(path: Path) -> dict:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames:format=duration",
        "-of", "json",
        str(path),
    ]
    res = run_cmd(cmd, capture=True)
    try:
        data = json.loads(res.stdout)
    except json.JSONDecodeError:
        die("ffprobe returned invalid JSON.")

    streams = data.get("streams", [])
    if not streams:
        die("No video stream found.")

    stream = streams[0]
    fmt = data.get("format", {})

    width = int(stream["width"])
    height = int(stream["height"])

    def parse_rate(s: str) -> float:
        if not s or s == "0/0":
            return 0.0
        if "/" in s:
            a, b = s.split("/", 1)
            a = float(a)
            b = float(b)
            return a / b if b != 0 else 0.0
        return float(s)

    fps = parse_rate(stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0/0")
    if fps <= 0:
        fps = parse_rate(stream.get("r_frame_rate", "0/0"))
    if fps <= 0:
        fps = 30.0

    duration = fmt.get("duration")
    if duration is None:
        die("Could not determine video duration.")
    duration = float(duration)

    nb_frames = stream.get("nb_frames")
    if nb_frames is not None and str(nb_frames).isdigit():
        frame_count = int(nb_frames)
    else:
        frame_count = max(1, int(round(duration * fps)))

    return {
        "width": width,
        "height": height,
        "fps": fps,
        "duration": duration,
        "frame_count": frame_count,
    }


def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_croptrim{input_path.suffix}")


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def sec_to_ms(s: float) -> int:
    return int(round(s * 1000.0))


def ms_to_sec(ms: int) -> float:
    return ms / 1000.0


def fmt_s(x: float) -> str:
    return f"{x:.3f}s"


# ============================================================
# Video frame access
# ============================================================

class VideoReader:
    def __init__(self, path: Path):
        self.path = path
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            die(f"Could not open video: {path}")

    def read_at_ms(self, ms: int):
        self.cap.set(cv2.CAP_PROP_POS_MSEC, float(ms))
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return None
        return frame

    def read_first_frame(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return None
        return frame

    def close(self):
        try:
            self.cap.release()
        except Exception:
            pass


# ============================================================
# Drawing helpers
# ============================================================

def draw_text(img, text, org, scale=0.8, fg=(255, 255, 255), bg=(0, 0, 0), thickness=2):
    x, y = org
    cv2.putText(
        img, text, (x, y),
        cv2.FONT_HERSHEY_SIMPLEX, scale, bg, thickness + 3, cv2.LINE_AA
    )
    cv2.putText(
        img, text, (x, y),
        cv2.FONT_HERSHEY_SIMPLEX, scale, fg, thickness, cv2.LINE_AA
    )


def draw_filled_panel(img, x1, y1, x2, y2, color=(0, 0, 0), alpha=0.72):
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img.shape[1], x2)
    y2 = min(img.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return
    roi = img[y1:y2, x1:x2]
    panel = np.full_like(roi, color, dtype=np.uint8)
    blended = cv2.addWeighted(panel, alpha, roi, 1.0 - alpha, 0.0)
    img[y1:y2, x1:x2] = blended


# ============================================================
# Crop UI
# ============================================================

def select_crop_box(video_path: Path):
    vr = VideoReader(video_path)
    frame = vr.read_first_frame()
    vr.close()

    if frame is None:
        die("Could not read first frame for crop selection.")

    win = "croptrim: crop selection"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, min(frame.shape[1], 1400), min(frame.shape[0], 900))

    print("Crop UI:")
    print("  Drag a rectangle on the first frame.")
    print("  Press ENTER or SPACE to confirm.")
    print("  Press C to cancel the current selection.")
    print("  Press ESC to abort.")

    roi = cv2.selectROI(win, frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow(win)

    x, y, w, h = map(int, roi)
    if w <= 0 or h <= 0:
        die("No crop box selected.")
    return x, y, w, h


# ============================================================
# Trim UI
# ============================================================

def trim_ui(video_path: Path, duration_s: float, fps: float):
    vr = VideoReader(video_path)

    total_ms = max(1, sec_to_ms(duration_s))
    frame_step_ms = max(1, int(round(1000.0 / max(fps, 1e-6))))

    win = "croptrim: trim selection"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1400, 900)

    state = {
        "cursor_ms": 0,
        "start_ms": 0,
        "end_ms": total_ms,
        "playing": False,
        "last_tick": time.time(),
        "dirty_frame": True,
        "last_rendered_ms": None,
        "frame": None,
    }

    cv2.createTrackbar("Cursor", win, 0, total_ms, lambda v: None)
    cv2.createTrackbar("Start", win, 0, total_ms, lambda v: None)
    cv2.createTrackbar("End", win, total_ms, total_ms, lambda v: None)

    cv2.setTrackbarPos("Cursor", win, state["cursor_ms"])
    cv2.setTrackbarPos("Start", win, state["start_ms"])
    cv2.setTrackbarPos("End", win, state["end_ms"])

    print("Trim UI:")
    print("  Drag trackbars or use keys.")
    print("  SPACE  play/pause")
    print("  a/d    step by 1 frame")
    print("  u/o    move cursor by 0.01 s")
    print("  j/l    move cursor by 0.1 s")
    print("  J/L    move cursor by 1.0 s")
    print("  s      start = cursor")
    print("  e      end   = cursor")
    print("  [ ]    start -=/+ 0.01 s")
    print("  { }    start -=/+ 0.1 s")
    print("  - =    end   -=/+ 0.01 s")
    print("  _ +    end   -=/+ 0.1 s")
    print("  r      reset")
    print("  ENTER  confirm")
    print("  ESC/q  abort")

    def sync_from_trackbars():
        cur = cv2.getTrackbarPos("Cursor", win)
        start = cv2.getTrackbarPos("Start", win)
        end = cv2.getTrackbarPos("End", win)

        if start > end:
            if abs(start - state["start_ms"]) >= abs(end - state["end_ms"]):
                start = end
                cv2.setTrackbarPos("Start", win, start)
            else:
                end = start
                cv2.setTrackbarPos("End", win, end)

        cur = clamp(cur, start, end)

        changed = (
            cur != state["cursor_ms"] or
            start != state["start_ms"] or
            end != state["end_ms"]
        )

        state["cursor_ms"] = cur
        state["start_ms"] = start
        state["end_ms"] = end

        if changed:
            state["dirty_frame"] = True

        if cv2.getTrackbarPos("Cursor", win) != cur:
            cv2.setTrackbarPos("Cursor", win, cur)

    def set_cursor(ms: int):
        ms = int(clamp(ms, state["start_ms"], state["end_ms"]))
        if ms != state["cursor_ms"]:
            state["cursor_ms"] = ms
            cv2.setTrackbarPos("Cursor", win, ms)
            state["dirty_frame"] = True

    def set_start(ms: int):
        ms = int(clamp(ms, 0, state["end_ms"]))
        if ms != state["start_ms"]:
            state["start_ms"] = ms
            cv2.setTrackbarPos("Start", win, ms)
            if state["cursor_ms"] < ms:
                set_cursor(ms)
            state["dirty_frame"] = True

    def set_end(ms: int):
        ms = int(clamp(ms, state["start_ms"], total_ms))
        if ms != state["end_ms"]:
            state["end_ms"] = ms
            cv2.setTrackbarPos("End", win, ms)
            if state["cursor_ms"] > ms:
                set_cursor(ms)
            state["dirty_frame"] = True

    def x_from_ms(ms, x0, width):
        return x0 + int(round((ms / max(total_ms, 1)) * width))

    def render_frame():
        if (
            not state["dirty_frame"]
            and state["last_rendered_ms"] == state["cursor_ms"]
            and state["frame"] is not None
        ):
            return state["frame"]

        frame = vr.read_at_ms(state["cursor_ms"])
        if frame is None:
            return None

        img = frame.copy()
        h, w = img.shape[:2]

        # Top HUD
        panel_h = 145
        draw_filled_panel(img, 0, 0, w, panel_h, color=(0, 0, 0), alpha=0.72)

        cursor_s = ms_to_sec(state["cursor_ms"])
        start_s = ms_to_sec(state["start_ms"])
        end_s = ms_to_sec(state["end_ms"])
        sel_s = end_s - start_s

        line1 = f"CURSOR  {cursor_s:9.3f}s"
        line2 = f"START   {start_s:9.3f}s"
        line3 = f"END     {end_s:9.3f}s"
        line4 = f"LENGTH  {sel_s:9.3f}s"

        draw_text(img, line1, (20, 38), scale=0.95)
        draw_text(img, line2, (20, 76), scale=0.95)
        draw_text(img, line3, (320, 76), scale=0.95)
        draw_text(img, line4, (620, 76), scale=0.95)

        status = "PLAYING" if state["playing"] else "PAUSED"
        draw_text(img, f"STATUS  {status}", (320, 38), scale=0.95)
        draw_text(img, "s=start@cursor   e=end@cursor   Enter=confirm   Esc=abort", (20, 116), scale=0.78)
        draw_text(img, "u/o ±0.01s   j/l ±0.1s   J/L ±1.0s   a/d ±1 frame", (620, 38), scale=0.78)
        draw_text(img, "[ ] start ±0.01   { } start ±0.1   - = end ±0.01   _ + end ±0.1", (620, 76), scale=0.78)

        # Bottom timeline
        tl_h = 105
        tl_y1 = h - tl_h
        draw_filled_panel(img, 0, tl_y1, w, h, color=(0, 0, 0), alpha=0.78)

        x0 = 30
        x1 = w - 30
        bar_w = max(10, x1 - x0)
        bar_y = h - 42

        # full timeline background
        cv2.rectangle(img, (x0, bar_y - 10), (x1, bar_y + 10), (90, 90, 90), -1)
        cv2.rectangle(img, (x0, bar_y - 10), (x1, bar_y + 10), (180, 180, 180), 1)

        xs = x_from_ms(state["start_ms"], x0, bar_w)
        xe = x_from_ms(state["end_ms"], x0, bar_w)
        xc = x_from_ms(state["cursor_ms"], x0, bar_w)

        # selected region
        cv2.rectangle(img, (xs, bar_y - 12), (xe, bar_y + 12), (0, 200, 255), -1)
        cv2.rectangle(img, (xs, bar_y - 12), (xe, bar_y + 12), (255, 255, 255), 1)

        # start/end markers
        cv2.line(img, (xs, bar_y - 28), (xs, bar_y + 28), (0, 255, 0), 3)
        cv2.line(img, (xe, bar_y - 28), (xe, bar_y + 28), (0, 0, 255), 3)

        # cursor marker
        cv2.line(img, (xc, bar_y - 34), (xc, bar_y + 34), (255, 255, 255), 2)

        # labels over the markers
        sx = clamp(xs - 45, 8, w - 110)
        ex = clamp(xe - 35, 8, w - 110)
        cx = clamp(xc - 45, 8, w - 110)

        draw_text(img, "START", (sx, h - 72), scale=0.65, fg=(0, 255, 0))
        draw_text(img, "END",   (ex, h - 72), scale=0.65, fg=(0, 0, 255))
        draw_text(img, "CURSOR",(cx, h - 12), scale=0.65, fg=(255, 255, 255))

        # endpoint time labels
        draw_text(img, "0.000s", (x0, h - 82), scale=0.65)
        total_label = f"{duration_s:.3f}s"
        draw_text(img, total_label, (max(x0, x1 - 120), h - 82), scale=0.65)

        state["frame"] = img
        state["last_rendered_ms"] = state["cursor_ms"]
        state["dirty_frame"] = False
        return img

    while True:
        sync_from_trackbars()

        now = time.time()
        if state["playing"]:
            dt = now - state["last_tick"]
            advance_ms = int(round(dt * 1000.0))
            if advance_ms > 0:
                new_ms = state["cursor_ms"] + advance_ms
                if new_ms >= state["end_ms"]:
                    new_ms = state["end_ms"]
                    state["playing"] = False
                set_cursor(new_ms)
        state["last_tick"] = now

        disp = render_frame()
        if disp is None:
            die("Failed to decode preview frame during trim selection.")

        cv2.imshow(win, disp)
        key = cv2.waitKey(15) & 0xFF

        if key == 255:
            continue

        if key in (27, ord('q')):
            cv2.destroyWindow(win)
            vr.close()
            die("Aborted by user.", code=130)

        if key in (13, 10):
            break

        if key == ord(' '):
            state["playing"] = not state["playing"]

        elif key == ord('a'):
            state["playing"] = False
            set_cursor(state["cursor_ms"] - frame_step_ms)

        elif key == ord('d'):
            state["playing"] = False
            set_cursor(state["cursor_ms"] + frame_step_ms)

        elif key == ord('u'):
            state["playing"] = False
            set_cursor(state["cursor_ms"] - 10)

        elif key == ord('o'):
            state["playing"] = False
            set_cursor(state["cursor_ms"] + 10)

        elif key == ord('j'):
            state["playing"] = False
            set_cursor(state["cursor_ms"] - 100)

        elif key == ord('l'):
            state["playing"] = False
            set_cursor(state["cursor_ms"] + 100)

        elif key == ord('J'):
            state["playing"] = False
            set_cursor(state["cursor_ms"] - 1000)

        elif key == ord('L'):
            state["playing"] = False
            set_cursor(state["cursor_ms"] + 1000)

        elif key == ord('s'):
            state["playing"] = False
            set_start(state["cursor_ms"])

        elif key == ord('e'):
            state["playing"] = False
            set_end(state["cursor_ms"])

        elif key == ord('['):
            set_start(state["start_ms"] - 10)

        elif key == ord(']'):
            set_start(state["start_ms"] + 10)

        elif key == ord('{'):
            set_start(state["start_ms"] - 100)

        elif key == ord('}'):
            set_start(state["start_ms"] + 100)

        elif key == ord('-'):
            set_end(state["end_ms"] - 10)

        elif key == ord('='):
            set_end(state["end_ms"] + 10)

        elif key == ord('_'):
            set_end(state["end_ms"] - 100)

        elif key == ord('+'):
            set_end(state["end_ms"] + 100)

        elif key == ord('r'):
            state["playing"] = False
            set_start(0)
            set_end(total_ms)
            set_cursor(0)

    start_s = ms_to_sec(state["start_ms"])
    end_s = ms_to_sec(state["end_ms"])

    cv2.destroyWindow(win)
    vr.close()

    if end_s <= start_s:
        die("Trim selection has non-positive duration.")

    return start_s, end_s


# ============================================================
# FFmpeg processing
# ============================================================

def build_ffmpeg_cmd(
    input_path: Path,
    output_path: Path,
    start_s,
    end_s,
    crop_box,
    reencode: bool,
):
    has_trim = start_s is not None and end_s is not None
    has_crop = crop_box is not None

    cmd = ["ffmpeg", "-hide_banner", "-y"]

    if has_trim and not has_crop and not reencode:
        duration = end_s - start_s
        cmd += [
            "-ss", f"{start_s:.6f}",
            "-i", str(input_path),
            "-t", f"{duration:.6f}",
            "-c", "copy",
            str(output_path),
        ]
        return cmd

    cmd += ["-i", str(input_path)]

    vf_parts = []
    if has_trim:
        vf_parts.append(f"trim=start={start_s:.6f}:end={end_s:.6f},setpts=PTS-STARTPTS")

    if has_crop:
        x, y, w, h = crop_box
        vf_parts.append(f"crop={w}:{h}:{x}:{y}")

    if vf_parts:
        cmd += ["-vf", ",".join(vf_parts)]

    if has_trim:
        cmd += ["-af", f"atrim=start={start_s:.6f}:end={end_s:.6f},asetpts=PTS-STARTPTS"]

    cmd += [
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "medium",
        "-c:a", "aac",
        "-b:a", "192k",
        "-movflags", "+faststart",
        str(output_path),
    ]
    return cmd


# ============================================================
# Main
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Interactively crop and/or trim a video."
    )
    p.add_argument("-i", "--input", required=True, help="Path to input video")
    p.add_argument("-o", "--output", help="Output path")
    p.add_argument("--crop", action="store_true", help="Open crop UI on the first frame")
    p.add_argument("--trim", action="store_true", help="Open trim UI")
    p.add_argument(
        "--reencode",
        action="store_true",
        help="Force re-encoding. Without this, trim-only uses stream copy when possible.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        die(f"Input does not exist: {input_path}")
    if not input_path.is_file():
        die(f"Input is not a file: {input_path}")

    output_path = Path(args.output).expanduser().resolve() if args.output else default_output_path(input_path)

    if output_path == input_path:
        die("Output path must be different from input path.")

    info = ffprobe_info(input_path)
    duration_s = info["duration"]
    fps = info["fps"]

    crop_box = None
    if args.crop:
        crop_box = select_crop_box(input_path)
        x, y, w, h = crop_box
        print(f"Selected crop: x={x}, y={y}, w={w}, h={h}")

    start_s = None
    end_s = None
    if args.trim:
        start_s, end_s = trim_ui(input_path, duration_s, fps)
        print(f"Selected trim: start={fmt_s(start_s)}, end={fmt_s(end_s)}")

    if not args.crop and not args.trim:
        die("Nothing to do. Use at least one of --crop or --trim.")

    if output_path.exists():
        print(f"Warning: output exists and will be overwritten: {output_path}")

    if args.crop and not args.reencode:
        print("Note: cropping requires re-encoding; proceeding with re-encode.")

    ffmpeg_cmd = build_ffmpeg_cmd(
        input_path=input_path,
        output_path=output_path,
        start_s=start_s,
        end_s=end_s,
        crop_box=crop_box,
        reencode=args.reencode or args.crop,
    )

    print("\nRunning:")
    print(" ".join(shlex.quote(x) for x in ffmpeg_cmd))
    run_cmd(ffmpeg_cmd, capture=False)

    print(f"\nWrote: {output_path}")


if __name__ == "__main__":
    main()

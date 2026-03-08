import cv2
import os
import glob
import json
import argparse

# --- Configuration ---
VIDEO_DIR = "../track/Videos"
VIDEO_EXTENSION = "MOV"

def select_crop(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"Error: Could not read {video_path}")
        return None

    h, w, _ = frame.shape
    coords = []
    temp_frame = frame.copy()

    window_name = f"Select Crop: {os.path.basename(video_path)}"

    # Enable resizing for high-res videos
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    def mouse_callback(event, x, y, flags, param):
        nonlocal coords, temp_frame
        if event == cv2.EVENT_LBUTTONDOWN and len(coords) < 2:
            coords.append(y)
            cv2.line(temp_frame, (0, y), (w, y), (0, 0, 255), 3)
            label = "TOP" if len(coords) == 1 else "BOTTOM"
            cv2.putText(
                temp_frame,
                label,
                (50, y - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (0, 0, 255),
                5,
            )
            cv2.imshow(window_name, temp_frame)

    cv2.setMouseCallback(window_name, mouse_callback)

    print(f"Interacting with: {os.path.basename(video_path)}")
    print(" 1. Click the TOP boundary.")
    print(" 2. Click the BOTTOM boundary.")
    print(" 3. Press ESC to skip.")

    while len(coords) < 2:
        cv2.imshow(window_name, temp_frame)
        if cv2.waitKey(1) & 0xFF == 27:
            cv2.destroyAllWindows()
            return None

    cv2.imshow(window_name, temp_frame)
    cv2.waitKey(500)
    cv2.destroyAllWindows()

    y_top = min(coords)
    y_bottom = max(coords)
    return {"top": y_top, "bottom": y_bottom}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--redo",
        action="store_true",
        help="Reprocess videos even if the JSON output already exists.",
    )
    args = parser.parse_args()

    video_files = sorted(glob.glob(os.path.join(VIDEO_DIR, f"*.{VIDEO_EXTENSION}")))

    if not video_files:
        print(f"No videos found in {VIDEO_DIR}")
        return

    for video_path in video_files:
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        json_path = os.path.join(VIDEO_DIR, f"{base_name}.json")

        if os.path.exists(json_path) and not args.redo:
            print(f"Already exists, skipping: {os.path.basename(json_path)}")
            continue

        result = select_crop(video_path)

        if result:
            with open(json_path, "w") as f:
                json.dump(result, f, indent=4)
            print(f"Saved: {json_path}\n")
        else:
            print(f"Skipped: {os.path.basename(video_path)}\n")

if __name__ == "__main__":
    main()

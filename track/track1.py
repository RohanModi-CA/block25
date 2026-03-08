import cv2
import numpy as np
import os
import sys
import argparse
import json
import msgpack
from dataclasses import asdict
from datetime import datetime

from tracking_classes import Track1Params, DetectionRecord, FrameDetections, VideoCentroids


def get_disk_kernel(radius: int):
    size = 2 * radius + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def imfill_holes(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(mask)
    cv2.drawContours(out, contours, -1, 255, thickness=cv2.FILLED)
    return out


def cleanup_color_mask(bw: np.ndarray, se_open: np.ndarray, se_close: np.ndarray) -> np.ndarray:
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, se_open)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, se_close)
    bw = imfill_holes(bw)
    return bw


def is_bbox_near_border(x, y, w, h, img_w, img_h, margin):
    return (x < margin) or (y < margin) or (x + w >= img_w - margin) or (y + h >= img_h - margin)


def extract_validated_detections(color_mask, white_mask, color_char, min_area, max_area,
                                 connectivity, se_inner, se_outer, min_white_frac,
                                 reject_near_border, border_margin_px, outer_radius,
                                 y_offset):

    dets = []

    if cv2.countNonZero(color_mask) == 0:
        return dets

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(color_mask, connectivity=connectivity)
    img_h, img_w = color_mask.shape

    for i in range(1, num_labels):

        area = stats[i, cv2.CC_STAT_AREA]
        if not (min_area <= area <= max_area):
            continue

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        cx, cy = centroids[i]

        if reject_near_border and is_bbox_near_border(x, y, w, h, img_w, img_h, border_margin_px):
            continue

        pad = outer_radius
        y1, y2 = max(0, y - pad), min(img_h, y + h + pad)
        x1, x2 = max(0, x - pad), min(img_w, x + w + pad)

        local_label_patch = labels[y1:y2, x1:x2]
        local_white_mask = white_mask[y1:y2, x1:x2]

        local_comp_mask = (local_label_patch == i).astype(np.uint8) * 255

        outer = cv2.dilate(local_comp_mask, se_outer)
        inner = cv2.dilate(local_comp_mask, se_inner)
        ring = cv2.bitwise_and(outer, cv2.bitwise_not(inner))

        n_ring = cv2.countNonZero(ring)
        if n_ring == 0:
            continue

        ring_white = cv2.bitwise_and(ring, local_white_mask)
        white_frac = cv2.countNonZero(ring_white) / n_ring

        if white_frac >= min_white_frac:
            dets.append(
                DetectionRecord(
                    x=float(cx),
                    y=float(cy) + y_offset,
                    color=color_char,
                    area=float(area)
                )
            )

    return dets


def main(args):

    min_area = 2500
    max_area = float('inf')
    minSat_color = 0.35
    minVal_color = 0.20

    redHueLow1, redHueHigh1 = 0.00, 0.05
    redHueLow2, redHueHigh2 = 0.95, 1.00

    greenHueLow, greenHueHigh = 0.22, 0.45

    whiteMaxSat = 0.42
    whiteMinVal = 0.00055

    colorOpenRadius = 1
    colorCloseRadius = 2
    whiteCloseRadius = 2

    ringInnerRadius = 1
    ringOuterRadius = 30

    minWhiteCoverageFraction = 0.70
    rejectNearImageBorder = True
    borderMarginPx = ringOuterRadius + 1

    ccConnectivity = 8

    params = Track1Params(
        inputVideoPath=args.input,
        outputMatPath=args.output,
        min_area=min_area,
        max_area=max_area,
        minSat_color=minSat_color,
        minVal_color=minVal_color,
        redHueLow1=redHueLow1,
        redHueHigh1=redHueHigh1,
        redHueLow2=redHueLow2,
        redHueHigh2=redHueHigh2,
        greenHueLow=greenHueLow,
        greenHueHigh=greenHueHigh,
        whiteMaxSat=whiteMaxSat,
        whiteMinVal=whiteMinVal,
        colorOpenRadius=colorOpenRadius,
        colorCloseRadius=colorCloseRadius,
        whiteCloseRadius=whiteCloseRadius,
        ringInnerRadius=ringInnerRadius,
        ringOuterRadius=ringOuterRadius,
        minWhiteCoverageFraction=minWhiteCoverageFraction,
        rejectNearImageBorder=rejectNearImageBorder,
        borderMarginPx=borderMarginPx,
        ccConnectivity=ccConnectivity,
        assumedInputType='MJPEG AVI',
        createdOn=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

    cap = cv2.VideoCapture(args.input)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    y_offset = 0
    crop_top, crop_bottom = 0, 0

    video_base = os.path.splitext(os.path.basename(args.input))[0]

    possible_json_paths = [
        os.path.join(os.path.dirname(args.input), f"{video_base}.json"),
        f"{video_base}.json"
    ]

    found_json = False

    for json_path in possible_json_paths:
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                crop_data = json.load(f)
                crop_top = int(crop_data['top'])
                crop_bottom = int(crop_data['bottom'])
                y_offset = crop_top
                found_json = True
                print(f"Loaded crop for {video_base}: Top={crop_top}, Bottom={crop_bottom}")
            break

    if args.start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    se_color_open = get_disk_kernel(colorOpenRadius)
    se_color_close = get_disk_kernel(colorCloseRadius)
    se_white_close = get_disk_kernel(whiteCloseRadius)
    se_ring_inner = get_disk_kernel(ringInnerRadius)
    se_ring_outer = get_disk_kernel(ringOuterRadius)

    cv_min_sat = int(np.clip(minSat_color * 255, 0, 255))
    cv_min_val = int(np.clip(minVal_color * 255, 0, 255))

    cv_white_max_sat = int(np.clip(whiteMaxSat * 255, 0, 255))
    cv_white_min_val = int(np.clip(whiteMinVal * 255, 0, 255))

    lower_red1 = np.array([int(redHueLow1 * 179), cv_min_sat, cv_min_val])
    upper_red1 = np.array([int(redHueHigh1 * 179), 255, 255])

    lower_red2 = np.array([int(redHueLow2 * 179), cv_min_sat, cv_min_val])
    upper_red2 = np.array([int(redHueHigh2 * 179), 255, 255])

    lower_green = np.array([int(greenHueLow * 179), cv_min_sat, cv_min_val])
    upper_green = np.array([int(greenHueHigh * 179), 255, 255])

    lower_white = np.array([0, 0, cv_white_min_val])
    upper_white = np.array([179, cv_white_max_sat, 255])

    frames_data = []
    frames_processed = 0
    total_detections_so_far = 0

    if not args.no_preview:
        window_name = f'Track1: {video_base}'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)

    while True:

        frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        t_frame = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        ret, bgr = cap.read()
        if not ret:
            break

        if found_json:
            bgr = bgr[crop_top:crop_bottom, :]

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)

        red_mask = cv2.bitwise_or(mask_red1, mask_red2)
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        white_mask = cv2.inRange(hsv, lower_white, upper_white)

        red_mask = cleanup_color_mask(red_mask, se_color_open, se_color_close)
        green_mask = cleanup_color_mask(green_mask, se_color_open, se_color_close)

        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, se_white_close)
        white_mask = imfill_holes(white_mask)

        det_r = extract_validated_detections(
            red_mask, white_mask, 'r',
            min_area, max_area, ccConnectivity,
            se_ring_inner, se_ring_outer,
            minWhiteCoverageFraction,
            rejectNearImageBorder, borderMarginPx,
            ringOuterRadius, y_offset
        )

        det_g = extract_validated_detections(
            green_mask, white_mask, 'g',
            min_area, max_area, ccConnectivity,
            se_ring_inner, se_ring_outer,
            minWhiteCoverageFraction,
            rejectNearImageBorder, borderMarginPx,
            ringOuterRadius, y_offset
        )

        detections = sorted(det_r + det_g, key=lambda d: d.x)

        total_detections_so_far += len(detections)

        frames_data.append(
            FrameDetections(
                frame_number=frame_num,
                frame_time_s=t_frame,
                detections=detections
            )
        )

        if not args.no_preview:

            display_img = bgr.copy()

            num_red = sum(1 for d in detections if d.color == 'r')
            num_green = sum(1 for d in detections if d.color == 'g')

            for d in detections:

                color = (0, 0, 255) if d.color == 'r' else (0, 255, 0)

                local_y = int(d.y - y_offset)
                local_x = int(d.x)

                cv2.drawMarker(
                    display_img,
                    (local_x, local_y),
                    color,
                    markerType=cv2.MARKER_CROSS,
                    markerSize=25,
                    thickness=3
                )

            status_text = f"Frame: {frame_num} | Detections: {len(detections)} (R:{num_red}, G:{num_green})"

            cv2.rectangle(display_img, (10, 10), (950, 70), (0, 0, 0), -1)

            cv2.putText(
                display_img,
                status_text,
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            cv2.imshow(window_name, display_img)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break

        frames_processed += 1

    cap.release()
    cv2.destroyAllWindows()

    vc = VideoCentroids(
        filepath=args.input,
        frames=frames_data,
        params=params,
        nFrames=frames_processed,
        fps=fps
    )

    with open(args.output, 'wb') as f:
        f.write(msgpack.packb(asdict(vc)))

    print("track1 complete.")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input', required=True)
    parser.add_argument('-o', '--output', required=True)

    parser.add_argument('--start-frame', type=int, default=0)
    parser.add_argument('--max-frames', type=int, default=None)

    parser.add_argument('--no-preview', action='store_true')

    args = parser.parse_args()

    main(args)

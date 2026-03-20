import os
import sys
import numpy as np
import scipy.io
import msgpack
from datetime import datetime
from dataclasses import asdict

# Import the shared classes from your project
from tracking_classes import VideoCentroids, FrameDetections, DetectionRecord, Track1Params

def convert_mat_to_msgpack(mat_path, video_path, output_path):
    print(f"Loading MATLAB file: {mat_path}")
    mat = scipy.io.loadmat(mat_path)

    found_keys = [k for k in mat.keys() if not k.startswith('__')]
    print(f"Variables found: {found_keys}")

    xR = mat.get('xR')
    yR = mat.get('yR')
    xG = mat.get('xG')
    yG = mat.get('yG')
    t_vec = mat.get('t')

    if t_vec is None:
        print("Error: Could not find time vector 't'.")
        return
    t_vec = t_vec.flatten()

    n_frames = len(t_vec)
    print(f"Processing {n_frames} frames...")

    frames_list = []
    Y_OFFSET = 300
    DUMMY_AREA = 2000.0  # Must be > 0 to pass Python verification

    for i in range(n_frames):
        detections = []
        
        # Red/Black blocks
        if xR is not None and yR is not None:
            for col in range(xR.shape[1]):
                x, y = xR[i, col], yR[i, col]
                if not np.isnan(x) and not np.isnan(y):
                    detections.append(DetectionRecord(
                        x=float(x), y=float(y) + Y_OFFSET,
                        color='r', area=DUMMY_AREA
                    ))

        # Green blocks
        if xG is not None and yG is not None:
            for col in range(xG.shape[1]):
                x, y = xG[i, col], yG[i, col]
                if not np.isnan(x) and not np.isnan(y):
                    detections.append(DetectionRecord(
                        x=float(x), y=float(y) + Y_OFFSET,
                        color='g', area=DUMMY_AREA
                    ))

        detections.sort(key=lambda d: d.x)
        frames_list.append(FrameDetections(
            frame_number=i,
            frame_time_s=float(t_vec[i]),
            detections=detections
        ))

    t1_params = Track1Params(
        inputVideoPath=video_path,
        outputMatPath=mat_path,
        assumedInputType='Converted from MATLAB (Fixed Area)',
        createdOn=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

    inferred_fps = 1.0 / np.mean(np.diff(t_vec)) if len(t_vec) > 1 else 30.0

    vc = VideoCentroids(
        filepath=video_path,
        frames=frames_list,
        params=t1_params,
        nFrames=n_frames,
        fps=inferred_fps
    )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(msgpack.packb(asdict(vc), use_bin_type=True))
    
    print(f"Conversion complete -> {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 convert_mat_to_btp.py <input.mat> <name>")
        sys.exit(1)
    convert_mat_to_msgpack(sys.argv[1], f"Videos/{sys.argv[2]}.mp4", f"data/{sys.argv[2]}/track1.msgpack")

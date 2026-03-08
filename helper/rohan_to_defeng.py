import os
import sys
import argparse
import msgpack
import numpy as np
import scipy.io as sio

# =============================================================================
# DATASET / PATH CONVENTION
# =============================================================================
# Assumes this script is in `../helper/` relative to the root, 
# and data is in `../track/data/`.
# =============================================================================

STANDARD_TRACK_ROOT = os.path.join("..", "track", "data")
STANDARD_TRACK2_NAME = "track2_x_permanence.msgpack"
DEFENG_MAT_NAME = "tracking_results.mat"

def dataset_dir_from_name(dataset: str) -> str:
    return os.path.join(STANDARD_TRACK_ROOT, dataset)

def default_input_path(dataset: str) -> str:
    return os.path.join(dataset_dir_from_name(dataset), STANDARD_TRACK2_NAME)

def default_output_path(dataset: str) -> str:
    return os.path.join(dataset_dir_from_name(dataset), DEFENG_MAT_NAME)

def resolve_paths(args):
    if args.dataset is None and args.input is None:
        raise ValueError("Provide either DATASET or -i/--input")

    input_path = args.input
    output_path = args.output

    if args.dataset is not None:
        if input_path is None:
            input_path = default_input_path(args.dataset)
        if output_path is None:
            output_path = default_output_path(args.dataset)
    
    # Fallback if no dataset provided but input is
    if output_path is None and input_path is not None:
        # Default to saving .mat next to the input .msgpack
        output_path = os.path.splitext(input_path)[0] + ".mat"

    return input_path, output_path

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert Rohan's Track2 MessagePack format to Defeng's MATLAB .mat format. "
            "Generates 'posX', 'posY', 'distPx', 'timeS', 'blockDefs', and 'fps'."
        )
    )

    parser.add_argument(
        "dataset",
        nargs="?",
        help=(
            "Dataset stem, e.g. 'IMG_0536'. Resolves input to "
            "../track/data/IMG_0536/track2_x_permanence.msgpack and "
            "output to ../track/data/IMG_0536/tracking_results.mat"
        )
    )
    parser.add_argument(
        "-i", "--input",
        default=None,
        help="Explicit input path to track2_x_permanence.msgpack."
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Explicit output path for the .mat file."
    )
    
    return parser

def convert_to_defeng_format(input_path, output_path):
    print(f"Loading: {input_path}")
    
    with open(input_path, 'rb') as f:
        # Unpack the Track2XPermanence dictionary
        data = msgpack.unpackb(f.read())

    # 1. Extract raw data
    # Matrix: nFrames x nBlocks
    x_positions = np.array(data['xPositions'], dtype=float) 
    
    # Vector: nFrames
    time_s = np.array(data['frameTimes_s'], dtype=float)
    
    # List of chars: ['g', 'r', 'g']
    block_colors_short = data['blockColors'] 
    
    n_frames, n_blocks = x_positions.shape
    
    # 2. Create Y Positions
    # Rohan's data is 1D. Defeng expects X and Y.
    # We fill Y with 0.0 where X is valid, and NaN where X is NaN.
    y_positions = np.zeros_like(x_positions)
    y_positions[np.isnan(x_positions)] = np.nan

    # 3. Create Block Definitions
    # Defeng expects a cell array (N x 2): {'B1', 'green'; 'B2', 'red'}
    color_map = {'r': 'red', 'g': 'green'}
    
    # We use an object array to emulate a MATLAB cell array of strings
    block_defs = np.empty((n_blocks, 2), dtype=object)
    
    for i, char_code in enumerate(block_colors_short):
        label_name = f"B{i+1}"
        full_color = color_map.get(char_code, 'unknown')
        block_defs[i, 0] = label_name
        block_defs[i, 1] = full_color

    # 4. Calculate Pairwise Distances (distPx)
    # Defeng calculates: distPx(:,p_i) = hypot(dx, dy);
    # Since our dy is 0, dist = abs(dx).
    # Matrix: nFrames x (nBlocks - 1)
    if n_blocks > 1:
        # Calculate diff between adjacent columns (Block i vs Block i+1)
        # Note: We take abs() because distance is magnitude
        dist_px = np.abs(np.diff(x_positions, axis=1))
    else:
        dist_px = np.empty((n_frames, 0))

    # 5. Calculate FPS
    # Estimate from average time delta
    if len(time_s) > 1:
        dt = np.diff(time_s)
        avg_dt = np.mean(dt)
        fps = 1.0 / avg_dt if avg_dt > 0 else 30.0
    else:
        fps = 30.0

    # 6. Prepare Dictionary for .mat
    # Note: Matlab expects column vectors for time
    mat_dict = {
        'posX': x_positions,
        'posY': y_positions,
        'distPx': dist_px,
        'timeS': time_s.reshape(-1, 1), # Column vector
        'blockDefs': block_defs,
        'fps': fps,
        'generated_by': 'rohan_to_defeng.py conversion'
    }

    # 7. Save
    sio.savemat(output_path, mat_dict)
    print(f"Success! Converted data saved to:\n  {output_path}")

def main():
    parser = build_parser()
    args = parser.parse_args()

    try:
        input_path, output_path = resolve_paths(args)
    except Exception as e:
        print(f"Error resolving paths: {e}", file=sys.stderr)
        sys.exit(1)

    if not os.path.isfile(input_path):
        print(f"Error: Input file not found at {input_path}", file=sys.stderr)
        sys.exit(1)

    # Ensure output directory exists
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    convert_to_defeng_format(input_path, output_path)

if __name__ == '__main__':
    main()

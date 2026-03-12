from __future__ import annotations

import numpy as np

from .models import SpacingDataset, Track2Dataset


def derive_pair_labels(block_colors: list[str]) -> list[str]:
    return [f"{block_colors[i]}{block_colors[i + 1]}" for i in range(max(0, len(block_colors) - 1))]


def derive_spacing_matrix(x_positions: np.ndarray) -> np.ndarray:
    x_positions = np.asarray(x_positions, dtype=float)
    if x_positions.ndim != 2:
        raise ValueError("x_positions must be a 2D array")
    if x_positions.shape[1] < 2:
        return np.empty((x_positions.shape[0], 0), dtype=float)
    return x_positions[:, 1:] - x_positions[:, :-1]


def derive_time_deltas(frame_times_s: np.ndarray) -> np.ndarray:
    frame_times_s = np.asarray(frame_times_s, dtype=float)
    if frame_times_s.ndim != 1:
        raise ValueError("frame_times_s must be 1D")
    out = np.empty_like(frame_times_s)
    if out.size == 0:
        return out
    out[0] = np.nan
    if out.size > 1:
        out[1:] = np.diff(frame_times_s)
    return out


def derive_velocity_matrix(x_positions: np.ndarray, frame_times_s: np.ndarray) -> np.ndarray:
    x_positions = np.asarray(x_positions, dtype=float)
    dt = derive_time_deltas(frame_times_s)

    if x_positions.ndim != 2:
        raise ValueError("x_positions must be 2D")
    if x_positions.shape[0] != dt.shape[0]:
        raise ValueError("x_positions row count must match frame_times_s length")

    out = np.full_like(x_positions, np.nan, dtype=float)
    if x_positions.shape[0] < 2:
        return out

    dx = np.diff(x_positions, axis=0)
    dt_row = dt[1:, None]
    valid_dt = np.isfinite(dt_row) & (dt_row > 0)
    out[1:] = np.where(valid_dt, dx / dt_row, np.nan)
    return out


def derive_spacing_dataset(track2: Track2Dataset) -> SpacingDataset:
    return SpacingDataset(
        track2=track2,
        pair_labels=derive_pair_labels(track2.block_colors),
        spacing_matrix=derive_spacing_matrix(track2.x_positions),
    )


def visible_counts_per_block(track2: Track2Dataset) -> np.ndarray:
    return np.sum(np.isfinite(track2.x_positions), axis=0)


def nonincreasing_visible_order_frames(track2: Track2Dataset) -> int:
    bad_order_frames = 0
    for row in track2.x_positions:
        finite_row = row[np.isfinite(row)]
        if finite_row.size >= 2 and np.any(np.diff(finite_row) <= 0):
            bad_order_frames += 1
    return bad_order_frames


def summarize_track2_positions(track2: Track2Dataset) -> dict:
    visible_mask = np.isfinite(track2.x_positions)
    nan_fraction = 1.0 - float(np.mean(visible_mask)) if visible_mask.size > 0 else float("nan")
    return {
        "n_frames": int(track2.x_positions.shape[0]),
        "n_blocks": int(track2.x_positions.shape[1]),
        "time_start_s": float(track2.frame_times_s[0]) if track2.frame_times_s.size > 0 else float("nan"),
        "time_end_s": float(track2.frame_times_s[-1]) if track2.frame_times_s.size > 0 else float("nan"),
        "nan_fraction": nan_fraction,
        "visible_counts": visible_counts_per_block(track2),
        "bad_order_frames": nonincreasing_visible_order_frames(track2),
    }

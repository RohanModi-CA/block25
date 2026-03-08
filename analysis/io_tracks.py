#!/usr/bin/env python3
"""
io_tracks.py

Shared path resolution and msgpack-loading helpers for Track2 / Track3 analysis
scripts.
"""

from __future__ import annotations

import os

import msgpack
import numpy as np


STANDARD_TRACK_ROOT = os.path.join("..", "track", "data")
STANDARD_TRACK2_NAME = "track2_x_permanence.msgpack"
STANDARD_TRACK3_NAME = "track3_analysis.msgpack"


def dataset_dir_from_name(dataset: str) -> str:
    return os.path.join(STANDARD_TRACK_ROOT, dataset)


def default_track2_path(dataset: str) -> str:
    return os.path.join(dataset_dir_from_name(dataset), STANDARD_TRACK2_NAME)


def default_track3_path(dataset: str) -> str:
    return os.path.join(dataset_dir_from_name(dataset), STANDARD_TRACK3_NAME)


def resolve_track2_track3_paths(
    dataset: str | None = None,
    track2_path: str | None = None,
    track3_path: str | None = None,
) -> tuple[str, str]:
    if dataset is None and track2_path is None and track3_path is None:
        raise ValueError("Provide either DATASET or both -i/--input and --track3")

    if dataset is not None:
        if track2_path is None:
            track2_path = default_track2_path(dataset)
        if track3_path is None:
            track3_path = default_track3_path(dataset)

    if track2_path is None or track3_path is None:
        raise ValueError("Could not determine both Track2 and Track3 paths")

    return track2_path, track3_path


def _require_file(path: str, label: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} not found: {path}")


def load_msgpack(path: str):
    with open(path, "rb") as f:
        return msgpack.unpackb(f.read(), raw=False)


def load_track2(path: str):
    _require_file(path, "Track2 file")
    return load_msgpack(path)


def load_track3(path: str):
    _require_file(path, "Track3 file")
    return load_msgpack(path)


def load_fft_dataset(
    dataset: str | None = None,
    track2_path: str | None = None,
    track3_path: str | None = None,
):
    resolved_track2, resolved_track3 = resolve_track2_track3_paths(
        dataset=dataset,
        track2_path=track2_path,
        track3_path=track3_path,
    )

    t2 = load_track2(resolved_track2)
    t3 = load_track3(resolved_track3)

    try:
        frame_times_s = np.asarray(t2["frameTimes_s"], dtype=float)
    except KeyError as exc:
        raise KeyError("Track2 is missing key 'frameTimes_s'") from exc

    try:
        spacing_matrix = np.asarray(t3["spacing_matrix"], dtype=float)
    except KeyError as exc:
        raise KeyError("Track3 is missing key 'spacing_matrix'") from exc

    try:
        pair_labels = t3["pair_colors"]
    except KeyError as exc:
        raise KeyError("Track3 is missing key 'pair_colors'") from exc

    if spacing_matrix.ndim != 2:
        raise ValueError(f"spacing_matrix must be 2D. Got shape {spacing_matrix.shape}")

    n_frames, _ = spacing_matrix.shape
    if frame_times_s.shape[0] != n_frames:
        raise ValueError(
            f"time vector length ({frame_times_s.shape[0]}) does not match "
            f"spacing rows ({n_frames})."
        )

    return {
        "track2_path": resolved_track2,
        "track3_path": resolved_track3,
        "frame_times_s": frame_times_s,
        "spacing_matrix": spacing_matrix,
        "pair_labels": pair_labels,
    }

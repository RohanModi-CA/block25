from __future__ import annotations

from pathlib import Path

import msgpack
import numpy as np

from .models import Track2Dataset


DEFAULT_TRACK_DATA_ROOT = (Path(__file__).resolve().parents[2] / "track" / "data").resolve()


def get_default_track_data_root() -> Path:
    return DEFAULT_TRACK_DATA_ROOT


def dataset_dir_from_name(dataset: str, track_data_root: str | Path | None = None) -> Path:
    root = Path(track_data_root) if track_data_root is not None else DEFAULT_TRACK_DATA_ROOT
    dataset_path: Path = root / dataset

    if not (Path.exists(dataset_path)):
        # check if we can make it exist with IMG_ in front of it.
        img_dataset = str("IMG_" + dataset)
        img_dataset_path: Path = root / img_dataset
        old_dataset_path = dataset_path
        dataset_path = img_dataset_path
        
        if not(Path.exists(dataset_path)):
            raise FileNotFoundError(f"Neither {str(old_dataset_path)} nor {str(img_dataset_path)} exists.")

    return dataset_path


def default_track2_path(dataset: str, track_data_root: str | Path | None = None) -> Path:
    return dataset_dir_from_name(dataset, track_data_root=track_data_root) / "track2_permanence.msgpack"


def resolve_track2_path(
    dataset: str | None = None,
    track2_path: str | Path | None = None,
    track_data_root: str | Path | None = None,
) -> Path:
    if dataset is None and track2_path is None:
        raise ValueError("Provide either DATASET or --track2")

    if track2_path is not None:
        return Path(track2_path)

    assert dataset is not None
    return default_track2_path(dataset, track_data_root=track_data_root)


def _require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def load_msgpack(path: str | Path):
    with open(path, "rb") as f:
        return msgpack.unpackb(f.read(), raw=False)


def load_track2_dataset(
    dataset: str | None = None,
    track2_path: str | Path | None = None,
    track_data_root: str | Path | None = None,
) -> Track2Dataset:

    resolved = resolve_track2_path(
        dataset=dataset,
        track2_path=track2_path,
        track_data_root=track_data_root,
    )
    _require_file(resolved, "Track2 permanence file")

    data = load_msgpack(resolved)

    try:
        block_colors = [str(c).lower() for c in data["blockColors"]]
    except KeyError as exc:
        raise KeyError("Track2 permanence is missing key 'blockColors'") from exc

    try:
        x_positions = np.asarray(data["xPositions"], dtype=float)
    except KeyError as exc:
        raise KeyError("Track2 permanence is missing key 'xPositions'") from exc

    try:
        frame_times_s = np.asarray(data["frameTimes_s"], dtype=float)
    except KeyError as exc:
        raise KeyError("Track2 permanence is missing key 'frameTimes_s'") from exc

    frame_numbers_raw = data.get("frameNumbers", None)
    if frame_numbers_raw is None:
        frame_numbers = np.arange(frame_times_s.shape[0], dtype=int)
    else:
        frame_numbers = np.asarray(frame_numbers_raw, dtype=int)

    if x_positions.ndim != 2:
        raise ValueError(f"xPositions must be 2D. Got shape {x_positions.shape}")

    n_frames, _ = x_positions.shape
    if frame_times_s.shape[0] != n_frames:
        raise ValueError(
            f"time vector length ({frame_times_s.shape[0]}) does not match xPositions rows ({n_frames})"
        )
    if frame_numbers.shape[0] != n_frames:
        raise ValueError(
            f"frame number vector length ({frame_numbers.shape[0]}) does not match xPositions rows ({n_frames})"
        )

    return Track2Dataset(
        dataset_name=dataset,
        track2_path=str(resolved),
        original_video_path=str(data.get("originalVideoPath", "")),
        tracking_results_path=str(data.get("trackingResultsPath", "")),
        block_colors=block_colors,
        x_positions=x_positions,
        frame_times_s=frame_times_s,
        frame_numbers=frame_numbers,
    )

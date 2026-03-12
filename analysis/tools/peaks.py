from __future__ import annotations

import csv
from pathlib import Path


def load_peaks_csv(path: str | Path) -> list[float]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Peaks file not found: {path}")

    peaks: list[float] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            for cell in row:
                text = cell.strip()
                if not text:
                    continue
                try:
                    val = float(text)
                except ValueError:
                    continue
                if val > 0:
                    peaks.append(val)

    if not peaks:
        raise ValueError("No valid positive floating point peaks found in CSV")

    return peaks


def select_active_peak_indices(
    peaks: list[float],
    *,
    disableplot: list[int] | None = None,
    onlyenableplots: list[int] | None = None,
) -> list[int]:
    all_indices = set(range(len(peaks)))
    if onlyenableplots is not None:
        active = set(int(i) for i in onlyenableplots)
    else:
        active = set(all_indices)

    active -= set(int(i) for i in (disableplot or []))
    active = {idx for idx in active if idx in all_indices}
    return sorted(active, key=lambda idx: peaks[idx], reverse=True)

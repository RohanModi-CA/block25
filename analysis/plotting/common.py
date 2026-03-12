from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize


COLORMAPS = {
    1: "viridis",
    2: "plasma",
    3: "inferno",
    4: "magma",
    5: "cividis",
    6: "turbo",
    7: "jet",
    8: "nipy_spectral",
    9: "ocean",
    10: "cubehelix",
}


def colormap_name(index: int) -> str:
    return COLORMAPS[int(index)]


def centers_to_edges(vals: np.ndarray, fallback_step: float = 1.0) -> np.ndarray:
    vals = np.asarray(vals, dtype=float)

    if vals.size == 0:
        return np.array([0.0, 1.0], dtype=float)

    if vals.size == 1:
        step = fallback_step if np.isfinite(fallback_step) and fallback_step > 0 else 1.0
        half = 0.5 * step
        return np.array([vals[0] - half, vals[0] + half], dtype=float)

    diffs = np.diff(vals)
    finite_diffs = diffs[np.isfinite(diffs) & (diffs != 0)]
    step = np.median(np.abs(finite_diffs)) if finite_diffs.size > 0 else fallback_step
    if not np.isfinite(step) or step <= 0:
        step = 1.0

    edges = np.empty(vals.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (vals[:-1] + vals[1:])
    edges[0] = vals[0] - 0.5 * (vals[1] - vals[0])
    edges[-1] = vals[-1] + 0.5 * (vals[-1] - vals[-2])

    if not np.isfinite(edges[0]):
        edges[0] = vals[0] - 0.5 * step
    if not np.isfinite(edges[-1]):
        edges[-1] = vals[-1] + 0.5 * step

    return edges


def robust_nonnegative_norm(data: np.ndarray, percentile: float = 99.0):
    finite = np.asarray(data, dtype=float)
    finite = finite[np.isfinite(finite)]

    if finite.size == 0:
        return None

    vmax = np.percentile(finite, percentile)
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = np.max(finite)

    if not np.isfinite(vmax) or vmax <= 0:
        return None

    return Normalize(vmin=0.0, vmax=vmax)


def ensure_parent_dir(path: str | Path) -> Path:
    path = Path(path)
    parent = path.parent
    if str(parent):
        parent.mkdir(parents=True, exist_ok=True)
    return path


def render_figure(fig, save: str | None = None) -> None:
    if save is not None:
        save_path = ensure_parent_dir(save)
        fig.savefig(save_path, dpi=300)
        print(f"Plot saved to: {save_path}")
    plt.show()

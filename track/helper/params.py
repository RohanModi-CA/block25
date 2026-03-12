"""
helper/params.py
TrackingParams dataclass — all tunable values for the pipeline in one place.
Supports JSON serialisation (None is used for optional/infinite values).
"""

import json
import os
from dataclasses import dataclass, asdict, fields as dc_fields
from typing import Optional


@dataclass
class TrackingParams:
    """
    All parameters used by the tracking pipeline.

    Pixel crop
    ----------
    crop_top    : int   — first row to keep (0 = no top crop)
    crop_bottom : int   — last row to keep, exclusive (0 = full height)

    Time window
    -----------
    time_start_s : float         — start time in seconds
    time_end_s   : float | None  — end time in seconds  (None = end of video)

    Area filter
    -----------
    min_area : int          — minimum blob area in pixels
    max_area : float | None — maximum blob area (None = no upper limit)

    HSV color thresholds  (all values normalised 0–1)
    --------------------
    minSat_color, minVal_color  — minimum saturation / value for color blobs
    redHueLow1/High1            — first red hue band  (wraps near 0)
    redHueLow2/High2            — second red hue band (wraps near 1)
    greenHueLow/High            — green hue band
    whiteMaxSat                 — maximum saturation for white pixels
    whiteMinVal                 — minimum value for white pixels

    Morphology radii  (pixels)
    -----------------
    colorOpenRadius  — opening radius applied to color masks
    colorCloseRadius — closing radius applied to color masks
    whiteCloseRadius — closing radius applied to white mask
    ringInnerRadius  — inner dilation radius for the white-ring test
    ringOuterRadius  — outer dilation radius for the white-ring test

    Validation
    ----------
    minWhiteCoverageFraction — fraction of the ring that must be white
    rejectNearImageBorder    — discard blobs whose bounding box touches the margin
    ccConnectivity           — connected-components connectivity (4 or 8)
    """

    # ---- Pixel crop ----
    crop_top:    int = 0
    crop_bottom: int = 0          # 0 = full height

    # ---- Time window ----
    time_start_s: float          = 0.0
    time_end_s:   Optional[float] = None

    # ---- Area filter ----
    min_area: int                = 2500
    max_area: Optional[float]    = None  # None = inf

    # ---- HSV thresholds ----
    minSat_color:  float = 0.35
    minVal_color:  float = 0.20
    redHueLow1:    float = 0.00
    redHueHigh1:   float = 0.05
    redHueLow2:    float = 0.95
    redHueHigh2:   float = 1.00
    greenHueLow:   float = 0.22
    greenHueHigh:  float = 0.45
    whiteMaxSat:   float = 0.42
    whiteMinVal:   float = 0.00055

    # ---- Morphology ----
    colorOpenRadius:  int = 1
    colorCloseRadius: int = 2
    whiteCloseRadius: int = 2
    ringInnerRadius:  int = 1
    ringOuterRadius:  int = 30

    # ---- Validation ----
    minWhiteCoverageFraction: float = 0.70
    rejectNearImageBorder:    bool  = True
    ccConnectivity:           int   = 8

    # ---- Derived (computed, not stored) ----

    @property
    def effective_max_area(self) -> float:
        """max_area as a float; None → inf."""
        return float('inf') if self.max_area is None else float(self.max_area)

    @property
    def border_margin_px(self) -> int:
        """Border exclusion margin derived from ringOuterRadius."""
        return self.ringOuterRadius + 1

    # ---- Serialisation ----

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'TrackingParams':
        known = {f.name for f in dc_fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, 'w') as fh:
            json.dump(self.to_dict(), fh, indent=2)
        print(f"  Params saved → {path}")

    @classmethod
    def load(cls, path: str) -> 'TrackingParams':
        with open(path, 'r') as fh:
            return cls.from_dict(json.load(fh))

    @classmethod
    def defaults(cls) -> 'TrackingParams':
        return cls()

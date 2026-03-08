#!/usr/bin/env python3
"""
fft_tools.py

Pure signal-processing helpers for the FFT / STFT analysis pipeline.
This module deliberately does not import matplotlib.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.signal as signal


@dataclass(frozen=True)
class ProcessedSignal:
    t: np.ndarray
    y: np.ndarray
    dt: float
    Fs: float
    nyquist: float
    proc_msg: str


@dataclass(frozen=True)
class FFTResult:
    freq: np.ndarray
    amplitude: np.ndarray


@dataclass(frozen=True)
class SpectrogramResult:
    f: np.ndarray
    t: np.ndarray
    S_complex: np.ndarray
    win_samp: int
    noverlap: int
    nfft: int


def get_longest_valid_segment(arr) -> tuple[int, int]:
    mask = ~np.isnan(arr)
    bounded = np.concatenate(([False], mask, [False]))
    diffs = np.diff(bounded.astype(int))

    starts = np.where(diffs == 1)[0]
    stops = np.where(diffs == -1)[0]

    if len(starts) == 0:
        return 0, 0

    lengths = stops - starts
    best_idx = int(np.argmax(lengths))
    return int(starts[best_idx]), int(stops[best_idx])


def get_longest_true_segment(mask) -> tuple[int, int]:
    mask = np.asarray(mask, dtype=bool)
    bounded = np.concatenate(([False], mask, [False]))
    diffs = np.diff(bounded.astype(int))

    starts = np.where(diffs == 1)[0]
    stops = np.where(diffs == -1)[0]

    if len(starts) == 0:
        return 0, 0

    lengths = stops - starts
    best_idx = int(np.argmax(lengths))
    return int(starts[best_idx]), int(stops[best_idx])


def collapse_duplicate_times(t, y):
    order = np.argsort(t)
    t_sorted = t[order]
    y_sorted = y[order]

    uniq_t, idx, counts = np.unique(t_sorted, return_index=True, return_counts=True)
    if uniq_t.size == 0:
        return uniq_t, np.array([], dtype=float)

    sums = np.add.reduceat(y_sorted, idx)
    uniq_y = sums / counts
    return uniq_t, uniq_y


def build_uniform_signal(t, y):
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    valid = np.isfinite(t) & np.isfinite(y)
    t = t[valid]
    y = y[valid]

    if t.size < 2:
        return None, None, None

    t, y = collapse_duplicate_times(t, y)
    if t.size < 2:
        return None, None, None

    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        return None, None, None

    median_dt = float(np.median(dt))
    if not np.isfinite(median_dt) or median_dt <= 0:
        return None, None, None

    t_uniform = np.arange(t[0], t[-1] + 0.5 * median_dt, median_dt, dtype=float)
    t_uniform = t_uniform[t_uniform <= (t[-1] + 1e-12)]
    if t_uniform.size < 2:
        return None, None, None

    y_uniform = np.interp(t_uniform, t, y)
    return t_uniform, y_uniform, median_dt


def preprocess_signal(
    T,
    y,
    *,
    longest: bool = False,
    handlenan: bool = False,
    min_samples: int = 10,
) -> tuple[ProcessedSignal | None, str | None]:
    T = np.asarray(T, dtype=float)
    y = np.asarray(y, dtype=float)

    if handlenan:
        valid_mask = np.isfinite(T) & np.isfinite(y)

        if longest:
            start, stop = get_longest_true_segment(valid_mask)
            if stop - start < min_samples:
                return None, "no valid data"

            t_sel = T[start:stop]
            y_sel = y[start:stop]
            finite_local = np.isfinite(t_sel) & np.isfinite(y_sel)
            t_sel = t_sel[finite_local]
            y_sel = y_sel[finite_local]
            proc_msg = f"Longest Seq: {stop - start}"
        else:
            t_sel = T[valid_mask]
            y_sel = y[valid_mask]
            if t_sel.size < 2:
                return None, "insufficient valid data"
            proc_msg = "Finite samples"
    else:
        if longest:
            start, stop = get_longest_valid_segment(y)
            if stop - start < min_samples:
                return None, "no valid data"

            t_sel = T[start:stop]
            y_sel = y[start:stop]
            proc_msg = f"Longest Seq: {stop - start}"
        else:
            nan_mask = np.isnan(y)
            if np.any(nan_mask):
                valid_idx = np.where(~nan_mask)[0]
                if valid_idx.size < 2:
                    return None, "insufficient valid data"
                y_sel = np.interp(np.arange(len(y)), valid_idx, y[valid_idx])
            else:
                y_sel = y.copy()

            t_sel = T.copy()
            proc_msg = "Interpolated"

    t_uniform, y_uniform, dt = build_uniform_signal(t_sel, y_sel)
    if t_uniform is None or y_uniform is None or dt is None:
        return None, "could not build uniform signal"

    if len(y_uniform) < min_samples:
        return None, "insufficient valid data"

    y_uniform = y_uniform - np.mean(y_uniform)
    y_uniform = signal.detrend(y_uniform, type="linear")

    Fs = 1.0 / dt
    nyquist = 0.5 * Fs

    return ProcessedSignal(
        t=t_uniform,
        y=y_uniform,
        dt=dt,
        Fs=Fs,
        nyquist=nyquist,
        proc_msg=proc_msg,
    ), None


def hann_window_symmetric(n: int):
    return signal.windows.hann(n, sym=True)


def hann_window_periodic(n: int):
    return signal.windows.hann(n, sym=False)


def compute_one_sided_fft(y, dt) -> FFTResult:
    y = np.asarray(y, dtype=float)
    n = len(y)

    w = hann_window_symmetric(n)
    x = y * w

    X = np.fft.rfft(x)
    P1 = np.abs(X / n)
    if P1.size > 2:
        P1[1:-1] *= 2.0

    f = np.fft.rfftfreq(n, d=dt)
    return FFTResult(freq=f, amplitude=P1)


def next_power_of_two(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << int(np.ceil(np.log2(n)))


def compute_complex_spectrogram(y, Fs: float, sliding_len_s: float) -> SpectrogramResult | None:
    n = len(y)
    win_samp = max(8, int(round(sliding_len_s * Fs)))
    win_samp = min(win_samp, n)

    if win_samp < 4:
        return None

    noverlap = min(int(round(0.90 * win_samp)), win_samp - 1)
    nfft = max(win_samp, next_power_of_two(win_samp))
    w = hann_window_periodic(win_samp)

    f_spec, t_spec, S_complex = signal.spectrogram(
        y,
        fs=Fs,
        window=w,
        nperseg=win_samp,
        noverlap=noverlap,
        nfft=nfft,
        detrend=False,
        return_onesided=True,
        scaling="spectrum",
        mode="complex",
    )

    return SpectrogramResult(
        f=f_spec,
        t=t_spec,
        S_complex=S_complex,
        win_samp=win_samp,
        noverlap=noverlap,
        nfft=nfft,
    )

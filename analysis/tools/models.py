from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class Track2Dataset:
    dataset_name: str | None
    track2_path: str
    original_video_path: str
    tracking_results_path: str
    block_colors: list[str]
    x_positions: np.ndarray
    frame_times_s: np.ndarray
    frame_numbers: np.ndarray


@dataclass(frozen=True)
class SpacingDataset:
    track2: Track2Dataset
    pair_labels: list[str]
    spacing_matrix: np.ndarray


@dataclass(frozen=True)
class DatasetSelection:
    include: bool
    discards: list[int]
    pair_ids: list[int]


@dataclass(frozen=True)
class SignalRecord:
    dataset_name: str
    entity_id: int
    local_index: int
    label: str
    signal_kind: Literal["bond", "site"]
    source_path: str
    t: np.ndarray
    y: np.ndarray


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
class ComplexFFTResult:
    freq: np.ndarray
    spectrum: np.ndarray
    amplitude: np.ndarray


@dataclass(frozen=True)
class SpectrogramResult:
    f: np.ndarray
    t: np.ndarray
    S_complex: np.ndarray
    win_samp: int
    noverlap: int
    nfft: int


@dataclass(frozen=True)
class TargetFrequencyResult:
    target_freq: float
    selected_freq: float
    index: int
    complex_value: complex
    amplitude: float
    found: bool


@dataclass(frozen=True)
class SpectrumContribution:
    record: SignalRecord
    processed: ProcessedSignal
    fft_result: FFTResult


@dataclass(frozen=True)
class AverageSpectrumResult:
    freq_grid: np.ndarray
    avg_amp: np.ndarray
    norm_low: float
    norm_high: float
    freq_low: float
    freq_high: float
    contributors: list[SpectrumContribution]


@dataclass(frozen=True)
class PairFrequencyAnalysisResult:
    pair_index: int
    label: str
    processed: ProcessedSignal | None
    fft_result: FFTResult | None
    spectrogram_result: SpectrogramResult | None
    error_message: str | None = None
    spectrogram_error_message: str | None = None


@dataclass(frozen=True)
class LocalizationProfile:
    peak_index: int
    frequency: float
    entity_ids: np.ndarray
    mean_amplitudes: np.ndarray
    std_amplitudes: np.ndarray

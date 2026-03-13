from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from tools.models import AverageSpectrumResult, PairFrequencyAnalysisResult, SiteAmplitudeAnalysisResult
from .common import centers_to_edges, colormap_name, robust_nonnegative_norm
from .indexed import overlay_indexed_points


def _set_panel_message(
    *,
    ax_fft=None,
    ax_spec=None,
    only: str | None = None,
    full_image: bool = False,
    title: str,
) -> None:
    if only != "sliding" and ax_fft is not None:
        ax_fft.set_title(title)
        ax_fft.set_xlabel("Frequency (Hz)")
        ax_fft.set_ylabel("Amplitude")

    if only != "fft" and ax_spec is not None:
        ax_spec.set_title(title)
        ax_spec.set_xlabel("Arbitrary X" if full_image else "Time (s)")
        ax_spec.set_ylabel("Frequency (Hz)")


def _plot_fft_curve_panel(
    ax,
    *,
    freq: np.ndarray,
    amp: np.ndarray,
    log_scale: bool,
    x_min: float,
    x_max: float,
    title: str,
) -> None:
    if log_scale:
        ax.semilogy(freq, amp, linewidth=1)
        positive_vals = amp[amp > 0]
        if positive_vals.size > 0:
            ymin = np.percentile(positive_vals, 0.1) * 0.7
            ymax = np.max(positive_vals) * 1.3
            if np.isfinite(ymin) and np.isfinite(ymax) and ymin > 0 and ymax > ymin:
                ax.set_ylim(ymin, ymax)
    else:
        ax.plot(freq, amp, linewidth=1)

    ax.set_title(title)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")
    ax.set_xlim(x_min, x_max)
    ax.grid(True, alpha=0.3)


def _plot_frequency_image(
    fig,
    ax,
    *,
    freq: np.ndarray,
    amp: np.ndarray,
    plot_scale: str,
    cmap_index: int,
    y_min: float,
    y_max: float,
    x_label: str = "Arbitrary X",
    x_max: float = 1.0,
    title: str,
    linear_color_label: str = "Amplitude",
    log_color_label: str = "Amplitude (dB)",
    reference_amp_for_norm: np.ndarray | None = None,
    overlay: dict | None = None,
) -> None:
    mask = (freq >= y_min) & (freq <= y_max)
    if not np.any(mask):
        ax.set_title(title + " - no frequencies in range")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Frequency (Hz)")
        return

    freq_img = freq[mask]
    amp_img = amp[mask]

    x_cols = 64
    image_2d = np.tile(amp_img[:, None], (1, x_cols))
    eps = np.finfo(float).eps

    if plot_scale == "log":
        image_plot = 20.0 * np.log10(image_2d + eps)
        pcm_norm = None
        color_label = log_color_label
    else:
        image_plot = image_2d
        color_label = linear_color_label
        norm_source = reference_amp_for_norm if reference_amp_for_norm is not None else image_plot
        pcm_norm = robust_nonnegative_norm(norm_source)

    x_edges = np.linspace(0.0, x_max, x_cols + 1)
    fallback_step = float(np.median(np.diff(freq_img))) if freq_img.size > 1 else max(1e-6, y_max - y_min)
    y_edges = centers_to_edges(freq_img, fallback_step=fallback_step)

    pcm = ax.pcolormesh(
        x_edges,
        y_edges,
        image_plot,
        shading="flat",
        cmap=colormap_name(cmap_index),
        norm=pcm_norm,
    )
    fig.colorbar(pcm, ax=ax, label=color_label)

    ax.set_xlabel(x_label)
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlim(0.0, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title(title)

    if overlay is None:
        if x_max == 1.0:
            ax.set_xticks([])
    else:
        overlay_indexed_points(
            ax,
            overlay["x_values"],
            overlay["y_values"],
            mode=overlay.get("mode", "scatter"),
            color=overlay.get("color", "red"),
            marker_size=float(overlay.get("marker_size", 30.0)),
        )
        if overlay.get("integer_ticks", True):
            x_ticks_stop = int(np.ceil(x_max))
            ax.set_xticks(np.arange(0, x_ticks_stop, 1))


def _plot_spectrogram_panel(
    fig,
    ax,
    *,
    f_spec: np.ndarray,
    t_spec: np.ndarray,
    s_complex: np.ndarray,
    t_start: float,
    plot_scale: str,
    cmap_index: int,
    y_min: float,
    y_max: float,
    title: str,
    time_interval: tuple[float, float] | None = None,
) -> None:
    mag = np.abs(s_complex)
    eps = np.finfo(float).eps

    if plot_scale == "log":
        s_plot = 20.0 * np.log10(mag + eps)
        pcm_norm = None
        color_label = "dB"
    else:
        s_plot = mag ** 2
        pcm_norm = robust_nonnegative_norm(s_plot)
        color_label = "Power"

    t_global = t_spec + t_start
    t_step = (t_global[1] - t_global[0]) if t_global.size > 1 else 1.0
    f_step = (f_spec[1] - f_spec[0]) if f_spec.size > 1 else max(1e-6, y_max - y_min)

    t_edges = centers_to_edges(t_global, fallback_step=t_step)
    f_edges = centers_to_edges(f_spec, fallback_step=f_step)

    pcm = ax.pcolormesh(
        t_edges,
        f_edges,
        s_plot,
        shading="flat",
        cmap=colormap_name(cmap_index),
        norm=pcm_norm,
    )
    fig.colorbar(pcm, ax=ax, label=color_label)

    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_ylim(y_min, y_max)
    if time_interval is not None:
        ax.set_xlim(*time_interval)


def plot_pair_frequency_grid(
    results: list[PairFrequencyAnalysisResult],
    *,
    fft_log: bool = False,
    sliding_plot_scale: str = "log",
    only: str | None = None,
    full_image: bool = False,
    fft_min_hz: float | None = None,
    fft_max_hz: float | None = None,
    sliding_min_hz: float | None = None,
    sliding_max_hz: float | None = None,
    time_interval: tuple[float, float] | None = None,
    cmap_index: int = 6,
    title: str | None = None,
):
    if len(results) == 0:
        raise ValueError("No pair results to plot")

    ncols = 1 if only in ("fft", "sliding") else 2
    fig, axes = plt.subplots(
        nrows=len(results),
        ncols=ncols,
        figsize=(14, 3.5 * len(results)),
        constrained_layout=True,
    )

    if ncols == 1:
        axes = np.atleast_1d(axes)
    else:
        axes = np.atleast_2d(axes)

    for row_idx, result in enumerate(results):
        if only == "fft":
            ax_fft = axes[row_idx]
            ax_spec = None
        elif only == "sliding":
            ax_fft = None
            ax_spec = axes[row_idx]
        else:
            ax_fft = axes[row_idx][0]
            ax_spec = axes[row_idx][1]

        pair_title_prefix = f"Pair {result.pair_index} ({str(result.label).upper()})"

        if result.error_message is not None or result.processed is None or result.fft_result is None:
            _set_panel_message(
                ax_fft=ax_fft,
                ax_spec=ax_spec,
                only=only,
                full_image=full_image,
                title=f"{pair_title_prefix} - {result.error_message or 'invalid data'}",
            )
            continue

        fft_xlim_min = max(0.0, fft_min_hz if fft_min_hz is not None else 0.0)
        fft_xlim_max = min(
            result.processed.nyquist,
            fft_max_hz if fft_max_hz is not None else result.processed.nyquist,
        )

        if only != "sliding":
            _plot_fft_curve_panel(
                ax_fft,
                freq=result.fft_result.freq,
                amp=result.fft_result.amplitude,
                log_scale=fft_log,
                x_min=fft_xlim_min,
                x_max=fft_xlim_max,
                title=f"{pair_title_prefix} FFT | {result.processed.proc_msg}",
            )

        if only == "fft":
            continue

        y_min = max(0.01, sliding_min_hz if sliding_min_hz is not None else 0.01)
        y_max = min(
            result.processed.nyquist,
            sliding_max_hz if sliding_max_hz is not None else result.processed.nyquist,
        )

        if y_max <= y_min:
            _set_panel_message(
                ax_fft=ax_fft,
                ax_spec=ax_spec,
                only=only,
                full_image=full_image,
                title=f"{pair_title_prefix} - invalid frequency range",
            )
            continue

        if full_image:
            _plot_frequency_image(
                fig,
                ax_spec,
                freq=result.fft_result.freq,
                amp=result.fft_result.amplitude,
                plot_scale=sliding_plot_scale,
                cmap_index=cmap_index,
                y_min=y_min,
                y_max=y_max,
                title=f"{pair_title_prefix} Full FFT Image",
                linear_color_label="Amplitude",
                log_color_label="Amplitude (dB)",
            )
            continue

        if result.spectrogram_result is None:
            _set_panel_message(
                ax_fft=ax_fft,
                ax_spec=ax_spec,
                only=only,
                full_image=full_image,
                title=f"{pair_title_prefix} - {result.spectrogram_error_message or 'window too short'}",
            )
            continue

        _plot_spectrogram_panel(
            fig,
            ax_spec,
            f_spec=result.spectrogram_result.f,
            t_spec=result.spectrogram_result.t,
            s_complex=result.spectrogram_result.S_complex,
            t_start=float(result.processed.t[0]),
            plot_scale=sliding_plot_scale,
            cmap_index=cmap_index,
            y_min=y_min,
            y_max=y_max,
            title=f"{pair_title_prefix} Sliding FFT",
            time_interval=time_interval,
        )

    if title:
        fig.suptitle(title, fontsize=14)

    return fig


def plot_average_spectrum(
    result: AverageSpectrumResult,
    *,
    full_image: bool = False,
    plot_scale: str = "linear",
    cmap_index: int = 6,
    title: str | None = None,
    reference_amp_for_norm: np.ndarray | None = None,
    overlay: dict | None = None,
):
    fig, ax = plt.subplots(figsize=(12, 5))

    if full_image:
        x_label = overlay.get("x_label", "Arbitrary X") if overlay is not None else "Arbitrary X"
        x_max = float(overlay.get("x_max", 1.0)) if overlay is not None else 1.0

        _plot_frequency_image(
            fig,
            ax,
            freq=result.freq_grid,
            amp=result.avg_amp,
            plot_scale=plot_scale,
            cmap_index=cmap_index,
            y_min=float(result.freq_grid[0]),
            y_max=float(result.freq_grid[-1]),
            x_label=x_label,
            x_max=x_max,
            title=title or "Average FFT",
            linear_color_label="Normalized Amplitude",
            log_color_label="Amplitude (dB)",
            reference_amp_for_norm=reference_amp_for_norm,
            overlay=overlay,
        )
    else:
        if plot_scale == "log":
            ax.semilogy(result.freq_grid, result.avg_amp, linewidth=1.5)
        else:
            ax.plot(result.freq_grid, result.avg_amp, linewidth=1.5)

        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Normalized Amplitude")
        ax.set_xlim(result.freq_grid[0], result.freq_grid[-1])
        ax.grid(True, alpha=0.3)
        ax.set_title(title or "Average FFT")

    fig.tight_layout()
    return fig


def plot_site_amplitude_previews(
    result: SiteAmplitudeAnalysisResult,
    *,
    title: str | None = None,
):
    if len(result.bonds) == 0:
        raise ValueError("No bond spectra available to preview")

    nrows = len(result.bonds)
    fig, axes = plt.subplots(
        nrows,
        1,
        figsize=(11, 2.8 * nrows),
        sharex=True,
        constrained_layout=True,
    )
    if nrows == 1:
        axes = [axes]

    for ax, bond in zip(axes, result.bonds):
        ax.plot(bond.roi_freq, bond.roi_normalized_amplitude, linewidth=1.5)
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.35)

        for peak in bond.peak_integrals:
            ax.axvspan(peak.low_hz, peak.high_hz, alpha=0.12)
            ax.axvline(peak.peak_hz, linestyle="--", linewidth=0.9, alpha=0.4, color="black")

        ax.set_ylabel("Norm. Amp.")
        ax.set_title(
            f"Bond {bond.display_bond_index} | contributors={len(bond.contributors)} | "
            f"area={bond.normalization_integral:.6g}"
        )
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Frequency (Hz)")
    axes[-1].set_xlim(result.roi_low, result.roi_high)

    if title is not None:
        fig.suptitle(title, fontsize=14)

    return fig

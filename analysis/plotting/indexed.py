from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from tools.models import LocalizationProfile


def overlay_indexed_points(
    ax,
    x_values,
    y_values,
    *,
    mode: str = "scatter",
    color: str = "red",
    marker_size: float = 30.0,
) -> None:
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)

    if mode == "scatter":
        ax.scatter(
            x_values,
            y_values,
            s=marker_size,
            c=color,
            marker="o",
            zorder=10,
        )
    elif mode == "line":
        ax.plot(
            x_values,
            y_values,
            marker="o",
            markersize=marker_size,
            markerfacecolor=color,
            markeredgecolor=color,
            color=color,
            linestyle="-",
            zorder=10,
        )
    else:
        raise ValueError("mode must be 'scatter' or 'line'")


def plot_localization_profiles(
    profiles: list[LocalizationProfile],
    *,
    xlabel: str,
    ylabel: str = "Norm. Amplitude",
    title: str | None = None,
    line_color: str | None = None,
):
    if len(profiles) == 0:
        raise ValueError("No localization profiles to plot")

    nrows = len(profiles)
    fig, axes = plt.subplots(
        nrows,
        1,
        figsize=(10, 3 * nrows),
        sharex=True,
        constrained_layout=True,
    )
    if nrows == 1:
        axes = [axes]

    all_ids = sorted(
        {
            int(entity_id)
            for profile in profiles
            for entity_id in profile.entity_ids.tolist()
        }
    )
    xticks = np.arange(min(all_ids), max(all_ids) + 1, 1) if all_ids else np.array([])

    for ax, profile in zip(axes, profiles):
        x_vals = profile.entity_ids
        y_vals = profile.mean_amplitudes
        y_errs = profile.std_amplitudes

        if x_vals.size == 0:
            ax.text(0.5, 0.5, "No Data", transform=ax.transAxes, ha="center", va="center")
            ax.set_title(f"Peak {profile.peak_index}: {profile.frequency} Hz")
            if xticks.size > 0:
                ax.set_xticks(xticks)
                ax.tick_params(axis="x", labelbottom=True)
            continue

        line_kwargs = dict(
            marker="o",
            linestyle="-",
            linewidth=1.5,
            markersize=5,
        )
        if line_color is not None:
            line_kwargs["color"] = line_color

        ax.plot(x_vals, y_vals, **line_kwargs)
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.35)

        if np.any(y_errs > 0):
            band_kwargs = dict(alpha=0.2)
            if line_color is not None:
                band_kwargs["color"] = line_color

            ax.fill_between(
                x_vals,
                y_vals - y_errs,
                y_vals + y_errs,
                **band_kwargs,
            )

        ax.set_title(f"Peak {profile.peak_index}: {profile.frequency} Hz")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

        if xticks.size > 0:
            ax.set_xticks(xticks)
            ax.set_xlim(xticks[0], xticks[-1])
            ax.tick_params(axis="x", labelbottom=True)

    for ax in axes:
        ax.set_xlabel(xlabel)

    if title is not None:
        fig.suptitle(title, fontsize=14)

    return fig

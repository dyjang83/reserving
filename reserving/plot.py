"""
Visualization module for the reserving library.

All plot functions accept a fitted method object (ChainLadder,
BornhuetterFerguson, or CapeCod) and return a matplotlib Figure
so callers can save, display, or further customize the output.

Usage:
    import reserving.plot as rvplot
    fig = rvplot.development_chart(cl)
    fig.savefig("development.png")
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from typing import Optional


# ------------------------------------------------------------------ #
# Helpers                                                             #
# ------------------------------------------------------------------ #

def _dollar_formatter(x: float, _) -> str:
    """Format axis tick as $Xk or $XM."""
    if abs(x) >= 1_000_000:
        return f"${x/1_000_000:.1f}M"
    if abs(x) >= 1_000:
        return f"${x/1_000:.0f}k"
    return f"${x:.0f}"


def _apply_style(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    """Apply consistent styling to an axes object."""
    ax.set_title(title, fontsize=13, fontweight="normal", pad=12)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=10)


# ------------------------------------------------------------------ #
# Development chart                                                   #
# ------------------------------------------------------------------ #

def development_chart(
    triangle,
    figsize: tuple = (11, 5),
    title: Optional[str] = None,
    color: str = "#4a90d9",
) -> plt.Figure:
    """
    Plot cumulative loss development curves for each accident year.

    Shows how losses develop over time for each accident year. Observed
    values are plotted as solid lines; future (projected) periods are
    not shown — use ultimates_chart() to see projections.

    Parameters
    ----------
    triangle : Triangle
        The triangle whose development curves to plot.
    figsize : tuple
        Figure size (width, height) in inches.
    title : str, optional
        Plot title. Defaults to 'Loss development curves'.
    color : str
        Base color for the lines.

    Returns
    -------
    matplotlib.figure.Figure
    """
    from reserving.triangle import Triangle
    if not isinstance(triangle, Triangle):
        raise TypeError(f"Expected Triangle, got {type(triangle).__name__}")

    fig, ax = plt.subplots(figsize=figsize)
    data = triangle.data
    origin_years = list(triangle.origin_years)
    cmap = plt.cm.Blues
    colors = [cmap(0.4 + 0.5 * i / max(len(origin_years) - 1, 1))
              for i in range(len(origin_years))]

    for i, ay in enumerate(origin_years):
        row = data.loc[ay].dropna()
        ax.plot(
            row.index, row.values,
            marker="o", markersize=5,
            color=colors[i], linewidth=1.8,
            label=str(ay)
        )

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_dollar_formatter))
    ax.set_xticks(list(triangle.dev_lags))
    _apply_style(
        ax,
        title=title or "Loss development curves",
        xlabel="Development lag",
        ylabel="Cumulative loss"
    )
    ax.legend(title="Accident year", fontsize=9, title_fontsize=9,
              loc="upper left", framealpha=0.7)
    plt.tight_layout()
    return fig


# ------------------------------------------------------------------ #
# Ultimates chart                                                     #
# ------------------------------------------------------------------ #

def ultimates_chart(
    model,
    figsize: tuple = (10, 5),
    title: Optional[str] = None,
    show_diagonal: bool = True,
    color: str = "#4a90d9",
) -> plt.Figure:
    """
    Plot ultimate loss estimates vs latest diagonal by accident year.

    Parameters
    ----------
    model : ChainLadder, BornhuetterFerguson, or CapeCod
        A fitted reserving model.
    figsize : tuple
        Figure size in inches.
    title : str, optional
        Plot title.
    show_diagonal : bool
        If True, also plot the latest diagonal for comparison.
    color : str
        Bar color for ultimate estimates.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _check_fitted(model)
    ultimates = model.ultimates()
    origin_years = [str(y) for y in ultimates.index]
    x = np.arange(len(origin_years))
    width = 0.35

    fig, ax = plt.subplots(figsize=figsize)

    if show_diagonal:
        diagonal = model._triangle.latest_diagonal
        ax.bar(x - width / 2, diagonal.values, width,
               label="Latest diagonal", color="#aaaaaa", alpha=0.85)
        ax.bar(x + width / 2, ultimates.values, width,
               label="Ultimate estimate", color=color, alpha=0.9)
    else:
        ax.bar(x, ultimates.values, width * 1.5,
               label="Ultimate estimate", color=color, alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(origin_years)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_dollar_formatter))
    _apply_style(
        ax,
        title=title or f"Ultimate estimates — {type(model).__name__}",
        xlabel="Accident year",
        ylabel="Loss"
    )
    ax.legend(fontsize=9, framealpha=0.7)
    plt.tight_layout()
    return fig


# ------------------------------------------------------------------ #
# IBNR chart                                                          #
# ------------------------------------------------------------------ #

def ibnr_chart(
    model,
    figsize: tuple = (10, 5),
    title: Optional[str] = None,
    color: str = "#4a90d9",
) -> plt.Figure:
    """
    Plot IBNR reserve estimates by accident year.

    Parameters
    ----------
    model : ChainLadder, BornhuetterFerguson, or CapeCod
        A fitted reserving model.
    figsize : tuple
        Figure size in inches.
    title : str, optional
        Plot title.
    color : str
        Bar color.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _check_fitted(model)
    ibnr = model.ibnr()
    origin_years = [str(y) for y in ibnr.index]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(origin_years, ibnr.values, color=color, alpha=0.85)

    # Label each bar with its value
    for bar, val in zip(bars, ibnr.values):
        if not np.isnan(val) and val != 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(ibnr.values) * 0.01,
                _dollar_formatter(val, None),
                ha="center", va="bottom", fontsize=9
            )

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_dollar_formatter))
    _apply_style(
        ax,
        title=title or f"IBNR reserves — {type(model).__name__}",
        xlabel="Accident year",
        ylabel="IBNR"
    )
    total = ibnr.sum()
    ax.set_xlabel(
        f"Accident year  (total IBNR: {_dollar_formatter(total, None)})",
        fontsize=11
    )
    plt.tight_layout()
    return fig


# ------------------------------------------------------------------ #
# Summary chart with confidence intervals                             #
# ------------------------------------------------------------------ #

def summary_chart(
    model,
    n_boot: int = 500,
    alpha: float = 0.05,
    figsize: tuple = (11, 5),
    title: Optional[str] = None,
    color: str = "#4a90d9",
) -> plt.Figure:
    """
    Plot ultimate estimates with bootstrap confidence interval bands.

    Parameters
    ----------
    model : ChainLadder, BornhuetterFerguson, or CapeCod
        A fitted reserving model.
    n_boot : int
        Number of bootstrap resamples for CIs.
    alpha : float
        Significance level (default 0.05 → 95% CI).
    figsize : tuple
        Figure size in inches.
    title : str, optional
        Plot title.
    color : str
        Color for the point estimate line and markers.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _check_fitted(model)
    summary = model.summary(alpha=alpha, n_boot=n_boot)
    origin_years = [str(y) for y in summary.index]
    x = np.arange(len(origin_years))
    ci_pct = int((1 - alpha) * 100)

    fig, ax = plt.subplots(figsize=figsize)

    # CI band
    ax.fill_between(
        x,
        summary["ci_lower"].values,
        summary["ci_upper"].values,
        alpha=0.18, color=color,
        label=f"{ci_pct}% CI"
    )

    # CI bounds
    ax.plot(x, summary["ci_lower"].values,
            color=color, linewidth=0.8, linestyle="--", alpha=0.6)
    ax.plot(x, summary["ci_upper"].values,
            color=color, linewidth=0.8, linestyle="--", alpha=0.6)

    # Point estimates
    ax.plot(x, summary["ultimate"].values,
            color=color, linewidth=2, marker="o", markersize=6,
            label="Ultimate estimate")

    # Latest diagonal
    ax.plot(x, summary["latest"].values,
            color="#aaaaaa", linewidth=1.5, marker="s", markersize=5,
            linestyle="--", label="Latest diagonal")

    ax.set_xticks(x)
    ax.set_xticklabels(origin_years)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_dollar_formatter))
    _apply_style(
        ax,
        title=title or f"Ultimates with {ci_pct}% bootstrap CI — {type(model).__name__}",
        xlabel="Accident year",
        ylabel="Loss"
    )
    ax.legend(fontsize=9, framealpha=0.7)
    plt.tight_layout()
    return fig


# ------------------------------------------------------------------ #
# Method comparison chart                                             #
# ------------------------------------------------------------------ #

def comparison_chart(
    models: dict,
    figsize: tuple = (11, 5),
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Compare ultimate estimates from multiple methods side by side.

    Parameters
    ----------
    models : dict
        Mapping of label to fitted model. Example:
        {"Chain-ladder": cl, "BF": bf, "Cape Cod": cc}
    figsize : tuple
        Figure size in inches.
    title : str, optional
        Plot title.

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    >>> fig = rvplot.comparison_chart({
    ...     "Chain-ladder": cl,
    ...     "BF": bf,
    ...     "Cape Cod": cc,
    ... })
    """
    if not models:
        raise ValueError("models dict must not be empty")

    for label, model in models.items():
        _check_fitted(model)

    colors = ["#4a90d9", "#1D9E75", "#EF9F27", "#D85A30", "#7F77DD"]
    first_model = next(iter(models.values()))
    origin_years = [str(y) for y in first_model.ultimates().index]
    n_years = len(origin_years)
    n_models = len(models)
    width = 0.8 / n_models
    x = np.arange(n_years)

    fig, ax = plt.subplots(figsize=figsize)

    for i, (label, model) in enumerate(models.items()):
        offset = (i - n_models / 2 + 0.5) * width
        ax.bar(
            x + offset,
            model.ultimates().values,
            width,
            label=label,
            color=colors[i % len(colors)],
            alpha=0.85
        )

    ax.set_xticks(x)
    ax.set_xticklabels(origin_years)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_dollar_formatter))
    _apply_style(
        ax,
        title=title or "Ultimate estimates — method comparison",
        xlabel="Accident year",
        ylabel="Ultimate loss"
    )
    ax.legend(fontsize=9, framealpha=0.7)
    plt.tight_layout()
    return fig


# ------------------------------------------------------------------ #
# Internal guard                                                      #
# ------------------------------------------------------------------ #

def _check_fitted(model) -> None:
    """Raise if model has not been fitted."""
    if not getattr(model, "_fitted", False):
        raise RuntimeError(
            f"{type(model).__name__} must be fitted before plotting. "
            "Call .fit() first."
        )

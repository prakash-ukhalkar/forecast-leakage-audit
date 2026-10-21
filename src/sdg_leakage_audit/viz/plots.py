"""Figure-generation functions: EDA plots, protocol timeline diagrams, inflation charts.

EDA plotting functions (time series, ACF/PACF, missingness, distribution shift) are
implemented now for notebook 03. Protocol-timeline and inflation-chart functions are
implemented in Phase 2 notebooks 05 and 08 respectively.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from sdg_leakage_audit.utils.logging import get_logger
from sdg_leakage_audit.utils.paths import to_project_relative

logger = get_logger(__name__)


def _save_or_show(fig: plt.Figure, save_path: Path | None) -> None:
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        logger.info("Saved figure -> %s", to_project_relative(save_path))
        plt.close(fig)
    else:
        plt.show()


def plot_time_series(series: pd.Series, title: str, save_path: Path | None = None) -> None:
    """Plot a single time series line chart.

    Args:
        series: Time-indexed numeric series to plot.
        title: Plot title (also used as the y-axis label).
        save_path: If given, save the figure to this path instead of showing it.
    """
    fig, ax = plt.subplots()
    ax.plot(series.index, series.values, linewidth=0.7)
    ax.set_title(title)
    ax.set_xlabel("time")
    ax.set_ylabel(series.name or "value")
    _save_or_show(fig, save_path)


def plot_acf_pacf(series: pd.Series, lags: int, save_path: Path | None = None) -> None:
    """Plot ACF and PACF side by side for a series (NaNs dropped before computing).

    Args:
        series: Time-indexed numeric series.
        lags: Number of lags to display.
        save_path: If given, save the figure to this path instead of showing it.
    """
    clean = series.dropna()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    plot_acf(clean, lags=lags, ax=axes[0])
    plot_pacf(clean, lags=lags, ax=axes[1], method="ywm")
    axes[0].set_title(f"ACF: {series.name}")
    axes[1].set_title(f"PACF: {series.name}")
    fig.tight_layout()
    _save_or_show(fig, save_path)


def plot_missingness_heatmap(df: pd.DataFrame, save_path: Path | None = None, max_rows: int = 2000) -> None:
    """Plot a missingness heatmap (rows = time, downsampled; columns = fields).

    Large series are downsampled to at most `max_rows` evenly spaced rows before
    plotting, since per-row rendering of millions of timestamps is both unreadable
    and slow.

    Args:
        df: DataFrame to visualize missingness for.
        save_path: If given, save the figure to this path instead of showing it.
        max_rows: Maximum number of rows to render after downsampling.
    """
    numeric_df = df.select_dtypes(include="number")
    if len(numeric_df) > max_rows:
        step = len(numeric_df) // max_rows
        numeric_df = numeric_df.iloc[::step]

    missing = numeric_df.isna().to_numpy().T
    fig, ax = plt.subplots(figsize=(10, max(2, 0.3 * len(numeric_df.columns))))
    ax.imshow(missing, aspect="auto", cmap="Greys", interpolation="none")
    ax.set_yticks(range(len(numeric_df.columns)))
    ax.set_yticklabels(numeric_df.columns)
    ax.set_xlabel(f"time (downsampled, {len(numeric_df)} points)")
    ax.set_title("Missingness map (dark = missing)")
    fig.tight_layout()
    _save_or_show(fig, save_path)


def plot_distribution_shift(
    early: pd.Series, late: pd.Series, title: str, save_path: Path | None = None
) -> None:
    """Plot overlaid histograms comparing an early-period vs late-period distribution.

    A visible shift here is a leakage-risk signal: if P2/P4-style protocols fit a
    scaler/imputer/feature on the full series, a shifted test-period distribution
    means the fitted statistic is contaminated by information the model would not
    have in a genuinely online/causal deployment.

    Args:
        early: Values from the early period.
        late: Values from the late period.
        title: Plot title.
        save_path: If given, save the figure to this path instead of showing it.
    """
    fig, ax = plt.subplots()
    bins = 50
    ax.hist(early.dropna(), bins=bins, alpha=0.5, density=True, label="early period")
    ax.hist(late.dropna(), bins=bins, alpha=0.5, density=True, label="late period")
    ax.set_title(title)
    ax.legend()
    _save_or_show(fig, save_path)


_SPLIT_PROTOCOL_ANNOTATIONS = {
    "P0": "Scaler/imputer fit on train only; tuning on validation block preceding test.",
    "P2": "LEAK: scaler/imputer fit on train+test combined (dashed span), not train only.",
    "P3": "LEAK: purge gap removed; windows straddling the boundary are kept (hatched overlap).",
    "P4": "LEAK: a centered rolling feature near the boundary reads into the test side (arrow).",
    "P5": "LEAK: hyperparameters are selected by scoring directly on the test block (arrow), "
    "not a preceding validation block.",
}


def _plot_split_timeline(ax: plt.Axes, protocol_id: str, n: int) -> None:
    from sdg_leakage_audit.protocols.splitters import chronological_split_with_gap

    horizon = max(1, n // 20)
    gap = 0 if protocol_id == "P3" else horizon
    train_idx, test_idx = chronological_split_with_gap(n, train_frac=0.6, test_frac=0.2, horizon=gap)
    n_train = int(train_idx.max()) + 1
    test_start = int(test_idx.min())
    test_end = int(test_idx.max()) + 1

    ax.broken_barh([(0, n_train)], (0, 1), facecolors="#4C72B0", label="train")
    if gap > 0:
        ax.broken_barh([(n_train, gap)], (0, 1), facecolors="white", edgecolor="black", hatch="//", label="purge gap")
    ax.broken_barh([(test_start, test_end - test_start)], (0, 1), facecolors="#DD8452", label="test")

    if protocol_id == "P2":
        ax.broken_barh([(0, test_end)], (1.3, 0.3), facecolors="none", edgecolor="red", linestyle="--", label="scaler fit span")
    elif protocol_id == "P3":
        overlap_start = max(0, n_train - horizon // 2)
        overlap_end = min(test_end, n_train + horizon // 2)
        ax.broken_barh([(overlap_start, overlap_end - overlap_start)], (0, 1), facecolors="none", edgecolor="red", hatch="xx")
    elif protocol_id == "P4":
        ax.annotate(
            "", xy=(n_train + horizon // 2, 1.0), xytext=(n_train - horizon // 2, 0.5),
            arrowprops={"arrowstyle": "->", "color": "red"},
        )
    elif protocol_id == "P5":
        ax.annotate(
            "model selection", xy=((test_start + test_end) / 2, 1.0), xytext=((test_start + test_end) / 2, 1.6),
            arrowprops={"arrowstyle": "->", "color": "red"}, ha="center", color="red",
        )

    ax.set_xlim(0, n)
    ax.set_ylim(-0.3, 2.0)
    ax.set_yticks([])
    ax.set_xlabel("position")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title(f"{protocol_id}: {_SPLIT_PROTOCOL_ANNOTATIONS.get(protocol_id, '')}", fontsize=9, wrap=True)


def _plot_random_split_timeline(ax: plt.Axes, n: int) -> None:
    from sdg_leakage_audit.protocols.splitters import random_split

    train_idx, test_idx = random_split(n, train_frac=0.6, test_frac=0.2, seed=0)
    colors = np.full(n, "#DDDDDD", dtype=object)
    colors[train_idx] = "#4C72B0"
    colors[test_idx] = "#DD8452"
    ax.bar(np.arange(n), np.ones(n), width=1.0, color=colors)
    ax.set_xlim(0, n)
    ax.set_yticks([])
    ax.set_xlabel("position (temporal order preserved on the x-axis)")
    ax.set_title(
        "P1: LEAK: train (blue) and test (orange) are interleaved across time, "
        "ignoring chronological order.",
        fontsize=9,
        wrap=True,
    )


def _plot_cv_timeline(ax: plt.Axes, n: int) -> None:
    from sdg_leakage_audit.protocols.splitters import blocked_cv_splits, shuffled_kfold_splits

    horizon = max(1, n // 20)
    n_folds = 4
    blocked_folds = blocked_cv_splits(n, n_folds=n_folds, horizon=horizon, expanding=True)
    shuffled_folds = shuffled_kfold_splits(n, n_folds=n_folds, seed=0)

    n_rows = len(blocked_folds) + len(shuffled_folds) + 1  # +1 for a visual gap row
    grid = np.zeros((n_rows, n))
    for row, (train_idx, test_idx) in enumerate(blocked_folds):
        grid[row, train_idx] = 1
        grid[row, test_idx] = 2
    for row, (train_idx, test_idx) in enumerate(shuffled_folds, start=len(blocked_folds) + 1):
        grid[row, train_idx] = 1
        grid[row, test_idx] = 2

    cmap = plt.matplotlib.colors.ListedColormap(["white", "#4C72B0", "#DD8452"])
    ax.imshow(grid, aspect="auto", cmap=cmap, interpolation="none", vmin=0, vmax=2)
    ax.set_yticks(
        list(range(len(blocked_folds))) + list(range(len(blocked_folds) + 1, n_rows)),
        [f"blocked fold {i}" for i in range(len(blocked_folds))]
        + [f"shuffled fold {i}" for i in range(len(shuffled_folds))],
        fontsize=7,
    )
    ax.set_xlabel("position (temporal order)")
    ax.set_title(
        "P6: blocked/expanding CV (top, time-respecting) vs shuffled K-fold (bottom, LEAK: "
        "train/test interleaved within each fold)",
        fontsize=9,
        wrap=True,
    )


def plot_protocol_timeline(protocol_id: str, n: int, save_path: Path | None = None) -> None:
    """Plot a visual timeline (train/gap/test blocks) for a given protocol.

    `P0`/`P2`/`P3`/`P4`/`P5` share the same underlying chronological split (per
    `configs/protocols.yaml`, these protocols differ in fit-order/feature/tuning
    behavior, not split mechanics -- except P3, which removes the purge gap), so
    they are rendered as the same train/gap/test bar with a protocol-specific
    annotation marking where that protocol's leak actually occurs. `P1` (random
    split) and `P6` (shuffled K-fold CV) have genuinely different split structure,
    so they get dedicated visualizations.

    Args:
        protocol_id: One of "P0".."P6".
        n: Total number of positions in the illustrative series.
        save_path: If given, save the figure to this path instead of showing it.

    Raises:
        KeyError: If `protocol_id` is not one of "P0".."P6".
    """
    if protocol_id not in {"P0", "P1", "P2", "P3", "P4", "P5", "P6"}:
        raise KeyError(f"Unknown protocol id '{protocol_id}'")

    fig, ax = plt.subplots(figsize=(9, 3.2 if protocol_id != "P6" else 4.5))
    if protocol_id == "P1":
        _plot_random_split_timeline(ax, n)
    elif protocol_id == "P6":
        _plot_cv_timeline(ax, n)
    else:
        _plot_split_timeline(ax, protocol_id, n)

    fig.tight_layout()
    _save_or_show(fig, save_path)


def plot_inflation_bars(inflation_df: pd.DataFrame, metric: str, save_path: Path | None = None) -> None:
    """Plot a bar chart of mean per-protocol inflation for a given metric.

    Baseline (P0) rows are excluded, since inflation vs P0 is zero by
    construction. Error bars show +/- 1 standard error of the mean across the
    rows (datasets x models x horizons x seeds) contributing to each protocol's bar.

    Args:
        inflation_df: Output of `evaluation.inflation.compute_inflation`.
        metric: Base metric name (e.g. "mae"); the column plotted is
            `f"{metric}_inflation_pct"`.
        save_path: If given, save the figure to this path instead of showing it.
    """
    column = f"{metric}_inflation_pct"
    plot_df = inflation_df.loc[inflation_df["protocol_id"] != "P0", ["protocol_id", column]].dropna()
    grouped = plot_df.groupby("protocol_id")[column]
    means = grouped.mean().sort_index()
    sems = grouped.sem().sort_index()

    fig, ax = plt.subplots()
    colors = ["#2A9D8F" if v >= 0 else "#E76F51" for v in means.to_numpy()]
    ax.bar(means.index, means.to_numpy(), yerr=sems.to_numpy(), capsize=4, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel(f"{metric} inflation (%) vs P0")
    ax.set_xlabel("protocol")
    ax.set_title(f"Apparent-performance inflation by protocol: {metric}")
    fig.tight_layout()
    _save_or_show(fig, save_path)


def save_figure_for_export(fig: plt.Figure, base_path: Path) -> None:
    """Save a figure as both PNG and PDF at 300 dpi, per the project's figure policy.

    Used for final exported figures (notebook 11); earlier exploratory
    notebooks (03, 05, 08, 10) save PNG only via `_save_or_show`, a known gap
    flagged in notebook 11's findings rather than silently left unstated.

    Args:
        fig: The figure to save.
        base_path: Destination path; any suffix is ignored and replaced with
            `.png` and `.pdf`.
    """
    base_path = Path(base_path).with_suffix("")
    base_path.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out_path = base_path.with_suffix(f".{ext}")
        fig.savefig(out_path)
        logger.info("Saved figure -> %s", to_project_relative(out_path))
    plt.close(fig)


def plot_inflation_heatmap(
    pivot_df: pd.DataFrame, title: str, cbar_label: str, save_path: Path | None = None
) -> plt.Figure:
    """Plot an annotated, zero-centered diverging heatmap of a protocol x dataset
    (or any two-way) pivot table, e.g. mean inflation % per (protocol, dataset).

    Args:
        pivot_df: 2-D pivot table (rows/columns can be any two categorical axes,
            e.g. protocol_id x dataset_id).
        title: Plot title.
        cbar_label: Label for the color bar.
        save_path: If given, save via `save_figure_for_export` (PNG+PDF) instead
            of showing. If None, the figure is returned without saving (the
            caller may still show or save it manually).

    Returns:
        The created `Figure` (also returned when `save_path` is given, though by
        then it has already been closed by `save_figure_for_export`).
    """
    values = pivot_df.to_numpy(dtype=float)
    vmax = np.nanmax(np.abs(values)) if np.isfinite(values).any() else 1.0

    fig, ax = plt.subplots(figsize=(1.2 * len(pivot_df.columns) + 2, 0.8 * len(pivot_df.index) + 2))
    im = ax.imshow(values, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(pivot_df.columns)))
    ax.set_xticklabels(pivot_df.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(pivot_df.index)))
    ax.set_yticklabels(pivot_df.index)
    ax.set_title(title)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if np.isfinite(values[i, j]):
                ax.text(j, i, f"{values[i, j]:.1f}", ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, label=cbar_label)
    fig.tight_layout()

    if save_path is not None:
        save_figure_for_export(fig, Path(save_path))
    return fig

"""Shared matplotlib/seaborn styling for consistent, clean figures.

Used throughout notebooks 03-11 for a consistent visual style; the final polish pass
happens in notebook 11, but the style itself is defined once here so every earlier
figure already matches it.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns

_FIGURE_DPI = 300

PALETTE = "colorblind"


def apply_plot_style() -> None:
    """Apply the project's consistent matplotlib/seaborn style (fonts, palette, DPI).

    Intended to be called once at the top of any notebook that produces figures.
    Sets a colorblind-safe palette, a serif-free readable font, white grid background,
    and a 300 dpi default for `savefig` so figures are sharp and consistent without
    per-call overrides.
    """
    sns.set_theme(style="whitegrid", palette=PALETTE, font_scale=1.0)
    plt.rcParams.update(
        {
            "figure.dpi": 100,
            "savefig.dpi": _FIGURE_DPI,
            "savefig.bbox": "tight",
            "font.family": "sans-serif",
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "figure.figsize": (8, 4),
        }
    )

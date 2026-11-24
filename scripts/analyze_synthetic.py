"""Analyse the synthetic ground-truth study (results/synthetic/groundtruth_cells.csv).

Outputs (results/synthetic/): summary tables, comparison with the theoretical ceiling, and two figures.
Key questions: is the corrected P0 unbiased, which protocols are optimistic against the TRUE future error, how close do learners
get to the interpolation ceiling, and does the optimism grow with persistence and horizon as the theory predicts.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

OUT = ROOT / "results" / "synthetic"
plt.rcParams.update({"font.family": "Arial", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "savefig.dpi": 300})
RNG = np.random.default_rng(0)
LINEAR = ["ar1_0.3", "ar1_0.6", "ar1_0.9", "ar1_0.98", "seas_ar1_0.5", "seas_ar1_0.9", "ar2_osc"]
COL = {"ridge": "#4c78a8", "random_forest": "#54a24b", "lightgbm": "#e45756", "mlp": "#b279a2"}
MK = {"ridge": "o", "random_forest": "s", "lightgbm": "^", "mlp": "D"}


def boot(x, n=2000):
    x = np.asarray(x, float)
    m = RNG.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main() -> None:
    d = pd.read_csv(OUT / "groundtruth_cells.csv")
    ceil = pd.read_csv(OUT / "ceiling.csv")
    # seed-level means per process/horizon/model/protocol
    g = d.groupby(["process", "horizon", "model", "protocol"]).inflation_true_pct
    summ = g.agg(["mean", "std", "count"]).reset_index()
    summ.to_csv(OUT / "summary_by_cell.csv", index=False)

    # 1. unbiasedness of the corrected baseline and the pessimism of blocked CV, pooled over everything
    rows = []
    for p in ["P0", "P6B", "P1", "P6", "P6S", "P4"]:
        x = d[d.protocol == p].groupby(["process", "horizon", "model"]).inflation_true_pct.mean().to_numpy()
        m, lo, hi = boot(x)
        rows.append({"protocol": p, "units": len(x), "mean_pct": m, "ci_lo": lo, "ci_hi": hi, "share_positive": float((x > 0).mean())})
    pooled = pd.DataFrame(rows)
    pooled.to_csv(OUT / "pooled_by_protocol.csv", index=False)

    # 2. by model family
    byfam = d.groupby(["model", "protocol"]).inflation_true_pct.mean().unstack("protocol").round(2)
    byfam.to_csv(OUT / "by_model.csv")

    # 3. ceiling comparison for linear Gaussian processes (interleaved protocols)
    m = summ[summ.process.isin(LINEAR) & summ.protocol.isin(["P1", "P6", "P6S"])].merge(ceil, on=["process", "horizon"])
    m["exploited"] = m["mean"] / m["ceiling_mae_pct"]
    m.to_csv(OUT / "ceiling_comparison.csv", index=False)
    stats = []
    for p in ["P1", "P6", "P6S"]:
        for model in COL:
            s = m[(m.protocol == p) & (m.model == model)]
            rho, pv = spearmanr(s.ceiling_mae_pct, s["mean"])
            stats.append({"protocol": p, "model": model, "n": len(s), "spearman_rho": rho, "p": pv,
                          "mean_exploited": s.exploited.mean(), "max_exploited": s.exploited.max(),
                          "n_above_ceiling": int((s["mean"] > s.ceiling_mae_pct + 2.0).sum())})
    st = pd.DataFrame(stats)
    st.to_csv(OUT / "ceiling_correlation.csv", index=False)

    # 4. look-ahead features: inflation by horizon (reach of the windows is 11 steps)
    p4 = d[d.protocol == "P4"].groupby(["horizon", "model"]).inflation_true_pct.mean().unstack("model").round(1)
    p4.to_csv(OUT / "p4_by_horizon.csv")

    # figures
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.2), sharey=True)
    for ax, p, lab in zip(axs, ["P1", "P6S"], ["(a) P1 random split", "(b) P6S size-matched shuffled CV"]):
        for model in COL:
            s = m[(m.protocol == p) & (m.model == model)]
            ax.scatter(s.ceiling_mae_pct, s["mean"], c=COL[model], marker=MK[model], s=22, label=model, alpha=0.85, linewidths=0)
        ax.plot([0, 80], [0, 80], color="#666", lw=0.8, ls="--")
        ax.set_xlabel("Theoretical ceiling, MAE (%)")
        ax.text(0.03, 0.96, lab, transform=ax.transAxes, fontsize=9, weight="bold", va="top")
    axs[0].set_ylabel("Ground-truth inflation (%)")
    axs[1].legend(frameon=False, loc="lower right")
    for ext in ("png", "pdf", "eps"):
        fig.savefig(OUT / f"fig_ceiling.{ext}", bbox_inches="tight")
    fig.savefig(OUT / "fig_ceiling.tiff", bbox_inches="tight", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    order = ["P0", "P6B", "P1", "P6", "P6S", "P4"]
    for i, p in enumerate(order):
        r = pooled[pooled.protocol == p].iloc[0]
        ax.errorbar(i, r.mean_pct, yerr=[[r.mean_pct - r.ci_lo], [r.ci_hi - r.mean_pct]], fmt="D", color="black", capsize=4, ms=5)
    ax.axhline(0, color="#666", lw=0.8)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order)
    ax.set_ylabel("Ground-truth inflation (%)")
    for ext in ("png", "pdf", "eps"):
        fig.savefig(OUT / f"fig_groundtruth_pooled.{ext}", bbox_inches="tight")
    fig.savefig(OUT / "fig_groundtruth_pooled.tiff", bbox_inches="tight", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)

    md = ["# Synthetic ground-truth study", "", f"Cells: {len(d)}; processes {sorted(d.process.unique())}", "", "## Pooled, ground-truth inflation (%)",
          pooled.round(2).to_markdown(index=False), "", "## By model", byfam.to_markdown(), "", "## Ceiling comparison (linear Gaussian processes)",
          st.round(3).to_markdown(index=False), "", "## P4 by horizon", p4.to_markdown()]
    (OUT / "SYNTHETIC_SUMMARY.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()

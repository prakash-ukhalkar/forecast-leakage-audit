"""Link the interpolation ceiling to the real datasets.

For every dataset and horizon the ceiling is computed from an AR model fitted to the deseasonalised target (the same rows the
experiments use), and compared with the measured inflation of P1 (difficulty-adjusted) and P6S (against P6B) per model family.
Run after scripts/analyze_extension.py.  Output: results/final/extension/tables/theory_real_*.csv, results/final/extension/figures/fig_theory_real.*
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
from statsmodels.regression.linear_model import yule_walker  # noqa: E402

from sdg_leakage_audit.audit.theory import leakage_ceiling  # noqa: E402
from sdg_leakage_audit.evaluation import runner as R  # noqa: E402
from sdg_leakage_audit.utils.io import read_yaml  # noqa: E402

OUT = ROOT / "results" / "final" / "extension"
FAMILY = {"ridge": "linear", "random_forest": "tree_ensemble", "lightgbm": "tree_ensemble", "mlp": "neural", "lstm": "neural", "gru": "neural"}
plt.rcParams.update({"font.family": "Arial", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "savefig.dpi": 300})


def deseason(x: np.ndarray, season: int) -> np.ndarray:
    prof = np.array([np.nanmean(x[i::season]) for i in range(season)])
    return x - prof[np.arange(len(x)) % season]


def fit_ar(x: np.ndarray) -> tuple[np.ndarray, float, int]:
    """AR(p) by Yule-Walker, order chosen by BIC among a small grid."""
    best = None
    n = len(x)
    for p in (1, 2, 4, 8, 12, 24, 48):
        if p >= n // 10:
            continue
        rho, sigma = yule_walker(x - x.mean(), order=p, method="mle")
        bic = n * np.log(sigma**2) + p * np.log(n)
        if best is None or bic < best[0]:
            best = (bic, rho, sigma**2, p)
    return best[1], best[2], best[3]


def main() -> None:
    cfg = read_yaml(ROOT / "configs" / "datasets.yaml")["datasets"]
    U = pd.read_csv(OUT / "tables" / "inflation_by_unit_all.csv")
    UA = pd.read_csv(OUT / "tables" / "adjusted_inflation_by_unit.csv")
    UA["family"] = UA.model_name.map(FAMILY)
    rows = []
    for d in sorted(U.dataset_id.unique()):
        c = cfg[d.partition("@")[0]]
        df = R._load_dataset_frame(d)
        y = df[c["target_column"]].astype(float).iloc[-20000:].to_numpy()
        y = y[np.isfinite(y)]
        x = deseason(y, c["season_length"])
        ar, s2, p = fit_ar(x)
        for h in c["forecast_horizons"]:
            ceil = 100 * leakage_ceiling(ar, h, sigma2=s2, metric="mae")
            for fam in ("linear", "tree_ensemble", "neural"):
                a1 = UA[(UA.dataset_id == d) & (UA.horizon == h) & (UA.protocol_id == "P1") & (UA.family == fam)].adj.mean()
                a6 = UA[(UA.dataset_id == d) & (UA.horizon == h) & (UA.protocol_id == "P6S") & (UA.family == fam)].adj.mean()
                m6s = U[(U.dataset_id == d) & (U.horizon == h) & (U.protocol_id == "P6S") & (U.family == fam)].mae_inflation_pct.mean()
                rows.append({"dataset": d, "horizon": h, "ar_order": p, "ceiling_mae_pct": ceil, "family": fam,
                             "p1_adjusted_pct": a1, "p6s_adjusted_pct": a6, "p6s_vs_p6b_pct": m6s})
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "tables" / "theory_real_by_cell.csv", index=False)
    stats = []
    for fam in ("linear", "tree_ensemble", "neural"):
        for col in ("p1_adjusted_pct", "p6s_adjusted_pct", "p6s_vs_p6b_pct"):
            s = T[T.family == fam].dropna(subset=[col])
            rho, pv = spearmanr(s.ceiling_mae_pct, s[col])
            stats.append({"family": fam, "measure": col, "n": len(s), "spearman_rho": rho, "p": pv})
    S = pd.DataFrame(stats)
    S.to_csv(OUT / "tables" / "theory_real_correlation.csv", index=False)

    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.8), sharey=True)
    mk = {"linear": "o", "tree_ensemble": "^", "neural": "s"}
    for ax, fam, lab in zip(axs, ("linear", "tree_ensemble", "neural"), ("(a) linear", "(b) tree ensembles", "(c) neural")):
        s = T[T.family == fam]
        ax.scatter(s.ceiling_mae_pct, s.p6s_adjusted_pct, marker=mk[fam], c="#333333", s=18, alpha=0.8, linewidths=0)
        ax.text(0.04, 0.96, lab, transform=ax.transAxes, fontsize=9, weight="bold", va="top")
        ax.set_xlabel("Ceiling, MAE (%)")
    axs[0].set_ylabel("Measured P6S inflation, adjusted (%)")
    for ext in ("png", "pdf", "eps"):
        fig.savefig(OUT / "figures" / f"fig_theory_real.{ext}", bbox_inches="tight")
    fig.savefig(OUT / "figures" / "fig_theory_real.tiff", bbox_inches="tight", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print(T.groupby("dataset")[["ceiling_mae_pct"]].mean().round(1).to_string())
    print(S.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

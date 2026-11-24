"""Analysis of the extension runs: size-matched control (P6S) and replication on additional datasets/sites.

Inputs (per-run metrics):
  results/final/metrics/colab_grid_metrics.csv       original grid (5 datasets, protocols P0..P6B)
  results/metrics/ext_p6s_grid_metrics.csv           P6S on the original 5 datasets
  results/metrics/ext_sites_grid_metrics.csv         all protocols (incl. P6S) on ETTh1 and two more Beijing stations
Outputs: results/final/extension/{tables,figures}/..., results/final/EXTENSION_SUMMARY.md

P6 decomposition (per unit = dataset, model, horizon; seeds averaged), with B = MAE(P6B), S = MAE(P6S), K = MAE(P6):
  total      = 100 (B - K) / B     shuffled CV vs blocked CV (the main P6 figure)
  matched    = 100 (B - S) / B     same train/test sizes as blocked CV, random positions: leakage net of training size
  size       = 100 (S - K) / S     gain from the extra training rows that shuffled folds have
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
from scipy.stats import wilcoxon  # noqa: E402
from statsmodels.stats.multitest import multipletests  # noqa: E402

from sdg_leakage_audit.evaluation.inflation import compute_inflation  # noqa: E402
from sdg_leakage_audit.stats.tests import effect_size_cliffs_delta  # noqa: E402

LEARNED = ["ridge", "random_forest", "lightgbm", "mlp", "lstm", "gru"]
FAMILY = {"ridge": "linear", "random_forest": "tree_ensemble", "lightgbm": "tree_ensemble",
          "mlp": "neural", "lstm": "neural", "gru": "neural"}
LEAKY = ["P1", "P2", "P3", "P4", "P5", "P6"]
UNIT = ["dataset_id", "model_name", "horizon"]
RNG = np.random.default_rng(0)
OUT = ROOT / "results" / "final" / "extension"
(OUT / "tables").mkdir(parents=True, exist_ok=True)
(OUT / "figures").mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "Arial", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "savefig.dpi": 300})


def boot(x, n=2000):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return (np.nan,) * 3
    m = RNG.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def load() -> pd.DataFrame:
    parts = []
    for p in (ROOT / "results" / "final" / "metrics" / "colab_grid_metrics.csv",
              ROOT / "results" / "metrics" / "ext_p6s_grid_metrics.csv",
              ROOT / "results" / "metrics" / "ext_sites_grid_metrics.csv"):
        if p.is_file():
            d = pd.read_csv(p)
            parts.append(d[d["error"].isna()] if "error" in d else d)
    df = pd.concat(parts, ignore_index=True)
    return df.drop_duplicates(subset=["dataset_id", "protocol_id", "model_name", "horizon", "seed"], keep="last")


def headline(U: pd.DataFrame, protos) -> pd.DataFrame:
    rows = []
    for p in protos:
        u = U[U.protocol_id == p]
        if u.empty:
            continue
        d = u["mae_inflation_pct"].to_numpy()
        m, lo, hi = boot(d)
        pv = wilcoxon(d).pvalue if len(d) > 5 and np.any(d != 0) else np.nan
        rows.append({"protocol": p, "n_units": len(u), "mean_mae_inflation_pct": m, "ci95_lo": lo, "ci95_hi": hi,
                     "median_pct": float(np.median(d)), "share_units_inflated": float((d > 0).mean()),
                     "mean_skill_gain_abs": float(u["skill_score_inflation_abs"].mean()), "wilcoxon_p": pv,
                     "cliffs_delta": effect_size_cliffs_delta(u["mae_baseline"], u["mae"])})
    h = pd.DataFrame(rows, columns=["protocol", "n_units", "mean_mae_inflation_pct", "ci95_lo", "ci95_hi", "median_pct", "share_units_inflated", "mean_skill_gain_abs", "wilcoxon_p", "cliffs_delta"])
    ok = h["wilcoxon_p"].notna()
    h.loc[ok, "p_holm"] = multipletests(h.loc[ok, "wilcoxon_p"], method="holm")[1]
    return h


def main() -> None:
    df = load()
    infl = compute_inflation(df)
    L = infl[infl.model_name.isin(LEARNED)].copy()
    L["family"] = L.model_name.map(FAMILY)
    U = L.groupby(["protocol_id", *UNIT, "family"], as_index=False)[
        ["mae_inflation_pct", "skill_score_inflation_abs", "mae", "mae_baseline"]].mean()

    U.to_csv(OUT / "tables" / "inflation_by_unit_all.csv", index=False)
    orig = ["household_power_consumption", "appliances_energy_prediction", "air_quality_uci", "beijing_multisite_air_quality", "jena_climate"]
    new = sorted(set(U.dataset_id) - set(orig))

    # ---- P6 decomposition
    M = df[df.model_name.isin(LEARNED) & df.protocol_id.isin(["P6", "P6B", "P6S"])]
    W = M.groupby(["dataset_id", "model_name", "horizon", "protocol_id"]).mae.mean().unstack("protocol_id").dropna()
    W = W.reset_index()
    W["family"] = W.model_name.map(FAMILY)
    W["total"] = 100 * (W.P6B - W.P6) / W.P6B
    W["matched"] = 100 * (W.P6B - W.P6S) / W.P6B
    W["size"] = 100 * (W.P6S - W.P6) / W.P6S
    W.to_csv(OUT / "tables" / "p6_decomposition_by_unit.csv", index=False)
    rows = []
    for name, sub in [("all datasets", W), ("original five", W[W.dataset_id.isin(orig)]), ("additional", W[~W.dataset_id.isin(orig)])]:
        if sub.empty:
            continue
        r = {"subset": name, "n_units": len(sub)}
        for k in ("total", "matched", "size"):
            m, lo, hi = boot(sub[k])
            r[k], r[k + "_lo"], r[k + "_hi"] = m, lo, hi
        d = sub["matched"].to_numpy()
        r["matched_wilcoxon_p"] = wilcoxon(d).pvalue if len(d) > 5 else np.nan
        r["leak_share_of_total"] = r["matched"] / r["total"] if r["total"] else np.nan
        rows.append(r)
    S = pd.DataFrame(rows)
    S.to_csv(OUT / "tables" / "p6_decomposition_summary.csv", index=False)
    Wo = W[W.dataset_id.isin(orig)]
    fam = Wo.groupby("family")[["total", "matched", "size"]].mean().round(2)  # original five datasets (main text, Fig. 9)
    fam.to_csv(OUT / "tables" / "p6_decomposition_by_family.csv")
    W.groupby("family")[["total", "matched", "size"]].mean().round(2).to_csv(OUT / "tables" / "p6_decomposition_by_family_all.csv")
    dsx = W.groupby("dataset_id")[["total", "matched", "size"]].mean().round(2)
    dsx.to_csv(OUT / "tables" / "p6_decomposition_by_dataset.csv")

    # ---- difficulty-adjusted inflation: improvement in skill relative to the naive forecast on the SAME rows,
    #      100 * (S_p - S_b) / (1 - S_b), so a protocol is not credited for evaluating on easier rows
    Lc = L.copy()
    denom = 1.0 - Lc["skill_score_baseline"]
    Lc["adj"] = np.where(denom > 1e-6, 100.0 * (Lc["skill_score"] - Lc["skill_score_baseline"]) / denom, np.nan)
    UA = Lc.groupby(["protocol_id", *UNIT], as_index=False)["adj"].mean()
    UA.to_csv(OUT / "tables" / "adjusted_inflation_by_unit.csv", index=False)
    adj_rows = []
    for name, sel in (("original", UA.dataset_id.isin(orig)), ("additional", ~UA.dataset_id.isin(orig)), ("pooled", UA.dataset_id.notna())):
        for p_ in ["P1", "P4", "P6", "P6S"]:
            v = UA[sel & (UA.protocol_id == p_)]["adj"].to_numpy()
            if len(v) == 0:
                continue
            m_, lo_, hi_ = boot(v)
            adj_rows.append({"subset": name, "protocol": p_, "n_units": len(v), "adj_mean": m_, "adj_lo": lo_, "adj_hi": hi_,
                             "share_positive": float((v > 0).mean())})
    pd.DataFrame(adj_rows).to_csv(OUT / "tables" / "headline_difficulty_adjusted.csv", index=False)

    # ---- replication headline
    h_orig = headline(U[U.dataset_id.isin(orig)], LEAKY + ["P6S"])
    h_new = headline(U[~U.dataset_id.isin(orig)], LEAKY + ["P6S"])
    h_all = headline(U, LEAKY + ["P6S"])
    for n, t in (("original", h_orig), ("additional", h_new), ("pooled", h_all)):
        t.to_csv(OUT / "tables" / f"headline_{n}.csv", index=False)
    byds = U[U.protocol_id.isin(LEAKY + ["P6S"]) & ~U.dataset_id.isin(orig)].pivot_table(
        index="dataset_id", columns="protocol_id", values="mae_inflation_pct", aggfunc="mean").round(2)
    byds.to_csv(OUT / "tables" / "additional_datasets_by_protocol.csv")
    byfam_all = U[U.protocol_id.isin(LEAKY + ["P6S"])].pivot_table(
        index="family", columns="protocol_id", values="mae_inflation_pct", aggfunc="mean").round(2)
    byfam_all.to_csv(OUT / "tables" / "pooled_by_family.csv")

    # ---- figures
    labels = {"linear": "Linear", "tree_ensemble": "Tree ensembles", "neural": "Neural"}
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    groups = ["tree_ensemble", "neural", "linear"]
    x = np.arange(len(groups) + 1)
    tot = [fam.loc[g, "total"] for g in groups] + [Wo.total.mean()]
    mat = [fam.loc[g, "matched"] for g in groups] + [Wo.matched.mean()]
    ax.bar(x - 0.19, tot, 0.36, color="#4c78a8", label="P6 vs P6B (total)")
    ax.bar(x + 0.19, mat, 0.36, color="#f2cf5b", edgecolor="#333", hatch="//", label="P6S vs P6B (size-matched)")
    ax.set_xticks(x)
    ax.set_xticklabels([labels[g] for g in groups] + ["All units"])
    ax.axhline(0, color="#666", lw=0.8)
    ax.set_ylabel("MAE inflation (%)")
    ax.legend(frameon=False)
    for ext in ("png", "pdf", "eps"):
        fig.savefig(OUT / "figures" / f"fig_p6_decomposition.{ext}", bbox_inches="tight")
    fig.savefig(OUT / "figures" / "fig_p6_decomposition.tiff", bbox_inches="tight", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.2), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axs[0]
    for i, p in enumerate(LEAKY):
        for off, (h, c, mk, lab) in zip((-0.2, 0.0, 0.2), [(h_orig, "#4c78a8", "o", "original five"), (h_new, "#e08a1e", "s", "additional"), (h_all, "#222222", "D", "pooled")]):
            r = h[h.protocol == p]
            if r.empty:
                continue
            r = r.iloc[0]
            ax.errorbar(i + off, r.mean_mae_inflation_pct, yerr=[[r.mean_mae_inflation_pct - r.ci95_lo], [r.ci95_hi - r.mean_mae_inflation_pct]],
                        fmt=mk, color=c, capsize=3, ms=5, label=lab if i == 0 else None)
    ax.axhline(0, color="#666", lw=0.8)
    ax.set_xticks(range(6))
    ax.set_xticklabels(LEAKY)
    ax.set_ylabel("MAE inflation (%)")
    ax.text(0.02, 0.96, "(a)", transform=ax.transAxes, fontsize=9, weight="bold", va="top")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    ax = axs[1]
    adj = pd.DataFrame(adj_rows)
    for i, p in enumerate(["P1", "P4", "P6", "P6S"]):
        for off, (nm, c, mk) in zip((-0.2, 0.0, 0.2), [("original", "#4c78a8", "o"), ("additional", "#e08a1e", "s"), ("pooled", "#222222", "D")]):
            r = adj[(adj["subset"] == nm) & (adj.protocol == p)]
            if r.empty:
                continue
            r = r.iloc[0]
            ax.errorbar(i + off, r.adj_mean, yerr=[[r.adj_mean - r.adj_lo], [r.adj_hi - r.adj_mean]], fmt=mk, color=c, capsize=3, ms=5)
    ax.axhline(0, color="#666", lw=0.8)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["P1", "P4", "P6", "P6S"])
    ax.set_ylabel("Difficulty-adjusted inflation (%)")
    ax.text(0.02, 0.96, "(b)", transform=ax.transAxes, fontsize=9, weight="bold", va="top")
    fig.tight_layout()
    for ext in ("png", "pdf", "eps"):
        fig.savefig(OUT / "figures" / f"fig_replication.{ext}", bbox_inches="tight")
    fig.savefig(OUT / "figures" / "fig_replication.tiff", bbox_inches="tight", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)

    md = ["# Extension summary", "", f"Datasets: original {len(orig)}; additional {new}. Rows: {len(df)}.", "",
          "## P6 decomposition (size-matched control)", S.round(2).to_markdown(index=False), "", fam.to_markdown(), "", dsx.to_markdown(), "",
          "## Headline inflation, original five", h_orig.round(3).to_markdown(index=False), "",
          "## Headline inflation, additional datasets", h_new.round(3).to_markdown(index=False), "",
          "## Headline inflation, pooled", h_all.round(3).to_markdown(index=False), "", byds.to_markdown(), "", byfam_all.to_markdown(), "",
          "## Difficulty-adjusted inflation (skill-based)", pd.DataFrame(adj_rows).round(2).to_markdown(index=False)]
    (ROOT / "results" / "final" / "EXTENSION_SUMMARY.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md[:14]))


if __name__ == "__main__":
    main()

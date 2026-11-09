"""Turn a finished grid (results/metrics/<profile>_grid_metrics.csv) into paper-ready outputs.

Usage:  python scripts/analyze_results.py --profile colab
Writes: results/final/{tables,figures}/...  and results/final/SUMMARY.md

Design choices (each addresses a weakness of the earlier fast-profile analysis):
  * Inflation uses fair baselines: P5 vs P0V (validation-tuned), P6 vs P6B (blocked CV),
    everything else vs P0 (see evaluation/inflation.py).
  * Only LEARNED models enter the inflation statistics. Persistence / seasonal-naive have no
    fitted parameters, so protocols that change fitting are not meaningful for them.
  * Seeds are replicates: they are averaged within a unit = (dataset, model, horizon); the
    unit is the statistical sample, and CIs come from a cluster bootstrap over units.
  * Per-protocol one-sample Wilcoxon on unit inflation (Holm-corrected) + Cliff's delta.
  * Explanatory analysis: does inflation scale with target autocorrelation at the horizon lag?
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr, wilcoxon  # noqa: E402
from statsmodels.stats.multitest import multipletests  # noqa: E402

from sdg_leakage_audit.evaluation.inflation import compute_inflation  # noqa: E402
from sdg_leakage_audit.evaluation.runner import _load_dataset_frame  # noqa: E402
from sdg_leakage_audit.sdg import impact  # noqa: E402
from sdg_leakage_audit.stats.tests import effect_size_cliffs_delta, friedman_nemenyi  # noqa: E402
from sdg_leakage_audit.utils.io import read_yaml  # noqa: E402

LEARNED = ["ridge", "random_forest", "lightgbm", "mlp", "lstm", "gru"]
FAMILY = {"ridge": "linear", "random_forest": "tree_ensemble", "lightgbm": "tree_ensemble",
          "mlp": "neural", "lstm": "neural", "gru": "neural"}
LEAKY = ["P1", "P2", "P3", "P4", "P5", "P6"]
SDG_FN = {"household_power_consumption": impact.estimate_reserve_margin_impact,
          "appliances_energy_prediction": impact.estimate_reserve_margin_impact,
          "air_quality_uci": impact.estimate_exceedance_alert_impact,
          "beijing_multisite_air_quality": impact.estimate_exceedance_alert_impact,
          "jena_climate": impact.estimate_storage_sizing_impact}
UNIT = ["dataset_id", "model_name", "horizon"]
RNG = np.random.default_rng(0)


def boot_ci(x: np.ndarray, n: int = 2000) -> tuple[float, float, float]:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return (np.nan,) * 3
    means = RNG.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def acf_at(series: np.ndarray, lag: int) -> float:
    s = series[np.isfinite(series)]
    s = s - s.mean()
    return float((s[:-lag] * s[lag:]).sum() / (s * s).sum()) if lag < len(s) else np.nan


def save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=200)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="colab")
    a = ap.parse_args()
    out = ROOT / "results" / "final"
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    src_csv = ROOT / "results" / "metrics" / f"{a.profile}_grid_metrics.csv"
    if not src_csv.is_file():  # fall back to the committed copy of the per-run metrics
        src_csv = ROOT / "results" / "final" / "metrics" / f"{a.profile}_grid_metrics.csv"
    raw = pd.read_csv(src_csv)
    n_err = int(raw["error"].notna().sum()) if "error" in raw else 0
    df = raw[raw["error"].isna()] if "error" in raw else raw
    infl = compute_inflation(df)
    infl.to_csv(out / "tables" / "inflation_by_cell.csv", index=False)

    L = infl[infl["model_name"].isin(LEARNED) & infl["protocol_id"].isin(["P0V", "P6B", *LEAKY])].copy()
    L["family"] = L["model_name"].map(FAMILY)
    # unit-level (seeds averaged)
    U = L.groupby(["protocol_id", *UNIT, "family"], as_index=False)[
        ["mae_inflation_pct", "rmse_inflation_pct", "skill_score_inflation_abs", "mae", "mae_baseline"]].mean()
    U.to_csv(out / "tables" / "inflation_by_unit.csv", index=False)

    # ---- 1. headline: per protocol, mean inflation + cluster-bootstrap CI + Wilcoxon + Cliff
    rows = []
    for p in LEAKY:
        u = U[U.protocol_id == p]
        m, lo, hi = boot_ci(u["mae_inflation_pct"].to_numpy())
        d = u["mae_inflation_pct"].to_numpy()
        pv = wilcoxon(d).pvalue if len(d) > 5 and np.any(d != 0) else np.nan
        rows.append({"protocol": p, "n_units": len(u), "mean_mae_inflation_pct": m, "ci95_lo": lo, "ci95_hi": hi,
                     "median_pct": float(np.median(d)), "share_units_inflated": float((d > 0).mean()),
                     "mean_skill_gain_abs": boot_ci(u["skill_score_inflation_abs"].to_numpy())[0],
                     "skill_gain_ci95_lo": boot_ci(u["skill_score_inflation_abs"].to_numpy())[1],
                     "skill_gain_ci95_hi": boot_ci(u["skill_score_inflation_abs"].to_numpy())[2],
                     "wilcoxon_p": pv, "cliffs_delta_mae_vs_baseline": effect_size_cliffs_delta(u["mae_baseline"], u["mae"]) if len(u) else np.nan})
    head = pd.DataFrame(rows)
    ok = head["wilcoxon_p"].notna()
    head.loc[ok, "p_holm"] = multipletests(head.loc[ok, "wilcoxon_p"], method="holm")[1]
    head["significant_holm_0.05"] = head["p_holm"] < 0.05
    head.to_csv(out / "tables" / "headline_inflation.csv", index=False)
    (out / "tables" / "headline_inflation.tex").write_text(head.round(3).to_latex(index=False), encoding="utf-8")

    # ---- 2. breakdowns
    def piv(by):
        return U[U.protocol_id.isin(LEAKY)].pivot_table(index=by, columns="protocol_id", values="mae_inflation_pct", aggfunc="mean")[LEAKY]
    by_ds, by_fam, by_model, by_h = piv("dataset_id"), piv("family"), piv("model_name"), piv("horizon")
    for n, t in [("by_dataset", by_ds), ("by_family", by_fam), ("by_model", by_model), ("by_horizon", by_h)]:
        t.round(2).to_csv(out / "tables" / f"inflation_{n}.csv")

    # P2 only changes fitted scalers, so scale-invariant tree models are a guaranteed null; report scaled models separately
    sc = U[(U.protocol_id == "P2") & U.model_name.isin(["ridge", "mlp"])]["mae_inflation_pct"].to_numpy()
    p2s = pd.DataFrame([dict(zip(["mean_pct", "ci95_lo", "ci95_hi"], boot_ci(sc)), n_units=len(sc))])
    p2s.to_csv(out / "tables" / "P2_scaled_models_only.csv", index=False)

    # tuning gain & CV-scheme gap (the two controls)
    ctrl = U[U.protocol_id.isin(["P0V", "P6B"])].groupby("protocol_id")["mae_inflation_pct"].agg(["mean", "median", "count"])
    ctrl.to_csv(out / "tables" / "control_arms_vs_P0.csv")

    # ---- 3. omnibus across protocols (learned models, seed-averaged)
    try:
        fb = L.groupby(["protocol_id", *UNIT], as_index=False)["mae"].mean().assign(seed=0)
        fn = friedman_nemenyi(fb, "mae")
        fn["nemenyi_pvalues"].to_csv(out / "tables" / "nemenyi_pvalues_mae.csv")
        friedman_txt = f"Friedman chi2={fn['friedman_statistic']:.2f}, p={fn['friedman_pvalue']:.3g}"
    except ValueError as e:  # not enough complete blocks
        friedman_txt = f"Friedman not run: {e}"

    # ---- 4. explanatory: inflation vs autocorrelation of the target at lag = horizon
    acf_rows = []
    for (ds, h), _ in U.groupby(["dataset_id", "horizon"]):
        series = _load_dataset_frame(ds)[read_yaml(ROOT / "configs" / "datasets.yaml")["datasets"][ds]["target_column"]].to_numpy()
        acf_rows.append({"dataset_id": ds, "horizon": h, "acf_at_horizon": acf_at(series, int(h))})
    acf = pd.DataFrame(acf_rows)
    corr_rows = []
    for p in LEAKY:
        j = U[U.protocol_id == p].groupby(["dataset_id", "horizon"], as_index=False)["mae_inflation_pct"].mean().merge(acf)
        if len(j) >= 4:
            r, pv = spearmanr(j["acf_at_horizon"], j["mae_inflation_pct"])
            corr_rows.append({"protocol": p, "n_dataset_horizons": len(j), "spearman_rho": r, "p": pv})
    pd.DataFrame(corr_rows).to_csv(out / "tables" / "inflation_vs_autocorrelation.csv", index=False)
    acf.to_csv(out / "tables" / "acf_at_horizon.csv", index=False)

    # ---- 5. SDG decision impact (illustrative RMSE-buffer assumption; see sdg/impact.py)
    sdg_rows = []
    for ds, fn_ in SDG_FN.items():
        sub = infl[(infl.dataset_id == ds) & infl.model_name.isin(LEARNED)]
        if not sub.empty:
            sdg_rows.append(fn_(sub, ds).reset_index().assign(dataset_id=ds))
    if sdg_rows:
        pd.concat(sdg_rows).to_csv(out / "tables" / "sdg_decision_impact.csv", index=False)

    # ---- figures
    fig, ax = plt.subplots(figsize=(7, 3.6))
    im = ax.imshow(by_ds.to_numpy(), cmap="RdBu_r", vmin=-np.nanmax(abs(by_ds.to_numpy())), vmax=np.nanmax(abs(by_ds.to_numpy())), aspect="auto")
    ax.set_xticks(range(len(LEAKY)), LEAKY); ax.set_yticks(range(len(by_ds)), by_ds.index)
    for i in range(by_ds.shape[0]):
        for j in range(by_ds.shape[1]):
            ax.text(j, i, f"{by_ds.iat[i, j]:.1f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, label="MAE inflation (%), + = flattering")
    ax.set_title("Mean MAE inflation by dataset and protocol (learned models)")
    save(fig, out / "figures" / "fig_inflation_heatmap")

    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.bar(head["protocol"], head["mean_mae_inflation_pct"],
           yerr=[head["mean_mae_inflation_pct"] - head["ci95_lo"], head["ci95_hi"] - head["mean_mae_inflation_pct"]], capsize=4, color="#4C78A8")
    ax.axhline(0, color="k", lw=.8); ax.set_ylabel("MAE inflation (%)"); ax.set_title("Headline inflation with 95% cluster-bootstrap CI")
    save(fig, out / "figures" / "fig_headline_inflation")

    fig, ax = plt.subplots(figsize=(7, 3.6))
    for fam, r in by_fam.iterrows():
        ax.plot(LEAKY, r.to_numpy(), marker="o", label=fam)
    ax.axhline(0, color="k", lw=.8); ax.legend(); ax.set_ylabel("MAE inflation (%)"); ax.set_title("Inflation by model family")
    save(fig, out / "figures" / "fig_inflation_by_family")

    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, p in zip(axs, ["P1", "P6"]):
        j = U[U.protocol_id == p].groupby(["dataset_id", "horizon"], as_index=False)["mae_inflation_pct"].mean().merge(acf)
        ax.scatter(j["acf_at_horizon"], j["mae_inflation_pct"])
        ax.set_xlabel("target ACF at lag = horizon"); ax.set_title(p)
    axs[0].set_ylabel("MAE inflation (%)")
    save(fig, out / "figures" / "fig_inflation_vs_acf")

    # ---- SUMMARY.md
    md = [f"# Results summary (profile `{a.profile}`)", "",
          f"Grid rows: {len(raw)} ({n_err} errored and excluded). Learned-model units: {U[['dataset_id','model_name','horizon']].drop_duplicates().shape[0]}.", "",
          "## Headline inflation (MAE %, + = flattering)", head.round(3).to_markdown(index=False), "",
          "## P2 restricted to scale-sensitive models (ridge, mlp)", p2s.round(3).to_markdown(index=False), "",
          "Caveats: P1/P6 draw different test rows than P0, so MAE inflation mixes leakage with test-period difficulty "
          "(see `skill_gain` columns, which are relative to seasonal-naive on the same rows); P6 also trains on ~80% of the data "
          "per fold vs the smaller expanding windows of P6B.", "",
          "## Control arms vs P0 (not leakage)", ctrl.round(2).to_markdown(), "",
          f"## Omnibus\n{friedman_txt}", "",
          "## By dataset", by_ds.round(2).to_markdown(), "", "## By family", by_fam.round(2).to_markdown(), "",
          "## Inflation vs autocorrelation (Spearman)", pd.DataFrame(corr_rows).round(3).to_markdown(index=False) if corr_rows else "n/a"]
    (out / "SUMMARY.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()

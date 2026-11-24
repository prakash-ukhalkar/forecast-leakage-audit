"""Validate the protocol audit (src/sdg_leakage_audit/audit) on the protocols and datasets of this project.

For every (dataset, horizon, protocol) the index sets that the experiment runner really uses are rebuilt and
passed to the audit. The audit should (i) stay silent on the corrected protocols P0, P0V and P6B, and (ii) raise an
error on each deliberately flawed protocol. The black-box look-ahead probe is run on a causal and a centred feature
builder, and the declared-window rule is compared with the measured P4 inflation.

Usage: python scripts/validate_audit.py
Writes results/final/tables/audit_*.csv and results/final/AUDIT_VALIDATION.md
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sdg_leakage_audit.audit import (  # noqa: E402
    audit,
    check_feature_windows,
    detect_lookahead,
    detect_test_dependence,
    window_reach,
)
from sdg_leakage_audit.evaluation import runner as R  # noqa: E402
from sdg_leakage_audit.features.build import build_feature_matrix  # noqa: E402
from sdg_leakage_audit.protocols.splitters import (  # noqa: E402
    blocked_cv_splits,
    chronological_split_with_gap,
    random_split,
    shuffled_kfold_splits,
    size_matched_shuffled_splits,
)
from sdg_leakage_audit.utils.io import read_yaml  # noqa: E402

OUT = ROOT / "results" / "final"
TAB = OUT / "tables"
DATASETS = ["household_power_consumption", "appliances_energy_prediction", "air_quality_uci",
            "beijing_multisite_air_quality", "jena_climate", "ett_h1"]
PROTOCOLS = ["P0", "P0V", "P6B", "P1", "P2", "P3", "P4", "P5", "P6", "P6S"]
CLEAN = {"P0", "P0V", "P6B"}
MAX_ROWS = 20000


def n_rows(df: pd.DataFrame, target: str, season: int, horizon: int) -> int:
    X, y, _ = R._build_direct_horizon_xy(df, target, horizon)
    sf = R._seasonal_forecast_for_index(df, target, season, horizon, X.index)
    ok = ~np.isnan(sf)
    return min(int(ok.sum()), MAX_ROWS)


def structural(df, target, season, horizon, protocol, n, seed=0):
    windows = [int(c[len(target) + 5 : -5]) for c in df.columns if c.startswith(f"{target}_roll") and c.endswith("_mean")]
    chrono = chronological_split_with_gap(n, 0.7, 0.2, horizon)
    kw: dict = {}
    if protocol in ("P0", "P2", "P4", "P5", "P0V"):
        kw["split"] = chrono
    if protocol == "P1":
        kw["split"] = random_split(n, 0.7, 0.2, seed=seed)
    if protocol == "P3":
        kw["split"] = chronological_split_with_gap(n, 0.7, 0.2, 0)
    if protocol == "P2":
        kw["fit_idx"] = np.concatenate(kw["split"])  # scaler/imputer fitted on train + evaluation rows
    elif protocol in ("P0", "P0V", "P4", "P5"):
        kw["fit_idx"] = kw["split"][0]
    if protocol == "P5":
        kw["selection_idx"] = kw["split"][1]
    if protocol == "P0V":
        tr = kw["split"][0]
        kw["selection_idx"] = tr[int(len(tr) * 0.8):]
    if protocol == "P4":
        kw["windows"], kw["centered"] = windows, True
    elif protocol in ("P0", "P0V", "P5", "P1", "P2", "P3"):
        kw["windows"], kw["centered"] = windows, False
    if protocol == "P6":
        kw["folds"] = shuffled_kfold_splits(n, 5, seed=seed)
    if protocol == "P6B":
        kw["folds"] = blocked_cv_splits(n, 5, horizon, expanding=True)
    if protocol == "P6S":
        kw["folds"] = size_matched_shuffled_splits(n, 5, horizon, seed=seed)
    # selection_idx for P0/P1/P2/P3 is validation-free; evaluation indices needed for the scope checks
    if "split" not in kw and ("fit_idx" in kw or "selection_idx" in kw):
        kw["test_idx"] = None
    return audit(horizon, **kw)


def main() -> None:
    cfgs = read_yaml(ROOT / "configs" / "datasets.yaml")["datasets"]
    rows = []
    for d in DATASETS:
        c = cfgs[d]
        df = R._load_dataset_frame(d)
        target, season = c["target_column"], c["season_length"]
        for h in c["forecast_horizons"]:
            n = n_rows(df, target, season, h)
            for p in PROTOCOLS:
                rep = structural(df, target, season, h, p, n)
                rows.append({"dataset": d, "horizon": h, "protocol": p, "flawed_by_design": p not in CLEAN,
                             "errors": ";".join(sorted(rep.codes("error"))), "warnings": ";".join(sorted(rep.codes("warn"))),
                             "flagged": bool(rep.errors), "flagged_any": bool(rep.errors or rep.warnings)})
    S = pd.DataFrame(rows)
    S.to_csv(TAB / "audit_structural_by_cell.csv", index=False)
    summ = S.groupby("protocol").agg(cells=("flagged", "size"), flagged=("flagged", "sum"), flagged_any=("flagged_any", "sum"), flawed=("flawed_by_design", "first"))
    summ["flag_rate"] = summ["flagged"] / summ["cells"]
    summ = summ.loc[PROTOCOLS]
    summ["codes"] = [", ".join(sorted({c for e in S[S.protocol == p].errors for c in e.split(";") if c})) or "none" for p in summ.index]
    summ.to_csv(TAB / "audit_structural_summary.csv")
    tp = int(S[S.flawed_by_design].flagged.sum()); fn = int((~S[S.flawed_by_design].flagged).sum())
    fp = int(S[~S.flawed_by_design].flagged.sum()); tn = int((~S[~S.flawed_by_design].flagged).sum())

    # ---- black-box look-ahead probe on causal vs centred builders (first 6,000 rows keeps the probe fast)
    lrows = []
    for d in DATASETS:
        c = cfgs[d]
        df = R._load_dataset_frame(d)[[c["target_column"]]].iloc[:6000].astype(float)
        target, H = c["target_column"], c["forecast_horizons"]

        def causal(x, target=target, H=H):
            out = build_feature_matrix(x, target, lags=H, rolling_windows=H, include_calendar=False)
            return out.drop(columns=[target])

        def centred(x, target=target, H=H):
            out = causal(x)
            for col in out.columns:
                if col.startswith(f"{target}_roll") and col.endswith("_mean"):
                    w = int(col[len(target) + 5 : -5])
                    out[col] = x[target].rolling(window=w, center=True, min_periods=1).mean()
            return out
        wmax = max(H)
        for kind, fn_ in (("causal", causal), ("centred", centred)):
            f, reach = detect_lookahead(fn_, df, target, horizon=min(H), n_probes=2, max_lead=min(150, wmax))
            lrows.append({"dataset": d, "builder": kind, "flagged_columns": len(reach),
                          "measured_max_reach": max(reach.values()) if reach else 0,
                          "rule_reach": window_reach(wmax, True) if kind == "centred" else 0})
    L = pd.DataFrame(lrows)
    L.to_csv(TAB / "audit_lookahead_probe.csv", index=False)

    # ---- fit-leak probe on the two scalers used in P0 and P2
    rng = np.random.default_rng(0)
    Xtr, Xte = rng.normal(size=(300, 6)), rng.normal(size=(100, 6))

    def correct(a, b):
        mu, sd = a.mean(0), a.std(0)
        return (a - mu) / sd, (b - mu) / sd

    def leaky(a, b):
        both = np.vstack([a, b]); mu, sd = both.mean(0), both.std(0)
        return (a - mu) / sd, (b - mu) / sd
    fit = {"P0 scaler": detect_test_dependence(correct, Xtr, Xte)[0].severity, "P2 scaler": detect_test_dependence(leaky, Xtr, Xte)[0].severity}

    # ---- declared-window rule vs measured P4 inflation (existing grid results)
    unit = pd.read_csv(TAB / "inflation_by_unit.csv")
    p4 = unit[(unit.protocol_id == "P4")].groupby(["dataset_id", "horizon"]).mae_inflation_pct.mean().reset_index()
    wmax = {d: max(cfgs[d]["forecast_horizons"]) for d in DATASETS}
    wmax["household_power_consumption"] = 60
    p4["rule_exposed"] = [window_reach(wmax[d], True) >= h for d, h in zip(p4.dataset_id, p4.horizon)]
    p4["empirical_leak_5pct"] = p4.mae_inflation_pct > 5.0
    p4.to_csv(TAB / "audit_rule_vs_empirical_p4.csv", index=False)
    agree = float((p4.rule_exposed == p4.empirical_leak_5pct).mean())
    nec = float(p4[p4.empirical_leak_5pct].rule_exposed.mean())
    # ---- report
    md = ["# Audit validation", "",
          f"Structural checks over {len(S)} (dataset, horizon, protocol) cells: flawed protocols flagged as errors {tp}/{tp + fn} "
          f"(error or warning: {int(S[S.flawed_by_design].flagged_any.sum())}/{tp + fn}); corrected protocols flagged {fp}/{fp + tn} (false positives).", "",
          summ.to_markdown(), "", "## Black-box look-ahead probe", "", L.to_markdown(index=False), "",
          f"## Fit-leak probe: {fit}", "",
          f"## Declared-window rule vs measured P4 inflation (>5%): agreement {agree:.2f}; share of empirically leaking cells that the rule marks exposed: {nec:.2f}", "",
          p4.round(2).to_markdown(index=False)]
    (OUT / "AUDIT_VALIDATION.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md[:6])); print(L.to_string()); print(fit, "agreement", agree, "necessary", nec)


if __name__ == "__main__":
    main()

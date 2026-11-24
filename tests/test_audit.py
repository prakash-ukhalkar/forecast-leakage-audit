"""Tests for the protocol audit module (structural checks and black-box probes)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from sdg_leakage_audit.audit import (
    audit,
    check_cv_folds,
    check_feature_windows,
    check_selection_scope,
    check_split,
    detect_lookahead,
    detect_test_dependence,
    window_reach,
)
from sdg_leakage_audit.features.build import build_feature_matrix
from sdg_leakage_audit.protocols.splitters import (
    blocked_cv_splits,
    chronological_split_with_gap,
    random_split,
    shuffled_kfold_splits,
    size_matched_shuffled_splits,
)


def _codes(findings, sev=None):
    return {f.code for f in findings if sev is None or f.severity == sev}


def test_split_checks_pass_for_p0_and_flag_p1_and_p3():
    n, h = 500, 6
    tr, te = chronological_split_with_gap(n, 0.7, 0.2, h)
    assert _codes(check_split(tr, te, h), "error") == set()
    tr1, te1 = random_split(n, 0.7, 0.2, seed=0)
    assert "SPLIT-ORDER" in _codes(check_split(tr1, te1, h))
    tr3, te3 = chronological_split_with_gap(n, 0.7, 0.2, 0)
    assert "SPLIT-GAP" in _codes(check_split(tr3, te3, h))


def test_cv_checks_flag_shuffled_and_size_matched_but_not_blocked():
    n, h = 600, 6
    assert _codes(check_cv_folds(blocked_cv_splits(n, 5, h, expanding=True), h), "error") == set()
    assert "CV-LEAK" in _codes(check_cv_folds(shuffled_kfold_splits(n, 5, seed=1), h))
    assert "CV-LEAK" in _codes(check_cv_folds(size_matched_shuffled_splits(n, 5, h, seed=1), h))
    # blocked expanding folds have unequal training sizes: a warning, not an error
    assert "CV-SIZE" in _codes(check_cv_folds(blocked_cv_splits(n, 5, h, expanding=True), h), "warn")


def test_selection_scope_flags_test_selection():
    te = np.arange(80, 100)
    assert "SELECT-LEAK" in _codes(check_selection_scope(te, te))
    assert "SELECT-OK" in _codes(check_selection_scope(np.arange(60, 70), te))


def test_feature_window_rule_matches_horizon_reach():
    assert window_reach(24, True) == 11 and window_reach(25, True) == 12 and window_reach(24, False) == 0
    assert "FEAT-LOOKAHEAD" in _codes(check_feature_windows([1, 6, 24], horizon=6, centered=True), "error")
    assert "FEAT-LOOKAHEAD" not in _codes(check_feature_windows([1, 6, 24], horizon=24, centered=True))
    assert "FEAT-FUTURE" in _codes(check_feature_windows([1, 6, 24], horizon=24, centered=True), "warn")
    assert "FEAT-OK" in _codes(check_feature_windows([1, 6, 24], horizon=6, centered=False))


def _frame(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    y = np.sin(np.arange(n) / 9.0) + 0.1 * rng.normal(size=n)
    return pd.DataFrame({"y": y}, index=pd.date_range("2020-01-01", periods=n, freq="h"))


def _causal(df):
    out = build_feature_matrix(df, "y", lags=[1, 6], rolling_windows=[6, 24], include_calendar=False)
    return out[[c for c in out.columns if c != "y"]]


def _centred(df):
    out = _causal(df)
    for w in (6, 24):
        r = df["y"].rolling(window=w, center=True, min_periods=1)
        out[f"y_roll{w}_mean"] = r.mean()
    return out


def test_blackbox_lookahead_detects_centred_windows_and_clears_causal_ones():
    df = _frame()
    ok, reach = detect_lookahead(_causal, df, "y", horizon=6, n_probes=2, max_lead=48)
    assert reach == {} and _codes(ok) == {"FEAT-OK"}
    bad, reach = detect_lookahead(_centred, df, "y", horizon=6, n_probes=2, max_lead=48)
    assert "FEAT-LOOKAHEAD" in _codes(bad, "error")
    assert reach["y_roll24_mean"] == 11 and reach["y_roll6_mean"] == 2  # matches window_reach(w, True)
    # beyond the reach the same builder is only a warning: horizon 48 exceeds the largest lead probed
    warn, _ = detect_lookahead(_centred, df, "y", horizon=48, n_probes=2, max_lead=48)
    assert "FEAT-FUTURE" in _codes(warn, "warn")


def test_blackbox_fit_leak_detection():
    rng = np.random.default_rng(0)
    Xtr, Xte = rng.normal(size=(100, 3)), rng.normal(size=(30, 3))

    def correct(a, b):
        mu, sd = a.mean(0), a.std(0)
        return (a - mu) / sd, (b - mu) / sd

    def leaky(a, b):
        both = np.vstack([a, b])
        mu, sd = both.mean(0), both.std(0)
        return (a - mu) / sd, (b - mu) / sd

    assert "FIT-OK" in _codes(detect_test_dependence(correct, Xtr, Xte))
    assert "FIT-LEAK" in _codes(detect_test_dependence(leaky, Xtr, Xte), "error")


def test_audit_report_orchestration():
    n, h = 500, 6
    tr, te = chronological_split_with_gap(n, 0.7, 0.2, h)
    rep = audit(h, split=(tr, te), fit_idx=tr, selection_idx=np.arange(300, 320), windows=[6, 24], centered=False)
    assert rep.passed
    rep_bad = audit(h, split=(tr, te), fit_idx=np.concatenate([tr, te]), selection_idx=te, windows=[6, 24], centered=True)
    assert not rep_bad.passed and {"FIT-LEAK", "SELECT-LEAK", "FEAT-LOOKAHEAD"} <= rep_bad.codes("error")
    assert "| severity |" in rep_bad.to_markdown()

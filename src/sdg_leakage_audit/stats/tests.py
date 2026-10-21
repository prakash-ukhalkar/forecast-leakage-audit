"""Significance tests: stationarity (ADF/KPSS), paired tests, Friedman/Nemenyi.

`stationarity_tests` is implemented for notebook 03; `paired_tests_with_correction`,
`friedman_nemenyi`, and `effect_size_cliffs_delta` are implemented for notebook 08.
"""

from __future__ import annotations

import warnings
from itertools import combinations

import numpy as np
import pandas as pd
import scikit_posthocs as sp
from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import InterpolationWarning, adfuller, kpss

from sdg_leakage_audit.utils.logging import get_logger

logger = get_logger(__name__)

_KEY_COLS = ["dataset_id", "model_name", "horizon", "seed"]


def stationarity_tests(series: pd.Series, regression: str = "c") -> dict[str, object]:
    """Run ADF (null: non-stationary/unit root) and KPSS (null: stationary) tests.

    The two tests have opposite null hypotheses, so running both and comparing their
    conclusions is more informative than either alone: ADF rejecting + KPSS not
    rejecting is the clean "stationary" case; both tests disagreeing flags a series
    that needs closer inspection (e.g. trend-stationary, or borderline sample size).

    Args:
        series: Time-indexed numeric series (NaNs are dropped before testing).
        regression: Deterministic trend term for both tests ("c" constant only,
            "ct" constant + trend).

    Returns:
        Dict with keys: `adf_statistic`, `adf_pvalue`, `adf_rejects_unit_root`
        (True if p < 0.05), `kpss_statistic`, `kpss_pvalue`,
        `kpss_rejects_stationary` (True if p < 0.05), and `conclusion` (one of
        "stationary", "non_stationary", "trend_stationary", "inconclusive").
    """
    clean = series.dropna()
    if len(clean) < 20:
        raise ValueError(f"Need at least 20 non-NaN observations for stationarity tests, got {len(clean)}")

    adf_stat, adf_pvalue, *_ = adfuller(clean, regression=regression)

    with np.errstate(all="ignore"), warnings.catch_warnings():
        # KPSS's test statistic routinely falls outside its lookup table's p-value
        # range for the large sample sizes used here; the p-value is still a valid
        # (clamped) bound, so this is expected and not worth surfacing as a warning.
        warnings.simplefilter("ignore", category=InterpolationWarning)
        kpss_stat, kpss_pvalue, *_ = kpss(clean, regression=regression, nlags="auto")

    adf_rejects_unit_root = bool(adf_pvalue < 0.05)
    kpss_rejects_stationary = bool(kpss_pvalue < 0.05)

    if adf_rejects_unit_root and not kpss_rejects_stationary:
        conclusion = "stationary"
    elif not adf_rejects_unit_root and kpss_rejects_stationary:
        conclusion = "non_stationary"
    elif adf_rejects_unit_root and kpss_rejects_stationary:
        conclusion = "trend_stationary"
    else:
        conclusion = "inconclusive"

    return {
        "adf_statistic": float(adf_stat),
        "adf_pvalue": float(adf_pvalue),
        "adf_rejects_unit_root": adf_rejects_unit_root,
        "kpss_statistic": float(kpss_stat),
        "kpss_pvalue": float(kpss_pvalue),
        "kpss_rejects_stationary": kpss_rejects_stationary,
        "conclusion": conclusion,
    }


def paired_tests_with_correction(
    results_df: pd.DataFrame, metric: str, method: str = "holm"
) -> pd.DataFrame:
    """Run paired Wilcoxon signed-rank tests between every pair of protocols, with
    multiple-comparison correction.

    Rows are matched into pairs by (dataset_id, model_name, horizon, seed); only
    keys present for both protocols in a pair are used. A pair with fewer than 2
    matched keys, or with all-zero differences (Wilcoxon is undefined for a
    degenerate zero-variance input), is skipped and excluded from correction.

    Args:
        results_df: Long-format results DataFrame with a `protocol_id` column and
            `dataset_id`/`model_name`/`horizon`/`seed` key columns.
        metric: Metric column name to test (e.g. "mae").
        method: Multiple-comparison correction method (e.g. "holm", "bonferroni"),
            passed through to `statsmodels.stats.multitest.multipletests`.

    Returns:
        DataFrame with columns: protocol_a, protocol_b, n_pairs, statistic,
        pvalue, pvalue_corrected, significant. Empty if fewer than 2 protocols
        have any matched pairs.
    """
    protocols = sorted(results_df["protocol_id"].unique())
    pivot = results_df.pivot_table(index=_KEY_COLS, columns="protocol_id", values=metric)

    rows = []
    for protocol_a, protocol_b in combinations(protocols, 2):
        paired = pivot[[protocol_a, protocol_b]].dropna()
        if len(paired) < 2:
            logger.debug("paired_tests_with_correction: skipping %s vs %s (< 2 matched pairs)", protocol_a, protocol_b)
            continue
        diffs = paired[protocol_a] - paired[protocol_b]
        if (diffs == 0).all():
            logger.debug("paired_tests_with_correction: skipping %s vs %s (identical values)", protocol_a, protocol_b)
            continue
        statistic, pvalue = wilcoxon(paired[protocol_a], paired[protocol_b])
        rows.append(
            {
                "protocol_a": protocol_a,
                "protocol_b": protocol_b,
                "n_pairs": len(paired),
                "statistic": float(statistic),
                "pvalue": float(pvalue),
            }
        )

    result_df = pd.DataFrame(rows)
    if result_df.empty:
        return result_df

    reject, pvalues_corrected, _, _ = multipletests(result_df["pvalue"], method=method)
    result_df["pvalue_corrected"] = pvalues_corrected
    result_df["significant"] = reject
    return result_df


def friedman_nemenyi(results_df: pd.DataFrame, metric: str) -> dict[str, object]:
    """Run a Friedman omnibus test followed by Nemenyi post-hoc pairwise comparisons.

    Rows are pivoted to (block, protocol) = ((dataset_id, model_name, horizon,
    seed), protocol_id); only blocks with a value for every protocol are kept
    (Friedman requires a complete, unreplicated block design).

    Args:
        results_df: Long-format results DataFrame.
        metric: Metric column name to test.

    Returns:
        Dict with keys "friedman_statistic", "friedman_pvalue", "protocols"
        (column/row order of the Nemenyi matrix), and "nemenyi_pvalues" (a
        DataFrame of pairwise p-values indexed/columned by protocol id).

    Raises:
        ValueError: If fewer than 2 complete blocks remain after dropping rows
            with any missing protocol.
    """
    pivot = results_df.pivot_table(index=_KEY_COLS, columns="protocol_id", values=metric).dropna()
    protocols = sorted(pivot.columns)
    if len(pivot) < 2:
        raise ValueError(f"Need at least 2 complete blocks for Friedman test, got {len(pivot)}")

    groups = [pivot[p].to_numpy() for p in protocols]
    friedman_statistic, friedman_pvalue = friedmanchisquare(*groups)

    nemenyi_pvalues = sp.posthoc_nemenyi_friedman(pivot[protocols].to_numpy())
    nemenyi_pvalues.index = protocols
    nemenyi_pvalues.columns = protocols

    return {
        "friedman_statistic": float(friedman_statistic),
        "friedman_pvalue": float(friedman_pvalue),
        "protocols": protocols,
        "nemenyi_pvalues": nemenyi_pvalues,
    }


def effect_size_cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Cliff's delta effect size between two independent samples.

    Args:
        x: First sample.
        y: Second sample.

    Returns:
        Cliff's delta in [-1, 1]: `(#{x_i > y_j} - #{x_i < y_j}) / (len(x) * len(y))`.
        Positive means `x` tends to exceed `y`; 0 means no stochastic dominance
        either way.

    Raises:
        ValueError: If either sample is empty.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) == 0 or len(y) == 0:
        raise ValueError("Both samples must be non-empty")

    comparisons = x[:, None] - y[None, :]
    n_greater = np.sum(comparisons > 0)
    n_less = np.sum(comparisons < 0)
    return float((n_greater - n_less) / (len(x) * len(y)))

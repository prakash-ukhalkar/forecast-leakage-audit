"""Protocol audit: automated detection of evaluation leakage in time-series prediction pipelines.

Two kinds of checks are provided.

Structural checks inspect the index sets of a protocol (what was used to train, fit
pre-processing, select hyperparameters and evaluate):

* ``check_split``           chronological order and purge gap of a train/test split
* ``check_cv_folds``        the same for every fold of a cross-validation scheme
* ``check_fit_scope``       whether pre-processing was fitted on evaluation rows
* ``check_selection_scope`` whether model selection scored candidates on evaluation rows
* ``check_feature_windows`` whether declared feature windows reach the forecast target

Black-box probes need only callables, so they work on pipelines whose internals are not
inspectable:

* ``detect_lookahead``       perturbs the series after a forecast origin and reports which
                             feature columns change, together with how far ahead they read
* ``detect_test_dependence`` perturbs the evaluation rows and reports whether anything the
                             pipeline derives for the training rows changes (fit leakage)

Every check returns a list of ``Finding`` objects; ``audit`` collects them in a report.
A finding of severity "error" means the protocol violates a leakage-safety rule; "warn"
means the design is questionable but the target itself is not necessarily exposed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Finding:
    code: str  # short identifier, e.g. "SPLIT-ORDER"
    severity: str  # "error" | "warn" | "ok"
    message: str

    def __str__(self) -> str:  # pragma: no cover - formatting only
        return f"[{self.severity.upper():5s}] {self.code}: {self.message}"


@dataclass
class AuditReport:
    findings: list[Finding] = field(default_factory=list)

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "error"]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "warn"]

    @property
    def passed(self) -> bool:
        return not self.errors

    def codes(self, severity: str | None = None) -> set[str]:
        return {f.code for f in self.findings if severity is None or f.severity == severity}

    def to_markdown(self) -> str:
        lines = ["| severity | check | message |", "|---|---|---|"]
        lines += [f"| {f.severity} | {f.code} | {f.message} |" for f in self.findings]
        return "\n".join(lines)


# ----------------------------------------------------------------------------- structural checks
def check_split(train_idx: Sequence[int], test_idx: Sequence[int], horizon: int, name: str = "split") -> list[Finding]:
    """Chronological order, overlap and purge gap of one train/test split (positions in time order)."""
    tr, te = np.asarray(train_idx), np.asarray(test_idx)
    out: list[Finding] = []
    if len(np.intersect1d(tr, te)) > 0:
        out.append(Finding("SPLIT-OVERLAP", "error", f"{name}: train and evaluation rows overlap"))
    if len(tr) == 0 or len(te) == 0:
        return out + [Finding("SPLIT-EMPTY", "error", f"{name}: empty train or evaluation set")]
    if te.min() < tr.max():
        n_before = int((te < tr.max()).sum())
        out.append(Finding("SPLIT-ORDER", "error",
                           f"{name}: {n_before} of {len(te)} evaluation rows precede the last training row (not chronological)"))
        return out  # gap is meaningless when order is violated
    gap = int(te.min() - tr.max() - 1)
    if gap < horizon:
        out.append(Finding("SPLIT-GAP", "error",
                           f"{name}: purge gap {gap} is shorter than the horizon {horizon}; training targets overlap the evaluation window"))
    if not out:
        out.append(Finding("SPLIT-OK", "ok", f"{name}: chronological with purge gap {gap} >= horizon {horizon}"))
    return out


def check_cv_folds(folds: Iterable[tuple[Sequence[int], Sequence[int]]], horizon: int) -> list[Finding]:
    """Apply `check_split` to every fold and summarise; also reports the spread of training sizes."""
    folds = list(folds)
    out: list[Finding] = []
    bad = 0
    for i, (tr, te) in enumerate(folds):
        fs = [f for f in check_split(tr, te, horizon, name=f"fold {i}") if f.severity == "error"]
        bad += bool(fs)
        out += fs[:1]
    if bad:
        out.insert(0, Finding("CV-LEAK", "error", f"{bad} of {len(folds)} folds violate chronological order or the purge gap"))
    else:
        out.append(Finding("CV-OK", "ok", f"all {len(folds)} folds are chronological with purge gap >= horizon"))
    sizes = [len(tr) for tr, _ in folds]
    if sizes and max(sizes) > 1.5 * max(min(sizes), 1):
        out.append(Finding("CV-SIZE", "warn",
                           f"training sizes range from {min(sizes)} to {max(sizes)}; compare with a size-matched control before reading CV error as a deployment estimate"))
    return out


def check_fit_scope(fit_idx: Sequence[int], test_idx: Sequence[int], what: str = "pre-processing") -> list[Finding]:
    """Rows used to fit imputers/scalers/encoders must not include evaluation rows."""
    n = len(np.intersect1d(np.asarray(fit_idx), np.asarray(test_idx)))
    if n:
        return [Finding("FIT-LEAK", "error", f"{what} was fitted on {n} evaluation rows")]
    return [Finding("FIT-OK", "ok", f"{what} fitted on training rows only")]


def check_selection_scope(selection_idx: Sequence[int], test_idx: Sequence[int]) -> list[Finding]:
    """Rows used to score hyperparameter candidates must not include evaluation rows."""
    n = len(np.intersect1d(np.asarray(selection_idx), np.asarray(test_idx)))
    if n:
        return [Finding("SELECT-LEAK", "error", f"model selection scored candidates on {n} evaluation rows")]
    return [Finding("SELECT-OK", "ok", "model selection used rows disjoint from evaluation")]


def window_reach(window: int, centered: bool) -> int:
    """Steps into the future a rolling window of length `window` reads (0 for trailing windows)."""
    return (window - 1) // 2 if centered else 0


def check_feature_windows(windows: Iterable[int], horizon: int, centered: bool) -> list[Finding]:
    """Declared-window rule: a centred window of length w reaches (w-1)//2 steps ahead and exposes
    the target when (w-1)//2 >= horizon (the horizon-reach rule). Any future reach is illegitimate."""
    ws = list(windows)
    if not centered:
        return [Finding("FEAT-OK", "ok", "all rolling windows are trailing (no future reach)")]
    wmax = max(ws)
    reach = window_reach(wmax, True)
    if reach >= horizon:
        return [Finding("FEAT-LOOKAHEAD", "error",
                        f"centred window {wmax} reaches {reach} steps ahead, which covers the horizon {horizon}: the target can leak into the features")]
    if reach > 0:
        return [Finding("FEAT-FUTURE", "warn",
                        f"centred window {wmax} reads {reach} future steps; below the horizon {horizon}, but future values are still used")]
    return [Finding("FEAT-OK", "ok", "no future reach")]


# ----------------------------------------------------------------------------- black-box probes
def detect_lookahead(
    build_features: Callable[[pd.DataFrame], pd.DataFrame],
    df: pd.DataFrame,
    target_column: str,
    horizon: int,
    n_probes: int = 3,
    max_lead: int = 200,
    seed: int = 0,
    tol: float = 1e-9,
    history: int = 300,
) -> tuple[list[Finding], dict[str, int]]:
    """Find feature columns that read values after the forecast origin.

    For a random origin t0, the target series is perturbed after t0 and the features are rebuilt.
    A feature whose value AT t0 changes uses the future. The perturbation is applied one lead at a
    time (t0+1 ... t0+max_lead), so the largest lead that changes a column is its reach.

    Args:
        build_features: callable mapping a DataFrame (with ``target_column``) to a feature DataFrame
            aligned on the same index.
        df: input data; only ``target_column`` is perturbed.
        horizon: forecast horizon in steps.
        n_probes: number of random origins; the reach is the maximum over probes.
        max_lead: largest lead examined (steps), scanned one step at a time.
        history: rows of history kept before the origin (must cover the longest lag or window).

    Returns:
        ``(findings, reach)`` where ``reach`` maps each leaking column to its largest lead.
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    hist = max(max_lead, history)
    reach: dict[str, int] = {}
    lo, hi = hist, n - max_lead - 2
    origins = rng.integers(lo, hi, size=n_probes) if hi > lo else [n // 2]
    for t0 in origins:
        t0 = int(t0)
        # features at the origin depend only on a local slice: `hist` rows of history, `max_lead` of future
        sub = df.iloc[t0 - hist : t0 + max_lead + 2]
        loc = hist
        ref = build_features(sub).iloc[loc].to_numpy(dtype=float)
        cols = list(build_features(sub).columns)
        for lead in range(1, max_lead + 1):
            d2 = sub.copy()
            d2.iloc[loc + lead, d2.columns.get_loc(target_column)] += 1.0e3
            changed = build_features(d2).iloc[loc].to_numpy(dtype=float)
            diff = np.abs(changed - ref)
            for c, dv in zip(cols, diff):
                if np.isfinite(dv) and dv > tol:
                    reach[c] = max(reach.get(c, 0), lead)
    if not reach:
        return [Finding("FEAT-OK", "ok", "no feature column depends on values after the forecast origin")], {}
    top = max(reach.values())
    sev = "error" if top >= horizon else "warn"
    cols = ", ".join(sorted(reach)[:4]) + (" ..." if len(reach) > 4 else "")
    return [Finding("FEAT-LOOKAHEAD" if sev == "error" else "FEAT-FUTURE", sev,
                    f"{len(reach)} feature column(s) read future values (reach up to {top} steps; horizon {horizon}): {cols}")], reach


def detect_test_dependence(
    pipeline: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]],
    X_train: np.ndarray,
    X_test: np.ndarray,
    seed: int = 0,
    tol: float = 1e-9,
) -> list[Finding]:
    """Black-box test for fit leakage in pre-processing.

    ``pipeline(X_train, X_test)`` must fit whatever it fits and return the transformed
    ``(train, test)`` arrays. If the transformed TRAINING rows change when only the evaluation rows
    are perturbed, the pipeline has used evaluation data to fit its transformation.
    """
    rng = np.random.default_rng(seed)
    a, _ = pipeline(np.array(X_train, dtype=float), np.array(X_test, dtype=float))
    Xt2 = np.array(X_test, dtype=float) + rng.normal(5.0, 1.0, size=np.shape(X_test))
    b, _ = pipeline(np.array(X_train, dtype=float), Xt2)
    if np.nanmax(np.abs(np.asarray(a) - np.asarray(b))) > tol:
        return [Finding("FIT-LEAK", "error", "transformed training rows depend on the evaluation rows (pre-processing fitted on evaluation data)")]
    return [Finding("FIT-OK", "ok", "training transformation is independent of evaluation rows")]


# ----------------------------------------------------------------------------- orchestration
def audit(
    horizon: int,
    split: tuple[Sequence[int], Sequence[int]] | None = None,
    folds: Iterable[tuple[Sequence[int], Sequence[int]]] | None = None,
    fit_idx: Sequence[int] | None = None,
    selection_idx: Sequence[int] | None = None,
    test_idx: Sequence[int] | None = None,
    windows: Iterable[int] | None = None,
    centered: bool = False,
) -> AuditReport:
    """Run every structural check for which inputs are supplied and collect the findings."""
    rep = AuditReport()
    if split is not None:
        rep.findings += check_split(split[0], split[1], horizon)
        test_idx = split[1] if test_idx is None else test_idx
    if folds is not None:
        rep.findings += check_cv_folds(folds, horizon)
    if fit_idx is not None and test_idx is not None:
        rep.findings += check_fit_scope(fit_idx, test_idx)
    if selection_idx is not None and test_idx is not None:
        rep.findings += check_selection_scope(selection_idx, test_idx)
    if windows is not None:
        rep.findings += check_feature_windows(windows, horizon, centered)
    return rep

"""Automated protocol audit (structural checks and black-box probes) for time-series evaluation pipelines."""
from sdg_leakage_audit.audit.checks import (  # noqa: F401
    AuditReport,
    Finding,
    audit,
    check_cv_folds,
    check_feature_windows,
    check_fit_scope,
    check_selection_scope,
    check_split,
    detect_lookahead,
    detect_test_dependence,
    window_reach,
)

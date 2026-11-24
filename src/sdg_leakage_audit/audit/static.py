"""Static detectors for evaluation errors in Python forecasting code and notebooks.

The detectors parse source with ``ast`` and report evidence lines for five error families. They are heuristic: they
look for call patterns, not for data flow, so every family is reported together with a manually checked precision
estimate (see scripts/prevalence_audit.py). Families:

* RANDOM_SPLIT     ``train_test_split`` without ``shuffle=False`` (default shuffling) or with ``shuffle=True``
* NONCHRONO_CV     cross-validation that is not time ordered: ``KFold`` / ``ShuffleSplit`` / ``StratifiedKFold`` objects,
                   or an integer ``cv`` passed to ``cross_val_score`` / ``cross_val_predict`` / ``GridSearchCV`` and similar
* FIT_BEFORE_SPLIT a scaler, imputer, PCA or similar is fitted on an argument without ``train`` in its name before the
                   first split in source order
* LOOKAHEAD        ``rolling(..., center=True)``, or a negative ``shift`` assigned to a column whose name does not look
                   like a target
* TEST_SELECTION   ``fit(..., validation_data=<expression naming a test set>)``

Compliant practices are reported too (TimeSeriesSplit, ``shuffle=False``, slicing splits) to give a denominator.
"""
from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field

ML_IMPORT = re.compile(r"\b(sklearn|keras|tensorflow|torch|xgboost|lightgbm|catboost|statsmodels|prophet|darts|sktime|neuralprophet)\b")
TS_CUES = re.compile(r"(forecast|time[ _-]?series|timeseries|lstm|gru|arima|sarima|prophet|date_range|to_datetime|parse_dates|DatetimeIndex|"
                     r"\.resample\(|\.rolling\(|\.shift\(|lags?)", re.I)
SCALERS = {"StandardScaler", "MinMaxScaler", "RobustScaler", "MaxAbsScaler", "Normalizer", "SimpleImputer", "KNNImputer", "IterativeImputer",
           "PCA", "PowerTransformer", "QuantileTransformer", "SelectKBest", "TruncatedSVD"}  # label and category encoders are excluded on purpose
NONCHRONO_CV = {"KFold", "StratifiedKFold", "ShuffleSplit", "RepeatedKFold", "GroupKFold", "StratifiedShuffleSplit", "LeaveOneOut"}
CV_FUNCS = {"cross_val_score", "cross_val_predict", "cross_validate", "GridSearchCV", "RandomizedSearchCV", "RandomizedSearchCV",
            "HalvingGridSearchCV", "learning_curve", "validation_curve"}
TARGET_NAME = re.compile(r"(target|label|^y|_y$|next|future|forecast|pred|ahead|tomorrow|output|response|t\+|horizon|lead)", re.I)


@dataclass
class FileReport:
    qualifies: bool = False  # ML library + time-series cue + evaluation call
    flags: dict[str, list[str]] = field(default_factory=dict)  # family -> evidence lines
    compliant: dict[str, int] = field(default_factory=dict)  # practice -> count


def code_from_source(text: str, is_notebook: bool) -> str:
    """Return plain Python from a .py file or the code cells of a .ipynb file (magics removed)."""
    if not is_notebook:
        return text
    try:
        nb = json.loads(text)
    except Exception:  # noqa: BLE001
        return ""
    cells = nb.get("cells") or [c for ws in nb.get("worksheets", []) for c in ws.get("cells", [])]
    out = []
    for c in cells:
        if c.get("cell_type") == "code":
            src = c.get("source") or c.get("input") or []
            src = "".join(src) if isinstance(src, list) else src
            out.append("\n".join(("# " + ln if ln.lstrip().startswith(("%", "!", "?")) else ln) for ln in src.split("\n")))
    return "\n\n".join(out)


def _name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _kw(call: ast.Call, key: str):
    for k in call.keywords:
        if k.arg == key:
            return k.value
    return None


def _text(src: str, node: ast.AST) -> str:
    try:
        return ast.get_source_segment(src, node) or ""
    except Exception:  # noqa: BLE001
        return ""


def _const(node):
    return node.value if isinstance(node, ast.Constant) else None


def analyse_source(src: str) -> FileReport:
    rep = FileReport()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return rep
    rep.qualifies = bool(ML_IMPORT.search(src)) and bool(TS_CUES.search(src)) and bool(
        re.search(r"(train_test_split|\.fit\(|cross_val|GridSearchCV|TimeSeriesSplit|\.predict\()", src))
    scaler_vars: set[str] = set()
    ts_vars: set[str] = set()
    first_split_line: int | None = None
    fit_calls: list[tuple[int, str]] = []

    def add(fam: str, node: ast.AST, extra: str = "") -> None:
        rep.flags.setdefault(fam, []).append(f"L{getattr(node, 'lineno', 0)}: {(_text(src, node) or extra)[:160]}")

    parents = {}
    for par in ast.walk(tree):
        for ch in ast.iter_child_nodes(par):
            parents[ch] = par

    def _guarded(n: ast.AST) -> bool:
        """True if the call sits in a branch or helper that is meant to fit on training data only."""
        cur = parents.get(n)
        while cur is not None:
            if isinstance(cur, ast.If) and re.search(r"(None|fit|train)", _text(src, cur.test), re.I):
                return True
            if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef)) and any(a.arg.lower() in {"fit", "scaler", "is_train", "train", "training"} for a in cur.args.args):
                return True
            cur = parents.get(cur)
        return False

    # first pass: scaler variables
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    if _name(node.value.func) in SCALERS:
                        scaler_vars.add(t.id)
                    if _name(node.value.func) == "TimeSeriesSplit":
                        ts_vars.add(t.id)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = _name(node.func)
            if fn == "train_test_split":
                first_split_line = min(first_split_line or node.lineno, node.lineno)
                sh = _kw(node, "shuffle")
                if sh is not None and _const(sh) is False:
                    rep.compliant["split_shuffle_false"] = rep.compliant.get("split_shuffle_false", 0) + 1
                else:
                    add("RANDOM_SPLIT", node)
            elif fn == "TimeSeriesSplit":
                rep.compliant["time_series_split"] = rep.compliant.get("time_series_split", 0) + 1
            elif fn in NONCHRONO_CV:
                add("NONCHRONO_CV", node)
            elif fn in CV_FUNCS:
                cv = _kw(node, "cv")
                txt = _text(src, cv).lower().replace("_", "") if cv is not None else ""
                is_ts = "timeseries" in txt or "tscv" in txt or (isinstance(cv, ast.Name) and cv.id in ts_vars)
                if cv is None:
                    if fn in {"cross_val_score", "cross_val_predict", "cross_validate", "GridSearchCV", "RandomizedSearchCV"}:
                        add("NONCHRONO_CV", node, extra=f"{fn} with default cv")
                elif is_ts:
                    rep.compliant["time_series_cv"] = rep.compliant.get("time_series_cv", 0) + 1
                elif isinstance(cv, ast.Constant) and isinstance(cv.value, int):
                    add("NONCHRONO_CV", node)
            elif fn in {"fit", "fit_transform"} and isinstance(node.func, ast.Attribute):
                obj = node.func.value
                oname = _name(obj)
                if (oname in scaler_vars or re.search(r"(scaler|scale|normaliz|imputer|pca)$", oname, re.I)) and node.args and not _guarded(node):
                    fit_calls.append((node.lineno, _text(src, node.args[0])))
                kw = _kw(node, "validation_data")
                if kw is not None and re.search(r"test", _text(src, kw), re.I) and not re.search(r"(val|valid)", _text(src, kw).replace("test", ""), re.I):
                    if re.search(r"(EarlyStopping|ModelCheckpoint|ReduceLROnPlateau)", src):
                        add("TEST_SELECTION", node)
            elif fn == "rolling":
                c = _kw(node, "center")
                if c is not None and _const(c) is True:
                    add("LOOKAHEAD", node)
            elif fn == "shift" and node.args:
                a0 = node.args[0]
                neg = (isinstance(a0, ast.UnaryOp) and isinstance(a0.op, ast.USub)) or (isinstance(a0, ast.Constant) and isinstance(a0.value, int) and a0.value < 0)
                if neg:
                    rep.compliant["_negative_shift_seen"] = rep.compliant.get("_negative_shift_seen", 0) + 1
        elif isinstance(node, ast.Subscript) and isinstance(getattr(node, "slice", None), (ast.Slice,)):
            seg = _text(src, node)
            if re.search(r"(train_size|split_idx|split_index|train_len|n_train|train_end|cutoff|split_point|split_at|int\(len)", seg) and isinstance(node.slice, ast.Slice):
                first_split_line = min(first_split_line or node.lineno, node.lineno)
                rep.compliant["slice_split"] = rep.compliant.get("slice_split", 0) + 1
    if "time_series_split" in rep.compliant or "time_series_cv" in rep.compliant:
        rep.flags.pop("NONCHRONO_CV", None)  # files that also use TimeSeriesSplit compare schemes deliberately
    # negative shift assigned to a feature column
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Subscript):
            val_txt = _text(src, node.value)
            if re.search(r"\.shift\(\s*-", val_txt):
                tgt = _text(src, node.targets[0].slice) if hasattr(node.targets[0], "slice") else ""
                if not TARGET_NAME.search(tgt) and not TARGET_NAME.search(val_txt.split(".shift")[0]):
                    add("LOOKAHEAD", node)
    # fit before split (source order; argument not naming a train set)
    if first_split_line is not None:
        for ln, arg in fit_calls:
            if ln < first_split_line and not re.search(r"train", arg, re.I):
                rep.flags.setdefault("FIT_BEFORE_SPLIT", []).append(f"L{ln}: fit on {arg[:80]} before the split at L{first_split_line}")
    return rep


def analyse_file(text: str, is_notebook: bool) -> FileReport:
    return analyse_source(code_from_source(text, is_notebook))

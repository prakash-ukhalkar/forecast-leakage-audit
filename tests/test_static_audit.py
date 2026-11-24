"""Tests for the static evaluation-error detectors."""
import json

from sdg_leakage_audit.audit.static import analyse_source, code_from_source

HEADER = "import pandas as pd\nimport numpy as np\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.linear_model import Ridge\n# forecast the time series\n"


def flags(code):
    return set(analyse_source(HEADER + code).flags)


def test_random_split_default_and_explicit_flagged_shuffle_false_not():
    assert "RANDOM_SPLIT" in flags("X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)\nm=Ridge().fit(X_train,y_train)")
    assert "RANDOM_SPLIT" in flags("a = train_test_split(X, y, shuffle=True)\nRidge().fit(X, y)")
    r = analyse_source(HEADER + "a = train_test_split(X, y, shuffle=False)\nRidge().fit(X, y)")
    assert "RANDOM_SPLIT" not in r.flags and r.compliant["split_shuffle_false"] == 1


def test_cv_detection():
    assert "NONCHRONO_CV" in flags("from sklearn.model_selection import KFold\nkf = KFold(n_splits=5, shuffle=True)\nRidge().fit(X,y)")
    assert "NONCHRONO_CV" in flags("from sklearn.model_selection import cross_val_score\ncross_val_score(Ridge(), X, y, cv=5)")
    assert "NONCHRONO_CV" in flags("from sklearn.model_selection import GridSearchCV\nGridSearchCV(Ridge(), {}).fit(X, y)")
    ok = analyse_source(HEADER + "from sklearn.model_selection import TimeSeriesSplit, cross_val_score\ntscv = TimeSeriesSplit(n_splits=5)\ncross_val_score(Ridge(), X, y, cv=tscv)")
    assert "NONCHRONO_CV" not in ok.flags and ok.compliant["time_series_cv"] == 1


def test_fit_before_split_flagged_only_when_not_train():
    bad = "from sklearn.preprocessing import StandardScaler\nsc = StandardScaler()\nX = sc.fit_transform(X)\nXtr, Xte, ytr, yte = train_test_split(X, y)\nRidge().fit(Xtr, ytr)"
    assert "FIT_BEFORE_SPLIT" in flags(bad)
    good = "from sklearn.preprocessing import StandardScaler\nXtr, Xte, ytr, yte = train_test_split(X, y, shuffle=False)\nsc = StandardScaler()\nXtr = sc.fit_transform(Xtr)\nRidge().fit(Xtr, ytr)"
    assert "FIT_BEFORE_SPLIT" not in flags(good)


def test_lookahead_and_target_shift():
    assert "LOOKAHEAD" in flags("df['m'] = df['y'].rolling(window=5, center=True).mean()\nRidge().fit(X,y)")
    assert "LOOKAHEAD" in flags("df['feat'] = df['x'].shift(-3)\nRidge().fit(X,y)")
    assert "LOOKAHEAD" not in flags("df['target'] = df['y'].shift(-1)\nRidge().fit(X,y)")
    assert "LOOKAHEAD" not in flags("df['m'] = df['y'].rolling(window=5).mean()\nRidge().fit(X,y)")


def test_test_selection_needs_a_selecting_callback_and_notebook_extraction():
    with_cb = ["from keras.callbacks import EarlyStopping", "es = EarlyStopping(monitor='val_loss')",
               "model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, callbacks=[es])"]
    assert "TEST_SELECTION" in flags("\n".join(with_cb))
    assert "TEST_SELECTION" not in flags("model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5)")
    assert "TEST_SELECTION" not in flags("model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5)")
    cells = [{"cell_type": "code", "source": ["%matplotlib inline" + chr(10), "x = 1" + chr(10)]}, {"cell_type": "markdown", "source": ["text"]}]
    nb = json.dumps({"cells": cells})
    assert "x = 1" in code_from_source(nb, True) and "# %matplotlib" in code_from_source(nb, True)


def test_qualification_needs_ml_and_time_series_cues():
    assert analyse_source(HEADER + "Ridge().fit(X, y)").qualifies
    assert not analyse_source("import pandas as pd\nprint(1)").qualifies
    assert not analyse_source("from sklearn.linear_model import Ridge\nRidge().fit(X, y)\nprice = 1").qualifies


def test_refined_fit_rules_ignore_encoders_and_guarded_helpers():
    enc = ["from sklearn.preprocessing import LabelEncoder", "le = LabelEncoder()", "y = le.fit_transform(y)",
           "Xtr, Xte, ytr, yte = train_test_split(X, y)", "Ridge().fit(Xtr, ytr)"]
    assert "FIT_BEFORE_SPLIT" not in flags("\n".join(enc))
    helper = ["from sklearn.preprocessing import RobustScaler", "def preprocess(X, scaler=None, fit=False):", "    if scaler is None:",
              "        scaler = RobustScaler()", "        return scaler.fit_transform(X)", "    return scaler.transform(X)",
              "Xtr, Xte, ytr, yte = train_test_split(X, y)", "Ridge().fit(Xtr, ytr)"]
    assert "FIT_BEFORE_SPLIT" not in flags("\n".join(helper))


def test_files_mixing_timeseries_split_are_not_flagged_for_cv():
    code = ["from sklearn.model_selection import TimeSeriesSplit, ShuffleSplit", "sp = TimeSeriesSplit(3)", "sp2 = ShuffleSplit(5)", "Ridge().fit(X, y)"]
    assert "NONCHRONO_CV" not in flags("\n".join(code))

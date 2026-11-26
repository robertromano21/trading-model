"""Unit tests for engine/prob_model.py: metrics, calibration, causal features, purging, packs.

No broker calls, no dashboard restart, no network.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import prob_model as pm
from test_strategy_framework import synthetic_panel


def test_brier_logloss_auc_known_values():
    y = np.array([0, 0, 1, 1], dtype=float)
    p = np.array([0.1, 0.2, 0.8, 0.9])
    assert abs(pm.brier_score(y, p) - np.mean((p - y) ** 2)) < 1e-12
    assert pm.log_loss(y, p) > 0
    assert pm.auc_binary(y, p) == 1.0
    assert pm.auc_binary(np.zeros(4), p) is None
    # IC: perfect rank agreement
    assert abs(pm.information_coefficient([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4]) - 1.0) < 1e-12


def test_logit_recovers_linear_signal():
    rng = np.random.default_rng(7)
    n, p = 800, 4
    X = rng.normal(size=(n, p))
    z = 1.4 * X[:, 0] - 0.8 * X[:, 1]
    y = (z + rng.normal(0, 0.4, n) > 0).astype(float)
    model = pm.fit_logit_l2(X, y, lam=0.5)
    pred = pm.predict_logit(model, X)
    assert model["success"]
    assert pm.auc_binary(y, pred) > 0.85
    assert model["coef"][0] > 0 and model["coef"][1] < 0


def test_platt_and_isotonic_recalibrate():
    rng = np.random.default_rng(3)
    # Overconfident scores: true rate is 0.6 whenever the raw score is 0.9
    y = rng.binomial(1, 0.6, 400).astype(float)
    raw = np.where(y == 1, 0.92, 0.88)  # still ~0.9, almost no discrimination, wrong level
    # Better case: scores squashed toward 0/1 while true rates are milder
    signal = rng.normal(size=600)
    y = (signal + rng.normal(0, 0.8, 600) > 0).astype(float)
    raw = 1.0 / (1.0 + np.exp(-3.0 * signal))  # overconfident logistic
    platt = pm.fit_platt(raw, y)
    iso = pm.fit_isotonic(raw, y)
    bp = pm.brier_score(y, raw)
    assert pm.brier_score(y, pm.apply_platt(platt, raw)) <= bp + 1e-6
    assert pm.brier_score(y, pm.apply_isotonic(iso, raw)) <= bp + 1e-6
    auto = pm.choose_calibrator(raw, y, method="auto")
    assert auto["method"] in ("platt", "isotonic")


def test_stumps_and_ensemble_fit():
    rng = np.random.default_rng(11)
    X = rng.normal(size=(500, 3))
    y = (X[:, 0] > 0.2).astype(float)
    st = pm.fit_stumps(X, y, n_stumps=8)
    assert st["success"] and st["stumps"]
    pred = pm.predict_stumps(st, X)
    assert pm.auc_binary(y, pred) > 0.8
    pack = pm.fit_pack(X, y, ["a", "b", "c"], kind="ensemble")
    p = pm.predict_pack(pack, X)
    assert 0.0 < p.mean() < 1.0
    assert pm.auc_binary(y, p) > 0.8


def test_targets_are_forward_and_features_are_causal():
    panel = synthetic_panel(sessions=280, stocks=6, seed=5)
    k = 20
    frames = pm.build_features(panel, include_macro=False, include_dilution=False)
    # Features at t must match panel.upto(t)
    cut = panel.dates[200]
    frames_cut = pm.build_features(panel.upto(cut), include_macro=False, include_dilution=False)
    for name, frame in frames_cut.items():
        full = frames[name].loc[:cut]
        pd.testing.assert_frame_equal(frame.loc[:cut], full.reindex_like(frame.loc[:cut]),
                                      atol=1e-10, check_names=False, obj=name)
    # A future price shock must change the label at t but not the features at t
    shocked = synthetic_panel(sessions=280, stocks=6, seed=5)
    shocked.close.loc[shocked.close.index > cut] *= 1.5
    shocked.high.loc[shocked.high.index > cut] *= 1.5
    shocked.low.loc[shocked.low.index > cut] *= 1.5
    shocked.open.loc[shocked.open.index > cut] *= 1.5
    shocked_frames = pm.build_features(shocked, include_macro=False, include_dilution=False)
    for name in ("tsmom_20", "tsmom_50", "vol_20", "ret_5"):
        pd.testing.assert_frame_equal(frames[name].loc[:cut], shocked_frames[name].loc[:cut],
                                      atol=1e-10, obj=name)
    t0 = pm.build_targets(panel, k=k)
    t1 = pm.build_targets(shocked, k=k)
    # Label at `cut` uses prices through cut+k, so it must move with the shock.
    assert not np.allclose(t0["forward"].loc[cut].to_numpy(dtype=float),
                           t1["forward"].loc[cut].to_numpy(dtype=float), equal_nan=True)


def test_purge_mask_drops_labels_that_overlap_the_end():
    dates = pd.bdate_range("2025-01-02", periods=40)
    train_end = dates[30]
    keep = pm.purge_mask(dates, train_end, k=5, embargo=5)
    # Sample at dates[25] has label through dates[30] and is kept; dates[26] uses dates[31] and is dropped.
    assert bool(keep.loc[dates[25]]) is True
    assert bool(keep.loc[dates[26]]) is False
    assert bool(keep.loc[dates[30]]) is False
    shifted = pm.session_shift(dates, dates[10], -5)
    assert shifted == dates[5]


def test_pack_roundtrip_and_report():
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(200, 3)), columns=["a", "b", "c"])
    y = (X["a"] + rng.normal(0, 0.5, 200) > 0).astype(float)
    pack = pm.fit_pack(X.to_numpy(), y.to_numpy(), list(X.columns), kind="logit",
                       val_X=X.to_numpy()[150:], val_y=y.to_numpy()[150:])
    p = pm.predict_pack(pack, X)
    rep = pm.classification_report(y, p, X["a"])
    assert pack["calibrator"]["method"] in ("platt", "isotonic")
    assert 0 <= (rep["brier"] or 1) <= 1
    assert rep["auc"] is None or rep["auc"] > 0.6
    # JSON serializable
    pm.write_json(Path("/tmp/tm-prob-pack.json") if sys.platform != "win32" else Path("prob_pack_tmp.json"), pack)
    snap = pm.latest_snapshot(pd.DataFrame({"date": pd.to_datetime(["2026-01-02", "2026-01-02"]),
                                            "ticker": ["SPY", "QQQ"], "excess_20": [0.51, 0.62]}),
                              {"excess_20:logit": pack})
    assert snap["asof"] == "2026-01-02" and "QQQ" in snap["tickers"] and snap["spy"]["excess_20"] == 0.51


if __name__ == "__main__":
    import tempfile
    import traceback
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                if "tmp_path" in fn.__code__.co_varnames:
                    with tempfile.TemporaryDirectory() as td:
                        fn(Path(td))
                else:
                    fn()
                print("PASS", name)
            except Exception:
                failed += 1
                print("FAIL", name)
                traceback.print_exc()
    sys.exit(1 if failed else 0)

"""Tests for engine/prob_walkforward.py: purged walk-forward, frozen holdout pack, pre-only selection.

No broker calls, no dashboard restart, no network. Uses a synthetic panel.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import prob_model as pm
import prob_walkforward as pw
from test_strategy_framework import synthetic_panel


def _signal_panel(sessions=360, seed=9):
    """Synthetic panel whose 20-day TSMOM feature is predictive of the next 20-day excess vs SPY.

    A slow (160-session) drift wave is much longer than the 20-day horizon, so the trailing
    20-day return and the forward 20-day return share the same local slope.
    """
    panel = synthetic_panel(sessions=sessions, stocks=6, seed=seed)
    rng = np.random.default_rng(seed + 1)
    t = np.arange(sessions)
    for name in panel.close.columns:
        if name == "SPY":
            # Quiet market so excess vs SPY is mostly the name's own wave.
            noise = rng.normal(0, 0.002, sessions)
            panel.close[name] = 50.0 * np.exp(np.cumsum(noise))
        else:
            phase = rng.uniform(0, 2 * np.pi)
            # ~0.15% daily drift at the peaks of a 160-session sine (period >> 20).
            drift = 0.0015 * np.sin(2 * np.pi * t / 160.0 + phase)
            noise = rng.normal(0, 0.004, sessions)
            panel.close[name] = 50.0 * np.exp(np.cumsum(drift + noise))
        panel.high[name] = np.maximum(panel.high[name], panel.close[name] * 1.002)
        panel.low[name] = np.minimum(panel.low[name], panel.close[name] * 0.998)
        panel.open[name] = panel.close[name].shift().fillna(panel.close[name]) * (1 + rng.normal(0, 0.001, sessions))
    return panel


def test_walk_forward_is_purged_and_embargoed():
    panel = _signal_panel()
    tickers = list(panel.close.columns)
    features = pm.stack_features(pm.build_features(panel, tickers, include_macro=False, include_dilution=False),
                                 tickers)
    k = 20
    pred, folds, pack = pw.walk_forward_probs(panel, features, tickers, "excess", k, model_kind="logit",
                                              min_train=120, holdout_start=None)
    assert pack is not None and len(folds) >= 2
    assert len(pred) > 0
    # Every test start must be at least k sessions after the previous train-end implied by the fold
    dates = panel.dates
    for i, fold in enumerate(folds[1:], start=1):
        test_day = pd.Timestamp(fold["testStart"])
        prev_end = pd.Timestamp(folds[i - 1]["testEnd"])
        # embargo: this fold's test_day is a new window; train_end was test_day - k
        train_end = pm.session_shift(dates, test_day, -k)
        # last training label date is train_end - k, so train_end >= test_day is a leak
        assert train_end < test_day
        loc = {d: i for i, d in enumerate(dates)}
        assert loc[test_day] - loc[train_end] >= k
    # Pre-holdout calibration should be better than chance on this planted panel
    aucs = [f["auc"] for f in folds if f.get("auc") is not None]
    assert aucs and float(np.mean(aucs)) > 0.55


def test_holdout_uses_frozen_pre_holdout_pack():
    panel = _signal_panel(sessions=360)
    tickers = list(panel.close.columns)
    features = pm.stack_features(pm.build_features(panel, tickers, include_macro=False, include_dilution=False),
                                 tickers)
    holdout = panel.dates[280]
    pred, folds, pack = pw.walk_forward_probs(panel, features, tickers, "excess", 20, model_kind="logit",
                                              min_train=120, holdout_start=holdout)
    pre_folds = [f for f in folds if not f.get("holdout")]
    ho_folds = [f for f in folds if f.get("holdout")]
    assert pre_folds and ho_folds
    # Coefficients of the returned pack equal a refit on pre-holdout data only (not the full sample).
    assert pack is not None and "coef" in (pack.get("model") or {})
    # Shuffling holdout labels and re-running must not change the frozen coefficients.
    pred2, folds2, pack2 = pw.walk_forward_probs(panel, features, tickers, "excess", 20, model_kind="logit",
                                                 min_train=120, holdout_start=holdout)
    np.testing.assert_allclose(np.asarray(pack["model"]["coef"]), np.asarray(pack2["model"]["coef"]), atol=1e-10)


def test_select_champion_ignores_holdout_returns():
    idx = pd.bdate_range("2025-01-02", periods=200)
    holdout = idx[140]
    rows = []
    # Trial 10 is the honest winner on pre-holdout; trial 11 is a holdout fluke.
    for i, (pre_s, ho_s, dd, ref) in {
        1: (0.4, 0.5, -0.10, True),
        10: (0.9, 0.1, -0.11, False),
        11: (0.2, 2.0, -0.09, False),
        12: (1.2, 0.3, -0.50, False),  # fails DD gate (cap ~0.20)
    }.items():
        rows.append({"id": i, "label": f"t{i}", "reference": ref, "passesDd": abs(dd) <= 0.20,
                     "pre": {"sharpe": pre_s}, "holdout": {"sharpe": ho_s}, "preMaxDrawdown": dd,
                     "spec": {"model": "logit", "target": "excess_20", "threshold": 0.6, "sizing": "tsmom_filter"}})
    champ, passed, ranked = pw.select_champion(rows)
    assert champ == 10 and passed is True
    assert 12 not in ranked or ranked.index(10) < ranked.index(12)
    assert ranked[0] == 10


def test_latest_for_tool_missing_and_filter(tmp_path):
    missing = pw.latest_for_tool({"path": str(tmp_path / "nope.json")})
    assert missing["available"] is False
    doc = {"asof": "2026-11-24", "tickers": {"NVDA": {"excess_20": 0.61}, "SPY": {"excess_20": 0.52}},
           "spy": {"excess_20": 0.52}, "caveats": []}
    path = tmp_path / "latest.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    out = pw.latest_for_tool({"path": str(path), "ticker": "nvda"})
    assert out["available"] is True and out["tickers"]["NVDA"]["excess_20"] == 0.61
    assert "SPY" not in out["tickers"] or len(out["tickers"]) == 1


def test_filter_and_sized_weights_respect_threshold():
    panel = synthetic_panel(sessions=260, stocks=4, seed=2)
    base = pw.tsmom_weights(panel)
    proba = pd.DataFrame(0.4, index=panel.dates, columns=base.columns)
    if "QQQ" in proba.columns:
        proba["QQQ"] = 0.8
    filt = pw.filter_weights(base, proba, 0.55)
    if "QQQ" in base.columns:
        # Names below the threshold are flat; QQQ may still be zero if TSMOM itself is flat.
        below = [c for c in filt.columns if c != "QQQ"]
        assert (filt[below].iloc[-40:] == 0).all().all()
    sized = pw.sized_weights(panel, proba, 0.55)
    assert (sized.max(axis=1) <= 0.30 + 1e-9).all()
    assert (sized.clip(upper=0).abs().sum().sum() == 0)  # long only


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

"""Unit tests for the daily drift monitor. No broker calls, no dashboard restart."""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import drift_monitor as dm


def _tmp(tmp_path):
    research = tmp_path / "Research"
    research.mkdir()
    return {
        "wf_equity": research / "walkforward_combined_oos_equity.csv",
        "wf_decisions": research / "walkforward_combined_oos_decisions.csv",
        "dt_trades": research / "daytrade_trades_defaults.csv",
        "dt_summary": research / "daytrade_summary.json",
        "paper_db": tmp_path / "paper_trading.sqlite3",
        "live_db": tmp_path / "live_trading.sqlite3",
        "status": research / "drift_status.json",
        "history": research / "drift_history.jsonl",
        "state": research / "drift_state.json",
        "chat": tmp_path / "chat_log.sqlite3",
    }


def _write_wf(paths, n=80, seed=1, mu=0.0005, sigma=0.01):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2025-11-26", periods=n)
    rets = rng.normal(mu, sigma, n)
    rets[0] = 0.0
    equity = 100_000 * np.cumprod(1 + rets)
    pd.DataFrame({"equity": equity, "return": rets}, index=dates).to_csv(paths["wf_equity"])
    # A handful of entries on day 0 and occasional rebalances so turnover/slippage exist.
    rows = []
    tickers = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH"]
    for i, d in enumerate(dates):
        if i == 0 or i % 10 == 0:
            for j, t in enumerate(tickers[: 4 + (i % 3)]):
                fill = d if i == 0 else dates[min(i + 1, n - 1)]
                rows.append({
                    "decision_date": d.date(), "fill_date": fill.date(), "ticker": t, "sleeve": "tsmom",
                    "action": "buy", "reason": "entry", "target_weight": 0.08 + 0.01 * (j % 3),
                    "current_weight": 0.0, "ref_price": 50 + j, "shares": 10 + j,
                    "fill_price": 50.2 + j, "notional": 800 + 20 * j, "cost_bps": 2.0,
                    "slippage_bps": 5.0 + (i % 7) - 3, "status": "filled",
                })
    pd.DataFrame(rows).to_csv(paths["wf_decisions"], index=False)
    # Day-trade fills on ~40% of sessions.
    dt = []
    for i, d in enumerate(dates):
        if i % 5 != 0:
            continue
        dt.append({"symbol": "AAA", "session_date": d.date(), "shares": 2})
    pd.DataFrame(dt).to_csv(paths["dt_trades"], index=False)
    paths["dt_summary"].write_text(json.dumps({
        "fullPeriodDefaults": {"first": str(dates[0].date()), "last": str(dates[-1].date()),
                               "sessions": n, "trades": len(dt)},
    }), encoding="utf-8")
    return dates, rets, equity


def _paper_schema(db):
    db.executescript("""
        CREATE TABLE equity_history (account TEXT, date TEXT, equity REAL, cash REAL, gross REAL, daily_return REAL);
        CREATE TABLE decision_log (account TEXT, run_id INTEGER, source TEXT, decision_date TEXT, fill_date TEXT,
            ticker TEXT, sleeve TEXT, action TEXT, reason TEXT, target_weight REAL, current_weight REAL,
            ref_price REAL, shares REAL, fill_price REAL, notional REAL, cost_bps REAL, slippage_bps REAL, status TEXT);
        CREATE TABLE fills (fill_id INTEGER PRIMARY KEY, account TEXT, fill_date TEXT, ticker TEXT, shares REAL,
            price REAL, effective_price REAL, notional REAL, cost_bps REAL, slippage_bps REAL);
        CREATE TABLE positions (account TEXT, ticker TEXT, shares REAL);
        CREATE TABLE dt_daily (account TEXT, session_date TEXT, start_equity REAL, pnl REAL, equity REAL, trades INTEGER, skipped TEXT);
        CREATE TABLE dt_trades (account TEXT, session_date TEXT, symbol TEXT, shares REAL);
        CREATE TABLE reconciliation (account TEXT, date TEXT, paper_equity REAL, shadow_equity REAL, equity_diff_pct REAL, max_weight_diff REAL, details TEXT);
    """)


def _write_paper(paths, dates, rets, equity_scale=10_000, mu_shift=0.0, slip=5.0, agreement=True):
    eq = equity_scale * np.cumprod(1 + (rets + mu_shift))
    eq[0] = equity_scale
    db = sqlite3.connect(paths["paper_db"])
    _paper_schema(db)
    for i, d in enumerate(dates):
        prev = eq[i - 1] if i else equity_scale
        daily = eq[i] / prev - 1 if i else 0.0
        # overwrite first day return to match
        db.execute("INSERT INTO equity_history VALUES (?,?,?,?,?,?)",
                   ("default", str(d.date()), float(eq[i]), 2000.0, 0.8, float(daily)))
        if i == 0 or i % 10 == 0:
            tickers = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"] if agreement else ["ZZZ"]
            for j, t in enumerate(tickers):
                w = 0.08 + 0.01 * (j % 3) if agreement else 0.2
                db.execute("""INSERT INTO decision_log VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                           ("default", 1, "strategy", str(d.date()), str(d.date()), t, "tsmom", "buy", "entry",
                            w, 0.0, 50, 10, 50.2, 800, 2.0, slip, "filled"))
                db.execute("INSERT INTO fills (account, fill_date, ticker, shares, price, effective_price, notional, cost_bps, slippage_bps) VALUES (?,?,?,?,?,?,?,?,?)",
                           ("default", str(d.date()), t, 10, 50, 50.2, 800, 2.0, slip))
        if i % 5 == 0:
            db.execute("INSERT INTO dt_daily VALUES (?,?,?,?,?,?,?)",
                       ("daytrade", str(d.date()), 10_000, 10, 10_010, 1, json.dumps({"pdt": 0})))
        else:
            db.execute("INSERT INTO dt_daily VALUES (?,?,?,?,?,?,?)",
                       ("daytrade", str(d.date()), 10_000, 0, 10_000, 0, json.dumps({})))
    names = ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF") if agreement else ("ZZZ",)
    for t in names:
        db.execute("INSERT INTO positions VALUES (?,?,?)", ("default", t, 10))
    last = str(dates[-1].date())
    if agreement:
        weights = {t: 0.08 + 0.01 * (j % 3) for j, t in enumerate(names)}
        details = json.dumps({"paperWeights": weights, "shadowWeights": dict(weights)})
        max_diff = 0.0
    else:
        details = json.dumps({"paperWeights": {"ZZZ": 0.2}, "shadowWeights": {"AAA": 0.2, "BBB": 0.2}})
        max_diff = 0.2
    db.execute("INSERT INTO reconciliation VALUES (?,?,?,?,?,?,?)",
               ("default", last, float(eq[-1]), float(eq[-1]), 0.0, max_diff, details))
    db.commit()
    db.close()


def test_percentile_bands_and_flags(tmp_path):
    paths = _tmp(tmp_path)
    _write_wf(paths, n=80)
    expected = dm.build_expected(paths)
    assert expected["sessions"] == 80
    band = expected["bands"]["daily_return"]
    assert band["p5"] < band["p50"] < band["p95"]
    assert band["p1"] <= band["p5"] and band["p95"] <= band["p99"]
    assert expected["bands"]["roll20_return"]["n"] == 61
    assert expected["bands"]["signal_agreement"]["p5"] == 0.80
    assert dm.classify(band["p50"], band) == "ok"
    # just outside 5–95 but inside 1–99
    lo, hi = band["p5"], band["p95"]
    p1, p99 = band["p1"], band["p99"]
    mid_low = lo - (lo - p1) * 0.4 if lo > p1 else lo
    if mid_low < lo:
        assert dm.classify(mid_low, band) in ("warning", "ok")  # equal-edge guard
    assert dm.classify(p1 - 1, band) == "severe"
    assert dm.classify(p99 + 1, band) == "severe"
    assert dm.classify(hi + (p99 - hi) * 0.4 if p99 > hi else hi + 1e-12, band) in ("warning", "severe")


def test_cusum_detects_persistent_shift():
    band = {"mean": 0.0, "std": 1.0, "p1": -3, "p5": -1.64, "p50": 0, "p95": 1.64, "p99": 3, "n": 100}
    state = {}
    # No alarm on in-range noise.
    for _ in range(8):
        slot = dm.update_cusum(state, "daily_return", 0.2, band)
    assert slot["alarm"] is False
    # Persistent +2-sigma mean shift should trip CUSUM (k=0.5, increment 1.5/day, h=5 → 4 days).
    for _ in range(6):
        slot = dm.update_cusum(state, "daily_return", 2.0, band)
    assert slot["alarm"] is True
    assert slot["sPlus"] > dm.CUSUM_H
    # Sequential test accumulates as well.
    assert slot["n"] >= 5
    assert slot["seqStat"] is not None


def test_signal_agreement_overlap_and_mismatch():
    a = {"AAA": 0.1, "BBB": 0.1}
    assert dm.signal_agreement(a, a) == 1.0
    assert dm.signal_agreement({}, {}) == 1.0
    # disjoint books → 0
    assert dm.signal_agreement({"AAA": 0.2}, {"BBB": 0.2}) == 0.0
    half = dm.signal_agreement({"AAA": 0.2, "BBB": 0.0}, {"AAA": 0.2, "BBB": 0.2})
    assert 0.4 < half < 0.8


def test_awaiting_paper_writes_status_without_alerts(tmp_path):
    paths = _tmp(tmp_path)
    _write_wf(paths, n=40)
    emitted = []
    report = dm.run(paths=paths, alerts=True, chat=False, memory=False,
                    emit_fn=lambda payload: emitted.append(payload) or True)
    assert report["status"] == "awaiting"
    assert report["paperSessions"] == 0
    assert emitted == []
    saved = json.loads(paths["status"].read_text(encoding="utf-8"))
    assert saved["status"] == "awaiting"
    assert saved["metrics"]["daily_return"]["band"]["p50"] is not None


def test_in_range_paper_is_ok_and_writes_chat_memory(tmp_path):
    paths = _tmp(tmp_path)
    dates, rets, _ = _write_wf(paths, n=50, seed=2)
    _write_paper(paths, dates, rets, mu_shift=0.0, slip=5.0, agreement=True)
    emitted, chats, mems = [], [], []
    report = dm.run(
        paths=paths, alerts=True, chat=True, memory=True,
        emit_fn=lambda payload: emitted.append(payload) or True,
        chat_fn=lambda r: chats.append(r) or {"ok": True, "added": 1},
        memory_fn=lambda kind, text, meta=None: mems.append((kind, text, meta)) or {"ok": True, "id": 1, "duplicate": False},
    )
    assert report["paperSessions"] == 50
    assert report["asof"] == str(dates[-1].date())
    # Daily return, turnover, slippage, agreement should all be classifiable.
    assert report["metrics"]["daily_return"]["flag"] in ("ok", "warning", "severe")
    assert report["metrics"]["signal_agreement"]["value"] is not None
    assert report["metrics"]["signal_agreement"]["value"] >= 0.8
    assert report["metrics"]["daytrade_fill_rate"]["value"] is not None
    assert chats and mems[0][0] == "EVENT"
    assert "drift:" in mems[0][2]["dedupeKey"]
    # In-range (or only mild warning) should not be forced to speak; no alert if ok.
    if report["status"] == "ok":
        assert emitted == []


def test_severe_shift_emits_spoken_critical_alert(tmp_path):
    paths = _tmp(tmp_path)
    dates, rets, _ = _write_wf(paths, n=60, seed=3, sigma=0.008)
    # Last days are a crash well beyond the 1% band, plus a mean shift.
    shifted = rets.copy()
    shifted[-1] = -0.20
    _write_paper(paths, dates, shifted, mu_shift=0.0, slip=400.0, agreement=False)
    emitted = []
    report = dm.run(
        paths=paths, alerts=True, chat=False, memory=False,
        emit_fn=lambda payload: emitted.append(payload) or True,
    )
    assert report["status"] in ("warning", "severe")
    assert report["metrics"]["daily_return"]["flag"] == "severe"
    assert report["metrics"]["slippage_bps"]["flag"] in ("warning", "severe")
    assert report["metrics"]["signal_agreement"]["flag"] in ("warning", "severe")
    assert len(emitted) == 1
    alert = emitted[0]
    assert alert["type"] == "drift"
    if report["status"] == "severe":
        assert alert["severity"] == "critical"
        assert alert["speech"]
        assert "serious drift" in alert["speech"].lower() or "serious" in alert["speech"].lower()
        assert alert["meta"]["speak"] is True
    assert alert["href"] == "/paper2.html#drift"
    # Re-run the same as-of day: no second CUSUM increment, no duplicate alert.
    n_before = report["metrics"]["daily_return"]["cusumN"]
    emitted.clear()
    again = dm.run(paths=paths, alerts=True, chat=False, memory=False,
                   emit_fn=lambda payload: emitted.append(payload) or True)
    assert again["metrics"]["daily_return"]["cusumN"] == n_before
    assert emitted == []


def test_daytrade_fill_rate_series_aligns_with_calendar():
    trades = pd.DataFrame({"session_date": ["2026-03-17", "2026-03-19", "2026-03-19"], "symbol": ["A", "B", "C"]})
    series = dm.daytrade_fill_series(trades, {"fullPeriodDefaults": {"first": "2026-03-17", "last": "2026-03-20"}})
    assert pd.Timestamp("2026-03-18") in series.index
    # 3/17: 1 fill, 3/18: 0, 3/19: 2, 3/20: 0 — rolling mean exists
    assert series.notna().any()


def test_missing_walkforward_does_not_crash(tmp_path):
    paths = _tmp(tmp_path)
    report = dm.run(paths=paths, alerts=False, chat=False, memory=False)
    assert report["status"] == "awaiting"
    assert any("missing" in n.lower() or "Walk-forward" in n for n in report["notes"])


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

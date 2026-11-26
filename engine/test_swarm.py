"""Unit tests for the simulated-account swarm (engine/swarm.py). No broker calls, no dashboard restart."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import swarm as sw


def test_latin_margins_are_balanced():
    rows = sw.latin_levels(sw.SWING_SPACE, 48, 1)
    assert len(rows) == 48
    for key, levels in sw.SWING_SPACE.items():
        counts = [sum(1 for r in rows if r[key] == lvl) for lvl in levels]
        assert max(counts) - min(counts) <= 1, (key, counts)


def test_design_has_baselines_and_is_deterministic():
    a = sw.make_design(50, 7, tax_available=True, orb_available=True)
    b = sw.make_design(50, 7, tax_available=True, orb_available=True)
    assert a["accounts"] == b["accounts"]
    assert len(a["accounts"]) == 50
    assert [x["id"] for x in a["accounts"]] == list(range(50))
    assert a["accounts"][0]["swing"]["family"] == "production"
    assert a["accounts"][1]["swing"]["family"] == "tsmom_baseline"
    for acct in a["accounts"][2:]:
        s = acct["swing"]
        if not s["overlay"]:
            assert s["mr_entry_z"] is None and s["regime"] != "fixed"
        if s["regime"] != "hmm":
            assert s["regime_threshold"] is None
    no_tax = sw.make_design(50, 7, tax_available=False, orb_available=False)
    assert all(not x["swing"].get("tax_aware") for x in no_tax["accounts"])
    assert all(x["orb"] is None for x in no_tax["accounts"])


def test_design_rejects_tiny_swarm():
    try:
        sw.make_design(5)
    except SystemExit:
        return
    raise AssertionError("n=5 should be refused")


def _frame(T=300, N=20, seed=0, edge=None, size=0.003):
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0002, 0.01, size=(T, N))
    if edge is not None:
        x[:, edge] += size
    return pd.DataFrame(x, index=pd.bdate_range("2025-01-01", periods=T), columns=list(range(N)))


def test_selection_never_sees_holdout_rows():
    frame = _frame()
    cut = frame.index[220]
    pre = frame.loc[frame.index < cut]
    ranked, _ = sw.select(pre, set(frame.columns))
    poisoned = frame.copy()
    poisoned.loc[poisoned.index >= cut, ranked[-1]] += 0.5        # make the worst pre-holdout variant a holdout star
    ranked2, _ = sw.select(poisoned.loc[poisoned.index < cut], set(frame.columns))
    assert ranked == ranked2


def test_pbo_noise_vs_edge():
    noise = sw.pbo_cscv(_frame(T=360, N=30, seed=3).to_numpy(), 12)
    assert 0.2 <= noise["pbo"] <= 0.85, noise["pbo"]
    edged = sw.pbo_cscv(_frame(T=360, N=30, seed=3, edge=4).to_numpy(), 12)
    assert edged["pbo"] < 0.1, edged["pbo"]
    assert edged["splits"] == 924


def test_spa_and_reality_check():
    frame = _frame(T=250, N=15, seed=5)
    bench = frame[0].to_numpy()
    d = frame.drop(columns=[0]).to_numpy() - bench[:, None]
    res, p, t = sw.spa_test(d, B=500, seed=1)
    assert res["spaP"] > 0.05 and res["realityCheckP"] > 0.05
    strong = _frame(T=250, N=15, seed=5, edge=3, size=0.006)
    d2 = strong.drop(columns=[0]).to_numpy() - strong[0].to_numpy()[:, None]
    res2, p2, _ = sw.spa_test(d2, B=500, seed=1)
    assert res2["spaP"] < 0.05
    assert np.argmin(p2) == 2                  # column 3 after dropping column 0
    flat = np.zeros((100, 3))
    _, p3, _ = sw.spa_test(flat, B=200)
    assert np.all(p3 == 1.0)


def test_holm_and_bh():
    p = np.array([0.01, 0.04, 0.03, 0.20])
    h = sw.holm(p)
    assert np.allclose(h, [0.04, 0.09, 0.09, 0.20])
    b = sw.bh(p)
    assert np.allclose(b, [0.04, 0.0533333333, 0.0533333333, 0.20])
    assert np.all(h >= p) and np.all(b >= p)


def test_deflated_sharpe_falls_with_trials():
    x = np.random.default_rng(2).normal(0.001, 0.01, 300)
    trials_sr = np.random.default_rng(3).normal(0.5, 0.6, 50)
    few = sw.deflated_sharpe(x, 2, trials_sr)["deflatedSharpeProbability"]
    many = sw.deflated_sharpe(x, 50, trials_sr)["deflatedSharpeProbability"]
    assert many < few


def test_effective_trials():
    base = np.random.default_rng(1).normal(size=(200, 1))
    clones = np.hstack([base + 1e-6 * np.random.default_rng(i).normal(size=(200, 1)) for i in range(10)])
    assert sw.effective_trials(clones) < 1.5
    independent = np.random.default_rng(1).normal(size=(500, 10))
    assert sw.effective_trials(independent) > 7


def test_pdt_stats():
    dates = pd.bdate_range("2026-01-05", periods=10)
    trades = pd.DataFrame({"holding_sessions": [0, 0, 0, 0, 3], "exit_date": [dates[1], dates[2], dates[3], dates[4], dates[8]]})
    s = sw.pdt_stats(trades, dates)
    assert s["sameDayRoundTrips"] == 4 and s["maxIn5"] == 4 and s["breach"]
    ok = sw.pdt_stats(trades.iloc[1:], dates)
    assert not ok["breach"]


def _panel(T=30):
    from market_data import Panel
    idx = pd.bdate_range("2026-01-05", periods=T)
    close = pd.DataFrame({"AAA": np.linspace(37.3, 41.0, T), "BBB": np.linspace(123.4, 130.0, T)}, index=idx)
    vol = pd.DataFrame(1e6, index=idx, columns=close.columns)
    zero = pd.DataFrame(0.0, index=idx, columns=close.columns)
    return Panel(close, close * 1.01, close * 0.99, close, vol, zero, pd.Series({"AAA": "stock", "BBB": "stock"}))


def test_whole_shares_and_default_path():
    from costs import CostModel
    from portfolio_sim import simulate
    panel = _panel()
    w = pd.DataFrame({"AAA": 0.3, "BBB": 0.3}, index=panel.dates)
    frac = simulate(panel, w, None, CostModel.for_broker("schwab"), 10_000.0)
    whole = simulate(panel, w, None, CostModel.for_broker("schwab"), 10_000.0, whole_shares=True, min_order_value=100)
    assert any(abs(v - round(v)) > 1e-6 for v in frac.final_positions.values())
    assert all(abs(v - round(v)) < 1e-9 for v in whole.final_positions.values())
    assert whole.final_cash > frac.final_cash          # flooring leaves a little more cash


def test_forward_ledger_is_append_only(tmp_path):
    old = sw.FORWARD_DIR
    sw.FORWARD_DIR = Path(tmp_path)
    try:
        idx = pd.bdate_range("2026-12-01", periods=3)
        eq = pd.DataFrame({0: [10000.0, 10010.0, 10020.0], 1: [10000.0, 9990.0, 9980.0]}, index=idx)
        assert sw._append_ledger(eq.iloc[:2], "swing") == 0
        assert sw._append_ledger(eq, "swing") == 0
        lines = (Path(tmp_path) / "ledger.jsonl").read_text().splitlines()
        assert len(lines) == 6
        revised = eq.copy()
        revised.iloc[0, 0] = 10500.0
        assert sw._append_ledger(revised, "swing") == 1
        assert len((Path(tmp_path) / "ledger.jsonl").read_text().splitlines()) == 6
    finally:
        sw.FORWARD_DIR = old


def test_prompt_summary_and_verdict_shape():
    status = {"accounts": 50, "designHash": "x", "design": {"holdoutStart": "2026-07-09", "dataEnd": "2026-11-24"},
              "swing": {"champion": 3, "deflatedSharpePre": {"trials": 50, "deflatedSharpeProbability": 0.4},
                        "effectiveTrials": 6.0, "pboPre": {"pbo": 0.6},
                        "holdoutChampion": {"sharpe": 0.3},
                        "tests": {"tsmom_baseline:holdout": {"spaP": 0.7, "realityCheckP": 0.8, "significantHolm": [],
                                                             "significantBH": [], "championP": 0.4}}},
              "benchmarks": {}, "leaderboard": [{"id": 3, "swing": {"family": "tsmom_baseline"}, "pre": {}, "holdout": {}}]}
    v = sw._verdict(status["swing"], None, {})
    assert v["anyBeatsBaselineAfterCorrection"] is False
    status["verdict"] = v
    text = sw.prompt_summary(status)
    assert "[SWARM]" in text and "never automatic" in text


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

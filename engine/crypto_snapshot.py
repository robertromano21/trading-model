"""Lightweight crypto market snapshot for the dashboard and the AI analyst (read-only; never any orders).

Writes Massive/Research/crypto/crypto_snapshot.json and copies it under "market" into
paper/crypto_paper_status.json (served at /api/crypto/status). Safe to run every few minutes:
  * fresh spot from Coinbase public candles (hourly for 24h change, daily store for 7d/30d, vol, 50/200-day trend);
  * FRED CBBTCUSD / CBETHUSD as a fallback when Coinbase is unreachable or stale (FRED_API_KEY from .env.local if
    set, else the keyless graph CSV; the key is never printed or logged);
  * Schwab market data (quotes only) for CME futures, the $BTC/$ETH indexes and crypto ETPs, when authorized;
  * dollar / real-yield / liquidity context (publication-lagged), the macro nowcast read, the 20 paper accounts'
    standings and the backtest verdict;
  * a local symbol universe (coins, ETPs, futures) for pickers - pure lookup, no AI involved.
A PID lock prevents overlapping runs; a stale lock (dead PID or > 10 minutes old) is replaced.

  python crypto_snapshot.py [--no-schwab] [--max-age-min N]   # skip if the snapshot is younger than N minutes
"""

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from urllib.parse import urlencode

import numpy as np
import pandas as pd

import crypto_data as cd

SNAPSHOT = cd.CRYPTO_DIR / "crypto_snapshot.json"
LOCK = cd.CRYPTO_DIR / "crypto_snapshot.lock"
PAPER_STATUS = cd.CRYPTO_DIR / "paper" / "crypto_paper_status.json"
REPORT = cd.CRYPTO_DIR / "crypto_backtest_report.json"
NOWCAST = cd.DATA_DIR / "Research" / "macro" / "nowcast_latest.json"
SNAP_COINS = ["BTC", "ETH", "SOL", "AVAX", "LINK", "XRP", "DOGE", "LTC"]
SCHWAB_SYMBOLS = ["/BTC", "/MBT", "/ETH", "/MET", "$BTC", "$ETH", "IBIT", "FBTC", "GBTC", "BITB", "ETHA", "FETH",
                  "ETHE", "BITO", "BSOL", "GSOL", "XRPR", "SBIT", "BITI"]   # SBIT -2x / BITI -1x inverse BTC ETFs
SHORT_REPORT = cd.CRYPTO_DIR / "short" / "crypto_short_report.json"
SHORT_PAPER = cd.CRYPTO_DIR / "short" / "short_paper_status.json"
FRED_FALLBACK = {"BTC": "CBBTCUSD", "ETH": "CBETHUSD", "LTC": "CBLTCUSD"}


def log(msg):
    print(f"[crypto_snapshot] {msg}", file=sys.stderr, flush=True)


def _pid_alive(pid):
    if not pid:
        return False
    try:
        if os.name == "nt":
            import ctypes
            h = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))   # PROCESS_QUERY_LIMITED_INFORMATION
            if not h:
                return False
            code = ctypes.c_ulong()
            ctypes.windll.kernel32.GetExitCodeProcess(h, ctypes.byref(code))
            ctypes.windll.kernel32.CloseHandle(h)
            return code.value == 259                                         # STILL_ACTIVE
        os.kill(int(pid), 0)
        return True
    except Exception:   # noqa: BLE001
        return False


def acquire_lock():
    cd.CRYPTO_DIR.mkdir(parents=True, exist_ok=True)
    if LOCK.exists():
        try:
            info = json.loads(LOCK.read_text())
        except Exception:   # noqa: BLE001
            info = {}
        fresh = time.time() - float(info.get("at", 0)) < 600
        if fresh and _pid_alive(info.get("pid")):
            return False
    LOCK.write_text(json.dumps({"pid": os.getpid(), "at": time.time()}))
    return True


def release_lock():
    try:
        info = json.loads(LOCK.read_text())
        if info.get("pid") == os.getpid():
            LOCK.unlink()
    except Exception:   # noqa: BLE001
        pass


def _f(x, d=4):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return round(x, d) if math.isfinite(x) else None


# ---------------------------------------------------------------- prices
def hourly_last(product):
    """Last ~30 hourly candles from Coinbase -> (last price, price 24h ago, last candle time UTC)."""
    url = cd.COINBASE.format(product=product) + "?granularity=3600"
    rows = cd._get_json(url, timeout=15)
    if not rows:
        return None
    rows = sorted(rows, key=lambda r: r[0])
    last = rows[-1]
    t_last = last[0]
    ago = [r for r in rows if r[0] <= t_last - 24 * 3600]
    return {"price": float(last[4]), "price24hAgo": float(ago[-1][4]) if ago else None,
            "asOfUtc": datetime.fromtimestamp(t_last + 3600, timezone.utc).strftime("%Y-%m-%d %H:%M")}


def fred_fallback(series):
    """Latest CB*USD from FRED: API with FRED_API_KEY when set (never printed), else the keyless CSV."""
    from local_env import env_value
    key = env_value("FRED_API_KEY")
    try:
        if key:
            url = "https://api.stlouisfed.org/fred/series/observations?" + urlencode(
                {"series_id": series, "api_key": key, "file_type": "json", "sort_order": "desc", "limit": 40})
            obs = [o for o in cd._get_json(url, timeout=20).get("observations", []) if o.get("value") not in (".", "")]
            s = pd.Series({pd.Timestamp(o["date"]): float(o["value"]) for o in obs}).sort_index()
        else:
            from macro_data import download_fred
            s = download_fred([series], start=(pd.Timestamp.today() - pd.Timedelta(days=60)).date())[series].dropna()
        return s
    except Exception as error:   # noqa: BLE001 - message never contains the key (URL is not included)
        log(f"FRED fallback {series} failed: {type(error).__name__}")
        return None


def coin_stats(sym, daily, hourly):
    s = daily.dropna()
    out = {"coin": sym, "schwab": cd.COINS[sym]["schwab"], "source": "coinbase"}
    if hourly:
        price = hourly["price"]
        out.update({"price": _f(price, 6), "asOfUtc": hourly["asOfUtc"],
                    "chg24h": _f(price / hourly["price24hAgo"] - 1) if hourly.get("price24hAgo") else None})
    elif len(s):
        price = float(s.iloc[-1])
        out.update({"price": _f(price, 6), "asOfUtc": str(s.index[-1].date()) + " 24:00 (daily close)", "chg24h": None})
    else:
        return out
    if len(s) > 31:
        def back(days):
            ref = s.loc[:s.index[-1] - pd.Timedelta(days=days)]
            return float(ref.iloc[-1]) if len(ref) else None
        for k, d in (("chg7d", 7), ("chg30d", 30), ("chg90d", 90), ("chg365d", 365)):
            b = back(d)
            out[k] = _f(price / b - 1) if b else None
        r = np.log(s).diff().dropna()
        out["vol30d"] = _f(r.iloc[-30:].std() * math.sqrt(365))
        out["vol90d"] = _f(r.iloc[-90:].std() * math.sqrt(365))
        for n in (50, 200):
            if len(s) >= n:
                sma = float(s.iloc[-n:].mean())
                out[f"sma{n}"] = _f(sma, 6)
                out[f"vsSma{n}"] = _f(price / sma - 1)
        a50, a200 = out.get("vsSma50"), out.get("vsSma200")
        out["trend"] = ("uptrend" if a50 is not None and a200 is not None and a50 > 0 and a200 > 0 else
                        "downtrend" if a50 is not None and a200 is not None and a50 < 0 and a200 < 0 else "mixed")
        out["ret100dPositive"] = bool(len(s) > 100 and price > float(s.iloc[-101]))
        hi = float(s.iloc[-365:].max())
        out["drawdownFrom1yHigh"] = _f(price / hi - 1)
    return out


def macro_context(calendar):
    d = cd.load_drivers(calendar)
    last = d.iloc[-1]

    def chg(col, days, pct=True):
        if col not in d or d[col].dropna().empty:
            return None
        s = d[col]
        a, b = s.iloc[-1], s.iloc[-1 - days] if len(s) > days else np.nan
        return _f((a / b - 1) if pct else (a - b))

    ctx = {"uup": _f(last.get("UUP"), 3), "uup30d": chg("UUP", 30), "uup90d": chg("UUP", 90),
           "broadDollar30d": chg("DTWEXBGS", 30), "realYield10y": _f(last.get("DFII10"), 3),
           "realYield30dChange": chg("DFII10", 30, pct=False), "netLiquidityBn": _f(last.get("NETLIQ"), 0),
           "netLiquidity91d": chg("NETLIQ", 91), "vix": _f(last.get("VIXCLS"), 2),
           "note": "Inputs are lagged to their publication dates (H.10 weekly, H.4.1 Thursday, UUP next day)."}
    try:
        import crypto_research as cr
        idx = d.index
        ctx["overlays"] = {
            "dollarTrend90dUp": bool((ctx["uup90d"] or 0) > 0),
            "compositeExposure": _f(cr.overlay_multiplier({"type": "composite", "floor": 0.25}, d, idx).iloc[-1], 3),
            "dollarShockActive": bool(cr.overlay_multiplier({"type": "dollar_shock", "z": 2.0, "days": 10, "cut": 0.0},
                                                            d, idx).iloc[-1] < 1),
        }
    except Exception as error:   # noqa: BLE001
        ctx["overlays"] = {"error": f"{type(error).__name__}: {error}"}
    return ctx


def nowcast_read():
    try:
        L = json.loads(NOWCAST.read_text(encoding="utf-8"))
    except Exception:   # noqa: BLE001
        return None
    items = {}
    for n in L.get("nowcasts", []):
        if n.get("target") in ("CPI_MOM", "CORE_CPI_MOM", "UNRATE", "PAYROLLS", "GDP_SAAR") and n.get("nowcast") is not None:
            items[n["target"]] = {"nowcast": n.get("nowcast"), "lastPrint": n.get("lastPrint"),
                                  "next": n.get("nextRelease") or n.get("estimatedRelease")}
    return {"asof": L.get("asof"), "regime": (L.get("regime") or {}).get("label"), "items": items}


def schwab_quotes(enabled=True):
    if not enabled:
        return {"skipped": True}
    try:
        from schwab_data import SchwabMarketData
        q = SchwabMarketData().quotes(SCHWAB_SYMBOLS)
    except Exception as error:   # noqa: BLE001 - auth may be expired; never fatal
        return {"error": f"{type(error).__name__}: {str(error)[:160]}"}
    out = {}
    for k, v in (q or {}).items():
        if k == "errors":
            continue
        qq, ref = v.get("quote") or {}, v.get("reference") or {}
        last, close = qq.get("lastPrice"), qq.get("closePrice")
        out[k] = {"description": ref.get("description"), "type": v.get("assetMainType"), "last": _f(last, 4),
                  "close": _f(close, 4), "chgPct": _f(last / close - 1) if last and close else None,
                  "quoteTime": datetime.fromtimestamp(qq["quoteTime"] / 1000).strftime("%Y-%m-%d %H:%M")
                  if qq.get("quoteTime") else None}
    return out


def universe():
    return {"coins": [{"symbol": c, "pair": f"{c}-USD", "schwab": v["schwab"], "etps": v["etp"]} for c, v in cd.COINS.items()],
            "futures": ["/BTC", "/MBT", "/ETH", "/MET"], "indexes": ["$BTC", "$ETH"],
            "etps": sorted({e for v in cd.COINS.values() for e in v["etp"]} | {"BITO"}),
            "note": "Local lookup list for pickers; Schwab Crypto trades BTC and ETH (SOL/AVAX/LINK announced)."}


def paper_summary():
    try:
        st = json.loads(PAPER_STATUS.read_text(encoding="utf-8"))
    except Exception:   # noqa: BLE001
        return None
    accts = st.get("accounts") or []
    rows = [{"id": a["id"], "label": a["label"], "equity": _f(a.get("equity"), 2), "totalReturn": _f(a.get("totalReturn")),
             "targetWeights": a.get("targetWeights"), "dollarAware": a.get("dollarAware")} for a in accts]
    return {"forwardStart": st.get("forwardStart"), "dataEnd": st.get("dataEnd"), "days": st.get("days"),
            "leader": st.get("leader"), "invested": sum(1 for a in accts if a.get("targetWeights")),
            "accounts": len(accts), "btcBaselineReturn": st.get("btcBaselineReturn"), "rows": rows}


def verdict():
    try:
        r = json.loads(REPORT.read_text(encoding="utf-8"))
    except Exception:   # noqa: BLE001
        return None
    c, b = r.get("champion") or {}, (r.get("btcHold") or {}).get("pre") or {}
    dp, nb = (r.get("dollarPairs") or {}).get("spa") or {}, (r.get("nowcastPairs") or {}).get("spa") or {}
    hl = r.get("holdoutLook") or (r.get("previousHoldoutLook") or {}).get("look") or {}
    return {"designHash": r.get("designHash"), "variants": r.get("variants"),
            "btcHoldSharpe": _f(b.get("sharpe"), 2), "btcHoldMaxDD": _f(b.get("maxDrawdown"), 3),
            "championId": c.get("id"), "championLabel": c.get("label"), "championSharpe": _f((c.get("pre") or {}).get("sharpe"), 2),
            "championMaxDD": _f((c.get("pre") or {}).get("maxDrawdown"), 3),
            "pbo": _f((r.get("pbo") or {}).get("pbo"), 3), "spaVsBtcP": _f((r.get("spaVsBtc") or {}).get("spaP"), 3),
            "dollarOverlaySpaP": _f(dp.get("spaP"), 3), "nowcastOverlaySpaP": _f(nb.get("spaP"), 3),
            "holdoutChampionReturn": _f((hl.get("champion") or {}).get("totalReturn")),
            "holdoutBtcReturn": _f((hl.get("btcHold") or {}).get("totalReturn")),
            "summary": ("Trend rules beat BTC buy-and-hold on Sharpe and drawdown before the holdout but are not "
                        "statistically significant after multiple-testing (SPA vs BTC p~0.5, PBO~0.5). Dollar, "
                        "liquidity and nowcast filters reduce drawdowns but do not improve returns after Schwab's "
                        "0.75% fees (SPA p>0.9). The rising-dollar 90-day trend does predict weaker 30-day BTC "
                        "returns (BH p 0.03). Research only; paper accounts only.")}


def short_summary():
    """SBIT/BITI short research verdict (crypto_short.py) and the shadow accounts 220-223."""
    out = {}
    try:
        r = json.loads(SHORT_REPORT.read_text(encoding="utf-8"))
        hc = r.get("hoursCost") or {}
        so = hc.get("shortOnlyMedianSharpe") or {}
        out["research"] = {"window": r.get("window"), "champion": r.get("champion"),
                           "spaVsBtcP": _f((r.get("spaVsBtc") or {}).get("spaP"), 3),
                           "spaVsOwnBaseP": _f((r.get("spaVsOwnBase") or {}).get("spaP"), 3),
                           "passBHVsBase": r.get("passBHVsBase"), "pbo": _f((r.get("pbo") or {}).get("pbo"), 3),
                           "shortOnlyMedianSharpe": so, "calibration": r.get("calibration"),
                           "hoursGapSharpe": _f(hc.get("medianSharpeGapSbitAnyHourMinusSbitHours"), 3)}
    except Exception:   # noqa: BLE001
        pass
    try:
        p = json.loads(SHORT_PAPER.read_text(encoding="utf-8"))
        out["accounts"] = [{k: a.get(k) for k in ("id", "label", "equity", "totalReturn", "targetWeights", "shortSignalNow")}
                           for a in p.get("accounts") or []]
    except Exception:   # noqa: BLE001
        pass
    return out or None


def _pct(x, d=1):
    return "n/a" if x is None else f"{x * 100:+.{d}f}%"


def context_text(snap, max_chars=1700):
    parts = []
    for c in snap.get("coins", []):
        if c["coin"] in ("BTC", "ETH", "SOL") and c.get("price") is not None:
            parts.append(f"{c['coin']} ${c['price']:,.2f} (24h {_pct(c.get('chg24h'))}, 7d {_pct(c.get('chg7d'))}, "
                         f"30d {_pct(c.get('chg30d'))}, vol30 {_pct(c.get('vol30d'), 0).lstrip('+')}, {c.get('trend', 'n/a')}, "
                         f"vs200d {_pct(c.get('vsSma200'))})")
    m = snap.get("macro") or {}
    ov = m.get("overlays") or {}
    text = (f"[CRYPTO] as of {snap.get('asOf')} ET (Coinbase spot; Schwab trades BTC/ETH at 0.75%/side). "
            + "; ".join(parts) + ". "
            + f"Dollar UUP 30d {_pct(m.get('uup30d'))}, 90d {_pct(m.get('uup90d'))}; real yield 30d chg "
              f"{m.get('realYield30dChange')}; net liquidity 91d {_pct(m.get('netLiquidity91d'))}; macro composite "
              f"exposure x{ov.get('compositeExposure')}{', dollar shock active' if ov.get('dollarShockActive') else ''}. ")
    sq = snap.get("schwab") or {}
    fut = [f"{k} {v['last']}" for k, v in sq.items() if isinstance(v, dict) and k.startswith("/") and v.get("last")]
    if fut:
        text += "Schwab futures " + ", ".join(fut[:2]) + ". "
    nc = snap.get("nowcast") or {}
    if nc.get("regime"):
        text += f"Nowcast regime: {nc['regime']}. "
    p = snap.get("paper") or {}
    if p.get("rows"):
        top = sorted(p["rows"], key=lambda r: -(r.get("totalReturn") or -9))[:3]
        text += (f"Paper crypto accounts 200-223 (220-223 SBIT shadow) since {p.get('forwardStart')} ({p.get('days')} day(s)): "
                 f"{p.get('invested')}/{p.get('accounts')} invested; BTC baseline {_pct(p.get('btcBaselineReturn'), 2)}; leaders "
                 + ", ".join(f"#{r['id']} {_pct(r.get('totalReturn'), 2)}" for r in top) + ". ")
    v = snap.get("verdict") or {}
    if v:
        text += (f"Backtest: BTC hold Sharpe {v.get('btcHoldSharpe')} (maxDD {_pct(v.get('btcHoldMaxDD'), 0)}), best "
                 f"#{v.get('championId')} Sharpe {v.get('championSharpe')}; SPA vs BTC p {v.get('spaVsBtcP')}, dollar "
                 f"filters SPA p {v.get('dollarOverlaySpaP')}, nowcast filters p {v.get('nowcastOverlaySpaP')} (not significant).")
    sh = snap.get("short") or {}
    if sh:
        q = (snap.get("schwab") or {}).get("SBIT") or {}
        rs = sh.get("research") or {}
        so = rs.get("shortOnlyMedianSharpe") or {}
        on = [f"#{a['id']}" for a in sh.get("accounts") or [] if a.get("shortSignalNow")]
        last = f" last ${q['last']}" if q.get("last") else ""
        r2 = lambda x: round(x, 2) if isinstance(x, (int, float)) else "n/a"   # noqa: E731
        text += (f" Shorting BTC: SBIT = -2x daily inverse ETF (BITI -1x){last}; trades only in US market hours, so 24/7 BTC "
                 "signals fill next session (gap risk, stops only in hours). Short-sleeve backtest not significant "
                 f"(SPA vs own long base p {rs.get('spaVsOwnBaseP')}); short-only median Sharpe {r2(so.get('SBIT_hours'))} in "
                 f"market hours vs {r2(so.get('perp_24x7'))} for a 24/7 short. Shadow SBIT paper accounts 220-223"
                 f"{(' short now: ' + ', '.join(on)) if on else ''}.")
    return text if len(text) <= max_chars else text[:max_chars - 1] + "…"


def build(schwab=True):
    started = time.time()
    try:
        cd.update_spot(coins=SNAP_COINS)               # incremental: last few daily candles per coin
    except Exception as error:   # noqa: BLE001
        log(f"daily update failed: {type(error).__name__}: {error}")
    daily = cd.load_spot(cd.ALL_COINS)
    coins, sources = [], {}
    for sym in SNAP_COINS:
        hourly = None
        try:
            hourly = hourly_last(cd.COINS[sym]["product"])
        except Exception as error:   # noqa: BLE001
            log(f"{sym} hourly failed: {error}")
        row = coin_stats(sym, daily[sym] if sym in daily else pd.Series(dtype=float), hourly)
        stale = row.get("price") is None or (daily[sym].dropna().index[-1] < pd.Timestamp.today().normalize() - pd.Timedelta(days=3))
        if stale and sym in FRED_FALLBACK and hourly is None:
            s = fred_fallback(FRED_FALLBACK[sym])
            if s is not None and len(s):
                merged = pd.concat([daily[sym].dropna(), s]).sort_index()
                merged = merged[~merged.index.duplicated(keep="last")]
                row = coin_stats(sym, merged, None)
                row["source"] = f"FRED {FRED_FALLBACK[sym]} (fallback)"
        coins.append(row)
        sources[sym] = row.get("source")
    snap = {"asOf": datetime.now().strftime("%Y-%m-%d %H:%M"), "coins": coins, "sources": sources,
            "macro": macro_context(daily.index), "nowcast": nowcast_read(), "schwab": schwab_quotes(schwab),
            "paper": paper_summary(), "verdict": verdict(), "universe": universe(), "short": short_summary(),
            "safety": "Market data only. No orders or previews are ever sent; crypto accounts are paper simulations."}
    snap["context"] = context_text(snap)
    snap["seconds"] = round(time.time() - started, 1)
    cd.write_json(SNAPSHOT, snap)
    # expose on /api/crypto/status (served from the paper status file) without touching server.js
    try:
        st = json.loads(PAPER_STATUS.read_text(encoding="utf-8")) if PAPER_STATUS.exists() else {"status": "awaiting"}
        st["market"] = {k: snap.get(k) for k in ("asOf", "coins", "macro", "nowcast", "schwab", "verdict", "universe", "short", "context")}
        cd.write_json(PAPER_STATUS, st)
    except Exception as error:   # noqa: BLE001
        log(f"could not merge market into paper status: {error}")
    return snap


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-schwab", action="store_true")
    parser.add_argument("--max-age-min", type=float, default=0.0)
    args = parser.parse_args(argv)
    if args.max_age_min and SNAPSHOT.exists() and time.time() - SNAPSHOT.stat().st_mtime < args.max_age_min * 60:
        print(json.dumps({"status": "fresh"}))
        return 0
    if not acquire_lock():
        print(json.dumps({"status": "locked"}))
        return 0
    try:
        snap = build(not args.no_schwab)
        print(json.dumps({"status": "ok", "asOf": snap["asOf"], "seconds": snap["seconds"]}))
    except Exception as error:   # noqa: BLE001
        log(f"failed: {type(error).__name__}: {error}")
        print(json.dumps({"status": "error", "error": f"{type(error).__name__}: {error}"}))
        return 1
    finally:
        release_lock()
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)

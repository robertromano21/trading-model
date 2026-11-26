"""VIX level and term structure for the volatility sleeve (local, gitignored: Massive/Options/vix_daily.csv).

Sources, in order:
  1. FRED public CSV (no key): VIXCLS (CBOE VIX close) and VXVCLS (CBOE 3-month VIX, "VIX3M").
  2. Schwab market data price history for $VIX and $VIX3M once the token is reauthorized (read-only endpoint).
'Low' volatility is defined on the VIX index itself (level, rolling percentile, z-score) and on the term structure
ratio VIX / VIX3M (< 1 = contango, > 1 = backwardation), never on VIXY's share price, which decays by construction.

  python vix_data.py refresh        # download / update the local file
  python vix_data.py show
"""

import csv
import io
import json
import math
import sys
from datetime import datetime
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from fetch_and_model import DATA_DIR, write_atomically

VIX_PATH = DATA_DIR / "Options" / "vix_daily.csv"
FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={id}&cosd=2010-01-01"


def _fred(series_id):
    req = Request(FRED.format(id=series_id), headers={"User-Agent": "trading-model/1.0"})
    with urlopen(req, timeout=60) as r:
        text = r.read().decode("utf-8")
    out = {}
    for row in csv.DictReader(io.StringIO(text)):
        v = row.get(series_id)
        try:
            out[row["observation_date"]] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def refresh():
    vix, vix3m = _fred("VIXCLS"), _fred("VXVCLS")
    days = sorted(set(vix) | set(vix3m))
    lines = ["date,vix,vix3m,source"] + [f"{d},{vix.get(d, '')},{vix3m.get(d, '')},fred" for d in days]
    VIX_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_atomically(VIX_PATH, ("\n".join(lines) + "\n").encode())
    return {"rows": len(days), "first": days[0] if days else None, "last": days[-1] if days else None, "path": str(VIX_PATH)}


def load(path=VIX_PATH):
    if not path.exists():
        raise SystemExit(f"No VIX file at {path}. Run: python vix_data.py refresh")
    f = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    return f[["vix", "vix3m"]].astype(float)


def features(vix_frame, pct_window=252, z_window=60):
    """Causal VIX features per date: level, percentile (rank of today within the trailing window), z-score,
    term ratio VIX/VIX3M, days since the last spike, peak VIX in the last 20 sessions."""
    f = vix_frame.copy().ffill()
    v = f["vix"]
    out = pd.DataFrame(index=f.index)
    out["vix"] = v
    out["vix3m"] = f["vix3m"]
    out["ratio"] = v / f["vix3m"]
    out["pct"] = v.rolling(pct_window, min_periods=120).apply(lambda w: float((w <= w[-1]).mean()), raw=True)
    m, s = v.rolling(z_window, min_periods=40).mean(), v.rolling(z_window, min_periods=40).std()
    out["z"] = (v - m) / s
    out["peak20"] = v.rolling(20, min_periods=1).max()
    return out


def latest(path=VIX_PATH):
    f = features(load(path)).dropna(subset=["vix"])
    row = f.iloc[-1]
    return {"date": str(f.index[-1].date()), **{k: (None if pd.isna(row[k]) else round(float(row[k]), 4)) for k in f.columns}}


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "show"
    print(json.dumps(refresh() if cmd == "refresh" else latest(), indent=1))

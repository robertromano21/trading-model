"""OUTSIDE projections - display and accuracy comparison ONLY. Never a model input.

macro_nowcast forms its own estimates from raw data. This module fetches what others project so the
dashboard can show them next to ours, compute "ours minus theirs", and score both after the fact:

  * Cleveland Fed inflation nowcast (CPI, core CPI, PCE, core PCE m/m) - public JSON behind their chart.
  * Atlanta Fed GDPNow - FRED series GDPNOW (final estimate per quarter; FRED does not carry the daily path).
  * Philadelphia Fed Survey of Professional Forecasters - median real GDP growth (xlsx, parsed with stdlib).
  * Street consensus per release - NO free, terms-clean source was found (calendar APIs with consensus need a
    paid key or scraping). Pluggable slot: Massive/Macro/outside/consensus.csv with columns
    target,period,value,source,asof (target = CPI_MOM, CORE_CPI_MOM, CORE_PCE_MOM, UNRATE, PAYROLLS, GDP_SAAR).
  * NY Fed Staff Nowcast - their download link returns an HTML page, not a file; left as a slot.

Everything is stored under Massive/Macro/outside/.
"""
from __future__ import annotations

import io
import json
import math
import re
import sys
import zipfile
from datetime import date, datetime
from pathlib import Path
from urllib.request import Request, urlopen
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd

try:
    from fetch_and_model import DATA_DIR, write_atomically
except Exception:  # pragma: no cover
    DATA_DIR = Path(__file__).resolve().parents[1] / "Massive"

    def write_atomically(path, contents):
        Path(path).write_bytes(contents)

OUT_DIR = DATA_DIR / "Macro" / "outside"
CLEVELAND_URL = "https://www.clevelandfed.org/-/media/files/webcharts/inflationnowcasting/nowcast_month.json"
SPF_URL = ("https://www.philadelphiafed.org/-/media/frbp/assets/surveys-and-data/survey-of-professional-forecasters/"
           "data-files/files/median_rgdp_growth.xlsx")
CONSENSUS_CSV = OUT_DIR / "consensus.csv"
UA = "Mozilla/5.0 (Trading-Model-main personal research)"

# Every id that is someone else's projection. macro_nowcast checks its inputs against this list.
OUTSIDE_IDS = frozenset({"GDPNOW", "CLEVELAND_NOWCAST", "SPF_RGDP", "CONSENSUS", "NYFED_NOWCAST"})

CLEVELAND_SERIES = {"CPI Inflation": "CPI_MOM", "Core CPI Inflation": "CORE_CPI_MOM",
                    "PCE Inflation": "PCE_MOM", "Core PCE Inflation": "CORE_PCE_MOM"}
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _get(url, timeout=60):
    with urlopen(Request(url, headers={"User-Agent": UA}), timeout=timeout) as r:
        return r.read()


# ------------------------------------------------------------------ Cleveland Fed
def parse_cleveland(doc):
    """-> DataFrame rows: target, period (YYYY-MM-01), date, value, finalBeforeRelease(bool)."""
    rows = []
    for entry in doc or []:
        try:
            sub = entry["chart"]["subcaption"]
            y, m = (int(x) for x in sub.split("-")[:2])
        except Exception:  # noqa: BLE001
            continue
        labels = [c.get("label", "") for c in entry["categories"][0]["category"]]
        # date for each category; vline labels like "CPI Oct" mark releases
        dates, release_idx = [], {}
        for i, lab in enumerate(labels):
            mm = re.match(r"^(\d{2})/(\d{2})$", lab)
            if mm:
                mo, dy = int(mm.group(1)), int(mm.group(2))
                yr = y if mo >= m else y + 1
                dates.append(pd.Timestamp(yr, mo, dy))
            else:
                dates.append(None)
                rel = re.match(r"^(CPI|PCE)\s+(\w{3})$", lab)
                if rel and rel.group(2) == MONTHS[m - 1]:
                    release_idx[rel.group(1)] = i
        for ds in entry.get("dataset", []):
            target = CLEVELAND_SERIES.get(ds.get("seriesname"))
            if not target:
                continue
            fam = "PCE" if "PCE" in target else "CPI"
            cut = release_idx.get(fam)
            data = ds.get("data", [])
            last_before = None
            for i, (d, point) in enumerate(zip(dates, data)):
                try:
                    v = float(point.get("value"))
                except (TypeError, ValueError):
                    continue
                if d is None:
                    continue
                rows.append({"target": target, "period": pd.Timestamp(y, m, 1), "date": d, "value": v})
                if cut is None or i < cut:
                    last_before = len(rows) - 1
            if last_before is not None and cut is not None:
                rows[last_before]["finalBeforeRelease"] = True
    f = pd.DataFrame(rows)
    if not f.empty:
        f["finalBeforeRelease"] = f.get("finalBeforeRelease", False).fillna(False).astype(bool)
    return f


def update_cleveland():
    doc = json.loads(_get(CLEVELAND_URL).decode("utf-8-sig"))
    f = parse_cleveland(doc)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_atomically(OUT_DIR / "cleveland_nowcast.csv", f.to_csv(index=False, date_format="%Y-%m-%d").encode("utf-8"))
    return f


# ------------------------------------------------------------------ SPF (xlsx via stdlib)
def read_xlsx_first_sheet(raw):
    z = zipfile.ZipFile(io.BytesIO(raw))
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    shared = []
    if "xl/sharedStrings.xml" in z.namelist():
        for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", ns):
            shared.append("".join(t.text or "" for t in si.iter("{%s}t" % ns["m"])))
    sheet = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
    rows = []
    for r in sheet.iter("{%s}row" % ns["m"]):
        vals = {}
        for c in r.findall("m:c", ns):
            ref = re.match(r"([A-Z]+)", c.get("r")).group(1)
            v = c.find("m:v", ns)
            if v is None:
                continue
            vals[ref] = shared[int(v.text)] if c.get("t") == "s" else v.text
        rows.append(vals)
    if not rows:
        return pd.DataFrame()
    cols = sorted({k for r in rows for k in r}, key=lambda s: (len(s), s))
    header = [rows[0].get(c, c) for c in cols]
    return pd.DataFrame([[r.get(c) for c in cols] for r in rows[1:]], columns=header)


def update_spf():
    f = read_xlsx_first_sheet(_get(SPF_URL))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_atomically(OUT_DIR / "spf_median_rgdp.csv", f.to_csv(index=False).encode("utf-8"))
    return f


# ------------------------------------------------------------------ consensus slot
def load_consensus():
    if not CONSENSUS_CSV.is_file():
        return pd.DataFrame(columns=["target", "period", "value", "source", "asof"])
    f = pd.read_csv(CONSENSUS_CSV)
    f["period"] = pd.to_datetime(f["period"], errors="coerce")
    return f


def ensure_consensus_template():
    if not CONSENSUS_CSV.is_file():
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        CONSENSUS_CSV.write_text("target,period,value,source,asof\n", encoding="utf-8")


def update_all():
    out = {}
    for name, fn in (("cleveland", update_cleveland), ("spf", update_spf)):
        try:
            out[name] = int(len(fn()))
        except Exception as exc:  # noqa: BLE001
            out[name] = f"error: {str(exc)[:120]}"
    ensure_consensus_template()
    return out


# ------------------------------------------------------------------ display
def _period_label(p, quarterly=False):
    p = pd.Timestamp(p)
    return f"{p.year}Q{(p.month - 1) // 3 + 1}" if quarterly else str(p.date())[:7]


def latest_for_display():
    """{target: {source: {value, period, asof, note}}} for the dashboard. Display only."""
    out = {"_note": ("Outside estimates are shown for comparison only and are never inputs to our nowcast. "
                     "No free terms-clean Street consensus feed was found; add rows to Massive/Macro/outside/consensus.csv.")}
    cpath = OUT_DIR / "cleveland_nowcast.csv"
    if cpath.is_file():
        f = pd.read_csv(cpath, parse_dates=["period", "date"])
        for target, g in f.groupby("target"):
            g = g.sort_values(["period", "date"])
            for per in sorted(g["period"].unique())[-2:]:
                last = g[g["period"] == per].iloc[-1]
                out.setdefault(target, {})[f"clevelandFed:{_period_label(per)}"] = {
                    "value": round(float(last["value"]), 3), "period": _period_label(per), "asof": str(last["date"].date()),
                    "source": "Cleveland Fed inflation nowcast"}
        # keep a single 'clevelandFed' key = nearest period, for ours-minus-theirs
        for target in list(out):
            if target.startswith("_"):
                continue
            keys = sorted(k for k in out[target] if k.startswith("clevelandFed:"))
            if keys:
                out[target]["clevelandFed"] = out[target][keys[0]]
    try:
        import fred_macro as FM
        g = FM.load_series("GDPNOW")
        if g is not None and len(g):
            out.setdefault("GDP_SAAR", {})["gdpNow"] = {"value": round(float(g.iloc[-1]), 2), "period": _period_label(g.index[-1], True),
                                                        "source": "Atlanta Fed GDPNow via FRED (final per quarter)"}
    except Exception:  # noqa: BLE001
        pass
    spath = OUT_DIR / "spf_median_rgdp.csv"
    if spath.is_file():
        try:
            f = pd.read_csv(spath)
            last = f.dropna(subset=["YEAR"]).iloc[-1]
            col = "DRGDP2" if "DRGDP2" in f.columns else None
            if col:
                out.setdefault("GDP_SAAR", {})["spfMedian"] = {
                    "value": round(float(last[col]), 2), "period": f"{int(float(last['YEAR']))}Q{int(float(last['QUARTER']))}",
                    "source": "Philadelphia Fed SPF median (current quarter, survey mid-quarter)"}
        except Exception:  # noqa: BLE001
            pass
    cons = load_consensus()
    for target, g in cons.groupby("target") if len(cons) else []:
        last = g.sort_values("period").iloc[-1]
        out.setdefault(target, {})["consensus"] = {"value": float(last["value"]), "period": str(last["period"].date())[:7],
                                                   "source": str(last.get("source") or "manual"), "asof": str(last.get("asof") or "")}
    return out


# ------------------------------------------------------------------ after-the-fact accuracy (ours vs theirs)
def _dm(e_a, e_b):
    d = np.asarray(e_b, float) ** 2 - np.asarray(e_a, float) ** 2   # >0 means A (ours) better
    d = d[np.isfinite(d)]
    if len(d) < 8:
        return None, None
    t = d.mean() / (d.std(ddof=1) / math.sqrt(len(d)))
    p = 0.5 * math.erfc(abs(t) / math.sqrt(2)) * 2
    return float(t), float(p)


def compare_accuracy(all_rows, end=None):
    """Final pre-release nowcasts: ours vs Cleveland Fed (inflation) and GDPNow (GDP). Two-sided DM test."""
    out = {}
    cpath = OUT_DIR / "cleveland_nowcast.csv"
    if cpath.is_file():
        cf = pd.read_csv(cpath, parse_dates=["period", "date"])
        cf = cf[cf.get("finalBeforeRelease", False) == True]  # noqa: E712
        for target in ("CPI_MOM", "CORE_CPI_MOM", "CORE_PCE_MOM"):
            rows = all_rows.get(target)
            if rows is None:
                continue
            ours = rows[(rows["k"] == 0) & rows["pred"].notna() & rows["y"].notna()][["period", "release", "pred", "y"]]
            if end is not None:
                ours = ours[ours["release"] < end]
            theirs = cf[cf["target"] == target][["period", "value"]]
            m = ours.merge(theirs, on="period")
            if len(m) < 8:
                out[target] = {"n": int(len(m))}
                continue
            e_o, e_t = m["y"] - m["pred"], m["y"] - m["value"]
            t, p = _dm(e_o, e_t)
            out[target] = {"vs": "Cleveland Fed nowcast", "n": int(len(m)), "first": str(m["period"].min().date()),
                           "rmseOurs": float(np.sqrt((e_o ** 2).mean())), "rmseTheirs": float(np.sqrt((e_t ** 2).mean())),
                           "dmT": t, "dmP": p}
    try:
        import fred_macro as FM
        g = FM.load_series("GDPNOW")
        rows = all_rows.get("GDP_SAAR")
        if g is not None and rows is not None:
            ours = rows[(rows["k"] == 0) & rows["pred"].notna() & rows["y"].notna()][["period", "release", "pred", "y"]]
            if end is not None:
                ours = ours[ours["release"] < end]
            m = ours.merge(g.rename("value").rename_axis("period").reset_index(), on="period")
            if len(m) >= 8:
                e_o, e_t = m["y"] - m["pred"], m["y"] - m["value"]
                t, p = _dm(e_o, e_t)
                out["GDP_SAAR"] = {"vs": "Atlanta Fed GDPNow (final)", "n": int(len(m)), "first": str(m["period"].min().date()),
                                   "rmseOurs": float(np.sqrt((e_o ** 2).mean())), "rmseTheirs": float(np.sqrt((e_t ** 2).mean())),
                                   "dmT": t, "dmP": p}
    except Exception as exc:  # noqa: BLE001
        out["GDP_SAAR"] = {"error": str(exc)[:120]}
    out["note"] = ("dmT > 0 means our final nowcast had the smaller squared error. Outside numbers are compared "
                   "after the fact only; they never enter our model.")
    return out


def main(argv=None):
    print(json.dumps(update_all(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

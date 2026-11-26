"""Security classification and causal liquidity filters for strategy universes.

Classification uses the public Nasdaq Trader symbol directory (nasdaqtraded.txt: ETF and test-issue flags,
security names), cached locally at Massive/Reference/nasdaqtraded.txt. Symbols missing from today's directory
(delisted names) are kept as stocks when they look like ordinary 1-4 letter tickers, so the universe is not
limited to today's survivors. OTC-only symbols, warrants, units, rights and preferreds are excluded.
"""

import re
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

from fetch_and_model import DATA_DIR, write_atomically

REFERENCE = DATA_DIR / "Reference" / "nasdaqtraded.txt"
DIRECTORY_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqtraded.txt"
# Liquid, unlevered ETFs used by the trend sleeve (broad equity, size, sectors, rates, gold, international).
CORE_ETFS = ("SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU",
             "XLRE", "XLC", "TLT", "IEF", "GLD", "EFA", "EEM")
EXCLUDED_WORDS = ("warrant", "unit", " right", "rights", "preferred", "depositary shares representing",
                  " notes", "% ", "debenture", "test stk", "test issue")


def refresh_directory(path=REFERENCE):
    request = Request(DIRECTORY_URL, headers={"User-Agent": "Mozilla/5.0 Trading-Model"})
    with urlopen(request, timeout=60) as response:
        write_atomically(Path(path), response.read())
    return Path(path)


def read_directory(path=REFERENCE):
    path = Path(path)
    if not path.is_file():
        return None
    table = pd.read_csv(path, sep="|", dtype=str, keep_default_na=False)
    table = table[table["Symbol"].ne("") & ~table["Nasdaq Traded"].str.startswith("File Creation")]
    return table.drop_duplicates("Symbol").set_index("Symbol")


def classify(tickers, directory=None):
    """Return a Series ticker -> 'stock' | 'etf' | 'other' | 'test'."""
    directory = read_directory() if directory is None else directory
    labels = {}
    for ticker in tickers:
        if directory is not None and ticker in directory.index:
            row = directory.loc[ticker]
            name = f" {row['Security Name'].lower()} "
            if row.get("Test Issue") == "Y":
                labels[ticker] = "test"
            elif row.get("ETF") == "Y":
                labels[ticker] = "etf"
            elif any(word in name for word in EXCLUDED_WORDS):
                labels[ticker] = "other"
            else:
                labels[ticker] = "stock"
        elif ticker in CORE_ETFS:
            labels[ticker] = "etf"
        else:
            # Not listed today: delisted or OTC. Keep plain 1-4 letter symbols (likely delisted listed stocks).
            labels[ticker] = "stock" if re.fullmatch(r"[A-Z]{1,4}", ticker) else "other"
    return pd.Series(labels, dtype=object)


def dollar_volume(panel, window=20):
    """Trailing median dollar volume through each date (uses data <= t only)."""
    return (panel.close * panel.volume).rolling(window, min_periods=max(5, window * 3 // 4)).median()


def liquid_mask(panel, min_price=5.0, min_dollar_volume=20e6, window=20, max_dollar_volume=None):
    adv = dollar_volume(panel, window)
    mask = (panel.close > min_price) & (adv > min_dollar_volume)
    if max_dollar_volume is not None:
        mask &= adv <= max_dollar_volume
    return mask.fillna(False)


def eligible(panel, kind="stocks", **filters):
    """Boolean DataFrame of tradable names per date for kind 'stocks', 'etf' (CORE_ETFS) or 'all'."""
    classes = panel.classes.reindex(panel.close.columns).fillna("other")
    if kind == "etf":
        columns = [ticker for ticker in CORE_ETFS if ticker in panel.close.columns]
    elif kind == "stocks":
        columns = list(classes.index[classes.eq("stock")])
    elif kind == "all":
        columns = list(classes.index[classes.isin(["stock", "etf"])])
    else:
        raise ValueError(f"Unknown universe {kind!r}.")
    mask = pd.DataFrame(False, index=panel.close.index, columns=panel.close.columns)
    if columns:
        if kind == "etf":
            mask[columns] = panel.close[columns].notna() & liquid_mask(panel, **{"min_dollar_volume": 5e6, **filters})[columns]
        else:
            mask[columns] = liquid_mask(panel, **filters)[columns]
    return mask

"""Config-gated option / downside overlays on top of the shared strategy interface (strategies.*).

An overlay never changes the equity strategy's target weights. Each session it looks at that strategy's holdings,
the causal regime / top-detector signals and the account, and returns intents for the paper options book
(options_book.OptionsBook) or, for inverse ETFs / paper short stock, simple share intents. The same class drives
options_research.py (walk-forward backtests) and options_paper.py (daily paper runs), so backtest and paper share
the signal code, exactly like Strategy.target_weights / target_weights_at.

Overlays (``mode``):
  protective_put   buy SPY/IWM puts (or put spreads) sized to hedge part of the book's delta while the regime is adverse
  covered_call     write OTM calls against holdings of >= 100 shares (covered by the equity book)
  csp_entry        sell cash-secured puts on in-trend names the book does not hold, from idle cash only
  bearish_puts     long puts / bear put spreads when the top detector fires, sized by max loss
  inverse_etf      buy SH / PSQ / SDS when the top detector fires, sized so a stop-out loses at most the budget
  short_stock      paper only and config-gated: short SPY (or the inverse-ETF underlying) with borrow fee, Reg-T
                   150% margin, gap-aware squeeze stop and dividends owed

All are OFF unless ``enabled=True``. Defaults are deliberately small (risk budget 1-2% of equity per position).
"""

import math

import numpy as np
import pandas as pd

from options_pricing import build_strategy, year_fraction

MODES = ("none", "protective_put", "covered_call", "csp_entry", "bearish_puts", "inverse_etf", "short_stock")
INVERSE = {"SPY": ("SH", -1.0), "QQQ": ("PSQ", -1.0), "SDS": ("SDS", -2.0)}


class OptionsOverlay:
    name = "options_overlay"
    defaults = {
        "enabled": False, "mode": "none",
        "trigger": "hmm",            # hmm | top | always
        "hmm_threshold": 0.6,        # P(risk_off)+P(stress)+0.5 P(chop)
        "top_k": 2,                  # top-detector signal count
        "underlyings": ("SPY", "IWM", "QQQ"),
        "dte": 45, "roll_dte": 10, "otm_pct": 0.05, "width_pct": 0.05,
        "structure": "put",          # put | put_spread  (protective_put / bearish_puts)
        "hedge_ratio": 0.5,          # fraction of book delta to hedge
        "risk_budget": 0.02,         # max loss per position / equity
        "take_profit": 1.0,          # close long options when value >= (1 + take_profit) * cost
        "call_otm_pct": 0.05, "call_dte": 30, "min_shares": 100,
        "csp_otm_pct": 0.05, "csp_dte": 30, "csp_max_names": 2,
        "inverse": "SH", "stop_pct": 0.08,
        "fractional": False,         # reference mode: allow fractional contracts (NOT tradable at $10k)
        "exit_days_off": 3,          # signal must be off this many sessions before a hedge is removed
    }

    def __init__(self, **params):
        unknown = set(params) - set(self.defaults)
        if unknown:
            raise ValueError(f"{self.name}: unknown parameters {sorted(unknown)}")
        self.params = {**self.defaults, **params}
        if self.params["mode"] not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.off_streak = 0

    def describe(self):
        return {"overlay": self.name, "params": {k: list(v) if isinstance(v, tuple) else v for k, v in self.params.items()}}

    # ------------------------------------------------------------------------------------------ signals
    def adverse(self, signal_row):
        p = self.params
        if p["trigger"] == "always":
            return True
        if p["trigger"] == "top":
            return float(signal_row.get("count", 0) or 0) >= p["top_k"]
        return float(signal_row.get("hmm_prob", 0) or 0) >= p["hmm_threshold"]

    @staticmethod
    def pick_expiration(asof, dte):
        """Nearest Friday on/after asof + dte calendar days (listed weeklies exist on SPY/QQQ/IWM)."""
        d = pd.Timestamp(asof) + pd.Timedelta(days=int(dte))
        d += pd.Timedelta(days=(4 - d.weekday()) % 7)
        return d.strftime("%Y-%m-%d")

    @staticmethod
    def _strike(x, spot):
        step = 0.5 if spot < 25 else 1.0 if spot < 200 else 5.0
        return max(step, round(x / step) * step)

    # ------------------------------------------------------------------------------------------ intents
    def intents(self, asof, ctx):
        """ctx: {"spots": {t: price}, "holdings": {t: shares}, "equity": float, "idle_cash": float,
        "book_delta": float (equity book delta dollars), "signal": row (dict-like), "positions": open option
        positions (book dicts), "share_positions": {t: shares} (inverse ETF / short sleeve), "candidates": [t...],
        "quote": callable(underlying, leg_dict) -> {"mid", "greeks", ...}}. Returns a list of intent dicts:
        {"action": "open", "strategy", "underlying", "legs", "covered_shares", "tag"} | {"action": "close", "id", "reason"}
        | {"action": "shares", "ticker", "target_shares", "reason"} | {"action": "short", ...} | {"action": "cover", ...}."""
        p = self.params
        if not p["enabled"] or p["mode"] == "none":
            return []
        mode = p["mode"]
        adverse = self.adverse(ctx.get("signal", {}))
        self.off_streak = 0 if adverse else self.off_streak + 1
        out = []
        mine = [x for x in ctx.get("positions", []) if x.get("tag") == mode]
        # roll / exit housekeeping for option overlays
        for pos in mine:
            front = min((l["expiration"] for l in pos["legs"] if l["kind"] != "stock"), default=None)
            dte = (pd.Timestamp(front) - pd.Timestamp(asof)).days if front else 999
            if dte <= p["roll_dte"]:
                out.append({"action": "close", "id": pos["id"], "reason": f"roll/exit at {dte} DTE"})
            elif mode in ("protective_put", "bearish_puts") and self.off_streak >= p["exit_days_off"]:
                out.append({"action": "close", "id": pos["id"], "reason": "adverse signal off"})
            elif mode == "covered_call" and ctx.get("holdings", {}).get(pos["underlying"], 0) < pos.get("coveredShares", 0):
                out.append({"action": "close", "id": pos["id"], "reason": "equity book sold the covering shares"})
            elif mode in ("protective_put", "bearish_puts") and p["take_profit"] and ctx.get("value_of"):
                value = ctx["value_of"](pos)
                cost = -pos["entryCash"]
                if cost > 0 and value >= (1 + p["take_profit"]) * cost:
                    out.append({"action": "close", "id": pos["id"], "reason": "take profit"})
        closing = {i["id"] for i in out if i["action"] == "close"}
        still_open = [x for x in mine if x["id"] not in closing]
        eq = ctx["equity"]
        if mode in ("protective_put", "bearish_puts") and adverse and not still_open:
            out += self._put_intents(asof, ctx, mode)
        elif mode == "covered_call":
            covered = {x["underlying"] for x in still_open}
            for t, sh in sorted(ctx.get("holdings", {}).items()):
                n = int(sh // 100) if not p["fractional"] else math.floor(sh) / 100.0
                if n > 0 and (sh >= p["min_shares"] or p["fractional"]) and t not in covered and t in ctx["spots"]:
                    spot = ctx["spots"][t]
                    k = self._strike(spot * (1 + p["call_otm_pct"]), spot)
                    exp = self.pick_expiration(asof, p["call_dte"])
                    legs = [{"kind": "call", "qty": -n, "strike": k, "expiration": exp}]
                    out.append({"action": "open", "strategy": "covered_call", "underlying": t, "legs": legs,
                                "covered_shares": n * 100, "tag": mode, "reason": f"write {n} call(s) {k} vs {int(sh)} shares"})
        elif mode == "csp_entry" and not adverse:
            have = {x["underlying"] for x in still_open}
            budget = ctx.get("idle_cash", 0.0) - sum(x.get("reserve", 0.0) for x in still_open)
            for t in ctx.get("candidates", [])[: p["csp_max_names"] * 3]:
                if len(have) >= p["csp_max_names"] or t in have or ctx.get("holdings", {}).get(t, 0) > 0 or t not in ctx["spots"]:
                    continue
                spot = ctx["spots"][t]
                k = self._strike(spot * (1 - p["csp_otm_pct"]), spot)
                if k * 100 > budget and not p["fractional"]:
                    out.append({"action": "skip", "underlying": t, "reason": f"CSP needs ${k * 100:,.0f} cash; idle cash ${budget:,.0f}"})
                    continue
                qty = 1 if not p["fractional"] else max(0.01, min(1.0, budget / (k * 100)))
                out.append({"action": "open", "strategy": "cash_secured_put", "underlying": t, "tag": mode,
                            "legs": [{"kind": "put", "qty": -qty, "strike": k, "expiration": self.pick_expiration(asof, p["csp_dte"])}],
                            "reason": f"cash-secured put {k} on in-trend {t}"})
                have.add(t)
                budget -= k * 100 * qty
        elif mode in ("inverse_etf", "short_stock"):
            ticker = p["inverse"] if mode == "inverse_etf" else "SPY"
            held = ctx.get("share_positions", {}).get(ticker, 0)
            if adverse and held == 0 and ticker in ctx["spots"]:
                px = ctx["spots"][ticker]
                dollars = p["risk_budget"] * eq / p["stop_pct"]
                shares = math.floor(dollars / px) if not p["fractional"] else dollars / px
                if shares > 0:
                    out.append({"action": "buy" if mode == "inverse_etf" else "short", "ticker": ticker, "shares": shares,
                                "stop": px * (1 - p["stop_pct"]) if mode == "inverse_etf" else px * (1 + p["stop_pct"]),
                                "reason": "adverse regime / top signal"})
            elif held and self.off_streak >= p["exit_days_off"]:
                out.append({"action": "sell" if mode == "inverse_etf" else "cover", "ticker": ticker, "reason": "adverse signal off"})
        return out

    def _put_intents(self, asof, ctx, mode):
        p = self.params
        eq = ctx["equity"]
        budget = p["risk_budget"] * eq
        exp = self.pick_expiration(asof, p["dte"])
        best, notes = None, []
        for u in p["underlyings"]:
            spot = ctx["spots"].get(u)
            if not spot:
                continue
            k = self._strike(spot * (1 - p["otm_pct"]), spot)
            if p["structure"] == "put_spread":
                k2 = self._strike(k - spot * p["width_pct"], spot)
                legs = [{"kind": "put", "qty": 1, "strike": k, "expiration": exp}, {"kind": "put", "qty": -1, "strike": k2, "expiration": exp}]
                strategy = "bear_put_spread"
            else:
                legs = [{"kind": "put", "qty": 1, "strike": k, "expiration": exp}]
                strategy = "long_put"
            q = [ctx["quote"](u, l) for l in legs]
            cost = sum(l["qty"] * 100 * (qq["ask"] if l["qty"] > 0 else qq["bid"]) for l, qq in zip(legs, q))
            delta = sum(l["qty"] * 100 * qq["greeks"]["delta"] * spot for l, qq in zip(legs, q))
            if cost <= 0 or delta >= 0:
                continue
            if mode == "protective_put":
                want = p["hedge_ratio"] * max(ctx.get("book_delta", 0.0), 0.0)
                n = want / -delta
            else:
                n = budget / cost
            n = min(n, budget / cost)
            n_int = math.floor(n + 1e-9) if mode == "bearish_puts" else int(round(n))
            n_int = min(n_int, math.floor(budget / cost + 1e-9))
            n_use = n if p["fractional"] else n_int
            if n_use <= 0:
                notes.append(f"{u}: one {strategy} costs ${cost:,.0f} (delta ${delta:,.0f}); budget ${budget:,.0f} -> 0 contracts")
                continue
            score = abs(n_use * delta - (p["hedge_ratio"] * ctx.get("book_delta", 0.0) if mode == "protective_put" else 0))
            if best is None or score < best[0]:
                best = (score, u, strategy, legs, n_use)
        if not best:
            return [{"action": "skip", "reason": "; ".join(notes) or "no hedge underlying priced"}]
        _, u, strategy, legs, n = best
        return [{"action": "open", "strategy": strategy, "underlying": u, "tag": mode,
                 "legs": [{**l, "qty": l["qty"] * n} for l in legs], "reason": f"adverse regime: {strategy} x{n:g} on {u}"}]


__all__ = ["OptionsOverlay", "MODES"]

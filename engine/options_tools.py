"""OPTIONS analyst ops for quant_tools (chain, greeks, payoff, probability of profit, expected move, top-risk and
vol-regime readings). Market data only: chains come from options_data.get_chain (Schwab market-data GET ->
local snapshot -> MODELED chain). Nothing here can place or preview an order."""

import json
import math
from datetime import date

from fetch_and_model import DATA_DIR

RESEARCH = DATA_DIR / "Research"


def ops(ToolError):
    """Return the op registry bound to quant_tools' ToolError (so errors come back typed)."""

    def bad(msg):
        raise ToolError("BAD_ARGS", msg)

    def num(args, key, default=None, lo=None, hi=None):
        v = args.get(key, default)
        if v is None:
            bad(f"{key} is required")
        try:
            v = float(v)
        except (TypeError, ValueError):
            bad(f"{key} must be a number")
        if not math.isfinite(v) or (lo is not None and v < lo) or (hi is not None and v > hi):
            bad(f"{key} out of range")
        return v

    def _sym(args):
        return str(args.get("symbol") or (args.get("tickers") or [""])[0] or "").upper()

    def _spot_vol(args):
        sym = _sym(args)
        spot, vol = args.get("spot"), args.get("vol")
        if sym and (spot is None or vol is None):
            from options_data import local_history
            from options_pricing import base_vol_forecast
            s, _, rets, _ = local_history(sym)
            spot = s if spot is None else spot
            vol = base_vol_forecast(rets) * 1.12 if vol is None else vol
        return num({"spot": spot}, "spot", lo=0.01), num({"vol": vol}, "vol", lo=0.01, hi=5)

    def op_options_chain(args):
        """Option chain for a symbol: live Schwab market data when authorized (snapshot stored locally), else the
        latest local snapshot, else a MODELED chain. args: symbol, strikes (per side, default 8), expirations (default 4),
        prefer (schwab|snapshot|model)."""
        from options_data import get_chain
        sym = _sym(args)
        if not sym.replace(".", "").isalnum():
            bad("symbol is required")
        prefer = args.get("prefer", "schwab")
        if prefer not in ("schwab", "snapshot", "model", "auto"):
            bad("prefer must be schwab, snapshot or model")
        chain = get_chain(sym, prefer=prefer, strike_count=int(num(args, "strikes", 8, 1, 40)) * 2)
        n_exp = int(num(args, "expirations", 4, 1, 20))
        spot = chain.get("underlyingPrice")
        exps = sorted({c["expiration"] for c in chain["contracts"]})[:n_exp]
        k = int(num(args, "strikes", 8, 1, 40))
        rows = []
        for e in exps:
            cs = [c for c in chain["contracts"] if c["expiration"] == e]
            strikes = sorted({c["strike"] for c in cs}, key=lambda x: abs(x - (spot or x)))[:2 * k]
            rows += [c for c in cs if c["strike"] in strikes]
        return {"symbol": sym, "source": chain.get("source"), "modeled": chain.get("source") == "model",
                "asof": chain.get("asof") or chain.get("snapshotAt"), "underlyingPrice": spot, "expirations": exps,
                "contracts": sorted(rows, key=lambda c: (c["expiration"], c["strike"], c["kind"])),
                "fallbackReason": chain.get("fallbackReason"),
                "note": "MODELED prices (no live chain available)" if chain.get("source") == "model" else None}

    def op_options_greeks(args):
        """Black-Scholes price and greeks. args: spot (or symbol), strike, dte (calendar days), vol (or symbol for a
        modeled vol), kind call|put, rate (0.04), div_yield (0)."""
        from options_pricing import bsm_greeks
        spot, vol = _spot_vol(args)
        kind = args.get("kind", "call")
        if kind not in ("call", "put", "C", "P"):
            bad("kind must be call or put")
        g = bsm_greeks(spot, num(args, "strike", lo=0.01), num(args, "dte", lo=0, hi=3650) / 365.0, vol,
                       num(args, "rate", 0.04, -0.05, 0.5), num(args, "div_yield", 0.0, 0, 0.5), kind)
        return {"spot": spot, "vol": vol, **g, "units": "per share; vega per 1 vol point; theta per calendar day"}

    def _legs(args, spot, vol):
        from options_pricing import build_strategy, price_legs
        asof = str(args.get("asof") or date.today())
        if args.get("legs"):
            legs = args["legs"]
            if not isinstance(legs, list) or not 1 <= len(legs) <= 6:
                bad("legs must be a list of 1-6 legs")
            need = [l for l in legs if l.get("kind") != "stock" and l.get("price") is None]
            if need:
                priced = price_legs(legs, spot, asof, vol)
                priced = priced[0] if isinstance(priced, tuple) else priced
                legs = [p.as_dict() if hasattr(p, "as_dict") else p for p in priced]
            return legs, asof
        name = args.get("strategy")
        exp = args.get("expiration")
        if not name or not exp:
            bad("give legs, or strategy + expiration")
        try:
            raw = build_strategy(name, spot, exp, strikes=args.get("strikes"), qty=int(num(args, "qty", 1, 1, 100)),
                                 back_expiration=args.get("back_expiration"), width=args.get("width"))
        except ValueError as error:
            bad(str(error))
        priced = price_legs(raw, spot, asof, vol)
        priced = priced[0] if isinstance(priced, tuple) else priced
        return [p.as_dict() if hasattr(p, "as_dict") else p for p in priced], asof

    def op_options_payoff(args):
        """Payoff, max loss / max gain, breakevens, greeks and model probability of profit for a strategy.
        args: spot/vol or symbol; legs [{kind, qty (+long/-short), strike, expiration, price?}] or strategy
        (long_call, bear_put_spread, iron_condor, ...) + expiration (+ strikes, back_expiration, qty). Unpriced legs are
        filled at MODELED bid/ask."""
        from options_pricing import OptionFees, analyze_legs, position_greeks, probability_of_profit
        spot, vol = _spot_vol(args)
        legs, asof = _legs(args, spot, vol)
        fees = OptionFees()
        fee = sum(fees.fee(abs(l["qty"]), l.get("price") or 0, "buy_to_open" if l["qty"] > 0 else "sell_to_open")
                  for l in legs if l["kind"] != "stock")
        info = analyze_legs(legs, spot, asof, vol=vol, fees=fee)
        curve = info.pop("curve", [])
        step = max(1, len(curve) // 60)
        return {"spot": spot, "vol": vol, "legs": legs, **info, "estimatedFees": round(fee, 2),
                "greeks": position_greeks(legs, spot, asof, vol=vol),
                "probabilityOfProfit": probability_of_profit(legs, spot, asof, vol, fees=fee),
                "curve": curve[::step], "note": "Model values (Black-Scholes, lognormal). POP is not an edge estimate."}

    def op_options_pop(args):
        """Model probability of profit at expiry for a strategy (same args as options_payoff)."""
        r = op_options_payoff(args)
        return {k: r[k] for k in ("probabilityOfProfit", "maxLoss", "maxGain", "breakevens", "spot", "vol", "note")}

    def op_expected_move(args):
        """1-sigma expected move: spot * vol * sqrt(days/365). args: spot/vol or symbol, days, straddle (optional)."""
        from options_pricing import expected_move
        spot, vol = _spot_vol(args)
        return expected_move(spot, vol, num(args, "days", 30, 0, 3650), args.get("straddle"))

    def _read(name):
        p = RESEARCH / name
        if not p.exists():
            raise ToolError("NO_DATA", f"{name} not found; run the research job first")
        return json.loads(p.read_text(encoding="utf-8"))

    def op_top_risk(args):
        """Latest top / adverse-regime detector reading, base rates, honest out-of-sample verdict."""
        s = _read("top_detector.json")
        return {"latest": s.get("latest"), "verdict": s.get("verdict"), "baseRates": s.get("baseRates"),
                "survivingHolm05": s.get("survivingHolm05"), "robustSurvivors": s.get("robustSurvivors"),
                "walkForwardComposite": s.get("walkForwardComposite"), "generated": s.get("generated")}

    def op_vol_regime(args):
        """VIX regime (level, 252d percentile, z-score, VIX/VIX3M term structure) and the vol-sleeve paper state."""
        import vix_data
        out = {"vix": vix_data.latest()}
        p = DATA_DIR / "PaperOptions" / "vol_sleeve_paper.json"
        if p.exists():
            s = json.loads(p.read_text(encoding="utf-8"))
            out["sleeve"] = {k: s.get(k) for k in ("asof", "equity", "positions", "pending", "hardStops")}
        try:
            r = _read("vol_sleeve.json")
            wf = r.get("walkForward") or {}
            out["walkForward"] = {"oos": {"oosStart": wf.get("oosStart"), **(wf.get("outOfSample") or {}),
                                          "trades": wf.get("oosTrades"), "exitReasons": wf.get("oosExitReasons")},
                                  "pOosMeanReturnGt0": r.get("pOosMeanReturnGt0"),
                                  "pFullExitsBeatTimeOnly": r.get("pFullExitsBeatTimeOnly"), "honesty": r.get("honesty")}
        except ToolError:
            pass
        return out

    return {"options_chain": op_options_chain, "options_greeks": op_options_greeks, "options_payoff": op_options_payoff,
            "options_pop": op_options_pop, "expected_move": op_expected_move, "top_risk": op_top_risk,
            "vol_regime": op_vol_regime}


def card_section():
    """Compact options / downside / vol-sleeve facts for the analyst master prompt (strategy_card)."""
    import vol_sleeve
    from options_book import OptionsConfig
    from dataclasses import asdict
    out = {"volSleeveRules": vol_sleeve.__doc__.split("Decisions are made")[0].strip(), "volSleeveDefaults": vol_sleeve.DEFAULTS,
           "optionsLimits": asdict(OptionsConfig())}
    for key, name in (("volSleeve", "vol_sleeve.json"), ("overlays", "options_overlays.json"), ("topDetector", "top_detector.json")):
        p = RESEARCH / name
        if not p.exists():
            continue
        try:
            s = json.loads(p.read_text(encoding="utf-8"))
        except ValueError:
            continue
        if key == "volSleeve":
            wf = s.get("walkForward") or {}
            out[key] = {"generated": s.get("generated"), "oos": {"oosStart": wf.get("oosStart"), **(wf.get("outOfSample") or {}),
                                                              "trades": wf.get("oosTrades"), "exitReasons": wf.get("oosExitReasons")},
                        "pOosMeanReturnGt0": s.get("pOosMeanReturnGt0"), "pFullExitsBeatTimeOnly": s.get("pFullExitsBeatTimeOnly"),
                        "honesty": s.get("honesty")}
        elif key == "overlays":
            out[key] = {"generated": s.get("generated"), "honesty": s.get("honesty"),
                        "results": [{"mode": r.get("mode"), "sizing": r.get("sizing"), "oosStart": r.get("oosStart"),
                                     "overlay": r.get("overlayOOS"), "base": r.get("baseOOS"),
                                     "pOverlayBeatsBase": r.get("pOverlayBeatsBase"), "holmP": r.get("holmP")}
                                    for r in s.get("results", [])]}
        else:
            out[key] = {k: s.get(k) for k in ("generated", "verdict", "baseRates", "survivingHolm05", "robustSurvivors", "latest")}
    return out


def install(OPS, ToolError):
    """Called from quant_tools: register the OPTIONS ops and add the options section to strategy_card."""
    OPS.update(ops(ToolError))
    base_card = OPS.get("strategy_card")
    if base_card and not getattr(base_card, "_options_wrapped", False):
        def strategy_card(args, _base=base_card):
            card = _base(args)
            try:
                card["options"] = card_section()
            except Exception as error:  # noqa: BLE001 - the card must still render
                card["options"] = {"error": f"{type(error).__name__}: {error}"[:300]}
            return card
        strategy_card.__doc__ = base_card.__doc__
        strategy_card._options_wrapped = True
        OPS["strategy_card"] = strategy_card

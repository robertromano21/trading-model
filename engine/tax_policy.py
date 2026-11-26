"""Tax-aware adjustments to target weights: near-LT trim hysteresis, loss-harvest swaps, wash-sale avoidance.

Risk always wins: forced exits (stops, stale / delisted write-offs, circuit-breaker flattening) are never deferred.
Policy is a no-op when TaxConfig.enabled is False, and each feature has its own flag so accounting-only mode
(track lots and pay tax, but do not change trades) stays available.

This is a planning model, not tax advice.
"""

from __future__ import annotations

from tax import LotLedger, TaxConfig, days_to_long_term, is_long_term, _as_date


def current_weights(ledger: LotLedger, prices: dict, equity: float):
    weights = {}
    if not equity:
        return weights
    for lot in ledger.open_lots():
        price = prices.get(lot.ticker)
        if price is None or not (price > 0):
            continue
        weights[lot.ticker] = weights.get(lot.ticker, 0.0) + lot.shares * price / equity
    return weights


def _held_weight(ticker, prices, equity, ledger):
    price = prices.get(ticker)
    if not equity or price is None or not (price > 0):
        return 0.0
    return ledger.shares_of(ticker) * price / equity


def adjust_targets(ledger, targets, prices, when, equity, config=None, current=None,
                   forced_exits=None, available=None):
    """Return (adjusted_target_weights, actions).

    `targets` and `prices` are ticker -> float maps. `forced_exits` is a set of tickers that must be
    flattened (stops etc.). `available` is the set of tickers that can be bought (panel columns).
    """
    config = config or (ledger.config if ledger is not None else TaxConfig())
    adjusted = {str(k).upper(): float(v) for k, v in dict(targets or {}).items()}
    actions = []
    if not config.enabled or ledger is None:
        return adjusted, actions
    when = _as_date(when)
    prices = {str(k).upper(): float(v) for k, v in dict(prices or {}).items()
              if v is not None and v == v and v > 0}
    available = set(available) if available is not None else set(prices)
    available = {str(t).upper() for t in available}
    forced = {str(t).upper() for t in (forced_exits or set())}
    current = {str(k).upper(): float(v) for k, v in (current or current_weights(ledger, prices, equity)).items()}

    # 1. Wash-sale avoidance on buys: do not repurchase a name (or a substantially identical ETF)
    #    sold at a loss in the last 30 days. Shift the intended weight onto a substitute when one exists.
    if config.avoid_wash_sales:
        for ticker, weight in list(adjusted.items()):
            held = current.get(ticker, 0.0)
            if weight <= held + 1e-12:
                continue
            if not ledger.recently_sold_at_loss(ticker, when):
                continue
            substitute = config.substitute(ticker, available)
            defer_weight = weight - held
            adjusted[ticker] = held  # do not add shares
            note = {"ticker": ticker, "kind": "tax_wash_avoid", "fromWeight": weight, "toWeight": held,
                    "substitute": substitute}
            if substitute and substitute not in forced:
                adjusted[substitute] = adjusted.get(substitute, current.get(substitute, 0.0)) + defer_weight
                note["kind"] = "tax_wash_avoid_substitute"
            actions.append(note)

    # 2. Trim hysteresis: defer selling lots with gains that are close to turning long-term when the
    #    signal is only a modest trim (target still long, weight cut <= defer_max_signal), not a flatten.
    if config.defer_near_lt:
        for ticker in sorted(set(current) | set(adjusted)):
            if ticker in forced:
                continue
            held = current.get(ticker, 0.0)
            weight = adjusted.get(ticker, 0.0)
            if weight >= held - 1e-12:
                continue
            if weight <= 1e-12:
                continue  # full exit: signal is strong enough; do not defer
            trim = held - weight
            if trim > config.defer_max_signal + 1e-12:
                continue
            price = prices.get(ticker)
            if not price or not equity:
                continue
            # How many shares would be sold, and how many of those are near-LT gains?
            shares_to_sell = trim * equity / price
            try:
                picked = ledger._select_lots(ticker, min(shares_to_sell, ledger.shares_of(ticker)), config.method)
            except ValueError:
                continue
            defer_shares = 0.0
            for lot, take in picked:
                if take <= 0:
                    continue
                gain = take * (price - lot.cost_per_share)
                if gain <= 0:
                    continue
                if is_long_term(lot.holding_period_start(), when):
                    continue
                days = days_to_long_term(lot.holding_period_start(), when)
                if 0 < days <= config.defer_window_days:
                    defer_shares += take
            if defer_shares > 1e-9:
                new_weight = weight + defer_shares * price / equity
                adjusted[ticker] = min(held, new_weight)
                actions.append({"ticker": ticker, "kind": "tax_defer_near_lt", "fromWeight": weight,
                                "toWeight": adjusted[ticker], "deferredShares": defer_shares})

    # 3. Loss harvest: sell material losers we still want to hold and swap into a correlated,
    #    not substantially identical, ETF so economic exposure is kept and the wash rule is not triggered.
    if config.harvest_losses:
        for candidate in ledger.harvest_candidates(prices, when):
            ticker = candidate["ticker"]
            if ticker in forced:
                continue
            # If the strategy is already exiting, the loss will be realized anyway; no swap needed.
            if adjusted.get(ticker, 0.0) <= 1e-12 and current.get(ticker, 0.0) > 0:
                continue
            substitute = candidate.get("substitute") or config.substitute(ticker, available)
            if not substitute or substitute not in available or substitute in forced:
                actions.append({"ticker": ticker, "kind": "tax_harvest_skipped",
                                "reason": "no_substitute", "loss": candidate["unrealized"]})
                continue
            if config.is_substantially_identical(ticker, substitute):
                continue
            if ledger.recently_sold_at_loss(substitute, when):
                actions.append({"ticker": ticker, "kind": "tax_harvest_skipped",
                                "reason": "substitute_wash_window", "substitute": substitute})
                continue
            held = current.get(ticker, 0.0)
            if held <= 1e-12:
                continue
            # Harvest the losing lots only (keep any gain lots).
            harvest_value = 0.0
            harvest_shares = 0.0
            for row in candidate["lots"]:
                harvest_shares += row["shares"]
                harvest_value += row["marketValue"]
            if harvest_shares <= 1e-12 or harvest_value <= 0:
                continue
            remaining_weight = max(0.0, held - harvest_value / equity)
            # If the strategy wanted a smaller weight than remaining gain lots, respect the smaller target.
            target_after = min(adjusted.get(ticker, held), remaining_weight)
            adjusted[ticker] = target_after
            swap_weight = harvest_value / equity
            adjusted[substitute] = adjusted.get(substitute, current.get(substitute, 0.0)) + swap_weight
            actions.append({"ticker": ticker, "kind": "tax_harvest", "toWeight": target_after,
                            "substitute": substitute, "swapWeight": swap_weight,
                            "loss": candidate["unrealized"]})

    # Drop exact zeros for cleanliness, but keep zeros that are explicit exits of a held name.
    cleaned = {}
    for ticker, weight in adjusted.items():
        if abs(weight) < 1e-12:
            if current.get(ticker, 0.0) > 1e-12 or ticker in forced:
                cleaned[ticker] = 0.0
        else:
            cleaned[ticker] = weight
    return cleaned, actions

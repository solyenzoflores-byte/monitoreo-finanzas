"""Option pricing models and helpers."""

from __future__ import annotations

from functools import lru_cache
from math import exp, sqrt
from typing import Dict

import numpy as np


_BINOMIAL_STEPS = 75  # lower step count keeps the tree manageable in real time


def _round_param(value: float, digits: int = 6) -> float:
    """Round float parameters in a consistent way for caching."""

    return round(float(value), digits)


@lru_cache(maxsize=4096)
def _binomial_tree_cached(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    option_type: str,
    steps: int,
) -> float:
    dt = T / steps
    u = exp(sigma * sqrt(dt))
    d = 1.0 / u
    discount = exp(-r * dt)
    growth = exp((r - q) * dt)
    p = (growth - d) / (u - d)
    p = min(max(p, 0.0), 1.0)

    prices = np.zeros((steps + 1, steps + 1), dtype=float)
    for i in range(steps + 1):
        for j in range(i + 1):
            prices[j, i] = S * (u ** (i - j)) * (d ** j)

    option_values = np.zeros_like(prices)
    payoff = np.maximum(0.0, prices[:, steps] - K) if option_type == "call" else np.maximum(0.0, K - prices[:, steps])
    option_values[: steps + 1, steps] = payoff

    for i in range(steps - 1, -1, -1):
        for j in range(i + 1):
            continuation = discount * (p * option_values[j, i + 1] + (1.0 - p) * option_values[j + 1, i + 1])
            intrinsic = prices[j, i] - K if option_type == "call" else K - prices[j, i]
            option_values[j, i] = max(continuation, intrinsic if intrinsic > 0 else 0.0)

    return float(option_values[0, 0])


def binomial_tree_american(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    option_type: str = "call",
    steps: int = _BINOMIAL_STEPS,
) -> float:
    if T <= 0:
        return max(0.0, S - K) if option_type == "call" else max(0.0, K - S)

    key = (
        _round_param(S),
        _round_param(K),
        _round_param(T, 8),
        _round_param(r, 8),
        _round_param(q, 8),
        _round_param(max(sigma, 1e-6), 6),
        option_type,
        int(steps),
    )
    return _binomial_tree_cached(*key)


def american_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    option_type: str = "call",
    steps: int = _BINOMIAL_STEPS,
) -> Dict[str, float]:
    h = 0.01
    price = binomial_tree_american(S, K, T, r, q, sigma, option_type, steps)
    price_up = binomial_tree_american(S + h, K, T, r, q, sigma, option_type, steps)
    price_down = binomial_tree_american(S - h, K, T, r, q, sigma, option_type, steps)
    delta = (price_up - price_down) / (2 * h)
    gamma = (price_up - 2 * price + price_down) / (h**2)

    sigma_shift = 0.01
    vega_price = binomial_tree_american(S, K, T, r, q, sigma + sigma_shift, option_type, steps)
    vega = (vega_price - price) / sigma_shift

    time_shift = min(T, 1 / 365)
    theta_price = binomial_tree_american(S, K, T - time_shift, r, q, sigma, option_type, steps) if T > time_shift else price
    theta = (theta_price - price) / time_shift

    rho_shift = 0.01
    rho_price = binomial_tree_american(S, K, T, r + rho_shift, q, sigma, option_type, steps)
    rho = (rho_price - price) / rho_shift

    return {
        "price": price,
        "delta": delta,
        "gamma": gamma,
        "vega": vega / 100.0,
        "theta": theta,
        "rho": rho / 100.0,
    }


def american_implied_volatility(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    option_type: str,
    steps: int = _BINOMIAL_STEPS,
    tol: float = 1e-3,
    max_iter: int = 60,
) -> float:
    """Simple bisection search for the implied volatility."""

    if market_price <= 0:
        return 0.0

    low, high = 1e-3, 3.0
    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        price = binomial_tree_american(S, K, T, r, q, mid, option_type, steps)
        diff = price - market_price
        if abs(diff) < tol:
            return mid
        if diff > 0:
            high = mid
        else:
            low = mid
    return 0.5 * (low + high)


__all__ = [
    "american_greeks",
    "american_implied_volatility",
    "binomial_tree_american",
]

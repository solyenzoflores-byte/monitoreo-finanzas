"""Fallback datasets used when the remote API is unavailable.

The production application relies on third-party HTTP endpoints that can
occasionally be unreachable when running locally or in automated tests.
To keep the experience usable we expose a small curated dataset that can
be used as a substitute.  The data is intentionally simple but contains
all the columns required by :class:`~core.processing.OptionsProcessor` so
that greeks and implied volatility can still be calculated.
"""

from __future__ import annotations

from datetime import datetime, timedelta


def _option_row(
    option_root: str,
    underlying: str,
    strike: float,
    option_type: str,
    price: float,
    bid: float,
    ask: float,
    volume: int,
    open_interest: int,
    days_to_expiry: int,
) -> dict[str, object]:
    expiration = (datetime.utcnow() + timedelta(days=days_to_expiry)).strftime("%Y-%m-%d")
    suffix = "C" if option_type == "call" else "V"
    symbol = f"{option_root}{int(strike):03d}{suffix}"
    return {
        "symbol": symbol,
        "underlying": underlying,
        "strike": strike,
        "b": bid,
        "a": ask,
        "c": price,
        "v": volume,
        "oi": open_interest,
        "expiration": expiration,
    }


_OPTIONS_DATA = [
    # ALUA (ALU)
    _option_row("ALU", "ALUA", 320.0, "call", 17.5, 17.0, 18.0, 150, 420, 45),
    _option_row("ALU", "ALUA", 320.0, "put", 14.0, 13.5, 14.5, 90, 310, 45),
    _option_row("ALU", "ALUA", 350.0, "call", 9.3, 9.0, 9.6, 120, 275, 75),
    # GGAL (GFG)
    _option_row("GFG", "GGAL", 1200.0, "call", 82.0, 81.0, 83.0, 210, 510, 30),
    _option_row("GFG", "GGAL", 1200.0, "put", 76.5, 75.5, 77.5, 160, 470, 30),
    _option_row("GFG", "GGAL", 1400.0, "call", 58.0, 57.0, 59.0, 110, 265, 60),
    # COME (COM)
    _option_row("COM", "COME", 120.0, "call", 6.2, 6.0, 6.4, 95, 180, 20),
    _option_row("COM", "COME", 120.0, "put", 5.4, 5.2, 5.6, 70, 150, 20),
    _option_row("COM", "COME", 140.0, "call", 3.8, 3.6, 4.0, 60, 120, 40),
]


_STOCKS_DATA = [
    {"symbol": "ALUA", "c": 335.5},
    {"symbol": "GGAL", "c": 1285.0},
    {"symbol": "COME", "c": 132.4},
]


FALLBACK_DATASETS = {
    "https://data912.com/live/arg_options": _OPTIONS_DATA,
    "https://data912.com/live/arg_stocks": _STOCKS_DATA,
}


__all__ = ["FALLBACK_DATASETS"]

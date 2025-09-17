"""High level data processing helpers."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import os

import numpy as np
import pandas as pd

from .data_client import DataClient
from .models import american_greeks, american_implied_volatility


@dataclass(slots=True)
class ProcessorConfig:
    risk_free_rate: float = 0.05
    dividend_yield: float = 0.0
    min_time_to_expiry: float = 1 / 365
    max_workers: Optional[int] = None


class OptionsProcessor:
    def __init__(
        self,
        options_df: pd.DataFrame,
        underlying_prices: Dict[str, float],
        config: Optional[ProcessorConfig] = None,
    ) -> None:
        self.df = options_df.copy() if not options_df.empty else pd.DataFrame()
        self.underlying_prices = underlying_prices
        self.config = config or ProcessorConfig()
        self._prepare_dataframe()

    def _prepare_dataframe(self) -> None:
        if self.df.empty:
            return

        if "option_root" not in self.df.columns and "symbol" in self.df.columns:
            self.df["option_root"] = self.df["symbol"].str.extract(r"^([A-Z]{3})")

        if "underlying" not in self.df.columns:
            prefix_map = {v: k for k, v in self.underlying_prices_map().items()}
            self.df["underlying"] = self.df.get("option_root", pd.Series(dtype=str)).map(prefix_map)

        parsed_strikes = None
        if "symbol" in self.df.columns:
            self.df["otype"] = self.df["symbol"].str[-1].map({"C": "call", "V": "put"})
            parsed = self.df.apply(
                lambda row: self._parse_symbol(row.get("symbol"), row.get("option_root")),
                axis=1,
            )
            self.df["expiration_code"] = parsed.str[0]
            parsed_strikes = pd.to_numeric(parsed.str[1], errors="coerce")

        if "strike" in self.df.columns:
            self.df["K"] = pd.to_numeric(self.df["strike"], errors="coerce")
        else:
            self.df["K"] = pd.NA

        if parsed_strikes is not None:
            self.df["K"] = pd.to_numeric(self.df["K"], errors="coerce").fillna(parsed_strikes)

        if self.df["K"].isna().any():
            extracted = (
                self.df["symbol"].str.extract(r"(\d+(?:[.,]\d+)?)[CV]$")[0]
                if "symbol" in self.df.columns
                else pd.Series(index=self.df.index, dtype=float)
            )
            extracted = pd.to_numeric(extracted.str.replace(",", ".", regex=False), errors="coerce")
            self.df.loc[self.df["K"].isna(), "K"] = extracted[self.df["K"].isna()]

        self.df["K"] = self.df["K"].astype(float)

        self.df["mkt_price"] = self.df.apply(self._market_price, axis=1)
        if "expiration" in self.df.columns:
            self.df["expiration"] = pd.to_datetime(self.df["expiration"], errors="coerce")
        else:
            self.df["expiration"] = pd.NaT

    @staticmethod
    def _parse_symbol(symbol: object, option_root: Optional[str]) -> Tuple[Optional[str], Optional[float]]:
        if not isinstance(symbol, str):
            return None, None
        token = symbol.strip().upper()
        root = (option_root or "").strip().upper()
        if root and token.startswith(root):
            token = token[len(root) :]
        if not token:
            return None, None
        if token[-1] in {"C", "V"}:
            token = token[:-1]
        if not token:
            return None, None
        idx = 0
        while idx < len(token) and token[idx].isalpha():
            idx += 1
        expiration_code = token[:idx] or None
        strike_part = token[idx:].replace(",", ".")
        try:
            strike_value = float(strike_part) if strike_part else None
        except ValueError:
            strike_value = None
        return expiration_code, strike_value

    @staticmethod
    def underlying_prices_map() -> Dict[str, str]:
        return DataClient.TARGET_UNDERLYINGS.copy()

    @staticmethod
    def _market_price(row: pd.Series) -> float:
        bid = row.get("b")
        ask = row.get("a")
        last = row.get("c")
        if pd.notna(bid) and pd.notna(ask) and ask:
            return float((bid + ask) / 2)
        if pd.notna(last) and last:
            return float(last)
        return float("nan")

    def _time_to_expiry(self, expiration: Optional[pd.Timestamp]) -> float:
        if pd.isna(expiration):
            return 90 / 365
        now = datetime.utcnow()
        delta = (expiration.to_pydatetime() - now).days
        if delta <= 0:
            return self.config.min_time_to_expiry
        return max(delta / 365, self.config.min_time_to_expiry)

    def _enrich_row(self, row: pd.Series) -> Optional[Dict[str, float]]:
        underlying = row.get("underlying")
        if not underlying:
            return None
        S = float(self.underlying_prices.get(underlying, 100.0))
        K = float(row.get("K", 0.0))
        option_type = row.get("otype")
        mkt_price = row.get("mkt_price")
        if any(pd.isna(val) for val in (S, K, option_type, mkt_price)):
            return None
        if mkt_price <= 0 or K <= 0:
            return None

        T = self._time_to_expiry(row.get("expiration"))
        if T <= 0:
            return None

        r = self.config.risk_free_rate
        q = self.config.dividend_yield
        iv = american_implied_volatility(mkt_price, S, K, T, r, q, option_type)
        if iv <= 0:
            return None

        greeks = american_greeks(S, K, T, r, q, iv, option_type)
        intrinsic = max(0.0, S - K) if option_type == "call" else max(0.0, K - S)
        time_value = mkt_price - intrinsic

        enriched = row.to_dict()
        enriched.update(
            {
                "S": S,
                "T": T,
                "iv": iv,
                "theo_price": greeks["price"],
                "delta": greeks["delta"],
                "gamma": greeks["gamma"],
                "vega": greeks["vega"],
                "theta": greeks["theta"],
                "rho": greeks["rho"],
                "moneyness": S / K if K else np.nan,
                "time_value": time_value,
            }
        )
        return enriched

    def enrich_with_greeks(self) -> pd.DataFrame:
        if self.df.empty:
            return pd.DataFrame()

        rows: List[Dict[str, float]] = []
        max_workers = self.config.max_workers or min(8, os.cpu_count() or 4)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._enrich_row, row): idx for idx, row in self.df.iterrows()}
            for future in as_completed(futures):
                result = future.result()
                if result:
                    rows.append(result)
        if not rows:
            return pd.DataFrame()
        enriched_df = pd.DataFrame(rows)
        numeric_cols = [
            "S",
            "K",
            "mkt_price",
            "T",
            "iv",
            "delta",
            "gamma",
            "vega",
            "theta",
            "rho",
            "time_value",
            "moneyness",
        ]
        for col in numeric_cols:
            if col in enriched_df.columns:
                enriched_df[col] = pd.to_numeric(enriched_df[col], errors="coerce")
        return enriched_df


__all__ = ["OptionsProcessor", "ProcessorConfig"]

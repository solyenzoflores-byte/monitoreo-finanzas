"""Data acquisition utilities."""

from __future__ import annotations

import threading
import time
from typing import Dict

import pandas as pd
import requests


class DataClient:
    BASES = {
        "Opciones": "https://data912.com/live/arg_options",
        "Bonos": "https://data912.com/live/arg_bonds",
        "MEP": "https://data912.com/live/mep",
        "Acciones": "https://data912.com/live/arg_stocks",
    }

    TARGET_UNDERLYINGS = {
        "ALUA": "ALU",
        "GGAL": "GFG",
        "COME": "COM",
    }

    _session = requests.Session()
    _cache: Dict[str, Dict[str, object]] = {}
    _lock = threading.Lock()

    @classmethod
    def clear_cache(cls) -> None:
        with cls._lock:
            cls._cache.clear()

    @classmethod
    def fetch(cls, url: str, ttl: int = 10, timeout: int = 10) -> pd.DataFrame:
        now = time.time()
        with cls._lock:
            cached = cls._cache.get(url)
            if cached and now - cached["timestamp"] <= ttl:
                return cached["data"].copy()

        try:
            response = cls._session.get(url, timeout=timeout)
            response.raise_for_status()
            payload = response.json()
            df = pd.DataFrame(payload)
        except Exception:
            df = pd.DataFrame()

        with cls._lock:
            cls._cache[url] = {"timestamp": now, "data": df.copy()}
        return df

    @classmethod
    def fetch_filtered_options(cls, ttl: int = 10) -> pd.DataFrame:
        options_df = cls.fetch(cls.BASES["Opciones"], ttl=ttl)
        if options_df.empty or "symbol" not in options_df.columns:
            return pd.DataFrame()

        frames = []
        for stock_symbol, option_prefix in cls.TARGET_UNDERLYINGS.items():
            mask = options_df["symbol"].str.startswith(option_prefix)
            underlying_options = options_df.loc[mask].copy()
            if underlying_options.empty:
                continue
            underlying_options["underlying"] = stock_symbol
            underlying_options["option_root"] = option_prefix
            frames.append(underlying_options)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    @classmethod
    def get_underlying_prices(cls, ttl: int = 10) -> Dict[str, float]:
        stocks_df = cls.fetch(cls.BASES["Acciones"], ttl=ttl)
        if stocks_df.empty:
            return {symbol: 100.0 for symbol in cls.TARGET_UNDERLYINGS.keys()}

        prices: Dict[str, float] = {}
        for stock_symbol in cls.TARGET_UNDERLYINGS.keys():
            stock_data = stocks_df[stocks_df["symbol"] == stock_symbol]
            if not stock_data.empty:
                if "c" in stock_data.columns:
                    price = stock_data["c"].iloc[0]
                elif "last" in stock_data.columns:
                    price = stock_data["last"].iloc[0]
                else:
                    price = float(stock_data.iloc[0].get("price", 100.0))
                prices[stock_symbol] = float(price) if price else 100.0
            else:
                prices[stock_symbol] = 100.0
        return prices


__all__ = ["DataClient"]

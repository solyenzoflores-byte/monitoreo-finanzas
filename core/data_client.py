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

    @classmethod
    def get_exchange_rates(cls, ttl: int = 10) -> Dict[str, float]:
        """Fetch and normalise exchange rate information.

        The remote endpoint can expose multiple representations of the
        exchange rate (e.g. MEP, CCL) and the field naming is not entirely
        consistent.  The method therefore performs a best-effort mapping by
        looking for well-known column names first and falling back to the
        first numerical value found on each row.  The resulting dictionary
        can be used to populate widgets that require one or more reference
        FX prices.
        """

        url = cls.BASES.get("MEP")
        if not url:
            return {}

        fx_df = cls.fetch(url, ttl=ttl)
        if fx_df.empty:
            return {}

        rates: Dict[str, float] = {}
        lowered_columns = {col.lower(): col for col in fx_df.columns if isinstance(col, str)}

        preferred_columns = [
            ("MEP", "mep"),
            ("CCL", "ccl"),
            ("Blue", "blue"),
            ("Oficial", "oficial"),
            ("Promedio", "promedio"),
        ]
        for label, column_key in preferred_columns:
            column_name = lowered_columns.get(column_key)
            if not column_name:
                continue
            series = pd.to_numeric(fx_df[column_name], errors="coerce").dropna()
            if not series.empty:
                rates[label] = float(series.mean())

        priority_fields = {
            "mep",
            "usd_mep",
            "usd",
            "dolar",
            "dólar",
            "price",
            "precio",
            "close",
            "last",
            "c",
            "value",
            "venta",
            "sell",
            "ask",
        }

        for idx, row in fx_df.iterrows():
            symbol = None
            for candidate in ("symbol", "ticker", "name", "title", "moneda", "instrumento", "tipo"):
                value = row.get(candidate)
                if isinstance(value, str) and value.strip():
                    symbol = value.strip()
                    break
            if not symbol:
                symbol = f"rate_{idx + 1}"

            price: float | None = None
            for column in row.index:
                if not isinstance(column, str):
                    continue
                if column.lower() not in priority_fields:
                    continue
                numeric_value = pd.to_numeric([row[column]], errors="coerce")[0]
                if pd.notna(numeric_value) and numeric_value > 0:
                    price = float(numeric_value)
                    break

            if price is None:
                numeric_row = pd.to_numeric(row, errors="coerce").dropna()
                if not numeric_row.empty:
                    price = float(numeric_row.iloc[0])

            if price is None or price <= 0:
                continue

            if symbol in rates:
                rates[symbol] = (rates[symbol] + price) / 2.0
            else:
                rates[symbol] = price

        if rates:
            has_mep_key = any("mep" in key.lower() for key in rates.keys())
            if not has_mep_key:
                average_rate = sum(rates.values()) / len(rates)
                rates.setdefault("MEP", float(average_rate))

        return rates


__all__ = ["DataClient"]

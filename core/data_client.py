"""Data acquisition utilities with graceful fallbacks."""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


LOGGER = logging.getLogger(__name__)


class DataClient:
    SOURCES = {
        "Opciones": ("https://data912.com/live/arg_options",),
        "Bonos": ("https://data912.com/live/arg_bonds",),
        "MEP": ("https://data912.com/live/mep",),
        "Acciones": ("https://data912.com/live/arg_stocks",),
    }

    FALLBACK_FILES = {
        "Opciones": Path(__file__).resolve().parent / "sample_data" / "options.json",
        "Acciones": Path(__file__).resolve().parent / "sample_data" / "stocks.json",
    }

    TARGET_UNDERLYINGS = {
        "ALUA": "ALU",
        "GGAL": "GFG",
        "COME": "COM",
    }

    _session = requests.Session()
    _session.headers.update(
        {
            "User-Agent": "MonitoreoFinanzas/1.0 (+https://github.com/monitoreo-finanzas)",
            "Accept": "application/json",
        }
    )
    _retry = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    _session.mount("https://", HTTPAdapter(max_retries=_retry))
    _session.mount("http://", HTTPAdapter(max_retries=_retry))

    _cache: Dict[str, Dict[str, object]] = {}
    _last_fetch_meta: Dict[str, Dict[str, object]] = {}
    _lock = threading.Lock()

    @classmethod
    def clear_cache(cls) -> None:
        with cls._lock:
            cls._cache.clear()
            cls._last_fetch_meta.clear()

    @classmethod
    def _iter_sources(cls, key: str) -> Iterable[str]:
        base_urls = list(cls.SOURCES.get(key, ()))
        env_key = f"MONITOREO_{key.upper()}_SOURCES"
        env_urls = os.getenv(env_key)
        if env_urls:
            base_urls.extend([item.strip() for item in env_urls.split(",") if item.strip()])
        return base_urls

    @classmethod
    def _record_meta(cls, key: str, **meta: object) -> None:
        with cls._lock:
            cls._last_fetch_meta[key] = meta

    @classmethod
    def _fetch_http(cls, url: str, ttl: int, timeout: int) -> Tuple[pd.DataFrame, Optional[str]]:
        now = time.time()
        with cls._lock:
            cached = cls._cache.get(url)
            if cached and now - cached["timestamp"] <= ttl:
                return cached["data"].copy(), None

        error_message: Optional[str] = None
        try:
            response = cls._session.get(url, timeout=timeout)
            response.raise_for_status()
            payload = response.json()
            df = pd.DataFrame(payload)
        except Exception as exc:  # pragma: no cover - defensive network handling
            error_message = str(exc)
            df = pd.DataFrame()

        with cls._lock:
            cls._cache[url] = {"timestamp": now, "data": df.copy()}

        return df, error_message

    @classmethod
    def fetch_dataset(cls, key: str, ttl: int = 10, timeout: int = 10) -> pd.DataFrame:
        last_error: Optional[str] = None
        for url in cls._iter_sources(key):
            df, error = cls._fetch_http(url, ttl=ttl, timeout=timeout)
            if not df.empty:
                cls._record_meta(key, source=url, fallback=False, error=None)
                return df
            if error:
                last_error = error
                LOGGER.warning("Failed to fetch %s from %s: %s", key, url, error)

        fallback_path = cls.FALLBACK_FILES.get(key)
        if fallback_path and fallback_path.exists():
            try:
                df = pd.read_json(fallback_path)
                cls._record_meta(
                    key,
                    source=str(fallback_path),
                    fallback=True,
                    error=last_error,
                )
                LOGGER.info("Loaded %s data from local fallback %s", key, fallback_path)
                return df
            except ValueError as exc:
                last_error = str(exc)
                LOGGER.error("Failed to load local fallback for %s: %s", key, exc)

        cls._record_meta(key, source=None, fallback=False, error=last_error)
        return pd.DataFrame()

    @classmethod
    def get_last_fetch_meta(cls, key: str) -> Dict[str, object]:
        with cls._lock:
            return cls._last_fetch_meta.get(key, {}).copy()

    @classmethod
    def fetch_filtered_options(cls, ttl: int = 10) -> pd.DataFrame:
        options_df = cls.fetch_dataset("Opciones", ttl=ttl)
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
        stocks_df = cls.fetch_dataset("Acciones", ttl=ttl)
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

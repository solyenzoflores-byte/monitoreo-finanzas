"""Database utilities for the monitoring project.

This module centralises all SQLite interactions so that both the
Streamlit application and background workers can reuse the same logic.
The implementation favours batch inserts in order to reduce the amount
of time that the UI thread spends waiting for disk operations.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterator, Optional

import pandas as pd
import sqlite3


@dataclass(slots=True)
class DatabaseConfig:
    """Configuration container for :class:`DatabaseManager`."""

    path: str = "options_data.db"
    detect_types: int = sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES


class DatabaseManager:
    """Simple wrapper around SQLite with helper methods.

    The class uses context managers so that connections are properly
    released even if an exception occurs.  It also exposes bulk insert
    helpers to keep write operations efficient when many option rows
    need to be persisted at once.
    """

    def __init__(self, config: Optional[DatabaseConfig] = None) -> None:
        self.config = config or DatabaseConfig()
        self._initialise_schema()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.config.path, detect_types=self.config.detect_types)
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _initialise_schema(self) -> None:
        with self.connect() as connection:
            cursor = connection.cursor()
            cursor.executescript(
                """
                CREATE TABLE IF NOT EXISTS options_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    symbol TEXT,
                    underlying TEXT,
                    option_type TEXT,
                    strike REAL,
                    expiration DATE,
                    last_price REAL,
                    bid REAL,
                    ask REAL,
                    volume INTEGER,
                    open_interest INTEGER,
                    iv REAL,
                    delta REAL,
                    gamma REAL,
                    vega REAL,
                    theta REAL,
                    rho REAL
                );

                CREATE TABLE IF NOT EXISTS underlying_prices (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    symbol TEXT,
                    price REAL,
                    volume INTEGER
                );

                CREATE TABLE IF NOT EXISTS saved_strategies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    strategy_data TEXT,
                    created_at DATETIME
                );
                """
            )

    def save_options_data(self, df: pd.DataFrame, underlying_symbol: str) -> int:
        """Persist the processed options rows for a specific underlying.

        Parameters
        ----------
        df:
            Enriched dataframe containing greeks and market information.
        underlying_symbol:
            The underlying ticker used to label the records inside the
            database.

        Returns
        -------
        int
            Number of rows inserted.
        """

        if df.empty:
            return 0

        timestamp = datetime.utcnow()
        columns = [
            "symbol",
            "otype",
            "K",
            "expiration",
            "mkt_price",
            "b",
            "a",
            "v",
            "oi",
            "iv",
            "delta",
            "gamma",
            "vega",
            "theta",
            "rho",
        ]
        available = [c for c in columns if c in df.columns]
        records = []
        for _, row in df[available].iterrows():
            expiration_value = row.get("expiration")
            if pd.isna(expiration_value):
                expiration_out = None
            elif isinstance(expiration_value, pd.Timestamp):
                expiration_out = expiration_value.to_pydatetime()
            else:
                expiration_out = expiration_value
            records.append(
                (
                    timestamp,
                    row.get("symbol", ""),
                    underlying_symbol,
                    row.get("otype", ""),
                    row.get("K", 0.0),
                    expiration_out,
                    row.get("mkt_price", 0.0),
                    row.get("b", 0.0),
                    row.get("a", 0.0),
                    row.get("v", 0),
                    row.get("oi", 0),
                    row.get("iv", 0.0),
                    row.get("delta", 0.0),
                    row.get("gamma", 0.0),
                    row.get("vega", 0.0),
                    row.get("theta", 0.0),
                    row.get("rho", 0.0),
                )
            )

        with self.connect() as connection:
            connection.executemany(
                """
                INSERT INTO options_data (
                    timestamp, symbol, underlying, option_type, strike, expiration,
                    last_price, bid, ask, volume, open_interest, iv,
                    delta, gamma, vega, theta, rho
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                records,
            )
        return len(records)

    def save_underlying_prices(self, price_dict: Dict[str, float]) -> int:
        if not price_dict:
            return 0
        timestamp = datetime.utcnow()
        rows = [
            (timestamp, symbol, float(price) if price is not None else 0.0, 0)
            for symbol, price in price_dict.items()
        ]
        with self.connect() as connection:
            connection.executemany(
                "INSERT INTO underlying_prices (timestamp, symbol, price, volume) VALUES (?, ?, ?, ?)",
                rows,
            )
        return len(rows)

    def get_historical_iv(self, underlying: str, days: int = 30) -> pd.DataFrame:
        query = """
            SELECT timestamp, AVG(iv) as iv
            FROM options_data
            WHERE underlying = ? AND timestamp >= datetime('now', ?)
            GROUP BY timestamp
            ORDER BY timestamp
        """
        window = f"-{int(days)} days"
        with self.connect() as connection:
            return pd.read_sql_query(query, connection, params=(underlying, window))

    def get_volume_open_interest(self, underlying: str, days: int = 30) -> pd.DataFrame:
        query = """
            SELECT timestamp, SUM(volume) as volume, SUM(open_interest) as open_interest
            FROM options_data
            WHERE underlying = ? AND timestamp >= datetime('now', ?)
            GROUP BY timestamp
            ORDER BY timestamp
        """
        window = f"-{int(days)} days"
        with self.connect() as connection:
            return pd.read_sql_query(query, connection, params=(underlying, window))

    def get_underlying_history(self, symbol: str, days: int = 30) -> pd.DataFrame:
        query = """
            SELECT timestamp, price, volume
            FROM underlying_prices
            WHERE symbol = ? AND timestamp >= datetime('now', ?)
            ORDER BY timestamp
        """
        window = f"-{int(days)} days"
        with self.connect() as connection:
            return pd.read_sql_query(query, connection, params=(symbol, window))


__all__ = ["DatabaseConfig", "DatabaseManager"]

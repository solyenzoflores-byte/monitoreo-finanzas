"""Background script to persist market data into SQLite."""

from __future__ import annotations

import argparse
import logging
import sys
import time

from core.data_client import DataClient
from core.database import DatabaseConfig, DatabaseManager
from core.processing import OptionsProcessor, ProcessorConfig


LOGGER = logging.getLogger("historical_ingestor")


def ingest_once(
    database: DatabaseManager,
    processor_config: ProcessorConfig,
    ttl: int = 10,
) -> None:
    LOGGER.info("Fetching latest market data")
    options_df = DataClient.fetch_filtered_options(ttl=ttl)
    underlying_prices = DataClient.get_underlying_prices(ttl=ttl)
    if options_df.empty:
        LOGGER.warning("No option data received from remote service")
        return

    processor = OptionsProcessor(options_df, underlying_prices, processor_config)
    enriched_df = processor.enrich_with_greeks()
    if enriched_df.empty:
        LOGGER.warning("Processed dataframe is empty; skipping insert")
        return

    inserted_prices = database.save_underlying_prices(underlying_prices)
    LOGGER.info("Saved %s underlying price entries", inserted_prices)

    for underlying in DataClient.TARGET_UNDERLYINGS.keys():
        subset = enriched_df[enriched_df["underlying"] == underlying]
        if subset.empty:
            continue
        inserted = database.save_options_data(subset, underlying)
        LOGGER.info("Saved %s option rows for %s", inserted, underlying)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Persist option market data into SQLite")
    parser.add_argument("--db", default="options_data.db", help="Path to the SQLite database file")
    parser.add_argument("--risk-free", type=float, default=5.0, help="Risk free rate in percent")
    parser.add_argument("--dividend", type=float, default=0.0, help="Dividend yield in percent")
    parser.add_argument("--ttl", type=int, default=10, help="Cache TTL used for HTTP requests")
    parser.add_argument(
        "--interval",
        type=int,
        default=0,
        help="If provided, the script will run continuously sleeping this many seconds between cycles",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    database = DatabaseManager(DatabaseConfig(path=args.db))
    processor_config = ProcessorConfig(
        risk_free_rate=args.risk_free / 100.0,
        dividend_yield=args.dividend / 100.0,
    )

    if args.interval <= 0:
        ingest_once(database, processor_config, ttl=args.ttl)
        return 0

    LOGGER.info("Starting continuous ingestion loop with %s second interval", args.interval)
    try:
        while True:
            start = time.time()
            ingest_once(database, processor_config, ttl=args.ttl)
            elapsed = time.time() - start
            sleep_for = max(args.interval - elapsed, 0)
            LOGGER.debug("Cycle finished in %.2fs, sleeping %.2fs", elapsed, sleep_for)
            time.sleep(sleep_for)
    except KeyboardInterrupt:
        LOGGER.info("Interrupted by user; exiting")
        return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

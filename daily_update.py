"""
Daily incremental OHLCV updater.

Meant to be run once a day by a Render Cron Job (see render.yaml:
`daily-stock-update`). For every symbol already stored in BOTH table sets
(Watchlist1 -> symbols/ohlcv_data, Nasdaq300 -> us_symbols/us_ohlcv_data),
it fetches only the days missing since that symbol's last stored date and
upserts them. This is deliberately incremental (not a full history
re-download) so the daily job stays fast and light on yfinance rate limits.

Run manually for testing:
    python daily_update.py
    python daily_update.py --target watchlist   # only Watchlist1
    python daily_update.py --target nasdaq300    # only Nasdaq300
"""

import os
import sys
import time
import argparse
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import psycopg2
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()  # no-op on Render (env vars are injected directly), used for local runs

# ---------------------------------------------------------------------------
# DB config — from environment only, no hardcoded secrets
# ---------------------------------------------------------------------------
DB_HOST = os.environ["DB_HOST"]
DB_PORT = os.environ.get("DB_PORT", "5432")
DB_NAME = os.environ.get("DB_NAME", "postgres")
DB_USER = os.environ["DB_USER"]
DB_PASSWORD = os.environ["DB_PASSWORD"]

TABLE_SETS = {
    "watchlist": {"symbols": "symbols", "ohlcv": "ohlcv_data"},
    "nasdaq300": {"symbols": "us_symbols", "ohlcv": "us_ohlcv_data"},
}

WORKERS = int(os.environ.get("DAILY_UPDATE_WORKERS", "6"))
RETRIES = 4


def get_connection(retries=5, delay=2):
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            conn = psycopg2.connect(
                host=DB_HOST, port=DB_PORT, dbname=DB_NAME,
                user=DB_USER, password=DB_PASSWORD, connect_timeout=10,
            )
            conn.autocommit = False
            return conn
        except psycopg2.OperationalError as e:
            last_err = e
            wait = delay * attempt
            print(f"  DB connect failed (attempt {attempt}/{retries}): {e}. Retrying in {wait}s...")
            time.sleep(wait)
    raise RuntimeError(f"Could not connect to database after {retries} attempts: {last_err}")


def get_symbols_with_last_date(cur, symbols_table, ohlcv_table):
    """Returns {symbol: (symbol_id, last_date_or_None)} for every symbol
    currently stored in this table set."""
    cur.execute(f"""
        SELECT s.id, s.symbol, MAX(o.date) AS last_date
        FROM {symbols_table} s
        LEFT JOIN {ohlcv_table} o ON o.symbol_id = s.id
        GROUP BY s.id, s.symbol
        ORDER BY s.symbol;
    """)
    return {row[1]: (row[0], row[2]) for row in cur.fetchall()}


def fetch_incremental(symbol, start_date, retries=RETRIES, base_delay=2):
    """Fetch OHLCV from start_date (inclusive) through today. If start_date
    is None, falls back to a longer default lookback (new symbol edge case)."""
    start = start_date.isoformat() if start_date else "2020-01-01"
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            df = yf.download(
                symbol,
                start=start,
                progress=False,
                auto_adjust=False,
                timeout=20,
            )
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            df = df.reset_index()
            return symbol, df, None
        except Exception as e:
            last_err = e
            wait = base_delay * (2 ** (attempt - 1))
            print(f"  [{symbol}] error (attempt {attempt}/{retries}): {e}. Retrying in {wait}s...")
            time.sleep(wait)
    return symbol, None, last_err


def store_ohlcv(cur, ohlcv_table, symbol_id, df):
    rows = [
        (
            symbol_id,
            row["Date"].date(),
            float(row["Open"]),
            float(row["High"]),
            float(row["Low"]),
            float(row["Close"]),
            int(row["Volume"]),
        )
        for _, row in df.iterrows()
    ]
    if not rows:
        return
    cur.executemany(
        f"""
        INSERT INTO {ohlcv_table} (symbol_id, date, open, high, low, close, volume)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (symbol_id, date) DO UPDATE SET
            open = EXCLUDED.open,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            close = EXCLUDED.close,
            volume = EXCLUDED.volume;
        """,
        rows,
    )


def update_table_set(target_name):
    tables = TABLE_SETS[target_name]
    symbols_table, ohlcv_table = tables["symbols"], tables["ohlcv"]

    conn = get_connection()
    cur = conn.cursor()

    symbol_map = get_symbols_with_last_date(cur, symbols_table, ohlcv_table)
    if not symbol_map:
        print(f"[{target_name}] No symbols found in {symbols_table}, skipping.")
        cur.close()
        conn.close()
        return

    print(f"[{target_name}] Updating {len(symbol_map)} symbols in {ohlcv_table} ...")

    today = datetime.date.today()
    fetch_jobs = []
    for symbol, (symbol_id, last_date) in symbol_map.items():
        # Nothing to do if we already have today's (or a future/odd) row.
        if last_date is not None and last_date >= today:
            continue
        start_date = (last_date + datetime.timedelta(days=1)) if last_date else None
        fetch_jobs.append((symbol, symbol_id, start_date))

    if not fetch_jobs:
        print(f"[{target_name}] Already up to date.")
        cur.close()
        conn.close()
        return

    ok, fail = 0, 0
    failed_symbols = []

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {
            executor.submit(fetch_incremental, symbol, start_date): (symbol, symbol_id)
            for symbol, symbol_id, start_date in fetch_jobs
        }

        for future in as_completed(futures):
            symbol, symbol_id = futures[future]
            _, df, err = future.result()

            if err is not None:
                print(f"  [{symbol}] FAILED: {err}")
                fail += 1
                failed_symbols.append(symbol)
                continue

            if df is None or df.empty:
                # No new rows today (market holiday, weekend backfill, etc.) — not an error.
                ok += 1
                continue

            for db_attempt in range(1, 4):
                try:
                    store_ohlcv(cur, ohlcv_table, symbol_id, df)
                    conn.commit()
                    ok += 1
                    print(f"  [{symbol}] +{len(df)} row(s) up to {df['Date'].max().date()}")
                    break
                except psycopg2.OperationalError as e:
                    print(f"  [{symbol}] DB write failed (attempt {db_attempt}/3): {e}. Reconnecting...")
                    conn.close()
                    conn = get_connection()
                    cur = conn.cursor()
                    time.sleep(2 * db_attempt)
            else:
                fail += 1
                failed_symbols.append(symbol)

    cur.close()
    conn.close()

    print(f"[{target_name}] Done. {ok} ok, {fail} failed.")
    if failed_symbols:
        print(f"[{target_name}] Failed symbols: {', '.join(failed_symbols)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        choices=["watchlist", "nasdaq300", "both"],
        default="both",
        help="which table set(s) to update (default: both)",
    )
    args = parser.parse_args()

    targets = ["watchlist", "nasdaq300"] if args.target == "both" else [args.target]

    exit_code = 0
    for target in targets:
        try:
            update_table_set(target)
        except Exception as e:
            print(f"[{target}] FATAL: {e}")
            exit_code = 1

    sys.exit(exit_code)


if __name__ == "__main__":
    main()

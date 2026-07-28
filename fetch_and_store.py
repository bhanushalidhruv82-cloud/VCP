import os
import time
import argparse
import signal
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import psycopg2
from psycopg2.extras import execute_values
import pandas as pd
import yfinance as yf
from requests.exceptions import RequestException

# ---------------------------------------------------------------------------
# Supabase connection
# ---------------------------------------------------------------------------
DB_HOST = os.environ["DB_HOST"]
DB_PORT = os.environ.get("DB_PORT", "5432")
DB_NAME = os.environ.get("DB_NAME", "postgres")
DB_USER = os.environ["DB_USER"]
DB_PASSWORD = os.environ["DB_PASSWORD"]  # set via env var, never hardcode

TABLE_SETS = {
    "watchlist": {"symbols": "symbols", "ohlcv": "ohlcv_data"},
    "nasdaq300": {"symbols": "us_symbols", "ohlcv": "us_ohlcv_data"},
}

DDL = """
CREATE TABLE IF NOT EXISTS symbols (
    id SERIAL PRIMARY KEY,
    symbol TEXT UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS ohlcv_data (
    id SERIAL PRIMARY KEY,
    symbol_id INTEGER NOT NULL REFERENCES symbols(id) ON DELETE CASCADE,
    date DATE NOT NULL,
    open DOUBLE PRECISION,
    high DOUBLE PRECISION,
    low DOUBLE PRECISION,
    close DOUBLE PRECISION,
    volume BIGINT,
    UNIQUE (symbol_id, date)
);

CREATE TABLE IF NOT EXISTS us_symbols (
    id SERIAL PRIMARY KEY,
    symbol TEXT UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS us_ohlcv_data (
    id SERIAL PRIMARY KEY,
    symbol_id INTEGER NOT NULL REFERENCES us_symbols(id) ON DELETE CASCADE,
    date DATE NOT NULL,
    open DOUBLE PRECISION,
    high DOUBLE PRECISION,
    low DOUBLE PRECISION,
    close DOUBLE PRECISION,
    volume BIGINT,
    UNIQUE (symbol_id, date)
);

CREATE INDEX IF NOT EXISTS idx_ohlcv_data_symbol_date ON ohlcv_data(symbol_id, date);
CREATE INDEX IF NOT EXISTS idx_us_ohlcv_data_symbol_date ON us_ohlcv_data(symbol_id, date);
"""

STOP = False  # set by Ctrl+C handler so in-flight work can wind down cleanly


def handle_sigint(signum, frame):
    global STOP
    if STOP:
        print("\nForce exiting.")
        sys.exit(1)
    print("\nCtrl+C received — finishing current symbol(s) then stopping. Press again to force quit.")
    STOP = True


signal.signal(signal.SIGINT, handle_sigint)


# ---------------------------------------------------------------------------
# DB helpers (with connection retry)
# ---------------------------------------------------------------------------
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


def ensure_tables(conn):
    with conn.cursor() as cur:
        cur.execute(DDL)
    conn.commit()


def get_or_create_symbol_id(cur, symbols_table, symbol):
    cur.execute(f"SELECT id FROM {symbols_table} WHERE symbol = %s", (symbol,))
    row = cur.fetchone()
    if row:
        return row[0]
    cur.execute(f"INSERT INTO {symbols_table} (symbol) VALUES (%s) RETURNING id", (symbol,))
    return cur.fetchone()[0]


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
    execute_values(
        cur,
        f"""
        INSERT INTO {ohlcv_table} (symbol_id, date, open, high, low, close, volume)
        VALUES %s
        ON CONFLICT (symbol_id, date) DO UPDATE SET
            open = EXCLUDED.open,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            close = EXCLUDED.close,
            volume = EXCLUDED.volume;
        """,
        rows,
        page_size=1000,
    )


# ---------------------------------------------------------------------------
# Fetching (network-bound, runs in worker threads, retried independently)
# ---------------------------------------------------------------------------
def fetch_ohlcv(symbol, start, end, retries=4, base_delay=2):
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            if start is None:
                df = yf.download(
                    symbol,
                    start="2020-01-01",
                    end="2026-07-17",
                    progress=False,
                    auto_adjust=False,
                    timeout=20,
                )
            else:
                df = yf.download(symbol, start=start, end=end, progress=False,
                                  auto_adjust=False, timeout=20)

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            df = df.reset_index()
            return symbol, df, None
        except (RequestException, ConnectionError, TimeoutError, OSError) as e:
            last_err = e
            wait = base_delay * (2 ** (attempt - 1))  # exponential backoff
            print(f"  [{symbol}] network error (attempt {attempt}/{retries}): {e}. Retrying in {wait}s...")
            time.sleep(wait)
        except Exception as e:
            # Non-network error (bad symbol, parsing issue, etc.) - don't retry forever
            last_err = e
            break
    return symbol, None, last_err


def load_symbols_from_file(path):
    with open(path) as f:
        return [line.strip().upper() for line in f if line.strip()]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("symbols", nargs="*", help="e.g. AAPL MSFT NVDA")
    parser.add_argument("--file", help="path to a text file with one symbol per line")
    parser.add_argument("--start", default=None, help="default: full available history")
    parser.add_argument("--end", default=None)
    parser.add_argument(
        "--target",
        choices=["watchlist", "nasdaq300"],
        default="watchlist",
        help="which table pair to store into: watchlist -> symbols/ohlcv_data, "
             "nasdaq300 -> us_symbols/us_ohlcv_data",
    )
    parser.add_argument("--skip-existing", action="store_true",
                         help="skip symbols that already have rows in the target table")
    parser.add_argument("--delay", type=float, default=0.0,
                         help="seconds to sleep between DB writes (politeness delay)")
    parser.add_argument("--workers", type=int, default=8,
                         help="number of symbols to fetch concurrently (default: 8)")
    args = parser.parse_args()

    symbols = list(args.symbols)
    if args.file:
        symbols += load_symbols_from_file(args.file)
    if not symbols:
        parser.error("provide symbols as arguments or via --file")
    symbols = [s.upper() for s in symbols]

    tables = TABLE_SETS[args.target]
    symbols_table = tables["symbols"]
    ohlcv_table = tables["ohlcv"]

    conn = get_connection()
    ensure_tables(conn)
    cur = conn.cursor()

    if args.skip_existing:
        cur.execute(f"""
            SELECT s.symbol FROM {symbols_table} s
            JOIN {ohlcv_table} o ON o.symbol_id = s.id
            GROUP BY s.symbol
        """)
        existing = {r[0] for r in cur.fetchall()}
        before = len(symbols)
        symbols = [s for s in symbols if s not in existing]
        print(f"Skipping {before - len(symbols)} symbols already in {ohlcv_table}")

    if not symbols:
        print("Nothing to fetch.")
        cur.close()
        conn.close()
        return

    total = len(symbols)
    ok_count = 0
    fail_count = 0
    failed_symbols = []

    print(f"Fetching {total} symbols with {args.workers} workers -> {ohlcv_table} ...")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(fetch_ohlcv, sym, args.start, args.end): sym
            for sym in symbols
        }

        for future in as_completed(futures):
            symbol, df, err = future.result()

            if err is not None:
                print(f"  [{symbol}] FAILED after retries: {err}")
                fail_count += 1
                failed_symbols.append(symbol)
                continue

            if df is None or df.empty:
                print(f"  [{symbol}] no data, skipping")
                fail_count += 1
                failed_symbols.append(symbol)
                continue

            # DB write with retry in case of transient connection drop
            written = False
            for db_attempt in range(1, 4):
                try:
                    symbol_id = get_or_create_symbol_id(cur, symbols_table, symbol)
                    store_ohlcv(cur, ohlcv_table, symbol_id, df)
                    conn.commit()
                    ok_count += 1
                    written = True
                    print(f"  [{symbol}] stored {len(df)} rows "
                          f"({df['Date'].min().date()} to {df['Date'].max().date()}) "
                          f"[{ok_count + fail_count}/{total}]")
                    break
                except psycopg2.OperationalError as e:
                    print(f"  [{symbol}] DB write failed (attempt {db_attempt}/3): {e}. Reconnecting...")
                    try:
                        conn.close()
                    except Exception:
                        pass
                    conn = get_connection()
                    cur = conn.cursor()
                    time.sleep(2 * db_attempt)

            if not written:
                print(f"  [{symbol}] giving up after DB retries")
                fail_count += 1
                failed_symbols.append(symbol)

            if args.delay:
                time.sleep(args.delay)

            if STOP:
                print("Stopping early due to Ctrl+C. In-flight fetches will still be drained.")
                break

    cur.close()
    conn.close()

    print(f"\nDone. {ok_count} succeeded, {fail_count} failed out of {total}.")
    if failed_symbols:
        print("Failed symbols:", ", ".join(failed_symbols))
        print("Re-run with --skip-existing to retry only the missing ones.")


if __name__ == "__main__":
    main()
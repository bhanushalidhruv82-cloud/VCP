"""
Fetch full daily historical OHLCV data for a custom list of tickers (read
from list_tickers.txt, one symbol per line — used as-is, no exchange suffix)
using yfinance and store it in the Supabase Postgres database (same project
used by fetch_and_store.py / app3.py). Optionally also saves a local CSV
copy of each symbol as a backup.

Install: pip install yfinance psycopg2-binary --break-system-packages
"""

import os
import time
import yfinance as yf
import pandas as pd
import psycopg2

# ---------------------------------------------------------------------------
# Supabase connection (same project as fetch_and_store.py / app3.py)
# ---------------------------------------------------------------------------
DB_HOST = os.environ["DB_HOST"]
DB_PORT = os.environ.get("DB_PORT", "5432")
DB_NAME = os.environ.get("DB_NAME", "postgres")
DB_USER = os.environ["DB_USER"]
DB_PASSWORD = os.environ["DB_PASSWORD"]  # set via env var, never hardcode

# This is its own table pair, separate from the "watchlist" (symbols/ohlcv_data),
# "nasdaq300" (us_symbols/us_ohlcv_data), and "nifty100" table sets used elsewhere.
SYMBOLS_TABLE = "symbols"
OHLCV_TABLE = "ohlcv_data"

DDL = f"""
CREATE TABLE IF NOT EXISTS {SYMBOLS_TABLE} (
    id SERIAL PRIMARY KEY,
    symbol TEXT UNIQUE  NOT NULL
);

CREATE TABLE IF NOT EXISTS {OHLCV_TABLE} (
    id SERIAL PRIMARY KEY,
    symbol_id INTEGER NOT NULL REFERENCES {SYMBOLS_TABLE}(id) ON DELETE CASCADE,
    date DATE NOT NULL,
    open DOUBLE PRECISION,
    high DOUBLE PRECISION,
    low DOUBLE PRECISION,
    close DOUBLE PRECISION,
    volume BIGINT,
    UNIQUE(symbol_id, date)
);

CREATE INDEX IF NOT EXISTS idx_{OHLCV_TABLE}_symbol_date ON {OHLCV_TABLE}(symbol_id, date);
"""

# --- Config ---
# Ticker list is read from this file, one symbol per line, used as-is
# (no ".NS" or other suffix added).
TICKERS_FILE = "list_tickers.txt"


def load_tickers(path):
    with open(path) as f:
        return [line.strip().upper() for line in f if line.strip()]


TICKER_LIST = load_tickers(TICKERS_FILE)
TICKERS = {sym: sym for sym in TICKER_LIST}

SAVE_CSV = True                    # also keep a local CSV backup per symbol
OUTPUT_DIR = "list_stock_data"
START_DATE = "2026-07-24"
END_DATE = None   # keep None for "till today"
INTERVAL = "1d"      # daily data
PAUSE_SECONDS = 1      # small delay between requests to avoid rate limiting


def get_connection():
    return psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME,
        user=DB_USER, password=DB_PASSWORD, connect_timeout=10,
    )


def ensure_tables(conn):
    with conn.cursor() as cur:
        cur.execute(DDL)
    conn.commit()


def get_or_create_symbol_id(cur, symbol):
    cur.execute(f"SELECT id FROM {SYMBOLS_TABLE} WHERE symbol = %s", (symbol,))
    row = cur.fetchone()
    if row:
        return row[0]
    cur.execute(f"INSERT INTO {SYMBOLS_TABLE} (symbol) VALUES (%s) RETURNING id", (symbol,))
    return cur.fetchone()[0]


def store_ohlcv(cur, symbol_id, df):
    rows = [
        (
            symbol_id,
            idx.date() if hasattr(idx, "date") else idx,
            float(row["Open"]),
            float(row["High"]),
            float(row["Low"]),
            float(row["Close"]),
            int(row["Volume"]),
        )
        for idx, row in df.iterrows()
    ]
    cur.executemany(
        f"""
        INSERT INTO {OHLCV_TABLE} (symbol_id, date, open, high, low, close, volume)
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


def fetch_and_save(name: str, ticker: str, conn, out_dir: str) -> None:
    print(f"Fetching {name} ({ticker}) ...")
    df = yf.download(
        ticker,
    start=START_DATE,
    end=END_DATE,
    interval=INTERVAL,
    auto_adjust=False,
    progress=False,
    )

    if df.empty:
        print(f"  No data returned for {ticker}, skipping.")
        return

    # Flatten MultiIndex columns if present (happens with newer yfinance versions)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index.name = "Date"

    # --- Store in Supabase ---
    with conn.cursor() as cur:
        symbol_id = get_or_create_symbol_id(cur, name)
        store_ohlcv(cur, symbol_id, df)
    conn.commit()
    print(f"  Stored {len(df)} rows in Supabase ({SYMBOLS_TABLE}/{OHLCV_TABLE}).")


    # --- Optional local CSV backup ---
    if SAVE_CSV:
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{name}.csv")
        df.to_csv(out_path)
        print(f"  Saved {len(df)} rows to {out_path}")


def main():
    conn = get_connection()
    ensure_tables(conn)

    failed = []
    for name, ticker in TICKERS.items():
        try:
            fetch_and_save(name, ticker, conn, OUTPUT_DIR)
        except Exception as e:
            print(f"  Error fetching {ticker}: {e}")
            failed.append(name)
            conn.rollback()
        time.sleep(PAUSE_SECONDS)

    conn.close()

    print("Done.")
    if failed:
        print(f"Failed tickers ({len(failed)}): {', '.join(failed)}")


if __name__ == "__main__":
    main()
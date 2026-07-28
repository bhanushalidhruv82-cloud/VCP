import os
import json
import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from flask import Flask, jsonify, send_from_directory, request
import psycopg2
from psycopg2.extras import RealDictCursor
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()  # loads .env locally; no-op on Render where env vars are injected directly

# ---------------------------------------------------------------------------
# Supabase connection (same project used by fetch_and_store.py)
# Required env vars: DB_HOST, DB_USER, DB_PASSWORD (no hardcoded fallbacks
# for secrets - the app fails fast at startup if these are missing).
# ---------------------------------------------------------------------------
DB_HOST = os.environ["DB_HOST"]
DB_PORT = os.environ.get("DB_PORT", "5432")
DB_NAME = os.environ.get("DB_NAME", "postgres")
DB_USER = os.environ["DB_USER"]
DB_PASSWORD = os.environ["DB_PASSWORD"]

# Both watchlists live in the SAME Supabase database now — they're just
# different table pairs within it (matches TABLE_SETS in fetch_and_store.py).
WATCHLIST_TABLE_MAP = {
    "Watchlist1": {"symbols": "symbols", "ohlcv": "ohlcv_data"},
    "Nasdaq300": {"symbols": "us_symbols", "ohlcv": "us_ohlcv_data"},
}
DEFAULT_WATCHLIST = "Watchlist1"

# Serve static files (index2.html) from the same directory as this script
app = Flask(__name__, static_folder=os.path.dirname(os.path.abspath(__file__)), static_url_path="")


def resolve_tables(watchlist_param):
    """Maps a watchlist selector value to its backing (symbols, ohlcv) table
    names, falling back to the default watchlist's tables for unknown values."""
    return WATCHLIST_TABLE_MAP.get(watchlist_param, WATCHLIST_TABLE_MAP[DEFAULT_WATCHLIST])


# def get_connection(db_name=None):
#     return psycopg2.connect(
#         host=DB_HOST, port=DB_PORT, dbname=(db_name or DB_NAME),
#         user=DB_USER, password=DB_PASSWORD
#     )
def get_connection(db_name=None):
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=(db_name or DB_NAME),
        user=DB_USER,
        password=DB_PASSWORD,
        sslmode=os.environ.get("DB_SSLMODE", "require"),
        connect_timeout=10,
    )


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "templates/index7.html")

@app.route("/health")
def health():
    return {"status": "ok"}, 200

@app.route("/api/symbols")
def api_symbols():
    watchlist = request.args.get("watchlist", DEFAULT_WATCHLIST)
    tables = resolve_tables(watchlist)
    symbols_table, ohlcv_table = tables["symbols"], tables["ohlcv"]
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # last_close / prev_close pulled via array_agg ordered by date desc
            cur.execute(f"""
                SELECT s.symbol AS symbol,
                       COUNT(o.id) AS row_count,
                       (ARRAY_AGG(o.close ORDER BY o.date DESC))[1] AS last_close,
                       (ARRAY_AGG(o.close ORDER BY o.date DESC))[2] AS prev_close
                FROM {symbols_table} s
                LEFT JOIN {ohlcv_table} o ON o.symbol_id = s.id
                GROUP BY s.symbol
                ORDER BY s.symbol;
            """)
            rows = cur.fetchall()

        result = []
        for r in rows:
            last_close = float(r["last_close"]) if r["last_close"] is not None else None
            prev_close = float(r["prev_close"]) if r["prev_close"] is not None else None
            change_pct = None
            if last_close is not None and prev_close:
                change_pct = round((last_close - prev_close) / prev_close * 100, 2)
            result.append({
                "symbol": r["symbol"],
                "row_count": r["row_count"],
                "last_close": last_close,
                "change_pct": change_pct,
            })
        return jsonify(result)
    finally:
        conn.close()


@app.route("/api/ohlcv/<symbol>")
def api_ohlcv(symbol):
    watchlist = request.args.get("watchlist", DEFAULT_WATCHLIST)
    tables = resolve_tables(watchlist)
    symbols_table, ohlcv_table = tables["symbols"], tables["ohlcv"]
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT o.date, o.open, o.high, o.low, o.close, o.volume
                FROM {ohlcv_table} o
                JOIN {symbols_table} s ON s.id = o.symbol_id
                WHERE s.symbol = %s
                ORDER BY o.date ASC;
            """, (symbol,))
            rows = cur.fetchall()

        if not rows:
            return jsonify({"error": f"no data for {symbol}"}), 404

        candles = []
        volumes = []
        for date, o, h, l, c, v in rows:
            t = date.isoformat()  # 'YYYY-MM-DD' -> lightweight-charts business-day format
            o, h, l, c = float(o), float(h), float(l), float(c)
            candles.append({"time": t, "open": o, "high": h, "low": l, "close": c})
            color = "#26a69a" if c >= o else "#ef5350"
            volumes.append({"time": t, "value": int(v), "color": color})

        return jsonify({"candles": candles, "volumes": volumes})
    finally:
        conn.close()


# =====================================================================
# WATCHLIST ADD / REMOVE
# =====================================================================

def get_or_create_symbol_id(cur, symbols_table, symbol):
    cur.execute(f"SELECT id FROM {symbols_table} WHERE symbol = %s", (symbol,))
    row = cur.fetchone()
    if row:
        return row[0]
    cur.execute(f"INSERT INTO {symbols_table} (symbol) VALUES (%s) RETURNING id", (symbol,))
    return cur.fetchone()[0]


def fetch_ohlcv_history(symbol):
    df = yf.download(symbol, start="2020-01-01", progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df = df.reset_index()
    return df


def store_ohlcv_rows(cur, ohlcv_table, symbol_id, df):
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


# @app.route("/api/watchlist/add", methods=["POST"])
# def api_watchlist_add():
#     data = request.json or {}
#     symbol = (data.get("symbol") or "").strip().upper()
#     watchlist = data.get("watchlist", DEFAULT_WATCHLIST)

#     if not symbol:
#         return jsonify({"status": "error", "message": "Ticker symbol is required"}), 400

#     tables = resolve_tables(watchlist)
#     symbols_table, ohlcv_table = tables["symbols"], tables["ohlcv"]
#     conn = get_connection()
#     try:
#         df = fetch_ohlcv_history(symbol)
#         if df is None or df.empty:
#             return jsonify({"status": "error", "message": f"No data found for '{symbol}'. Check the ticker symbol."}), 404

#         with conn.cursor() as cur:
#             symbol_id = get_or_create_symbol_id(cur, symbols_table, symbol)
#             store_ohlcv_rows(cur, ohlcv_table, symbol_id, df)
#         conn.commit()

#         return jsonify({
#             "status": "success",
#             "message": f"Added {symbol} ({len(df)} rows fetched).",
#             "symbol": symbol,
#             "rows": len(df),
#         })
#     except Exception as e:
#         conn.rollback()
#         return jsonify({"status": "error", "message": f"Failed to fetch/store {symbol}: {str(e)}"}), 500
#     finally:
#         conn.close()

@app.route("/api/watchlist/add", methods=["POST"])
def api_watchlist_add():
    data = request.json or {}
    symbol = (data.get("symbol") or "").strip().upper()
    watchlist = data.get("watchlist", DEFAULT_WATCHLIST)

    if not symbol:
        return jsonify({"status": "error", "message": "Ticker symbol is required"}), 400

    tables = resolve_tables(watchlist)
    symbols_table, ohlcv_table = tables["symbols"], tables["ohlcv"]
    conn = get_connection()
    try:
        df = fetch_ohlcv_history(symbol)
        if df is None or df.empty:
            return jsonify({"status": "error", "message": f"No data found for '{symbol}'. Check the ticker symbol."}), 404

        with conn.cursor() as cur:
            symbol_id = get_or_create_symbol_id(cur, symbols_table, symbol)
            store_ohlcv_rows(cur, ohlcv_table, symbol_id, df)
        conn.commit()

        return jsonify({
            "status": "success",
            "message": f"Added {symbol} ({len(df)} rows fetched).",
            "symbol": symbol,
            "rows": len(df),
        })
    except Exception as e:
        conn.rollback()
        return jsonify({"status": "error", "message": f"Failed to fetch/store {symbol}: {str(e)}"}), 500
    finally:
        conn.close()



@app.route("/api/watchlist/remove", methods=["POST"])
def api_watchlist_remove():
    data = request.json or {}
    symbol = (data.get("symbol") or "").strip().upper()
    watchlist = data.get("watchlist", DEFAULT_WATCHLIST)

    if not symbol:
        return jsonify({"status": "error", "message": "Ticker symbol is required"}), 400

    tables = resolve_tables(watchlist)
    symbols_table, ohlcv_table = tables["symbols"], tables["ohlcv"]
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(f"SELECT id FROM {symbols_table} WHERE symbol = %s", (symbol,))
            row = cur.fetchone()
            if not row:
                return jsonify({"status": "error", "message": f"{symbol} not found"}), 404
            symbol_id = row[0]
            cur.execute(f"DELETE FROM {ohlcv_table} WHERE symbol_id = %s", (symbol_id,))
            cur.execute(f"DELETE FROM {symbols_table} WHERE id = %s", (symbol_id,))
        conn.commit()
        return jsonify({"status": "success", "message": f"Removed {symbol}."})
    except Exception as e:
        conn.rollback()
        return jsonify({"status": "error", "message": f"Failed to remove {symbol}: {str(e)}"}), 500
    finally:
        conn.close()


# =====================================================================
# VCP ALERT SYSTEM BACKEND IMPLEMENTATION
# =====================================================================

CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "alert_config.json")
SENT_ALERTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sent_alerts.json")


def load_settings():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
        except:
            pass
    return {}


def save_settings(data):
    with open(CONFIG_FILE, "w") as f:
        json.dump(data, f, indent=4)


def load_sent_alerts():
    if os.path.exists(SENT_ALERTS_FILE):
        try:
            with open(SENT_ALERTS_FILE, "r") as f:
                return json.load(f)
        except:
            pass
    return {}


def save_sent_alerts(alerts):
    with open(SENT_ALERTS_FILE, "w") as f:
        json.dump(alerts, f, indent=4)


def build_candidates(candles, depth, from_index=0):
    points = []
    for i in range(max(depth, from_index), len(candles) - depth):
        cur = candles[i]
        is_high = True
        is_low = True
        for j in range(1, depth + 1):
            if candles[i - j]["high"] >= cur["high"] or candles[i + j]["high"] > cur["high"]:
                is_high = False
            if candles[i - j]["low"] <= cur["low"] or candles[i + j]["low"] < cur["low"]:
                is_low = False
        if is_high:
            points.append({"index": i, "type": "H", "value": cur["high"], "time": cur["time"]})
        elif is_low:
            points.append({"index": i, "type": "L", "value": cur["low"], "time": cur["time"]})
            
    if not points:
        return []
        
    alt = [points[0]]
    for i in range(1, len(points)):
        prev = alt[-1]
        curr = points[i]
        if curr["type"] == prev["type"]:
            if (curr["type"] == "H" and curr["value"] > prev["value"]) or \
               (curr["type"] == "L" and curr["value"] < prev["value"]):
                alt[-1] = curr
        else:
            alt.append(curr)
            
    seq = alt[1:] if alt[0]["type"] == "L" else alt
    
    candidates = []
    for k in range(0, len(seq) - 2, 2):
        p_start = seq[k]
        p_mid = seq[k + 1]
        p_end = seq[k + 2]
        leg_left = p_mid["index"] - p_start["index"]
        leg_right = p_end["index"] - p_mid["index"]
        if leg_left < 1 or leg_right < 1:
            continue
            
        drop_pct = (p_start["value"] - p_mid["value"]) / p_start["value"] * 100
        candidates.append({
            "pStart": p_start,
            "pMid": p_mid,
            "pEnd": p_end,
            "legLeft": leg_left,
            "legRight": leg_right,
            "dropPct": drop_pct,
            "low": p_mid["value"]
        })
    return candidates


def get_resistance_level(run):
    first_high = run[0]["pStart"]["value"]
    return first_high * 0.992


def calculate_wave_segments(candles, depth, min_waves=2):
    MIN_WAVES = min_waves
    SMALL_WAVE_DEPTH = 1
    VCP_RESISTANCE_APPROACH_PCT = 0.05
    VCP_BREAKOUT_KILL_PCT = 0.05
    
    candidates = build_candidates(candles, depth, 0)
    if not candidates:
        return []
        
    runs_data = []
    i = 0
    while i < len(candidates):
        run = [candidates[i]]
        j = i + 1
        breakout_idx = -1
        active_candidates = candidates
        
        while True:
            if j >= len(active_candidates):
                if len(run) >= 2 and active_candidates is candidates:
                    last_end_idx = run[-1]["pEnd"]["index"]
                    finer = [c for c in build_candidates(candles, SMALL_WAVE_DEPTH, last_end_idx)
                             if c["pStart"]["index"] >= last_end_idx]
                    if finer:
                        active_candidates = finer
                        j = 0
                        continue
                break
                
            prev = run[-1]
            curr = active_candidates[j]
            
            if len(run) >= 2:
                break_level = get_resistance_level(run)
            else:
                break_level = run[0]["pStart"]["value"]
                
            broken = False
            break_during_curr = False
            kill_level = float('inf') if break_level == float('inf') else break_level * (1 + VCP_BREAKOUT_KILL_PCT)
            
            for k in range(prev["pEnd"]["index"], curr["pEnd"]["index"] + 1):
                if candles[k]["close"] > kill_level:
                    broken = True
                    breakout_idx = k
                    if k >= curr["pStart"]["index"]:
                        break_during_curr = True
                    break
                    
            approach_allowed_gap = break_level * VCP_RESISTANCE_APPROACH_PCT
            approach_gap = break_level - curr["pEnd"]["value"]
            valid_approach = True if break_level == float('inf') else (approach_gap <= approach_allowed_gap)
            
            valid_low = curr["low"] >= prev["low"]
            
            if (broken and not break_during_curr) or not valid_approach or not valid_low:
                if len(run) >= 2 and active_candidates is candidates:
                    last_end_idx = prev["pEnd"]["index"]
                    finer = [c for c in build_candidates(candles, SMALL_WAVE_DEPTH, last_end_idx)
                             if c["pStart"]["index"] >= last_end_idx]
                    if finer:
                        active_candidates = finer
                        j = 0
                        breakout_idx = -1
                        continue
                if not broken and active_candidates is not candidates and j + 1 < len(active_candidates):
                    j += 1
                    continue
                break
                
            if broken and break_during_curr:
                run.append(curr)
                break
                
            run.append(curr)
            j += 1
            
        if breakout_idx == -1 and len(run) >= 2:
            res_lvl = get_resistance_level(run)
            kill_lvl = res_lvl * (1 + VCP_BREAKOUT_KILL_PCT)
            last_end_idx = run[-1]["pEnd"]["index"]
            for k in range(last_end_idx + 1, len(candles)):
                if candles[k]["close"] > kill_lvl:
                    breakout_idx = k
                    break
                    
        if len(run) >= MIN_WAVES:
            runs_data.append({
                "run": run,
                "breakoutIdx": breakout_idx
            })
            
        if active_candidates is candidates and j > i + 1:
            i = j
        else:
            i += 1
            
    return runs_data


def parse_date(d_str):
    try:
        return datetime.datetime.strptime(d_str.split("T")[0], "%Y-%m-%d").date()
    except Exception:
        return None


def is_within_months(date_val, months):
    if not date_val:
        return False
    delta = datetime.date.today() - date_val
    return delta.days <= (months * 31)


def get_symbol_live_entry(symbol, candles, min_waves=2):
    sensitivities = [6, 7, 8, 11, 15]
    lookback_months = 5
    max_entry_age_bars = 5
    
    for depth in sensitivities:
        runs = calculate_wave_segments(candles, depth, min_waves=min_waves)
        
        filtered_runs = []
        for r in runs:
            start_date_str = r["run"][0]["pStart"]["time"]
            start_date = parse_date(start_date_str)
            if is_within_months(start_date, lookback_months):
                filtered_runs.append(r)
                
        if not filtered_runs:
            continue
            
        most_recent = filtered_runs[-1]
        breakout_idx = most_recent["breakoutIdx"]
        
        if breakout_idx != -1:
            bars_ago = (len(candles) - 1) - breakout_idx
            if bars_ago <= max_entry_age_bars:
                resistance = get_resistance_level(most_recent["run"])
                entry_price = resistance * 1.05
                current_price = candles[-1]["close"]
                max_price = entry_price * 1.05
                
                if resistance * 0.98 <= current_price <= max_price:
                    return {
                        "sensitivity": depth,
                        "breakout_date": candles[breakout_idx]["time"],
                        "breakout_price": entry_price,
                        "current_price": current_price,
                        "bars_ago": bars_ago,
                        "resistance": resistance
                    }
    return None


def send_alert_email(config, live_entries):
    smtp_server = config.get("smtp_server", "smtp.gmail.com")
    smtp_port = int(config.get("smtp_port", 587))
    sender_email = config.get("sender_email")
    sender_password = config.get("sender_password")
    recipient_email = config.get("recipient_email")
    
    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = recipient_email
    msg["Subject"] = f"VCP Live Entry Alert: {len(live_entries)} Tickers Found"
    
    html = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; background-color: #131722; color: #d1d4dc; padding: 20px; }}
            h2 {{ color: #ffb300; border-bottom: 2px solid #2a2e39; padding-bottom: 10px; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 15px; background-color: #1c1f2b; }}
            th, td {{ padding: 12px; border: 1px solid #2a2e39; text-align: left; color: #d1d4dc; }}
            th {{ background-color: #2a2e39; color: #ffb300; }}
            tr:nth-child(even) {{ background-color: #1e222d; }}
            .green {{ color: #26a69a; font-weight: bold; }}
            .accent {{ color: #ffb300; }}
        </style>
    </head>
    <body>
        <h2>VCP Live Entry Alerts</h2>
        <p>The VCP Screener detected the following tickers with recent breakouts (live entries):</p>
        <table>
            <thead>
                <tr>
                    <th>Symbol</th>
                    <th>Current Price</th>
                    <th>Entry Price</th>
                    <th>Breakout Date</th>
                    <th>Days Ago</th>
                    <th>Sensitivity</th>
                </tr>
            </thead>
            <tbody>
    """
    
    for entry in live_entries:
        html += f"""
                <tr>
                    <td><strong class="accent">{entry['symbol']}</strong></td>
                    <td class="green">${entry['current_price']:.2f}</td>
                    <td>${entry['breakout_price']:.2f}</td>
                    <td>{entry['breakout_date']}</td>
                    <td>{entry['bars_ago']}</td>
                    <td>{entry['sensitivity']}</td>
                </tr>
        """
        
    html += """
            </tbody>
        </table>
        <p style="margin-top: 20px; font-size: 12px; color: #787b86;">
            Sent automatically from your VCP Trading Dashboard.
        </p>
    </body>
    </html>
    """
    
    msg.attach(MIMEText(html, "html"))
    
    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, recipient_email, msg.as_string())


@app.route("/api/alerts/settings", methods=["GET", "POST"])
def api_alerts_settings():
    if request.method == "POST":
        data = request.json or {}
        existing = load_settings()
        if data.get("sender_password") == "********":
            data["sender_password"] = existing.get("sender_password", "")
        save_settings(data)
        return jsonify({"status": "success", "message": "Settings saved successfully"})
    else:
        settings = load_settings()
        masked_settings = settings.copy()
        if "sender_password" in masked_settings and masked_settings["sender_password"]:
            masked_settings["sender_password"] = "********"
        return jsonify(masked_settings)


@app.route("/api/alerts/scan_and_send", methods=["POST"])
def api_alerts_scan_and_send():
    data = request.json or {}
    config = load_settings()
    
    for k in ["smtp_server", "smtp_port", "sender_email", "sender_password", "recipient_email", "min_waves"]:
        if k in data:
            if k == "sender_password" and data[k] == "********":
                continue
            config[k] = data[k]
            
    if data:
        save_settings(config)
        
    sender_email = config.get("sender_email")
    sender_password = config.get("sender_password")
    recipient_email = config.get("recipient_email")
    min_waves = int(config.get("min_waves", 2))
    
    if not sender_email or not sender_password or not recipient_email:
        return jsonify({"status": "error", "message": "Sender Email, Password, and Recipient Email are required"}), 400
        
    conn = get_connection()
    live_entries = []
    
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT symbol FROM symbols ORDER BY symbol;")
            symbols = [row[0] for row in cur.fetchall()]
            
            for symbol in symbols:
                cur.execute("""
                    SELECT o.date, o.open, o.high, o.low, o.close
                    FROM ohlcv_data o
                    JOIN symbols s ON s.id = o.symbol_id
                    WHERE s.symbol = %s
                    ORDER BY o.date ASC;
                """, (symbol,))
                rows = cur.fetchall()
                if not rows:
                    continue
                    
                candles = []
                for date, o, h, l, c in rows:
                    candles.append({
                        "time": date.isoformat(),
                        "open": float(o),
                        "high": float(h),
                        "low": float(l),
                        "close": float(c)
                    })
                    
                entry_info = get_symbol_live_entry(symbol, candles, min_waves=min_waves)
                if entry_info:
                    entry_info["symbol"] = symbol
                    live_entries.append(entry_info)
    except Exception as e:
        return jsonify({"status": "error", "message": f"Error scanning database: {str(e)}"}), 500
    finally:
        conn.close()
        
    sent_alerts = load_sent_alerts()
    new_alerts_to_send = []
    
    for entry in live_entries:
        symbol = entry["symbol"]
        breakout_date = entry["breakout_date"]
        if sent_alerts.get(symbol) == breakout_date:
            continue
        new_alerts_to_send.append(entry)
        
    if not new_alerts_to_send:
        return jsonify({
            "status": "success",
            "message": "Scan completed. No new live entries found to notify.",
            "total_live_entries_found": len(live_entries),
            "new_alerts_sent": 0
        })
        
    try:
        send_alert_email(config, new_alerts_to_send)
        for entry in new_alerts_to_send:
            sent_alerts[entry["symbol"]] = entry["breakout_date"]
        save_sent_alerts(sent_alerts)
        
        return jsonify({
            "status": "success",
            "message": f"Successfully scanned. Sent alert email for {len(new_alerts_to_send)} tickers.",
            "sent_tickers": [e["symbol"] for e in new_alerts_to_send],
            "total_live_entries_found": len(live_entries),
            "new_alerts_sent": len(new_alerts_to_send)
        })
    except Exception as e:
        return jsonify({"status": "error", "message": f"Failed to send email: {str(e)}"}), 500


@app.route("/api/alerts/test", methods=["POST"])
def api_alerts_test():
    data = request.json or {}
    config = load_settings()
    
    for k in ["smtp_server", "smtp_port", "sender_email", "sender_password", "recipient_email"]:
        if k in data:
            if k == "sender_password" and data[k] == "********":
                continue
            config[k] = data[k]
            
    sender_email = config.get("sender_email")
    sender_password = config.get("sender_password")
    recipient_email = config.get("recipient_email")
    
    if not sender_email or not sender_password or not recipient_email:
        return jsonify({"status": "error", "message": "Sender Email, Password, and Recipient Email are required"}), 400
        
    try:
        msg = MIMEMultipart()
        msg["From"] = sender_email
        msg["To"] = recipient_email
        msg["Subject"] = "VCP Email Alert Test"
        
        html = """
        <html>
        <body>
            <h3>VCP Alert System - Test Successful!</h3>
            <p>Your SMTP configuration is working correctly. You will receive alerts for any live entry tickers here.</p>
        </body>
        </html>
        """
        msg.attach(MIMEText(html, "html"))
        
        smtp_server = config.get("smtp_server", "smtp.gmail.com")
        smtp_port = int(config.get("smtp_port", 587))
        
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipient_email, msg.as_string())
            
        return jsonify({"status": "success", "message": "Test email sent successfully!"})
    except Exception as e:
        return jsonify({"status": "error", "message": f"Test failed: {str(e)}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
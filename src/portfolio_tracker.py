"""
portfolio_tracker.py

Portfolio state tracking for BMEM daily update pipeline.
Handles holdings persistence, trade recording, and nightly order generation.

Designed for max_holdings=1 (top-1 strategy) but parameterized for easy
extension to top-N.

State files (all in outputs/{broker_id}/daily/):
  holdings.json          -- current state, overwritten each run
  trades_log.csv         -- append-only trade history
  orders_{date}.json     -- nightly order suggestions for auto-trading
"""

import json
import csv
from datetime import datetime
from pathlib import Path

import pandas as pd


# ─── CONSTANTS ────────────────────────────────────────────────────────────────

TRAILING_STOP_RATIO = 0.8
LONG_ENTRY_BUFFER   = 0.02   # effective entry threshold = long_threshold + buffer

HOLDINGS_FILENAME   = "holdings.json"
TRADES_LOG_FILENAME = "trades_log.csv"
ORDERS_FILENAME_FMT = "orders_{date}.json"

TRADES_LOG_COLS = [
    "date", "stock_id", "stock_name", "action", "price",
    "reason", "entry_date", "entry_price", "return_pct", "signal_prob",
]


# ─── STATE I/O ────────────────────────────────────────────────────────────────

def load_holdings(daily_dir: Path) -> dict:
    """
    Load portfolio state from holdings.json.
    Returns an empty state dict on first run (file absent).

    State schema
    ------------
    holdings     : {stock_id -> {entry_date, entry_price, entry_prob_long,
                                  highest_price}}
    pending_buys : [{stock_id, stock_name, signal_prob_long, reason}]
                   -- confirmed at next run's today-open
    pending_sells: [{stock_id, stock_name, entry_date, entry_price,
                     signal_prob_short, reason}]
                   -- confirmed at next run's today-open
    last_updated : "YYYY-MM-DD"
    """
    path = daily_dir / HOLDINGS_FILENAME
    if not path.exists():
        return {
            "holdings":      {},
            "pending_buys":  [],
            "pending_sells": [],
            "last_updated":  None,
        }
    return json.loads(path.read_text(encoding="utf-8"))


def save_holdings(daily_dir: Path, state: dict) -> None:
    daily_dir.mkdir(parents=True, exist_ok=True)
    (daily_dir / HOLDINGS_FILENAME).write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_trades_log(daily_dir: Path) -> pd.DataFrame:
    path = daily_dir / TRADES_LOG_FILENAME
    if not path.exists():
        return pd.DataFrame(columns=TRADES_LOG_COLS)
    return pd.read_csv(path, encoding="utf-8-sig")


def _append_trade(daily_dir: Path, trade: dict) -> None:
    daily_dir.mkdir(parents=True, exist_ok=True)
    path = daily_dir / TRADES_LOG_FILENAME
    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=TRADES_LOG_COLS)
        if write_header:
            writer.writeheader()
        writer.writerow({k: trade.get(k, "") for k in TRADES_LOG_COLS})


def save_orders(
    daily_dir: Path,
    target_date: str,
    orders: list,
    broker_id: str,
) -> Path:
    """Write orders_{target_date}.json for nightly order placement / auto-trading."""
    daily_dir.mkdir(parents=True, exist_ok=True)
    path = daily_dir / ORDERS_FILENAME_FMT.format(date=target_date)
    payload = {
        "date":         target_date,
        "broker_id":    broker_id,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "orders":       orders,
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return path


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def _next_trading_day(date_str: str) -> str:
    """Return the next weekday after date_str as YYYY-MM-DD (holidays not considered)."""
    dt = pd.Timestamp(date_str) + pd.Timedelta(days=1)
    while dt.weekday() >= 5:
        dt += pd.Timedelta(days=1)
    return dt.strftime("%Y-%m-%d")


def _pct(sell: float, entry: float) -> float:
    return round((sell / entry) - 1, 4)


# ─── CORE UPDATE ──────────────────────────────────────────────────────────────

def update_portfolio(
    daily_dir: Path,
    target_date: str,
    signals_df: pd.DataFrame,
    price_df: pd.DataFrame,
    name_map: dict,
    broker_id: str,
    max_holdings: int = 1,
    long_threshold: float = 0.6,
    short_threshold: float = 0.8,
    trailing_stop_ratio: float = TRAILING_STOP_RATIO,
) -> tuple[list, dict, list]:
    """
    Process one day's portfolio update.

    Execution order (mirrors portfolio_backtest.py logic):
      1. Confirm pending sells from yesterday at today's open
      2. Confirm pending buys from yesterday at today's open
      3. Update highest_price for all holdings (use today's max)
      4. Trailing stop check: today_min <= highest_price * ratio -> sell today
      5. Short signal check -> pending sell for tomorrow's open
      6. Long signal / rebalancing -> pending buy for tomorrow's open
      7. Build tonight's orders (conditional stops + market-open orders)
      8. Persist holdings.json, append trades_log.csv, write orders_{date}.json

    Parameters
    ----------
    signals_df      : today's signals; columns: stock_id, pred_prob_long,
                      pred_prob_short (and any others — extras are ignored)
    price_df        : today's OHLCV for all relevant stocks; columns:
                      stock_id, open, close, max, min
    max_holdings    : portfolio capacity; 1 = top-1, N = top-N
    long_threshold  : base long signal threshold (entry uses threshold + LONG_ENTRY_BUFFER)
    short_threshold : short signal threshold for exit

    Returns
    -------
    executed_today  : list of trade dicts confirmed executed today
    holdings        : current holdings dict after today's update
    tonight_orders  : list of order dicts for nightly conditional/pending orders
    """
    entry_threshold = long_threshold + LONG_ENTRY_BUFFER

    state         = load_holdings(daily_dir)
    holdings      = state.get("holdings", {})
    pending_buys  = state.get("pending_buys", [])
    pending_sells = state.get("pending_sells", [])

    # Normalize stock_id to str in both lookup tables
    price_df = price_df.copy()
    price_df["stock_id"] = price_df["stock_id"].astype(str)
    price_lkp = price_df.set_index("stock_id").to_dict("index")

    signals_df = signals_df.copy()
    signals_df["stock_id"] = signals_df["stock_id"].astype(str)
    sig_lkp = signals_df.set_index("stock_id").to_dict("index")

    executed_today: list[dict] = []

    # ── 1. Confirm pending sells at today's open ──────────────────────────────
    for ps in pending_sells:
        sid = str(ps["stock_id"])
        if sid not in price_lkp:
            continue
        sell_price  = price_lkp[sid]["open"]
        entry_price = ps.get("entry_price")
        ret         = _pct(sell_price, entry_price) if entry_price else ""
        trade = {
            "date":        target_date,
            "stock_id":    sid,
            "stock_name":  ps.get("stock_name", ""),
            "action":      "SELL",
            "price":       round(sell_price, 2),
            "reason":      ps.get("reason", ""),
            "entry_date":  ps.get("entry_date", ""),
            "entry_price": entry_price or "",
            "return_pct":  ret,
            "signal_prob": ps.get("signal_prob_short", ""),
        }
        _append_trade(daily_dir, trade)
        executed_today.append(trade)

    # ── 2. Confirm pending buys at today's open ───────────────────────────────
    for pb in pending_buys:
        sid = str(pb["stock_id"])
        if sid not in price_lkp:
            continue
        open_price      = price_lkp[sid]["open"]
        entry_prob_long = pb.get("signal_prob_long", 0.0)
        holdings[sid] = {
            "entry_date":      target_date,
            "entry_price":     round(open_price, 2),
            "entry_prob_long": entry_prob_long,
            "peak_prob_long":  entry_prob_long,   # tracks highest long prob since entry
            "highest_price":   price_lkp[sid].get("max", open_price),
        }
        trade = {
            "date":        target_date,
            "stock_id":    sid,
            "stock_name":  pb.get("stock_name", ""),
            "action":      "BUY",
            "price":       round(open_price, 2),
            "reason":      pb.get("reason", "long_signal"),
            "entry_date":  target_date,
            "entry_price": round(open_price, 2),
            "return_pct":  "",
            "signal_prob": pb.get("signal_prob_long", ""),
        }
        _append_trade(daily_dir, trade)
        executed_today.append(trade)

    # ── 3. Update highest_price and peak_prob_long ───────────────────────────
    for sid, holding in holdings.items():
        if sid in price_lkp:
            today_max = price_lkp[sid].get("max", holding["highest_price"])
            if today_max > holding["highest_price"]:
                holding["highest_price"] = today_max
        if sid in sig_lkp:
            today_prob = float(sig_lkp[sid].get("pred_prob_long", 0.0))
            if today_prob > holding.get("peak_prob_long", 0.0):
                holding["peak_prob_long"] = round(today_prob, 4)

    # ── 4. Trailing stop check (executes today) ───────────────────────────────
    to_remove: list[str] = []
    for sid, holding in holdings.items():
        if sid not in price_lkp:
            continue
        row        = price_lkp[sid]
        stop_price = holding["highest_price"] * trailing_stop_ratio
        if row["min"] <= stop_price:
            sell_price = row["open"] if row["open"] < stop_price else stop_price
            trade = {
                "date":        target_date,
                "stock_id":    sid,
                "stock_name":  name_map.get(sid, ""),
                "action":      "SELL",
                "price":       round(sell_price, 2),
                "reason":      "trailing_stop",
                "entry_date":  holding["entry_date"],
                "entry_price": holding["entry_price"],
                "return_pct":  _pct(sell_price, holding["entry_price"]),
                "signal_prob": "",
            }
            _append_trade(daily_dir, trade)
            executed_today.append(trade)
            to_remove.append(sid)

    for sid in to_remove:
        del holdings[sid]

    # ── 5. Short signal → pending sell for tomorrow's open ────────────────────
    new_pending_sells: list[dict] = []
    to_remove = []
    for sid, holding in list(holdings.items()):
        if sid not in sig_lkp:
            continue
        short_prob = sig_lkp[sid].get("pred_prob_short", 0)
        if short_prob >= short_threshold:
            new_pending_sells.append({
                "stock_id":          sid,
                "stock_name":        name_map.get(sid, ""),
                "entry_date":        holding["entry_date"],
                "entry_price":       holding["entry_price"],
                "signal_prob_short": round(float(short_prob), 4),
                "reason":            "short_signal",
            })
            to_remove.append(sid)

    for sid in to_remove:
        del holdings[sid]

    # ── 6. Long signal — new entries and rebalancing ──────────────────────────
    new_pending_buys: list[dict] = []

    # Stocks already accounted for (held, pending sell, or pending buy this cycle)
    held_or_pending = (
        set(holdings.keys())
        | {str(ps["stock_id"]) for ps in new_pending_sells}
    )

    candidates = (
        signals_df[signals_df["pred_prob_long"] >= entry_threshold]
        .copy()
        .sort_values("pred_prob_long", ascending=False)
    )

    for _, cand in candidates.iterrows():
        sid  = str(cand["stock_id"])
        prob = float(cand["pred_prob_long"])
        if sid in held_or_pending:
            continue

        current_count = len(holdings) + len(new_pending_buys)

        if current_count < max_holdings:
            new_pending_buys.append({
                "stock_id":         sid,
                "stock_name":       name_map.get(sid, ""),
                "signal_prob_long": round(prob, 4),
                "reason":           "long_signal",
            })
            held_or_pending.add(sid)
        else:
            # Portfolio full: consider swapping out the weakest holding
            weakest_sid, weakest_prob = _find_weakest(holdings, new_pending_buys, sig_lkp)
            if weakest_sid is None or prob <= weakest_prob:
                break  # Candidates are sorted descending; no further swap possible

            # Evict weakest
            if weakest_sid in holdings:
                h = holdings.pop(weakest_sid)
                new_pending_sells.append({
                    "stock_id":          weakest_sid,
                    "stock_name":        name_map.get(weakest_sid, ""),
                    "entry_date":        h["entry_date"],
                    "entry_price":       h["entry_price"],
                    "signal_prob_short": weakest_prob,
                    "reason":            "rebalance",
                })
            else:
                new_pending_buys = [
                    pb for pb in new_pending_buys
                    if str(pb["stock_id"]) != weakest_sid
                ]
            held_or_pending.discard(weakest_sid)

            new_pending_buys.append({
                "stock_id":         sid,
                "stock_name":       name_map.get(sid, ""),
                "signal_prob_long": round(prob, 4),
                "reason":           "long_signal",
            })
            held_or_pending.add(sid)

    # ── 7. Build tonight's orders ─────────────────────────────────────────────
    tonight_orders: list[dict] = []
    execute_date = _next_trading_day(target_date)

    # Conditional stop orders — only when stop_price is reachable tomorrow
    # (i.e. stop_price >= today_close * 0.9, the 10% limit-down floor)
    for sid, holding in holdings.items():
        if sid not in price_lkp:
            continue
        today_close = price_lkp[sid]["close"]
        stop_price  = holding["highest_price"] * trailing_stop_ratio
        limit_down  = round(today_close * 0.9, 2)
        if stop_price < limit_down:
            continue  # Stop unreachable tomorrow; no order needed tonight
        ret = (today_close / holding["entry_price"]) - 1
        tonight_orders.append({
            "stock_id":              sid,
            "stock_name":            name_map.get(sid, ""),
            "action":                "SELL",
            "order_type":            "conditional_stop",
            "trigger_price":         round(stop_price, 2),
            "execute_price_type":    "market",
            "execute_date":          execute_date,
            "reason":                "trailing_stop_20pct",
            "entry_date":            holding["entry_date"],
            "entry_price":           holding["entry_price"],
            "highest_price":         holding["highest_price"],
            "unrealized_return_pct": round(ret, 4),
        })

    # Market-open sell orders (short signal or rebalance)
    for ps in new_pending_sells:
        sid         = str(ps["stock_id"])
        today_close = price_lkp.get(sid, {}).get("close")
        entry_price = ps.get("entry_price")
        ret = round((today_close / entry_price) - 1, 4) if today_close and entry_price else None
        tonight_orders.append({
            "stock_id":              sid,
            "stock_name":            ps["stock_name"],
            "action":                "SELL",
            "order_type":            "market_open",
            "execute_date":          execute_date,
            "reason":                ps["reason"],
            "signal_prob_short":     ps.get("signal_prob_short"),
            "entry_date":            ps.get("entry_date"),
            "entry_price":           ps.get("entry_price"),
            "unrealized_return_pct": ret,
        })

    # Market-open buy orders
    for pb in new_pending_buys:
        tonight_orders.append({
            "stock_id":         str(pb["stock_id"]),
            "stock_name":       pb["stock_name"],
            "action":           "BUY",
            "order_type":       "market_open",
            "execute_date":     execute_date,
            "reason":           pb["reason"],
            "signal_prob_long": pb["signal_prob_long"],
        })

    # ── 8. Persist state ──────────────────────────────────────────────────────
    new_state = {
        "holdings":      holdings,
        "pending_buys":  new_pending_buys,
        "pending_sells": new_pending_sells,
        "last_updated":  target_date,
    }
    save_holdings(daily_dir, new_state)
    save_orders(daily_dir, target_date, tonight_orders, broker_id)

    return executed_today, holdings, tonight_orders


# ─── INTERNAL HELPERS ─────────────────────────────────────────────────────────

def _find_weakest(
    holdings: dict,
    pending_buys: list,
    sig_lkp: dict,
) -> tuple[str | None, float]:
    """
    Return (stock_id, peak_prob_long) of the weakest position.

    For confirmed holdings: uses peak_prob_long (highest long prob seen since
    entry), mirroring portfolio_backtest.py lines 206-210 + 235.
    For pending_buys (not yet confirmed): uses the signal prob at entry since
    no history exists yet.
    """
    candidates: dict[str, float] = {}
    for sid, holding in holdings.items():
        candidates[sid] = float(holding.get("peak_prob_long", 0.0))
    for pb in pending_buys:
        sid = str(pb["stock_id"])
        candidates[sid] = float(pb.get("signal_prob_long", 0.0))
    if not candidates:
        return None, 0.0
    weakest_sid = min(candidates, key=lambda s: candidates[s])
    return weakest_sid, candidates[weakest_sid]

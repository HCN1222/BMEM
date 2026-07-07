"""
daily_update.py

Daily update pipeline for Merrill Lynch (1440) broker signals.

For a given target date this script:
  1. Ensures broker parquet is current (fetches gap from API if stale, saves to
     data/brokers/{broker_id}/incremental.parquet)
  2. Loads historical broker data (rolling 130-day window)
  3. Ensures stock parquets are current for all window stocks (fetches gap from
     API if stale, saves to data/stocks/{sid}_incremental.parquet)
  4. Loads historical stock price data
  5. Computes HMM observation features (z_t, c_t, a_t, s_t, m_t)
  6. Runs rolling HMM inference (no-lookahead, 120-day window)
  7. Runs XGBoost long/short signal generation
  8. Saves one CSV per run: outputs/{broker_id}/daily/signals_YYYY-MM-DD.csv

Parquet update behaviour
------------------------
* New rows are appended to incremental.parquet and deduplicated — original
  historical parquets are never modified.
* If target_date is a weekend the update step is skipped automatically.
* If the API returns no data (market holiday) the parquet is left unchanged and
  the pipeline exits cleanly with no signals for that date.

Usage
-----
    python src/daily_update.py --broker-id 1440
    python src/daily_update.py --broker-id 1440 --date 2026-04-16
"""

import os
import re
import sys
import json
import argparse
import smtplib
import traceback
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime
from pathlib import Path

import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from FinMind.data import DataLoader
from tqdm import tqdm

# Allow running from either the repo root or from src/
_SRC_DIR = Path(__file__).parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from pipeline_functions import (
    fetch_broker_activity,
    fetch_stock_prices,
    compute_observation_features,
    load_hmm_model,
    compute_rolling_hmm_proba,
    load_xgb_model,
    generate_signals,
    FEATURE_COLS,
)
from portfolio_tracker import update_portfolio, load_trades_log, TRAILING_STOP_RATIO
from utils.paths import (
    BrokerPaths,
    DEFAULT_DATA_ROOT,
    DEFAULT_OUTPUT_ROOT,
    add_broker_path_args,
)

BROKER_DIR       = DEFAULT_DATA_ROOT / "brokers"
STOCK_DIR        = DEFAULT_DATA_ROOT / "stocks"
STOCK_NAMES_FILE = STOCK_DIR / "stock_names.json"

# ─── PIPELINE PARAMETERS ─────────────────────────────────────────────────────
LOOKBACK_DAYS   = 130   # days of history loaded for rolling windows (>= 60 + buffer)
HMM_WINDOW      = 120   # rolling Viterbi window (matches training)
LONG_THRESHOLD  = 0.6   # matches portfolio_backtest.py LONG_PROB_THRESHOLD
SHORT_THRESHOLD = 0.8   # matches portfolio_backtest.py SHORT_PROB_THRESHOLD
OUTPUT_TOP_N    = 10    # stocks written to the daily output CSV
OUTPUT_HIST_DAYS = 20   # trading-day lookback for the wide output


# ─── HISTORY LOADERS ─────────────────────────────────────────────────────────

def _load_all_broker_parquets(broker_id: str) -> pd.DataFrame:
    """
    Load and concatenate every .parquet file in data/brokers/{broker_id}/.
    Deduplicates on (date, stock_id) and returns a sorted DataFrame.
    """
    broker_dir = BROKER_DIR / broker_id
    files = sorted(broker_dir.glob("*.parquet"))
    if not files:
        return pd.DataFrame()

    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_parquet(f))
        except Exception as e:
            print(f"  [warn] Could not read {f.name}: {e}")

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)
    df['date'] = pd.to_datetime(df['date'])
    df = df.drop_duplicates(subset=['date', 'stock_id'])
    return df.sort_values(['stock_id', 'date']).reset_index(drop=True)


def _load_stock_parquets(stock_ids: list) -> pd.DataFrame:
    """
    Load and concatenate every .parquet file matching data/stocks/{sid}_*.parquet
    for each stock_id in stock_ids.
    """
    all_dfs = []
    for sid in stock_ids:
        for f in STOCK_DIR.glob(f"{sid}_*.parquet"):
            try:
                all_dfs.append(pd.read_parquet(f))
            except Exception as e:
                print(f"  [warn] Could not read {f.name}: {e}")

    if not all_dfs:
        return pd.DataFrame()

    df = pd.concat(all_dfs, ignore_index=True)
    df['date'] = pd.to_datetime(df['date'])
    df = df.drop_duplicates(subset=['stock_id', 'date'])
    return df.sort_values(['stock_id', 'date']).reset_index(drop=True)


# ─── PARQUET UPDATERS ────────────────────────────────────────────────────────

_STOCK_ID_FROM_FILENAME = re.compile(r'^(.+?)_\d{4}-\d{2}-\d{2}')


def _all_stock_ids_in_dir() -> set[str]:
    """
    Return every stock ID that already has at least one parquet in data/stocks/.
    IDs are extracted from filenames of the form {sid}_{start}_to_{end}.parquet.
    """
    ids: set[str] = set()
    for f in STOCK_DIR.glob("*.parquet"):
        m = _STOCK_ID_FROM_FILENAME.match(f.stem)
        if m:
            ids.add(m.group(1))
    return ids


def _is_weekend(date_str: str) -> bool:
    """Return True if the given date falls on Saturday or Sunday."""
    return pd.to_datetime(date_str).weekday() >= 5  # 5=Sat, 6=Sun


def _next_business_day(dt: pd.Timestamp) -> pd.Timestamp:
    """Return the first weekday strictly after dt."""
    nxt = dt + pd.Timedelta(days=1)
    while nxt.weekday() >= 5:
        nxt += pd.Timedelta(days=1)
    return nxt


def _latest_date_in_parquets(files) -> pd.Timestamp | None:
    """Scan a list of parquet paths and return the maximum 'date' value found."""
    latest = None
    for f in files:
        try:
            df = pd.read_parquet(f, columns=['date'])
            mx = pd.to_datetime(df['date']).max()
            if latest is None or mx > latest:
                latest = mx
        except Exception:
            pass
    return latest


def _consolidate_and_save(
    directory: Path,
    glob_pattern: str,
    new_df: pd.DataFrame,
    dedup_cols: list,
    new_stem_fn,       # callable(start_date_str, end_date_str) -> stem string
    meta_extras: dict,
) -> Path:
    """
    Merge new_df with every parquet matching glob_pattern in directory,
    deduplicate, save as a single renamed parquet, delete the old files,
    and write a companion _meta.json.

    Parameters
    ----------
    directory      : parent directory
    glob_pattern   : e.g. "*.parquet" or "2330_*.parquet"
    new_df         : freshly fetched rows to add
    dedup_cols     : columns used for deduplication / sort
    new_stem_fn    : builds the new filename stem from (start_str, end_str)
    meta_extras    : extra keys merged into the metadata JSON

    Returns the path of the newly written parquet.
    """
    existing_files = sorted(directory.glob(glob_pattern))

    # Load + merge all existing parquets with the new data
    parts = [new_df]
    for f in existing_files:
        try:
            parts.append(pd.read_parquet(f))
        except Exception as e:
            print(f"  [warn] Could not read {f.name}: {e}")

    combined = pd.concat(parts, ignore_index=True)
    combined['date'] = pd.to_datetime(combined['date'])
    combined = (combined
                .drop_duplicates(subset=dedup_cols)
                .sort_values(dedup_cols)
                .reset_index(drop=True))

    start_str = combined['date'].min().strftime("%Y-%m-%d")
    end_str   = combined['date'].max().strftime("%Y-%m-%d")
    stem      = new_stem_fn(start_str, end_str)

    new_parquet = directory / f"{stem}.parquet"
    new_meta    = directory / f"{stem}_meta.json"

    # Write parquet first so we never lose data on a mid-run failure
    directory.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(new_parquet, index=False)

    # Remove old parquet files (and their sidecar JSONs) now that save succeeded
    for f in existing_files:
        if f == new_parquet:
            continue
        try:
            f.unlink()
        except Exception as e:
            print(f"  [warn] Could not delete old parquet {f.name}: {e}")
        for suffix in ["_meta.json", "_failed_dates.json"]:
            sidecar = f.with_name(f.stem + suffix)
            if sidecar.exists():
                try:
                    sidecar.unlink()
                except Exception:
                    pass

    # Write metadata JSON
    meta = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "start_date": start_str,
        "end_date":   end_str,
        "rows":       int(len(combined)),
        **meta_extras,
    }
    new_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return new_parquet


def _update_broker_parquet(get_api, broker_id: str, target_date: str) -> bool:
    """
    Ensure data/brokers/{broker_id}/ contains records up to target_date.

    * Skips immediately if target_date is a weekend.
    * Fetches only the gap (next business day after the current latest date).
    * The API is only authenticated (via get_api()) when a fetch is actually
      needed — if the parquet is already current no network call is made.
    * Merges with the existing parquet, renames it to reflect the new date
      range (e.g. 2021-06-30_to_2026-04-16.parquet), and writes a _meta.json.
    * Old parquet + sidecar files are deleted after the new file is saved.
    * If the API returns nothing (holiday / closed market) the parquet is left
      unchanged and False is returned.

    Returns True if new rows were saved.
    """
    if _is_weekend(target_date):
        print(f"  -> {target_date} is a weekend — skipping broker update.")
        return False

    target_dt  = pd.to_datetime(target_date)
    broker_dir = BROKER_DIR / broker_id
    latest_dt  = _latest_date_in_parquets(broker_dir.glob("*.parquet"))

    if latest_dt is not None and latest_dt >= target_dt:
        print(f"  -> Broker parquet is current (latest: {latest_dt.date()})")
        return False

    if latest_dt is None:
        print(f"  [warn] No existing broker parquet for {broker_id}. "
              f"Fetching target date only — run run_broker_activity.ps1 for full history.")
        from_str = target_date
    else:
        from_dt = _next_business_day(latest_dt)
        if from_dt > target_dt:
            print(f"  -> Broker parquet is current (latest: {latest_dt.date()})")
            return False
        from_str = from_dt.strftime("%Y-%m-%d")

    print(f"  -> Broker parquet stale (latest: {latest_dt.date() if latest_dt else 'none'}). "
          f"Fetching {from_str} -> {target_date} ...")

    new_data = fetch_broker_activity(get_api(), from_str, target_date, broker_id)

    if new_data.empty:
        print(f"  -> No broker activity for {from_str} -> {target_date} "
              f"(market closed / holiday). Parquet unchanged.")
        return False

    saved = _consolidate_and_save(
        directory     = broker_dir,
        glob_pattern  = "*.parquet",
        new_df        = new_data,
        dedup_cols    = ['date', 'stock_id'],
        new_stem_fn   = lambda s, e: f"{s}_to_{e}",
        meta_extras   = {
            "securities_trader_id": broker_id,
            "unique_stocks": int(new_data['stock_id'].nunique()),
        },
    )

    dates_added = sorted(pd.to_datetime(new_data['date']).dt.strftime("%Y-%m-%d").unique())
    print(f"  -> +{len(new_data):,} rows ({dates_added}). Saved -> {saved.name}")
    return True


def _update_stock_parquets(get_api, stock_ids: list, target_date: str) -> bool:
    """
    Ensure data/stocks/ contains price records up to target_date for every
    stock in stock_ids.

    * Skips immediately if target_date is a weekend.
    * Per stock: checks the latest date in data/stocks/{sid}_*.parquet and
      skips stocks that are already current.
    * Fetches all stale stocks in one batch call (using the earliest missing
      start date across all stale stocks).
    * For each stock that received new data: merges with the existing parquet,
      renames it (e.g. 2330_2021-06-30_to_2026-04-16.parquet), and writes a
      _meta.json. Old files are deleted after the new file is saved.

    Returns True if any new rows were saved.
    """
    if _is_weekend(target_date):
        print(f"  -> {target_date} is a weekend — skipping stock update.")
        return False

    target_dt = pd.to_datetime(target_date)

    # Union of passed-in stocks and every stock that already has a parquet,
    # so all existing parquets are kept current regardless of broker activity.
    all_sids = {str(s) for s in stock_ids} | _all_stock_ids_in_dir()

    # Per-stock staleness check
    stale: dict[str, pd.Timestamp] = {}   # sid -> earliest from_dt needed
    for sid in tqdm(sorted(all_sids), desc="Scanning parquets", unit="stock"):
        latest_dt = _latest_date_in_parquets(STOCK_DIR.glob(f"{sid}_*.parquet"))
        if latest_dt is not None and latest_dt >= target_dt:
            continue
        if latest_dt is None:
            from_dt = target_dt - pd.Timedelta(days=LOOKBACK_DAYS + 30)
        else:
            from_dt = _next_business_day(latest_dt)
            if from_dt > target_dt:
                continue
        stale[sid] = from_dt

    if not stale:
        print(f"  -> Stock parquets current for all {len(stock_ids)} stocks.")
        return False

    earliest_from = min(stale.values()).strftime("%Y-%m-%d")
    print(f"  -> {len(stale)} stocks need price updates "
          f"({earliest_from} -> {target_date}) ...")

    new_prices = fetch_stock_prices(get_api(), list(stale.keys()), earliest_from, target_date)

    if new_prices.empty:
        print(f"  -> No stock price data returned "
              f"(market closed / holiday). Parquets unchanged.")
        return False

    groups = list(new_prices.groupby('stock_id'))
    updated = 0
    for sid, grp in tqdm(groups, desc="Saving parquets", unit="stock"):
        sid = str(sid)
        _consolidate_and_save(
            directory    = STOCK_DIR,
            glob_pattern = f"{sid}_*.parquet",
            new_df       = grp.copy(),
            dedup_cols   = ['stock_id', 'date'],
            new_stem_fn  = lambda s, e, _sid=sid: f"{_sid}_{s}_to_{e}",
            meta_extras  = {"stock_id": sid},
        )
        updated += 1

    print(f"  -> Updated and renamed parquets for {updated} stocks.")
    return updated > 0


# ─── STOCK NAME LOOKUP ───────────────────────────────────────────────────────

def _lookup_stock_names(get_api, stock_ids: list) -> dict:
    """
    Return a dict mapping stock_id -> stock_name for every id in stock_ids.

    Names are read from STOCK_NAMES_FILE (data/stocks/stock_names.json).
    Any ids not yet in the file are fetched via taiwan_stock_info() and the
    file is updated in-place.  Unknown ids get an empty string fallback.
    """
    # Load existing cache
    if STOCK_NAMES_FILE.exists():
        cache: dict = json.loads(STOCK_NAMES_FILE.read_text(encoding='utf-8'))
    else:
        cache = {}

    missing = [sid for sid in stock_ids if str(sid) not in cache]

    if missing:
        print(f"  [names] Fetching names for {len(missing)} unknown stock(s) ...")
        try:
            info_df = get_api().taiwan_stock_info()
            info_df['stock_id'] = info_df['stock_id'].astype(str)
            new_names = info_df.set_index('stock_id')['stock_name'].to_dict()
            cache.update(new_names)
            STOCK_NAMES_FILE.write_text(
                json.dumps(cache, ensure_ascii=False, indent=2), encoding='utf-8'
            )
            print(f"  [names] Cache updated -> {STOCK_NAMES_FILE.name} ({len(cache)} entries)")
        except Exception as e:
            print(f"  [names] Could not fetch stock info: {e}")

    return {str(sid): cache.get(str(sid), "") for sid in stock_ids}


# ─── CHART GENERATION ────────────────────────────────────────────────────────

_UP_C   = '#e74c3c'   # red   – Taiwan convention: price up / 買超
_DOWN_C = '#27ae60'   # green – price down / 賣超


def _generate_stock_chart(
    sid: str,
    stock_name: str,
    ohlcv_df: pd.DataFrame,
    broker_df: pd.DataFrame,
    scores_df: pd.DataFrame,
    display_window: int = 20,
) -> bytes | None:
    """
    4-panel PNG for one stock.
    ohlcv_df may contain more rows than display_window — MAs are computed on the
    full history then only the last display_window candles are rendered.
    Returns raw PNG bytes, or None on failure.
    """
    import matplotlib.font_manager as _fm

    # Detect first available CJK-capable font; fall back to English labels if none
    _cjk_font = next(
        (f for f in ('Microsoft JhengHei', 'Microsoft YaHei', 'SimHei',
                     'Noto Sans CJK TC', 'Arial Unicode MS')
         if _fm.findfont(_fm.FontProperties(family=f), fallback_to_default=False)),
        None,
    )

    try:
        if ohlcv_df.empty or len(ohlcv_df) < 2:
            return None

        ohlcv_full = ohlcv_df.copy()
        ohlcv_full.index = pd.to_datetime(ohlcv_full.index)
        ohlcv_full = ohlcv_full.sort_index()

        # Compute MAs on full history, then slice to display window
        hi_col_full = 'max' if 'max' in ohlcv_full.columns else 'high'
        lo_col_full = 'min' if 'min' in ohlcv_full.columns else 'low'
        _ma_full = {}
        for _p in (5, 10, 20, 60):
            _ma_full[_p] = ohlcv_full['close'].rolling(_p, min_periods=1).mean()

        # Slice to display window
        ohlcv = ohlcv_full.iloc[-display_window:]
        for _p in (5, 10, 20, 60):
            _ma_full[_p] = _ma_full[_p].iloc[-display_window:]

        # Only keep numeric broker columns so fill_value=0 doesn't touch str cols
        _bcols = [c for c in ('buy', 'sell', 'net_buy', 'buy_amount', 'sell_amount', 'net_buy_amount')
                  if c in broker_df.columns]
        broker = broker_df[_bcols].reindex(ohlcv.index, fill_value=0)
        scores = scores_df.reindex(ohlcv.index)

        n = len(ohlcv)
        x = np.arange(n)
        date_labels = [d.strftime('%m/%d') for d in ohlcv.index]

        _rc = ({'font.sans-serif': [_cjk_font, 'DejaVu Sans'],
                'axes.unicode_minus': False}
               if _cjk_font else {'axes.unicode_minus': False})

        with matplotlib.rc_context(_rc):
            # ── Figure: 4 panels ─────────────────────────────────────────────
            fig = plt.figure(figsize=(12, 12), facecolor='white')
            fig.suptitle(f'{sid}  {stock_name}', fontsize=13, fontweight='bold', y=0.99)
            gs = gridspec.GridSpec(4, 1, height_ratios=[3, 1.5, 1.5, 1.5],
                                   hspace=0.08, top=0.95, bottom=0.06)
            ax0 = fig.add_subplot(gs[0])   # K線
            ax1 = fig.add_subplot(gs[1])   # 淨買超
            ax2 = fig.add_subplot(gs[2])   # 買入/賣出張數
            ax3 = fig.add_subplot(gs[3])   # 模型分數

            # ── Panel 0: Candlestick ─────────────────────────────────────────
            for i in range(n):
                o = ohlcv['open'].iat[i]
                h = ohlcv[hi_col_full].iat[i]
                l = ohlcv[lo_col_full].iat[i]
                c = ohlcv['close'].iat[i]
                color = _UP_C if c >= o else _DOWN_C
                ax0.plot([i, i], [l, h], color=color, linewidth=1)
                body_bot = min(o, c)
                body_h   = max(abs(c - o), o * 0.001)
                ax0.add_patch(Rectangle((i - 0.35, body_bot), 0.7, body_h,
                                        facecolor=color, edgecolor=color))
            # Moving averages (computed on full history, sliced to display window)
            for period, color, lw in [(5, '#e67e22', 1.2), (10, '#8e44ad', 1.2), (20, '#2980b9', 1.2), (60, '#7f8c8d', 1.0)]:
                ax0.plot(x, _ma_full[period].values, color=color, linewidth=lw, label=f'MA{period}')
            ax0.legend(loc='upper left', fontsize=7, framealpha=0.6)

            ax0.set_xlim(-0.5, n - 0.5)
            ax0.set_xticks([])
            ax0.set_ylabel('Price (TWD)', fontsize=8)
            ax0.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:,.0f}'))
            ax0.grid(True, alpha=0.3, linestyle='--')

            # ── Broker data ───────────────────────────────────────────────────
            _nb  = broker['net_buy'].fillna(0)       if 'net_buy'        in broker.columns else pd.Series(0,   index=ohlcv.index)
            _b   = broker['buy'].fillna(0)            if 'buy'            in broker.columns else pd.Series(0,   index=ohlcv.index)
            _sl  = broker['sell'].fillna(0)           if 'sell'           in broker.columns else pd.Series(0,   index=ohlcv.index)
            _ba  = broker['buy_amount'].fillna(0)     if 'buy_amount'     in broker.columns else pd.Series(0.0, index=ohlcv.index)
            _sa  = broker['sell_amount'].fillna(0)    if 'sell_amount'    in broker.columns else pd.Series(0.0, index=ohlcv.index)

            net_lots  = _nb / 1000
            buy_lots  = _b  / 1000
            sell_lots = _sl / 1000
            avg_buy   = np.where(_b  > 0, _ba / _b,  0.0)
            avg_sell  = np.where(_sl > 0, _sa / _sl, 0.0)

            # ── Panel 1: 淨買超 (張) ─────────────────────────────────────────
            lot_colors = [_UP_C if v >= 0 else _DOWN_C for v in net_lots]
            ax1.bar(x, net_lots.values, color=lot_colors, alpha=0.85, width=0.6)
            ax1.axhline(0, color='black', linewidth=0.7)
            ax1.set_xlim(-0.5, n - 0.5)
            ax1.set_xticks([])
            ax1.set_ylabel('淨買超 (張)', fontsize=8)
            ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:,.0f}'))
            ax1.grid(True, alpha=0.3, linestyle='--', axis='y')

            # ── Panel 2: 買入/賣出張數 bar chart + 均價標示 ────────────────────
            bar_w = 0.35
            ax2.bar(x - bar_w / 2, buy_lots.values,  width=bar_w,
                    color=_UP_C,   alpha=0.85, label='買入(張)')
            ax2.bar(x + bar_w / 2, sell_lots.values, width=bar_w,
                    color=_DOWN_C, alpha=0.85, label='賣出(張)')
            ax2.axhline(0, color='black', linewidth=0.7)

            # 均價標示 (小字，向右旋轉)
            _ymax = max(buy_lots.max(), sell_lots.max(), 1)
            _pad  = _ymax * 0.05
            for i in range(n):
                if buy_lots.iat[i] > 0 and avg_buy[i] > 0:
                    ax2.text(i - bar_w / 2, buy_lots.iat[i] + _pad,
                             f'{avg_buy[i]:,.0f}',
                             ha='center', va='bottom', fontsize=5,
                             color='#922b21', rotation=-90)
                if sell_lots.iat[i] > 0 and avg_sell[i] > 0:
                    ax2.text(i + bar_w / 2, sell_lots.iat[i] + _pad,
                             f'{avg_sell[i]:,.0f}',
                             ha='center', va='bottom', fontsize=5,
                             color='#1a5276', rotation=-90)

            ax2.set_xlim(-0.5, n - 0.5)
            ax2.set_xticks([])
            ax2.set_ylabel('張數', fontsize=8)
            ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:,.0f}'))
            ax2.legend(loc='upper left', fontsize=7, framealpha=0.6)
            ax2.grid(True, alpha=0.3, linestyle='--', axis='y')

            # ── Panel 3: 模型分數 ────────────────────────────────────────────
            prob_long  = scores['pred_prob_long'].copy()  if 'pred_prob_long'  in scores.columns else pd.Series(np.nan, index=ohlcv.index)
            prob_short = scores['pred_prob_short'].copy() if 'pred_prob_short' in scores.columns else pd.Series(np.nan, index=ohlcv.index)

            ax3.plot(x, prob_long.values,  color=_UP_C,   linewidth=2,
                     label=f'Long  (>={LONG_THRESHOLD:.0%})')
            ax3.plot(x, prob_short.values, color=_DOWN_C, linewidth=2,
                     label=f'Short (>={SHORT_THRESHOLD:.0%})')
            ax3.axhline(LONG_THRESHOLD,  color=_UP_C,   linestyle='--', alpha=0.45, linewidth=1)
            ax3.axhline(SHORT_THRESHOLD, color=_DOWN_C, linestyle='--', alpha=0.45, linewidth=1)
            ax3.set_xlim(-0.5, n - 0.5)
            ax3.set_ylabel('Probability', fontsize=8)
            ax3.set_xticks(x)
            ax3.set_xticklabels(date_labels, rotation=30, ha='right', fontsize=7)
            ax3.grid(True, alpha=0.3, linestyle='--')
            ax3.legend(loc='upper left', fontsize=7, framealpha=0.6)

            # ── Export ───────────────────────────────────────────────────────
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=90, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            buf.seek(0)
            return buf.read()

    except Exception as exc:
        print(f"  [chart] {sid}: {exc}")
        return None


# ─── EMAIL ───────────────────────────────────────────────────────────────────

def _send_email(
    target_date: str,
    *,
    error_msg: str | None = None,
    n_long: int = 0,
    n_short: int = 0,
    n_candidates: int = 0,
    output_path: Path | None = None,
    top_long_df: "pd.DataFrame | None" = None,
    charts: "list[bytes] | None" = None,
    portfolio_data: "dict | None" = None,
) -> None:
    """
    Send a daily update notification via Gmail SMTP (HTML body).

    Pass ``error_msg`` to send a failure notice; omit it for the normal
    success summary (which also attaches the output CSV).
    Credentials are read from .env.
    """
    load_dotenv()
    sender   = os.environ.get("MY_GMAIL")
    password = os.environ.get("MY_GMAIL_APP_PASSWORD")
    receiver = os.environ.get("My_RECEIVER")
    cc       = os.environ.get("MY_CC", "")

    if not all([sender, password, receiver]):
        print("  [email] Missing email credentials in .env — skipping email.")
        return

    if error_msg:
        subject  = f"[FAILED] BMEM Daily Update — {target_date}"
        body_txt = f"BMEM Daily Update FAILED: {target_date}\n{'='*50}\n\n{error_msg}\n"
        body_html = f"""
<html><body>
<h2 style="color:#c0392b;">BMEM Daily Update FAILED — {target_date}</h2>
<pre style="background:#f8f8f8;padding:12px;border-radius:4px;">{error_msg}</pre>
</body></html>"""
    else:
        subject = f"[OK] BMEM Daily Signals — {target_date}"

        # Plain-text fallback
        body_txt = (
            f"BMEM Daily Update: {target_date}\n{'='*50}\n\n"
            f"Candidates scored : {n_candidates}\n"
            f"Long  signals (>={LONG_THRESHOLD:.0%})  : {n_long}\n"
            f"Short signals (>={SHORT_THRESHOLD:.0%})  : {n_short}\n"
        )
        if top_long_df is not None and not top_long_df.empty:
            body_txt += f"\nTop long candidates:\n{top_long_df.to_string(index=False)}\n"

        # HTML table for top candidates
        table_html = ""
        if top_long_df is not None and not top_long_df.empty:
            col_labels = {
                '股票':           '股票',
                'pred_prob_long': 'Long 機率',
                'pred_prob_short':'Short 機率',
            }
            header_cells = "".join(
                f'<th style="padding:6px 12px;background:#2c3e50;color:#fff;'
                f'text-align:center;">{col_labels.get(c, c)}</th>'
                for c in top_long_df.columns
            )
            rows_html = ""
            for i, row in top_long_df.iterrows():
                bg = "#f2f2f2" if i % 2 == 0 else "#ffffff"
                cells = ""
                for c, v in row.items():
                    if c in ('pred_prob_long', 'pred_prob_short'):
                        cell_val = f"{v:.2%}"
                    else:
                        cell_val = str(v)
                    cells += (
                        f'<td style="padding:6px 12px;text-align:center;">'
                        f'{cell_val}</td>'
                    )
                rows_html += f'<tr style="background:{bg};">{cells}</tr>'

            table_html = f"""
<h3>Top Long 候選股票</h3>
<table style="border-collapse:collapse;font-family:Arial,sans-serif;font-size:14px;">
  <thead><tr>{header_cells}</tr></thead>
  <tbody>{rows_html}</tbody>
</table>"""

        # ── Portfolio sections ────────────────────────────────────────────────
        portfolio_html = ""
        if portfolio_data:
            executed   = portfolio_data.get("executed_today", [])
            holdings   = portfolio_data.get("holdings", {})
            orders     = portfolio_data.get("tonight_orders", [])
            price_lkp  = portfolio_data.get("price_lkp", {})
            name_map_p = portfolio_data.get("name_map", {})

            def _stock_label(sid, name: str = "") -> str:
                """Combined stock cell text, e.g. '3189 景碩' (code + space + name)."""
                return f'{sid} {name or name_map_p.get(str(sid), "")}'.strip()

            _cell = 'style="padding:5px 10px;text-align:center;border:1px solid #ddd;"'
            _hcell = (
                'style="padding:5px 10px;background:#2c3e50;color:#fff;'
                'text-align:center;border:1px solid #2c3e50;"'
            )

            # Section 1: executed today
            exec_html = ""
            if executed:
                rows = ""
                for t in executed:
                    action_color = "#27ae60" if t["action"] == "BUY" else "#e74c3c"
                    ret_str = f"{float(t['return_pct']):.2%}" if t.get("return_pct") != "" else "—"
                    rows += (
                        f'<tr>'
                        f'<td {_cell}><b style="color:{action_color};">{t["action"]}</b></td>'
                        f'<td {_cell}>{_stock_label(t["stock_id"], t.get("stock_name",""))}</td>'
                        f'<td {_cell}>{t.get("price","")}</td>'
                        f'<td {_cell}>{t.get("reason","")}</td>'
                        f'<td {_cell}>{ret_str}</td>'
                        f'</tr>'
                    )
                exec_html = f"""
<h3 style="margin-top:20px;">今日執行確認</h3>
<table style="border-collapse:collapse;font-size:13px;">
  <tr><th {_hcell}>動作</th><th {_hcell}>股票</th>
      <th {_hcell}>執行價</th><th {_hcell}>原因</th><th {_hcell}>報酬</th></tr>
  {rows}
</table>"""

            # Section 2: current holdings
            # Stocks queued for a tomorrow-open sell were already removed from
            # `holdings` by update_portfolio, but the position is still open
            # tonight — merge them back into the display with a (待賣出) tag.
            pending_sell_orders = [
                o for o in orders
                if o.get("order_type") == "market_open" and o.get("action") == "SELL"
            ]
            holdings_html = ""
            if holdings or pending_sell_orders:
                rows = ""
                for sid, h in holdings.items():
                    close = price_lkp.get(sid, {}).get("close")
                    ret_str = f"{(close/h['entry_price']-1):.2%}" if close else "—"
                    ret_color = "#27ae60" if close and close >= h["entry_price"] else "#e74c3c"
                    stop_price = round(h["highest_price"] * TRAILING_STOP_RATIO, 2)
                    peak_prob = h.get("peak_prob_long")
                    prob_str = f"{peak_prob:.2%}" if peak_prob is not None else "—"
                    rows += (
                        f'<tr>'
                        f'<td {_cell}>{_stock_label(sid)}</td>'
                        f'<td {_cell}>{h["entry_date"]}</td>'
                        f'<td {_cell}>{h["entry_price"]}</td>'
                        f'<td {_cell}>{close if close else "—"}</td>'
                        f'<td {_cell}><b style="color:{ret_color};">{ret_str}</b></td>'
                        f'<td {_cell}>{h["highest_price"]}</td>'
                        f'<td {_cell}>{stop_price}</td>'
                        f'<td {_cell}>{prob_str}</td>'
                        f'</tr>'
                    )
                for o in pending_sell_orders:
                    sid         = str(o["stock_id"])
                    entry_price = o.get("entry_price")
                    close       = price_lkp.get(sid, {}).get("close")
                    ret_str   = f"{(close/entry_price-1):.2%}" if close and entry_price else "—"
                    ret_color = "#27ae60" if close and entry_price and close >= entry_price else "#e74c3c"
                    peak_prob = o.get("peak_prob_long")
                    prob_str = f"{peak_prob:.2%}" if peak_prob is not None else "—"
                    rows += (
                        f'<tr>'
                        f'<td {_cell}>{_stock_label(sid, o.get("stock_name",""))} '
                        f'<span style="color:#e67e22;">(待賣出)</span></td>'
                        f'<td {_cell}>{o.get("entry_date","")}</td>'
                        f'<td {_cell}>{entry_price if entry_price is not None else "—"}</td>'
                        f'<td {_cell}>{close if close else "—"}</td>'
                        f'<td {_cell}><b style="color:{ret_color};">{ret_str}</b></td>'
                        f'<td {_cell}>—</td>'
                        f'<td {_cell}>—</td>'
                        f'<td {_cell}>{prob_str}</td>'
                        f'</tr>'
                    )
                holdings_html = f"""
<h3 style="margin-top:20px;">目前持倉</h3>
<table style="border-collapse:collapse;font-size:13px;">
  <tr><th {_hcell}>股票</th><th {_hcell}>進場日</th><th {_hcell}>進場價</th>
      <th {_hcell}>現價</th><th {_hcell}>報酬</th>
      <th {_hcell}>歷史高點</th><th {_hcell}>停損線</th>
      <th {_hcell}>最高Long機率</th></tr>
  {rows}
</table>"""
            else:
                holdings_html = '<p style="color:#888;margin-top:12px;">目前無持倉</p>'

            # Section 3: tonight's orders
            orders_html = ""
            if orders:
                cond_orders  = [o for o in orders if o["order_type"] == "conditional_stop"]
                mkt_sells    = [o for o in orders if o["order_type"] == "market_open" and o["action"] == "SELL"]
                mkt_buys     = [o for o in orders if o["order_type"] == "market_open" and o["action"] == "BUY"]

                def _order_rows(order_list: list, extra_col: str, extra_key: str) -> str:
                    out = ""
                    for o in order_list:
                        ret_str = f"{float(o['unrealized_return_pct']):.2%}" if o.get("unrealized_return_pct") is not None else "—"
                        extra   = o.get(extra_key, "")
                        extra   = f"{float(extra):.2%}" if extra else "—"
                        out += (
                            f'<tr>'
                            f'<td {_cell}>{_stock_label(o["stock_id"], o.get("stock_name",""))}</td>'
                            f'<td {_cell}>{o.get("reason","")}</td>'
                            f'<td {_cell}>{extra}</td>'
                            f'<td {_cell}>{ret_str}</td>'
                            f'</tr>'
                        )
                    return out

                cond_rows = ""
                for o in cond_orders:
                    ret_str = f"{float(o['unrealized_return_pct']):.2%}" if o.get("unrealized_return_pct") is not None else "—"
                    cond_rows += (
                        f'<tr>'
                        f'<td {_cell}>{_stock_label(o["stock_id"], o.get("stock_name",""))}</td>'
                        f'<td {_cell}><b>{o["trigger_price"]}</b></td>'
                        f'<td {_cell}>{ret_str}</td>'
                        f'</tr>'
                    )

                sell_rows = _order_rows(mkt_sells, "Short機率", "signal_prob_short")
                buy_rows  = _order_rows(mkt_buys,  "Long機率",  "signal_prob_long")

                cond_table = f"""
<b>條件單（停損）</b>
<table style="border-collapse:collapse;font-size:13px;margin-bottom:8px;">
  <tr><th {_hcell}>股票</th><th {_hcell}>觸發價</th><th {_hcell}>未實現報酬</th></tr>
  {cond_rows}
</table>""" if cond_orders else ""

                sell_table = f"""
<b>預約賣出（明日開盤）</b>
<table style="border-collapse:collapse;font-size:13px;margin-bottom:8px;">
  <tr><th {_hcell}>股票</th><th {_hcell}>原因</th><th {_hcell}>Short機率</th><th {_hcell}>未實現報酬</th></tr>
  {sell_rows}
</table>""" if mkt_sells else ""

                buy_table = f"""
<b>預約買入（明日開盤）</b>
<table style="border-collapse:collapse;font-size:13px;margin-bottom:8px;">
  <tr><th {_hcell}>股票</th><th {_hcell}>原因</th><th {_hcell}>Long機率</th><th {_hcell}>未實現報酬</th></tr>
  {buy_rows}
</table>""" if mkt_buys else ""

                orders_html = f"""
<h3 style="margin-top:20px;">今晚掛單建議</h3>
{cond_table}{sell_table}{buy_table}
<p style="font-size:11px;color:#888;">詳細下單參數見附件 orders_{target_date}.json</p>"""
            else:
                orders_html = '<p style="color:#888;margin-top:12px;">今晚無需掛單</p>'

            portfolio_html = exec_html + holdings_html + orders_html

        # Build CID image tags (Gmail blocks data: URIs; CID inline works)
        charts_html = ""
        if charts:
            charts_html = '<h3 style="margin-top:24px;">個股圖表 (K線 / 進出 / 分數)</h3>'
            for i in range(len(charts)):
                charts_html += (
                    f'<img src="cid:chart_{i}" '
                    f'style="width:100%;max-width:960px;display:block;margin:8px 0;" />'
                )

        body_html = f"""
<html><body style="font-family:Arial,sans-serif;">
<h2 style="color:#27ae60;">BMEM Daily Signals — {target_date}</h2>
<table style="font-size:14px;margin-bottom:16px;">
  <tr><td style="padding:4px 12px 4px 0;color:#555;">候選股票數</td>
      <td><strong>{n_candidates}</strong></td></tr>
  <tr><td style="padding:4px 12px 4px 0;color:#555;">Long 訊號 (&ge;{LONG_THRESHOLD:.0%})</td>
      <td><strong style="color:#27ae60;">{n_long}</strong></td></tr>
  <tr><td style="padding:4px 12px 4px 0;color:#555;">Short 訊號 (&ge;{SHORT_THRESHOLD:.0%})</td>
      <td><strong style="color:#e74c3c;">{n_short}</strong></td></tr>
</table>
{portfolio_html}
{table_html}
{charts_html}
</body></html>"""

    # ── Assemble MIME structure ───────────────────────────────────────────────
    # multipart/mixed
    # ├── multipart/alternative
    # │   ├── text/plain
    # │   └── multipart/related   (only when charts present)
    # │       ├── text/html
    # │       └── image/png × N   (Content-ID: chart_i)
    # └── text/csv (signals attachment)

    msg = MIMEMultipart('mixed')
    msg['From']    = sender
    msg['To']      = receiver
    if cc:
        msg['Cc']  = cc
    msg['Subject'] = subject

    alt = MIMEMultipart('alternative')
    alt.attach(MIMEText(body_txt, 'plain', 'utf-8'))

    if not error_msg and charts:
        related = MIMEMultipart('related')
        related.attach(MIMEText(body_html, 'html', 'utf-8'))
        for i, png_bytes in enumerate(charts):
            img = MIMEImage(png_bytes, 'png')
            img['Content-ID']          = f'<chart_{i}>'
            img['Content-Disposition'] = 'inline'
            related.attach(img)
        alt.attach(related)
    else:
        alt.attach(MIMEText(body_html, 'html', 'utf-8'))

    msg.attach(alt)

    if not error_msg and output_path is not None and output_path.exists():
        csv_part = MIMEBase('text', 'csv')
        csv_part.set_payload(output_path.read_bytes())
        encoders.encode_base64(csv_part)
        csv_part['Content-Disposition'] = f'attachment; filename="{output_path.name}"'
        msg.attach(csv_part)

    if not error_msg and portfolio_data:
        orders_path = portfolio_data.get("orders_path")
        if orders_path is not None and Path(orders_path).exists():
            json_part = MIMEBase('application', 'json')
            json_part.set_payload(Path(orders_path).read_bytes())
            encoders.encode_base64(json_part)
            json_part['Content-Disposition'] = (
                f'attachment; filename="{Path(orders_path).name}"'
            )
            msg.attach(json_part)

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(sender, password)
            smtp.send_message(msg)
        print(f"  [email] Email sent to {receiver}")
    except Exception as e:
        print(f"  [email] Failed to send email: {e}")


# ─── MAIN PIPELINE ────────────────────────────────────────────────────────────

def run_daily_update(
    target_date: str,
    broker_id: str,
    output_dir: Path | None = None,
    data_root: Path = DEFAULT_DATA_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> pd.DataFrame | None:
    """
    Execute the full daily update pipeline for one target date.

    Parameters
    ----------
    target_date : "YYYY-MM-DD" string
    broker_id   : broker trader code and path namespace
    output_dir  : directory where signals_{target_date}.csv is written

    Returns
    -------
    pd.DataFrame of signals (all candidate stocks for target_date),
    or None if no data was available (weekend / holiday / insufficient history).
    """
    paths = BrokerPaths(broker_id, data_root, output_root)
    broker_id = paths.broker_id
    hmm_params = paths.hmm_model_path
    xgb_long_path = paths.xgboost_model_path("long")
    xgb_short_path = paths.xgboost_model_path("short")
    output_dir = Path(output_dir) if output_dir else paths.daily_dir

    # Configure the shared-data helpers for optional non-default data roots.
    global BROKER_DIR, STOCK_DIR, STOCK_NAMES_FILE
    BROKER_DIR = paths.data_root / "brokers"
    STOCK_DIR = paths.stock_dir
    STOCK_NAMES_FILE = STOCK_DIR / "stock_names.json"

    print(f"\n{'='*60}")
    print(f"  BMEM Daily Update -- {target_date}")
    print(f"{'='*60}")

    # ── 0. Lazy API — only authenticates on first fetch ──────────────────────
    load_dotenv()
    _api_cache: list = [None]

    def _get_api():
        if _api_cache[0] is None:
            api_key = os.environ.get("FINMIND_API_KEY")
            if not api_key:
                raise EnvironmentError("FINMIND_API_KEY not set. Add it to your .env file.")
            _api_cache[0] = DataLoader()
            _api_cache[0].login_by_token(api_token=api_key)
            print("  [auth] Authenticated with FinMind API.")
        return _api_cache[0]

    # ── 1. Ensure broker parquet is current ───────────────────────────────────
    print(f"\n[1/5] Checking broker {broker_id} parquet ...")
    _update_broker_parquet(_get_api, broker_id, target_date)

    # ── 2. Load broker history; verify today has activity ─────────────────────
    print(f"\n[2/5] Loading broker history ...")
    cutoff_dt = pd.to_datetime(target_date) - pd.Timedelta(days=LOOKBACK_DAYS)
    target_dt = pd.to_datetime(target_date)

    all_broker = _load_all_broker_parquets(broker_id)

    if all_broker.empty:
        msg = "No broker data found. Nothing to score."
        print(f"  -> {msg}")
        _send_email(target_date, error_msg=msg)
        return None

    today_broker = all_broker[all_broker['date'] == target_dt]
    if today_broker.empty:
        msg = (f"No broker activity on {target_date} "
               f"(weekend / holiday / market closed). Nothing to score.")
        print(f"  -> {msg}")
        _send_email(target_date, error_msg=msg)
        return None

    today_stock_ids = today_broker['stock_id'].astype(str).unique().tolist()
    print(f"  -> {len(today_broker)} records | "
          f"{len(today_stock_ids)} stocks active today")

    # Apply rolling window
    combined_broker = all_broker[all_broker['date'] >= cutoff_dt].copy()
    if 'securities_trader_id' not in combined_broker.columns:
        combined_broker['securities_trader_id'] = broker_id

    window_stock_ids = combined_broker['stock_id'].astype(str).unique().tolist()
    print(f"  -> Window: {cutoff_dt.date()} -> {target_date} | "
          f"{len(window_stock_ids)} stocks in broker history")

    # ── 3. Ensure stock parquets are current; load prices ─────────────────────
    print(f"\n[3/5] Checking stock parquets ...")
    _update_stock_parquets(_get_api, window_stock_ids, target_date)

    combined_stocks = _load_stock_parquets(window_stock_ids)
    if not combined_stocks.empty:
        combined_stocks['date'] = pd.to_datetime(combined_stocks['date'])
        combined_stocks = combined_stocks[combined_stocks['date'] >= cutoff_dt]

    print(f"  -> Broker rows: {len(combined_broker):,} | "
          f"Stock rows: {len(combined_stocks):,}")

    # ── 4. Compute observation features ───────────────────────────────────────
    print(f"\n[4/5] Computing observation features ...")
    feature_df = compute_observation_features(
        combined_broker, combined_stocks, disable_standardize=True,
        contamination_lookback_days=180,  # covers full HMM 120-day window (~180 calendar days)
    )
    feature_df['date'] = pd.to_datetime(feature_df['date'])

    valid_df = feature_df.dropna(subset=FEATURE_COLS).copy()
    today_valid = valid_df[valid_df['date'] == target_dt]

    if today_valid.empty:
        msg = (f"No valid feature rows for {target_date}. "
               f"Possibly insufficient history (need >=60 trading days).")
        print(f"  -> {msg}")
        _send_email(target_date, error_msg=msg)
        return None

    print(f"  -> {len(today_valid)} valid signal candidates for {target_date}")

    # ── 5. HMM rolling inference ───────────────────────────────────────────────
    print(f"\n[5/5] Running HMM + XGBoost inference ...")
    if not hmm_params.exists():
        raise FileNotFoundError(f"HMM params not found: {hmm_params}")

    hmm_model = load_hmm_model(str(hmm_params))

    # Phase 1: all stocks × today only (1 predict_proba call per stock)
    hmm_input = valid_df.sort_values(['stock_id', 'securities_trader_id', 'date'])
    today_hmm = compute_rolling_hmm_proba(
        hmm_input, hmm_model, feature_cols=FEATURE_COLS, window=HMM_WINDOW,
        inference_dates=[target_dt],
    )
    print(f"  -> State probabilities computed for {len(today_hmm)} rows")

    # ── 6. XGBoost signal generation ──────────────────────────────────────────
    for path, label in [(xgb_long_path, "long"), (xgb_short_path, "short")]:
        if not path.exists():
            raise FileNotFoundError(f"XGBoost {label} model not found: {path}")

    clf_long  = load_xgb_model(str(xgb_long_path))
    clf_short = load_xgb_model(str(xgb_short_path))

    xgb_feature_cols = clf_long.get_booster().feature_names
    short_feature_cols = clf_short.get_booster().feature_names
    if xgb_feature_cols and short_feature_cols and xgb_feature_cols != short_feature_cols:
        raise ValueError("Long and short XGBoost models use different feature columns.")
    if not xgb_feature_cols:
        xgb_feature_cols = (
            ['z_t', 'c_t', 'a_t', 's_t', 'm_t', 'bias_60d', 'net_buy_amt_60d']
            + [f'prob_S{i}' for i in range(hmm_model.n_components)]
        )

    missing_feats = [c for c in xgb_feature_cols if c not in today_hmm.columns]
    if missing_feats:
        raise ValueError(f"Missing XGBoost input features: {missing_feats}")

    signals_df = generate_signals(
        today_hmm, clf_long, clf_short,
        feature_cols=xgb_feature_cols,
        long_threshold=LONG_THRESHOLD,
        short_threshold=SHORT_THRESHOLD,
    )

    # ── 7. Save output CSV ─────────────────────────────────────────────────────
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"signals_{target_date}.csv"

    # Top-N stocks by today's long probability
    top_stocks = (
        signals_df.sort_values('pred_prob_long', ascending=False)
        .head(OUTPUT_TOP_N)['stock_id'].astype(str).tolist()
    )

    # Last OUTPUT_HIST_DAYS trading dates from valid_df
    all_valid_dates = sorted(valid_df['date'].unique())
    hist_dates = [d for d in all_valid_dates if d <= target_dt][-OUTPUT_HIST_DAYS:]

    # Phase 2: top-N stocks × last OUTPUT_HIST_DAYS dates (20 predict_proba calls per stock)
    hist_input = hmm_input[hmm_input['stock_id'].astype(str).isin(top_stocks)]
    hmm_hist = compute_rolling_hmm_proba(
        hist_input, hmm_model, feature_cols=FEATURE_COLS, window=HMM_WINDOW,
        inference_dates=hist_dates, show_progress=False,
    )

    signals_hist = generate_signals(
        hmm_hist, clf_long, clf_short,
        feature_cols=xgb_feature_cols,
        long_threshold=LONG_THRESHOLD,
        short_threshold=SHORT_THRESHOLD,
    )
    signals_hist['stock_id'] = signals_hist['stock_id'].astype(str)
    signals_hist['date_str'] = pd.to_datetime(signals_hist['date']).dt.strftime('%Y-%m-%d')

    long_pivot = signals_hist.pivot(index='stock_id', columns='date_str', values='pred_prob_long')
    long_pivot = long_pivot[sorted(long_pivot.columns, reverse=True)]
    long_pivot.columns = [f"long_{c}" for c in long_pivot.columns]

    short_pivot = signals_hist.pivot(index='stock_id', columns='date_str', values='pred_prob_short')
    short_pivot = short_pivot[sorted(short_pivot.columns, reverse=True)]
    short_pivot.columns = [f"short_{c}" for c in short_pivot.columns]

    out = pd.concat([long_pivot, short_pivot], axis=1).reset_index()
    today_long_col = f"long_{target_date}"
    if today_long_col in out.columns:
        out = out.sort_values(today_long_col, ascending=False)
    out = out.reset_index(drop=True)

    # Insert stock_name after stock_id.
    # Include any stocks already in the portfolio so holdings/order cells get names
    # even when those stocks are not in today's top-N output.
    from portfolio_tracker import load_holdings as _load_holdings
    _prior_state = _load_holdings(output_dir)
    _portfolio_sids = (
        list(_prior_state.get("holdings", {}).keys())
        + [pb["stock_id"] for pb in _prior_state.get("pending_buys", [])]
        + [ps["stock_id"] for ps in _prior_state.get("pending_sells", [])]
    )
    _all_name_ids = list(dict.fromkeys(out['stock_id'].tolist() + [str(s) for s in _portfolio_sids]))
    name_map = _lookup_stock_names(_get_api, _all_name_ids)
    out.insert(1, 'stock_name', out['stock_id'].map(name_map))

    out.to_csv(output_path, index=False, encoding='utf-8-sig')

    # ── Summary ───────────────────────────────────────────────────────────────
    n_long  = int(signals_df['signal_long'].sum())
    n_short = int(signals_df['signal_short'].sum())

    print(f"\n{'─'*60}")
    print(f"  Results for {target_date}")
    print(f"  Candidates scored : {len(signals_df)}")
    print(f"  Long  signals (>={LONG_THRESHOLD:.0%})  : {n_long}")
    print(f"  Short signals (>={SHORT_THRESHOLD:.0%})  : {n_short}")
    print(f"  Output CSV        : {output_path}")
    print(f"{'─'*60}\n")

    top10_today = signals_df.sort_values('pred_prob_long', ascending=False).head(10).copy()
    top10_today['stock_name'] = top10_today['stock_id'].astype(str).map(name_map)
    top_long_df = top10_today[['stock_id', 'stock_name', 'pred_prob_long', 'pred_prob_short']].reset_index(drop=True)
    top_long_df.insert(0, '股票', top_long_df['stock_id'].astype(str) + ' ' + top_long_df['stock_name'].fillna(''))
    top_long_df['股票'] = top_long_df['股票'].str.strip()
    top_long_df = top_long_df.drop(columns=['stock_id', 'stock_name'])
    print("  Top-10 long candidates:")
    print(top_long_df.to_string(index=False))
    print()

    # ── 8. Generate per-stock charts ──────────────────────────────────────────
    print("  Generating stock charts ...")
    # Held stocks first, then the rest of the top-N (reuses _prior_state loaded above)
    held_sids = [str(s) for s in _prior_state.get("holdings", {}).keys()]
    _out_sids = out['stock_id'].tolist()
    chart_order = list(dict.fromkeys(held_sids))  # all held stocks first, deduped
    chart_order += [s for s in _out_sids if s not in set(chart_order)]
    charts: list[bytes] = []
    for sid in chart_order:
        # Full history for MA computation (up to LOOKBACK_DAYS); chart renders only last OUTPUT_HIST_DAYS candles
        ohlcv_20d = combined_stocks[
            combined_stocks['stock_id'].astype(str) == sid
        ].set_index('date').sort_index()

        broker_20d = combined_broker[
            (combined_broker['stock_id'].astype(str) == sid) &
            (combined_broker['date'].isin(hist_dates))
        ].set_index('date').sort_index()

        scores_20d = signals_hist[
            signals_hist['stock_id'] == sid
        ][['date', 'pred_prob_long', 'pred_prob_short']].set_index('date').sort_index()

        png = _generate_stock_chart(
            sid, name_map.get(sid, sid), ohlcv_20d, broker_20d, scores_20d,
            display_window=OUTPUT_HIST_DAYS,
        )
        if png:
            charts.append(png)

    print(f"  -> {len(charts)} chart(s) generated")

    # ── 9. Portfolio tracking ─────────────────────────────────────────────────
    print("  Updating portfolio tracker ...")
    today_price_df = combined_stocks[
        combined_stocks['date'] == target_dt
    ][['stock_id', 'open', 'close', 'max', 'min']].copy()

    executed_today, holdings, tonight_orders = update_portfolio(
        daily_dir=output_dir,
        target_date=target_date,
        signals_df=signals_df,
        price_df=today_price_df,
        name_map=name_map,
        broker_id=broker_id,
        max_holdings=1,
        long_threshold=LONG_THRESHOLD,
        short_threshold=SHORT_THRESHOLD,
    )

    # Build price lookup for email rendering (current close per held stock)
    today_price_df['stock_id'] = today_price_df['stock_id'].astype(str)
    price_lkp_email = today_price_df.set_index('stock_id').to_dict('index')

    orders_path = output_dir / f"orders_{target_date}.json"
    portfolio_data = {
        "executed_today": executed_today,
        "holdings":       holdings,
        "tonight_orders": tonight_orders,
        "price_lkp":      price_lkp_email,
        "name_map":       name_map,
        "orders_path":    orders_path,
    }

    n_exec = len(executed_today)
    n_ord  = len(tonight_orders)
    print(f"  -> {n_exec} trade(s) executed today | {n_ord} order(s) for tonight")

    _send_email(
        target_date=target_date,
        n_long=n_long,
        n_short=n_short,
        n_candidates=len(signals_df),
        output_path=output_path,
        top_long_df=top_long_df,
        charts=charts,
        portfolio_data=portfolio_data,
    )

    return signals_df


# ─── CLI ENTRY POINT ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "BMEM daily update: keep broker + stock parquets current, compute features, "
            "run HMM + XGBoost inference, and save signals to CSV."
        )
    )
    add_broker_path_args(parser)
    parser.add_argument(
        "--date",
        default=datetime.today().strftime("%Y-%m-%d"),
        help="Target trading date YYYY-MM-DD (default: today)",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        help="Override the broker-specific daily signal directory.",
    )
    args = parser.parse_args()

    try:
        run_daily_update(
            target_date=args.date,
            broker_id=args.broker_id,
            output_dir=args.outdir,
            data_root=args.data_root,
            output_root=args.output_root,
        )
    except Exception:
        tb = traceback.format_exc()
        print(tb)
        _send_email(args.date, error_msg=tb)
        sys.exit(1)


if __name__ == "__main__":
    main()

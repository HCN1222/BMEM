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
  8. Saves one CSV per run: outputs/daily/signals_YYYY-MM-DD.csv

Parquet update behaviour
------------------------
* New rows are appended to incremental.parquet and deduplicated — original
  historical parquets are never modified.
* If target_date is a weekend the update step is skipped automatically.
* If the API returns no data (market holiday) the parquet is left unchanged and
  the pipeline exits cleanly with no signals for that date.

Usage
-----
    python src/daily_update.py                     # target = today
    python src/daily_update.py --date 2026-04-16
    python src/daily_update.py --date 2026-04-16 --outdir ./outputs/daily
"""

import os
import re
import sys
import json
import argparse
import smtplib
import traceback
from email.message import EmailMessage
from datetime import datetime
from pathlib import Path

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
    XGB_FEATURE_COLS,
)

# ─── DEFAULT PATHS (relative to repo root) ───────────────────────────────────
_ROOT = _SRC_DIR.parent

BROKER_DIR     = _ROOT / "data" / "brokers"
STOCK_DIR      = _ROOT / "data" / "stocks"
HMM_PARAMS     = _ROOT / "outputs" / "exp3" / "states_10" / "trained_hmm_params.npz"
XGB_LONG_PATH  = _ROOT / "outputs" / "models" / "long"  / "xgb_trading_model.json"
XGB_SHORT_PATH = _ROOT / "outputs" / "models" / "short" / "xgb_trading_model.json"
DEFAULT_OUTDIR = _ROOT / "outputs" / "daily"

# ─── PIPELINE PARAMETERS ─────────────────────────────────────────────────────
LOOKBACK_DAYS   = 130   # days of history loaded for rolling windows (>= 60 + buffer)
HMM_WINDOW      = 120   # rolling Viterbi window (matches training)
LONG_THRESHOLD  = 0.6   # matches portfolio_backtest.py LONG_PROB_THRESHOLD
SHORT_THRESHOLD = 0.8   # matches portfolio_backtest.py SHORT_PROB_THRESHOLD


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


# ─── EMAIL ───────────────────────────────────────────────────────────────────

def _send_email(
    target_date: str,
    *,
    error_msg: str | None = None,
    n_long: int = 0,
    n_short: int = 0,
    n_candidates: int = 0,
    output_path: Path | None = None,
    top_long_text: str = "",
) -> None:
    """
    Send a daily update notification via Gmail SMTP.

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
        subject = f"[FAILED] BMEM Daily Update — {target_date}"
        body = (
            f"BMEM Daily Update FAILED: {target_date}\n"
            f"{'='*50}\n\n"
            f"{error_msg}\n"
        )
    else:
        subject = f"[OK] BMEM Daily Signals — {target_date}"
        body = (
            f"BMEM Daily Update: {target_date}\n"
            f"{'='*50}\n\n"
            f"Candidates scored : {n_candidates}\n"
            f"Long  signals (>={LONG_THRESHOLD:.0%})  : {n_long}\n"
            f"Short signals (>={SHORT_THRESHOLD:.0%})  : {n_short}\n"
        )
        if top_long_text:
            body += f"\nTop long candidates:\n{top_long_text}\n"

    msg = EmailMessage()
    msg["From"]    = sender
    msg["To"]      = receiver
    if cc:
        msg["Cc"] = cc
    msg["Subject"] = subject
    msg.set_content(body)

    if not error_msg and output_path is not None and output_path.exists():
        msg.add_attachment(
            output_path.read_bytes(),
            maintype="text",
            subtype="csv",
            filename=output_path.name,
        )

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(sender, password)
            smtp.send_message(msg)
        print(f"  [email] Email sent to {receiver}")
    except Exception as e:
        print(f"  [email] Failed to send email: {e}")


# ─── OUTPUT HELPERS ───────────────────────────────────────────────────────────

def _build_output_row_order() -> list:
    """Canonical column order for the output CSV."""
    prob_cols = [f'prob_S{i}' for i in range(10)]
    return (
        ['date', 'stock_id', 'securities_trader_id']
        + FEATURE_COLS
        + ['bias_60d', 'net_buy_amt_60d']
        + prob_cols
        + ['pred_prob_long', 'pred_prob_short', 'signal_long', 'signal_short']
    )


# ─── MAIN PIPELINE ────────────────────────────────────────────────────────────

def run_daily_update(
    target_date: str,
    broker_id: str = "1440",
    output_dir: Path = DEFAULT_OUTDIR,
) -> pd.DataFrame | None:
    """
    Execute the full daily update pipeline for one target date.

    Parameters
    ----------
    target_date : "YYYY-MM-DD" string
    broker_id   : broker trader code (default "1440" = Merrill Lynch)
    output_dir  : directory where signals_{target_date}.csv is written

    Returns
    -------
    pd.DataFrame of signals (all candidate stocks for target_date),
    or None if no data was available (weekend / holiday / insufficient history).
    """
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
        combined_broker, combined_stocks, disable_standardize=True
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
    if not HMM_PARAMS.exists():
        raise FileNotFoundError(f"HMM params not found: {HMM_PARAMS}")

    hmm_model = load_hmm_model(str(HMM_PARAMS))

    hmm_input  = valid_df.sort_values(['stock_id', 'securities_trader_id', 'date'])
    hmm_output = compute_rolling_hmm_proba(
        hmm_input, hmm_model, feature_cols=FEATURE_COLS, window=HMM_WINDOW
    )

    today_hmm = hmm_output[hmm_output['date'] == target_dt].copy()
    print(f"  -> State probabilities computed for {len(today_hmm)} rows")

    # ── 6. XGBoost signal generation ──────────────────────────────────────────
    for path, label in [(XGB_LONG_PATH, "long"), (XGB_SHORT_PATH, "short")]:
        if not path.exists():
            raise FileNotFoundError(f"XGBoost {label} model not found: {path}")

    clf_long  = load_xgb_model(str(XGB_LONG_PATH))
    clf_short = load_xgb_model(str(XGB_SHORT_PATH))

    missing_feats = [c for c in XGB_FEATURE_COLS if c not in today_hmm.columns]
    if missing_feats:
        raise ValueError(f"Missing XGBoost input features: {missing_feats}")

    signals_df = generate_signals(
        today_hmm, clf_long, clf_short,
        feature_cols=XGB_FEATURE_COLS,
        long_threshold=LONG_THRESHOLD,
        short_threshold=SHORT_THRESHOLD,
    )

    # ── 7. Save output CSV ─────────────────────────────────────────────────────
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"signals_{target_date}.csv"

    ordered_cols = [c for c in _build_output_row_order() if c in signals_df.columns]
    out = signals_df[ordered_cols].copy()
    out['date'] = out['date'].astype(str).str[:10]
    out = out.sort_values(
        ['signal_long', 'pred_prob_long'],
        ascending=[False, False],
    ).reset_index(drop=True)
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

    top_long_text = ""
    if n_long > 0:
        top = out[out['signal_long']][
            ['date', 'stock_id', 'pred_prob_long', 'pred_prob_short']
        ].head(10)
        top_long_text = top.to_string(index=False)
        print("  Top long candidates:")
        print(top_long_text)
        print()

    _send_email(
        target_date=target_date,
        n_long=n_long,
        n_short=n_short,
        n_candidates=len(signals_df),
        output_path=output_path,
        top_long_text=top_long_text,
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
    parser.add_argument(
        "--date",
        default=datetime.today().strftime("%Y-%m-%d"),
        help="Target trading date YYYY-MM-DD (default: today)",
    )
    parser.add_argument(
        "--broker-id",
        default="1440",
        help="FinMind broker trader ID (default: 1440 = Merrill Lynch)",
    )
    parser.add_argument(
        "--outdir",
        default=str(DEFAULT_OUTDIR),
        help=f"Output directory for signal CSV files (default: {DEFAULT_OUTDIR})",
    )
    args = parser.parse_args()

    try:
        run_daily_update(
            target_date=args.date,
            broker_id=args.broker_id,
            output_dir=Path(args.outdir),
        )
    except Exception:
        tb = traceback.format_exc()
        print(tb)
        _send_email(args.date, error_msg=tb)
        sys.exit(1)


if __name__ == "__main__":
    main()

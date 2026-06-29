# download_broker_activity.py
import os
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

import pandas as pd
from FinMind.data import DataLoader
from src.utils.paths import add_broker_path_args, paths_from_args
from src.utils.datetime import yesterday_str


def daterange_business_days(start: str, end: str) -> list[str]:
    """Return business days (Mon-Fri) between start and end inclusive."""
    dates = pd.date_range(start=start, end=end, freq="B")
    return [d.strftime("%Y-%m-%d") for d in dates]


def save_df(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".parquet":
        try:
            df.to_parquet(out_path, index=False)
            return
        except Exception:
            out_path = out_path.with_suffix(".csv")
    df.to_csv(out_path, index=False, encoding="utf-8-sig")


def latest_broker_date(broker_dir: Path) -> str | None:
    """
    Scan all parquet/csv files in broker_dir and return the latest 'date' value
    found across them, as 'YYYY-MM-DD'. Returns None if no files exist yet.
    """
    broker_dir = Path(broker_dir)
    files = sorted(broker_dir.glob("*.parquet")) + sorted(broker_dir.glob("*.csv"))
    # exclude meta/failed-dates json neighbours that share the stem
    data_files = [f for f in files if not any(kw in f.name for kw in ("_meta", "_failed"))]

    latest = None
    for path in data_files:
        try:
            if path.suffix.lower() == ".parquet":
                dates = pd.read_parquet(path, columns=["date"])["date"]
            else:
                dates = pd.read_csv(path, usecols=["date"])["date"]
            candidate = pd.to_datetime(dates).max()
            if pd.notna(candidate) and (latest is None or candidate > latest):
                latest = candidate
        except Exception:
            continue

    return latest.strftime("%Y-%m-%d") if latest is not None else None


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Download one broker's trading activity across Taiwan stocks via FinMind.\n"
            "By default, auto-detects the latest existing data and downloads only the missing gap up to yesterday."
        )
    )
    add_broker_path_args(parser)
    parser.add_argument(
        "--start",
        default=None,
        help="YYYY-MM-DD (inclusive). Defaults to the day after the latest existing data, "
             "or 2010-01-01 if no data exists yet.",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="YYYY-MM-DD (inclusive). Defaults to yesterday.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        help="Override the broker-specific output directory.",
    )
    parser.add_argument("--sleep", type=float, default=0.1)
    parser.add_argument("--format", choices=["parquet", "csv"], default="parquet")
    args = parser.parse_args()
    paths = paths_from_args(args)

    load_dotenv()
    token = os.environ["FINMIND_API_KEY"]

    outdir = args.outdir or paths.broker_raw_dir
    outdir.mkdir(parents=True, exist_ok=True)

    # --- Resolve end date ---
    end = args.end or yesterday_str()

    # --- Resolve start date (incremental: day after latest existing data) ---
    if args.start:
        start = args.start
    else:
        latest = latest_broker_date(outdir)
        if latest is None:
            print(f"[{args.broker_id}] No existing data found. Please provide --start.")
            raise SystemExit(1)
        # next business day after latest
        next_day = (pd.Timestamp(latest) + pd.offsets.BDay(1)).strftime("%Y-%m-%d")
        start = next_day
        print(f"[{args.broker_id}] Latest existing date: {latest}. Downloading from {start} to {end}.")

    if pd.Timestamp(start) > pd.Timestamp(end):
        print(f"[{args.broker_id}] Already up to date (latest={start}, end={end}). Nothing to download.")
        return

    api = DataLoader()
    api.login_by_token(api_token=token)

    dates = daterange_business_days(start, end)
    if not dates:
        print(f"[{args.broker_id}] No business days in range {start} ~ {end}. Nothing to download.")
        return

    print(f"[{args.broker_id}] Downloading {len(dates)} business days ({start} ~ {end})...")

    all_agg = []
    failed_dates = []

    for i, d in enumerate(dates, 1):
        try:
            df = api.taiwan_stock_trading_daily_report(
                securities_trader_id=args.broker_id,
                date=d,
            )

            if df is None or df.empty:
                continue

            df = df[(df["buy"] > 0) | (df["sell"] > 0)]
            if df.empty:
                continue

            df["buy_amount"] = df["buy"] * df["price"]
            df["sell_amount"] = df["sell"] * df["price"]

            agg = (
                df.groupby(["date", "stock_id"], as_index=False)
                .agg(
                    securities_trader=("securities_trader", "first"),
                    securities_trader_id=("securities_trader_id", "first"),
                    buy_amount=("buy_amount", "sum"),
                    sell_amount=("sell_amount", "sum"),
                    buy=("buy", "sum"),
                    sell=("sell", "sum"),
                )
            )
            agg["net_buy"] = agg["buy"] - agg["sell"]
            agg["net_buy_amount"] = agg["buy_amount"] - agg["sell_amount"]
            agg["avg_price"] = agg["net_buy_amount"] / agg["net_buy"]

            all_agg.append(agg)

        except Exception as e:
            failed_dates.append({"date": d, "error": repr(e)})

        time.sleep(args.sleep)

        if i % 20 == 0:
            print(f"[{i}/{len(dates)}] processed up to {d}")

    if all_agg:
        result = pd.concat(all_agg, ignore_index=True)
    else:
        result = pd.DataFrame(
            columns=["date", "stock_id", "securities_trader", "securities_trader_id",
                     "buy", "sell", "net_buy", "net_buy_amount", "buy_amount", "sell_amount", "avg_price"]
        )

    fname = f"{start}_to_{end}"
    out_file = outdir / (fname + (".parquet" if args.format == "parquet" else ".csv"))
    save_df(result, out_file)

    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "securities_trader_id": args.broker_id,
        "start_date": start,
        "end_date": end,
        "rows": int(len(result)),
        "unique_stocks": int(result["stock_id"].nunique()) if not result.empty else 0,
        "failed_dates_count": len(failed_dates),
        "output_file": str(out_file),
    }
    (outdir / (fname + "_meta.json")).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (outdir / (fname + "_failed_dates.json")).write_text(
        json.dumps(failed_dates, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Saved : {out_file}  ({len(result):,} rows)")
    print(f"Meta  : {outdir / (fname + '_meta.json')}")
    if failed_dates:
        print(f"Failed: {len(failed_dates)} dates (see {outdir / (fname + '_failed_dates.json')})")


if __name__ == "__main__":
    main()

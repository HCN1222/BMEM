# download_institutional_investors.py
import os
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

import requests
import pandas as pd

from src.utils.paths import DEFAULT_DATA_ROOT
from src.utils.datetime import yesterday_str

FINMIND_API_URL = "https://api.finmindtrade.com/api/v4/data"

NAME_TO_PSEUDO_ID = {
    "Foreign_Investor": "FOREIGN",
    "Investment_Trust": "TRUST",
    "Dealer_self": "DEALER_SELF",
    "Dealer_Hedging": "DEALER_HEDGE",
    "Foreign_Dealer_Self": "FOREIGN_DEALER",
}


def daterange_business_days(start: str, end: str) -> list[str]:
    dates = pd.date_range(start=start, end=end, freq="B")
    return [d.strftime("%Y-%m-%d") for d in dates]


def fetch_one_day(token: str, date: str) -> pd.DataFrame:
    resp = requests.get(
        FINMIND_API_URL,
        params={
            "dataset": "TaiwanStockInstitutionalInvestorsBuySell",
            "start_date": date,
            "end_date": date,
            "token": token,
        },
        timeout=60,
    )
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("status") != 200:
        raise RuntimeError(f"FinMind error: {payload.get('msg')}")
    return pd.DataFrame(payload.get("data", []))


def build_close_map(stock_dir: Path, stock_ids: set[str]) -> pd.DataFrame:
    frames = []
    for stock_id in stock_ids:
        files = sorted(stock_dir.glob(f"{stock_id}_*.parquet"))
        for f in files:
            try:
                df = pd.read_parquet(f, columns=["date", "close"])
                df["stock_id"] = stock_id
                frames.append(df)
            except Exception:
                continue
    if not frames:
        return pd.DataFrame(columns=["date", "stock_id", "close"])
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    return out.drop_duplicates(subset=["stock_id", "date"], keep="last")


def latest_existing_date(outdirs: list[Path]) -> str | None:
    latest = None
    for outdir in outdirs:
        if not outdir.exists():
            continue
        for path in sorted(outdir.glob("*.parquet")):
            if any(kw in path.name for kw in ("_meta", "_failed")):
                continue
            try:
                dates = pd.read_parquet(path, columns=["date"])["date"]
                candidate = pd.to_datetime(dates).max()
                if pd.notna(candidate) and (latest is None or candidate > latest):
                    latest = candidate
            except Exception:
                continue
    return latest.strftime("%Y-%m-%d") if latest is not None else None


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Download Taiwan institutional investors (三大法人) daily buy/sell via FinMind,\n"
            "and save them as pseudo-broker parquet files compatible with the existing\n"
            "preprocess/train pipeline (one pseudo-broker directory per investor type)."
        )
    )
    parser.add_argument("--start", default=None, help="YYYY-MM-DD (inclusive). Defaults to day after latest existing data.")
    parser.add_argument("--end", default=None, help="YYYY-MM-DD (inclusive). Defaults to yesterday.")
    parser.add_argument(
        "--names",
        nargs="+",
        default=["Foreign_Investor", "Investment_Trust", "Dealer_self", "Dealer_Hedging"],
        choices=list(NAME_TO_PSEUDO_ID.keys()),
        help="Which investor types to save (default: the main four).",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--sleep", type=float, default=0.1)
    parser.add_argument("--max-consecutive-failures", type=int, default=15,
                        help="Stop early after this many consecutive failed days "
                             "(e.g. API quota exhausted) and save progress so far.")
    parser.add_argument("--no-estimate-amounts", action="store_true",
                        help="Skip estimating buy/sell amounts from close price.")
    args = parser.parse_args()

    load_dotenv()
    token = os.environ["FINMIND_API_KEY"]

    brokers_root = args.data_root / "brokers"
    stock_dir = args.data_root / "stocks"
    outdirs = [brokers_root / NAME_TO_PSEUDO_ID[n] for n in args.names]

    end = args.end or yesterday_str()
    if args.start:
        start = args.start
    else:
        latest = latest_existing_date(outdirs)
        if latest is None:
            print("No existing institutional data found. Please provide --start.")
            raise SystemExit(1)
        start = (pd.Timestamp(latest) + pd.offsets.BDay(1)).strftime("%Y-%m-%d")
        print(f"Latest existing date: {latest}. Downloading from {start} to {end}.")

    if pd.Timestamp(start) > pd.Timestamp(end):
        print(f"Already up to date (latest={start}, end={end}). Nothing to download.")
        return

    dates = daterange_business_days(start, end)
    print(f"Downloading {len(dates)} business days ({start} ~ {end})...")

    all_rows = []
    failed_dates = []
    consecutive_failures = 0
    last_ok_date = None
    for i, d in enumerate(dates, 1):
        try:
            df = fetch_one_day(token, d)
            if not df.empty:
                all_rows.append(df)
                last_ok_date = d
            consecutive_failures = 0
        except Exception as e:
            failed_dates.append({"date": d, "error": repr(e)})
            consecutive_failures += 1
            if consecutive_failures >= args.max_consecutive_failures:
                print(f"\n{consecutive_failures} consecutive failures at {d} "
                      f"(likely quota exhausted). Stopping early; saving data up to {last_ok_date}.")
                break
        time.sleep(args.sleep)
        if i % 20 == 0:
            print(f"[{i}/{len(dates)}] processed up to {d}")

    if not all_rows:
        print("No data downloaded.")
        if failed_dates:
            print(f"Failed dates: {len(failed_dates)}")
        return

    raw = pd.concat(all_rows, ignore_index=True)
    raw = raw[raw["name"].isin(args.names)]
    raw["buy"] = raw["buy"].astype("int64")
    raw["sell"] = raw["sell"].astype("int64")
    raw = raw[(raw["buy"] > 0) | (raw["sell"] > 0)]
    raw["net_buy"] = raw["buy"] - raw["sell"]

    if args.no_estimate_amounts:
        raw["buy_amount"] = 0.0
        raw["sell_amount"] = 0.0
        raw["net_buy_amount"] = 0.0
        raw["avg_price"] = 0.0
    else:
        print("Estimating amounts from close prices...")
        close_map = build_close_map(stock_dir, set(raw["stock_id"].unique()))
        raw = raw.merge(close_map, on=["date", "stock_id"], how="left")
        raw["buy_amount"] = raw["buy"] * raw["close"]
        raw["sell_amount"] = raw["sell"] * raw["close"]
        raw["net_buy_amount"] = raw["net_buy"] * raw["close"]
        raw["avg_price"] = raw["close"]
        missing = raw["close"].isna().sum()
        if missing:
            print(f"WARNING: {missing:,} rows have no close price (stocks not in {stock_dir}). Amounts set to 0.")
            for col in ["buy_amount", "sell_amount", "net_buy_amount", "avg_price"]:
                raw[col] = raw[col].fillna(0.0)
        raw = raw.drop(columns=["close"])

    out_cols = ["date", "stock_id", "securities_trader", "securities_trader_id",
                "buy_amount", "sell_amount", "buy", "sell", "net_buy", "net_buy_amount", "avg_price"]

    fname = f"{start}_to_{end}"
    for name in args.names:
        pseudo_id = NAME_TO_PSEUDO_ID[name]
        sub = raw[raw["name"] == name].copy()
        if sub.empty:
            print(f"[{pseudo_id}] no rows, skipped.")
            continue
        sub["securities_trader"] = name
        sub["securities_trader_id"] = pseudo_id

        outdir = brokers_root / pseudo_id
        outdir.mkdir(parents=True, exist_ok=True)
        out_file = outdir / f"{fname}.parquet"
        sub[out_cols].to_parquet(out_file, index=False)

        meta = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "securities_trader_id": pseudo_id,
            "investor_name": name,
            "source_dataset": "TaiwanStockInstitutionalInvestorsBuySell",
            "start_date": start,
            "end_date": end,
            "rows": int(len(sub)),
            "unique_stocks": int(sub["stock_id"].nunique()),
            "amounts_estimated_from_close": not args.no_estimate_amounts,
            "failed_dates_count": len(failed_dates),
            "output_file": str(out_file),
        }
        (outdir / f"{fname}_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        (outdir / f"{fname}_failed_dates.json").write_text(
            json.dumps(failed_dates, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[{pseudo_id}] Saved {len(sub):,} rows ({sub['stock_id'].nunique()} stocks) -> {out_file}")

    print("Done.")
    print("Next : python -m src.experiments.preprocess --broker-id DEALER_SELF --disable_standardize")


if __name__ == "__main__":
    main()

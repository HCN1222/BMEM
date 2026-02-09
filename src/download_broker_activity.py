# download_merrill_activity.py
import os
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

import pandas as pd
from FinMind.data import DataLoader


def daterange_business_days(start: str, end: str) -> list[str]:
    """Return business days (Mon-Fri) between start and end inclusive."""
    dates = pd.date_range(start=start, end=end, freq="B")
    return [d.strftime("%Y-%m-%d") for d in dates]


def save_df(df: pd.DataFrame, out_path: Path) -> None:
    """Save dataframe as parquet if possible, otherwise csv."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Try parquet first (fast + smaller); fallback to csv if pyarrow/fastparquet not installed.
    if out_path.suffix.lower() == ".parquet":
        try:
            df.to_parquet(out_path, index=False)
            return
        except Exception:
            # fallback
            out_path = out_path.with_suffix(".csv")

    df.to_csv(out_path, index=False, encoding="utf-8-sig")


def main():
    parser = argparse.ArgumentParser(
        description="Download Merrill (1440) trading activity across all Taiwan stocks via FinMind."
    )
    parser.add_argument("--start", required=True, help="YYYY-MM-DD (inclusive)")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD (inclusive)")
    parser.add_argument("--trader-id", default="1440", help="券商代碼；美林預設 1440")
    parser.add_argument("--outdir", default="../data", help="輸出資料夾（預設 ../data)")
    parser.add_argument("--sleep", type=float, default=0.2, help="每次 request 間隔秒數（預設 0.2)")
    parser.add_argument("--format", choices=["parquet", "csv"], default="parquet", help="輸出格式")
    args = parser.parse_args()

    load_dotenv()
    token = os.environ["FINMIND_API_KEY"]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    api = DataLoader()
    api.login_by_token(api_token=token)

    dates = daterange_business_days(args.start, args.end)

    all_agg = []
    failed_dates = []

    for i, d in enumerate(dates, 1):
        try:
            df = api.taiwan_stock_trading_daily_report(
                securities_trader_id=str(args.trader_id),
                date=d,
            )

            if df is None or df.empty:
                continue

            # 只保留有行為（有買或有賣）
            df = df[(df["buy"] > 0) | (df["sell"] > 0)]
            if df.empty:
                continue
            
            # 計算成交金額
            df["buy_amount"] = df["buy"] * df["price"]
            df["sell_amount"] = df["sell"] * df["price"]

            # 同一 stock_id 同一天可能有多個成交價列 -> 加總成日彙總（方便回測）
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

            agg = agg.drop(["buy", "sell", "buy_amount", "sell_amount",], axis=1)

            all_agg.append(agg)

        except Exception as e:
            failed_dates.append({"date": d, "error": repr(e)})

        # 簡單節流，避免太密集
        time.sleep(args.sleep)

        if i % 20 == 0:
            print(f"[{i}/{len(dates)}] processed up to {d}")

    if all_agg:
        result = pd.concat(all_agg, ignore_index=True)
    else:
        result = pd.DataFrame(
            columns=["date", "stock_id", "securities_trader", "securities_trader_id", "buy", "sell", "net_buy"]
        )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"merrill_{args.trader_id}_activity_{args.start}_to_{args.end}_{stamp}"
    out_file = outdir / (fname + (".parquet" if args.format == "parquet" else ".csv"))
    save_df(result, out_file)

    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "securities_trader_id": str(args.trader_id),
        "start_date": args.start,
        "end_date": args.end,
        "rows": int(len(result)),
        "unique_stocks": int(result["stock_id"].nunique()) if not result.empty else 0,
        "failed_dates_count": len(failed_dates),
        "output_file": str(out_file),
    }
    (outdir / (fname + "_meta.json")).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (outdir / (fname + "_failed_dates.json")).write_text(
        json.dumps(failed_dates, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Saved: {out_file}")
    print(f"Meta : {outdir / (fname + '_meta.json')}")
    if failed_dates:
        print(f"Failed dates: {len(failed_dates)} (see {outdir / (fname + '_failed_dates.json')})")


if __name__ == "__main__":
    main()

# aggregate_institutional.py
"""
Aggregate multiple institutional-investor pseudo-brokers into ONE combined
"total institutional" flow by SUMMING buy/sell per (date, stock_id).

Unlike pool_broker_data.py (which STACKS, keeping each broker's identity as its
own sequence), this SUMS across institution types into a single flow per stock-day
-- a general "institutional money flow" baseline.

Default sums FOREIGN + TRUST + DEALER_SELF (excludes DEALER_HEDGE, which is
mechanical warrant-hedging noise, not a directional view).
"""
import json
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from src.utils.paths import DEFAULT_DATA_ROOT, validate_broker_id


SUM_COLS = ["buy", "sell", "net_buy", "buy_amount", "sell_amount", "net_buy_amount"]


def main():
    parser = argparse.ArgumentParser(
        description="Sum several institutional pseudo-brokers into one combined flow per (date, stock_id)."
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["FOREIGN", "TRUST", "DEALER_SELF"],
        help="Pseudo-broker dirs to sum (default: FOREIGN TRUST DEALER_SELF; excludes DEALER_HEDGE).",
    )
    parser.add_argument("--out-id", default="INST_TOTAL", help="Output pseudo-broker ID (default: INST_TOTAL).")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    args = parser.parse_args()

    out_id = validate_broker_id(args.out_id)
    brokers_root = args.data_root / "brokers"

    frames = []
    for sid in args.sources:
        files = sorted((brokers_root / sid).glob("*.parquet"))
        if not files:
            raise SystemExit(f"No parquet found for source '{sid}' in {brokers_root / sid}")
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        frames.append(df)
        print(f"  [{sid}] {len(df):,} rows")

    combined = pd.concat(frames, ignore_index=True)
    for c in SUM_COLS:
        combined[c] = pd.to_numeric(combined[c], errors="coerce").fillna(0)

    agg = combined.groupby(["date", "stock_id"], as_index=False)[SUM_COLS].sum()
    agg["securities_trader_id"] = out_id
    agg["securities_trader"] = "Institutional_Total"
    agg["avg_price"] = (agg["net_buy_amount"] / agg["net_buy"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0)

    out_cols = ["date", "stock_id", "securities_trader", "securities_trader_id",
                "buy_amount", "sell_amount", "buy", "sell", "net_buy", "net_buy_amount", "avg_price"]
    agg = agg[out_cols].sort_values(["stock_id", "date"]).reset_index(drop=True)

    start = pd.to_datetime(agg["date"]).min().strftime("%Y-%m-%d")
    end = pd.to_datetime(agg["date"]).max().strftime("%Y-%m-%d")

    outdir = brokers_root / out_id
    outdir.mkdir(parents=True, exist_ok=True)
    out_file = outdir / f"{out_id}_{start}_to_{end}.parquet"
    agg.to_parquet(out_file, index=False)

    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "securities_trader_id": out_id,
        "aggregated_from": args.sources,
        "method": "sum per (date, stock_id)",
        "start_date": start, "end_date": end,
        "rows": int(len(agg)),
        "unique_stocks": int(agg["stock_id"].nunique()),
    }
    (outdir / f"{out_id}_{start}_to_{end}_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Saved [{out_id}]: {out_file}  ({len(agg):,} rows, {meta['unique_stocks']} stocks)")
    print(f"Next : python -m src.experiments.preprocess --broker-id {out_id} --disable_standardize")


if __name__ == "__main__":
    main()

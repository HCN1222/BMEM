# pool_broker_data.py
import json
import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd

from src.utils.paths import DEFAULT_DATA_ROOT, validate_broker_id


def load_broker_dir(broker_dir: Path) -> pd.DataFrame:
    files = sorted(broker_dir.glob("*.parquet"))
    if not files:
        raise ValueError(f"No parquet files found in {broker_dir}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    if "securities_trader_id" not in df.columns:
        df["securities_trader_id"] = broker_dir.name
    df["securities_trader_id"] = df["securities_trader_id"].astype(str)
    return df


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Pool multiple brokers' raw activity parquet into a single pseudo-broker directory,\n"
            "so the existing preprocess/train pipeline can train one HMM on all brokers jointly.\n"
            "Sequences remain separated per (stock, broker) inside preprocess.py."
        )
    )
    parser.add_argument(
        "--broker-ids",
        nargs="+",
        default=None,
        help="Broker IDs to pool. Defaults to every subdirectory under data/brokers.",
    )
    parser.add_argument(
        "--pool-id",
        default="POOLED",
        help="Pseudo-broker ID for the pooled output directory (default: POOLED).",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--format", choices=["parquet", "csv"], default="parquet")
    args = parser.parse_args()

    pool_id = validate_broker_id(args.pool_id)
    brokers_root = args.data_root / "brokers"

    if args.broker_ids:
        broker_ids = [validate_broker_id(b) for b in args.broker_ids]
    else:
        broker_ids = sorted(
            d.name for d in brokers_root.iterdir()
            if d.is_dir() and d.name != pool_id
        )

    if not broker_ids:
        raise SystemExit("No broker directories found to pool.")

    print(f"Pooling {len(broker_ids)} brokers: {', '.join(broker_ids)}")

    frames = []
    per_broker_rows = {}
    for broker_id in broker_ids:
        broker_dir = brokers_root / broker_id
        df = load_broker_dir(broker_dir)
        per_broker_rows[broker_id] = len(df)
        frames.append(df)
        print(f"  [{broker_id}] {len(df):,} rows")

    pooled = pd.concat(frames, ignore_index=True)
    pooled = pooled.drop_duplicates(subset=["date", "stock_id", "securities_trader_id"])
    pooled["date"] = pd.to_datetime(pooled["date"])
    pooled = pooled.sort_values(["securities_trader_id", "stock_id", "date"]).reset_index(drop=True)

    start = pooled["date"].min().strftime("%Y-%m-%d")
    end = pooled["date"].max().strftime("%Y-%m-%d")

    outdir = brokers_root / pool_id
    outdir.mkdir(parents=True, exist_ok=True)
    fname = f"{start}_to_{end}"
    out_file = outdir / (fname + (".parquet" if args.format == "parquet" else ".csv"))

    pooled["date"] = pooled["date"].dt.strftime("%Y-%m-%d")
    if args.format == "parquet":
        pooled.to_parquet(out_file, index=False)
    else:
        pooled.to_csv(out_file, index=False, encoding="utf-8-sig")

    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "pool_id": pool_id,
        "source_brokers": broker_ids,
        "per_broker_rows": per_broker_rows,
        "start_date": start,
        "end_date": end,
        "rows": int(len(pooled)),
        "unique_stocks": int(pooled["stock_id"].nunique()),
        "output_file": str(out_file),
    }
    (outdir / (fname + "_meta.json")).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Saved : {out_file}  ({len(pooled):,} rows, {meta['unique_stocks']} stocks)")
    print(f"Meta  : {outdir / (fname + '_meta.json')}")
    print(f"Next  : python -m src.experiments.preprocess --broker-id {pool_id} --disable_standardize")


if __name__ == "__main__":
    main()

import os
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd

from FinMind.data import DataLoader
from src.utils.io import save_df
from src.utils.paths import DEFAULT_DATA_ROOT
from src.utils.stock_data import latest_stock_date

def normalize_stock_ids(values) -> list[str]:
    invalid = {"", "nan", "none", "null", "<na>"}
    ids = {
        str(value).strip()
        for value in values
        if str(value).strip().lower() not in invalid
    }
    return sorted(ids)


def load_stock_ids_json(path: Path) -> list[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"stock ids json not found: {p}")
    obj = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(obj, list):
        raise ValueError('stock ids json must be a top-level list, e.g. ["2330", "2317"]')
    return normalize_stock_ids(obj)


def load_stock_ids_from_brokers(broker_data_root: Path) -> tuple[list[str], int]:
    """Return the union of stock IDs in every broker parquet/CSV file."""
    broker_data_root = Path(broker_data_root)
    files = sorted(broker_data_root.glob("*/*.parquet"))
    files += sorted(broker_data_root.glob("*/*.csv"))
    if not files:
        raise FileNotFoundError(f"No broker parquet/CSV files found under: {broker_data_root}")

    stock_ids: set[str] = set()
    for path in files:
        try:
            if path.suffix.lower() == ".parquet":
                df = pd.read_parquet(path, columns=["stock_id"])
            else:
                df = pd.read_csv(path, usecols=["stock_id"], dtype={"stock_id": "string"})
        except Exception as e:
            raise RuntimeError(f"Failed to read stock IDs from {path}: {e}") from e

        stock_ids.update(normalize_stock_ids(df["stock_id"].dropna()))

    if not stock_ids:
        raise ValueError(f"Broker files under {broker_data_root} contain no stock IDs")
    return sorted(stock_ids), len(files)


def save_stock_ids_json(stock_ids: list[str], path: Path) -> None:
    """Atomically replace stock_ids.json with a normalized, sorted list."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(normalize_stock_ids(stock_ids), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temp_path.replace(path)

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Download Taiwan stock daily information (OHLCV, etc.) via FinMind.\n"
            "Modes:\n"
            "  - single: download a single stock by --stock-id\n"
            "  - list  : read stock_ids from a JSON file (top-level list) and download each stock\n"
            "  - brokers: rebuild stock_ids.json from all broker data, then download each stock\n"
        )
    )
    parser.add_argument("--mode", choices=["single", "list", "brokers"], required=True)
    parser.add_argument("--start", help="YYYY-MM-DD (inclusive)")
    parser.add_argument("--end", help="YYYY-MM-DD (inclusive)")
    parser.add_argument("--stock-id", help="Required if --mode single")
    parser.add_argument(
        "--stock-ids-json",
        type=Path,
        default=DEFAULT_DATA_ROOT / "stocks" / "stock_ids.json",
        help="Stock ID JSON input/output path",
    )
    parser.add_argument(
        "--broker-data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT / "brokers",
        help="Directory containing one subdirectory per broker",
    )
    parser.add_argument(
        "--refresh-only",
        action="store_true",
        help="With --mode brokers, update stock_ids.json without downloading prices",
    )
    parser.add_argument("--outdir", type=Path, default=DEFAULT_DATA_ROOT / "stocks")
    parser.add_argument("--sleep", type=float, default=0.1)
    parser.add_argument("--format", choices=["parquet", "csv"], default="parquet")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.mode == "single":
        if args.refresh_only:
            raise ValueError("--refresh-only is only valid with --mode brokers")
        if not args.stock_id:
            raise ValueError("--stock-id is required when --mode single")
        stock_ids = [str(args.stock_id).strip()]
    elif args.mode == "list":
        stock_ids = load_stock_ids_json(args.stock_ids_json)
    else:
        stock_ids, source_file_count = load_stock_ids_from_brokers(args.broker_data_root)
        save_stock_ids_json(stock_ids, args.stock_ids_json)
        print(
            f"Updated {args.stock_ids_json}: {len(stock_ids):,} unique stock IDs "
            f"from {source_file_count} broker files."
        )
        if args.refresh_only:
            return

    if not args.start or not args.end:
        raise ValueError("--start and --end are required when downloading stock data")

    load_dotenv()
    token = os.environ["FINMIND_API_KEY"]

    api = DataLoader()
    api.login_by_token(api_token=token)

    out_root = args.outdir
    out_root.mkdir(parents=True, exist_ok=True)

    failed = []
    skipped = 0
    total = len(stock_ids)
    requested_end = pd.Timestamp(args.end)

    for i, sid in enumerate(stock_ids, 1):
        try:
            fetch_start = args.start
            if not args.overwrite:
                latest_date = latest_stock_date(out_root, sid)
                if latest_date is not None and latest_date.normalize() >= requested_end:
                    skipped += 1
                    continue
                if latest_date is not None:
                    fetch_start = max(
                        pd.Timestamp(args.start),
                        latest_date.normalize() + pd.Timedelta(days=1),
                    ).strftime("%Y-%m-%d")

            fname = f"{sid}_{fetch_start}_to_{args.end}"
            out_file = out_root / (fname + (".parquet" if args.format == "parquet" else ".csv"))

            # Check existence BEFORE requesting API
            if out_file.exists() and not args.overwrite:
                skipped += 1
                continue

            df = api.taiwan_stock_daily(
                stock_id=str(sid),
                start_date=fetch_start,
                end_date=args.end,
            )

            if df is None or df.empty:
                raise RuntimeError("Empty dataframe returned from FinMind")

            save_df(df, out_file)

            meta = {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "mode": args.mode,
                "stock_id": str(sid),
                "start_date": fetch_start,
                "end_date": args.end,
                "rows": int(len(df)),
                "output_file": str(out_file),
            }
            (out_root / (fname + "_meta.json")).write_text(
                json.dumps(meta, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        except Exception as e:
            failed.append({
                "stock_id": str(sid),
                "reason": "exception",
                "error": repr(e),
            })

        time.sleep(args.sleep)

        if i % 20 == 0 or i == total:
            print(f"[{i}/{total}] processed (last: {sid})")

    run_fname = f"run_{args.mode}_{args.start}_to_{args.end}"
    (out_root / (run_fname + "_failed.json")).write_text(
        json.dumps(failed, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        f"Done. Downloaded/updated: {total - skipped - len(failed):,}; "
        f"already current: {skipped:,}; issues: {len(failed):,} "
        f"(see {out_root / (run_fname + '_failed.json')})"
    )


if __name__ == "__main__":
    main()

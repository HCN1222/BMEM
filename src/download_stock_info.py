import os
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

from FinMind.data import DataLoader
from utils.io import save_df

def load_stock_ids_json(path: str) -> list[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"stock ids json not found: {p}")
    obj = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(obj, list):
        raise ValueError('stock ids json must be a top-level list, e.g. ["2330", "2317"]')
    ids = [str(x).strip() for x in obj if str(x).strip()]
    return sorted(set(ids))

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Download Taiwan stock daily information (OHLCV, etc.) via FinMind.\n"
            "Modes:\n"
            "  - single: download a single stock by --stock-id\n"
            "  - list  : read stock_ids from a JSON file (top-level list) and download each stock\n"
        )
    )
    parser.add_argument("--mode", choices=["single", "list"], required=True)
    parser.add_argument("--start", required=True, help="YYYY-MM-DD (inclusive)")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD (inclusive)")
    parser.add_argument("--stock-id", help="Required if --mode single")
    parser.add_argument("--stock-ids-json", help="Required if --mode list; JSON top-level list of ids")
    parser.add_argument("--outdir", default="./data/stocks")
    parser.add_argument("--sleep", type=float, default=0.1)
    parser.add_argument("--format", choices=["parquet", "csv"], default="parquet")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.mode == "single":
        if not args.stock_id:
            raise ValueError("--stock-id is required when --mode single")
        stock_ids = [str(args.stock_id).strip()]
    else:
        if not args.stock_ids_json:
            raise ValueError("--stock-ids-json is required when --mode list")
        stock_ids = load_stock_ids_json(args.stock_ids_json)

    load_dotenv()
    token = os.environ["FINMIND_API_KEY"]

    api = DataLoader()
    api.login_by_token(api_token=token)

    out_root = Path(args.outdir)
    out_root.mkdir(parents=True, exist_ok=True)

    failed = []
    total = len(stock_ids)

    for i, sid in enumerate(stock_ids, 1):
        try:
            fname = f"{sid}_{args.start}_to_{args.end}"
            out_file = out_root / (fname + (".parquet" if args.format == "parquet" else ".csv"))

            # Check existence BEFORE requesting API
            if out_file.exists() and not args.overwrite:
                msg = f"Skip {sid}: file exists ({out_file})"
                print(msg)
                failed.append({
                    "stock_id": str(sid),
                    "reason": "skipped_exists",
                    "output_file": str(out_file),
                })
                continue

            df = api.taiwan_stock_daily(
                stock_id=str(sid),
                start_date=args.start,
                end_date=args.end,
            )

            if df is None or df.empty:
                raise RuntimeError("Empty dataframe returned from FinMind")

            save_df(df, out_file)

            meta = {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "mode": args.mode,
                "stock_id": str(sid),
                "start_date": args.start,
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

    print(f"Done. Issues logged: {len(failed)} (see {out_root / (run_fname + '_failed.json')})")


if __name__ == "__main__":
    main()
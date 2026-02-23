import pandas as pd
from pathlib import Path

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


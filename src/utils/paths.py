"""Centralized, broker-aware filesystem paths for BMEM."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = REPO_ROOT / "data"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs"
_BROKER_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def validate_broker_id(broker_id: str) -> str:
    """Return a safe broker directory name or raise a CLI-friendly error."""
    broker_id = str(broker_id).strip()
    if not broker_id or not _BROKER_ID_RE.fullmatch(broker_id):
        raise ValueError(
            "broker_id must contain only letters, numbers, underscores, or hyphens"
        )
    return broker_id


@dataclass(frozen=True)
class BrokerPaths:
    """All broker-specific and shared paths used by the pipeline."""

    broker_id: str
    data_root: Path = DEFAULT_DATA_ROOT
    output_root: Path = DEFAULT_OUTPUT_ROOT

    def __post_init__(self) -> None:
        object.__setattr__(self, "broker_id", validate_broker_id(self.broker_id))
        object.__setattr__(self, "data_root", Path(self.data_root).resolve())
        object.__setattr__(self, "output_root", Path(self.output_root).resolve())

    @property
    def broker_raw_dir(self) -> Path:
        return self.data_root / "brokers" / self.broker_id

    @property
    def stock_dir(self) -> Path:
        return self.data_root / "stocks"

    @property
    def preprocessed_dir(self) -> Path:
        return self.data_root / "preprocessed_data" / self.broker_id

    @property
    def hmm_data_dir(self) -> Path:
        return self.preprocessed_dir / "hmm"

    def xgboost_data_dir(self, side: str | None = None) -> Path:
        path = self.preprocessed_dir / "xgboost"
        return path / side if side else path

    @property
    def broker_output_dir(self) -> Path:
        return self.output_root / self.broker_id

    @property
    def hmm_runs_dir(self) -> Path:
        return self.broker_output_dir / "runs" / "hmm"

    @property
    def hmm_model_dir(self) -> Path:
        return self.broker_output_dir / "models" / "hmm"

    @property
    def hmm_model_path(self) -> Path:
        return self.hmm_model_dir / "trained_hmm_params.npz"

    def xgboost_model_dir(self, side: str) -> Path:
        return self.broker_output_dir / "models" / "xgboost" / side

    def xgboost_model_path(self, side: str) -> Path:
        return self.xgboost_model_dir(side) / "xgb_trading_model.json"

    @property
    def backtest_dir(self) -> Path:
        return self.broker_output_dir / "backtest"

    @property
    def analysis_dir(self) -> Path:
        return self.broker_output_dir / "analysis"

    @property
    def daily_dir(self) -> Path:
        return self.broker_output_dir / "daily"


def add_broker_path_args(parser: argparse.ArgumentParser) -> None:
    """Add the standard broker and root directory options to a parser."""
    parser.add_argument(
        "--broker-id",
        required=True,
        help="Broker trader code used as the data/output subdirectory name.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"Shared data root (default: {DEFAULT_DATA_ROOT})",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Output root (default: {DEFAULT_OUTPUT_ROOT})",
    )


def paths_from_args(args: argparse.Namespace) -> BrokerPaths:
    return BrokerPaths(args.broker_id, args.data_root, args.output_root)

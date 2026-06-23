"""
test_pipeline.py

Verification test bench for pipeline_functions.py.

Uses pre-computed reference files from the experiments to validate that the
extracted pipeline functions produce equivalent results.

Run from the repo root:
    conda run -n BMEM python src/test_pipeline.py

Tests
-----
Test 1   Feature consistency
         Checks that the feature arrays in final_vectors_eval.parquet and
         hmm_data_eval.npz are in sync (both produced by the same preprocess.py run).

Test 1b  compute_observation_features replication
         Calls compute_observation_features on the raw broker + stock parquets and
         compares every feature column against final_vectors_eval.parquet row-by-row.
         This is the full end-to-end check of the feature engineering code.

Test 2   HMM probability replication  [controlled by RUN_INFERENCE flag]
         Re-runs compute_rolling_hmm_proba on final_vectors_eval.parquet and
         compares the output prob_S0...S9 values against the reference stored in
         evaluation.parquet (produced by prepare_xgb_data.py).
         Two sub-tests:
           2a  groupby sequence_id   - exact replication of prepare_xgb_data.py
           2b  groupby stock+trader  - pipeline_functions.py grouping

Test 3   Long XGBoost signal quality
         Loads evaluation.parquet (ground-truth features + prob_S columns),
         runs generate_signals, and reports precision / recall / F1 at threshold 0.6.

Test 4   Short XGBoost signal quality
         Same as Test 3 but for the short model (threshold 0.8).
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# ─── Resolve paths ─────────────────────────────────────────────────────────
_SRC = Path(__file__).parent
_ROOT = _SRC.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pipeline_functions import (
    compute_observation_features,
    load_hmm_model,
    compute_rolling_hmm_proba,
    load_xgb_model,
    generate_signals,
    FEATURE_COLS,
)
from utils.paths import add_broker_path_args, paths_from_args

# ─── Reference file paths ──────────────────────────────────────────────────
BROKER_ID = None
FINAL_VECTORS_EVAL = None
HMM_EVAL_NPZ = None
XGB_LONG_EVAL = None
XGB_SHORT_EVAL = None
HMM_PARAMS = None
XGB_LONG_MODEL = None
XGB_SHORT_MODEL = None
BROKER_DATA_DIR = None
STOCK_DATA_DIR = None


def configure_paths(paths):
    global BROKER_ID, FINAL_VECTORS_EVAL, HMM_EVAL_NPZ
    global XGB_LONG_EVAL, XGB_SHORT_EVAL, HMM_PARAMS
    global XGB_LONG_MODEL, XGB_SHORT_MODEL, BROKER_DATA_DIR, STOCK_DATA_DIR

    BROKER_ID = paths.broker_id
    FINAL_VECTORS_EVAL = paths.hmm_data_dir / "final_vectors_eval.parquet"
    HMM_EVAL_NPZ = paths.hmm_data_dir / "hmm_data_eval.npz"
    XGB_LONG_EVAL = paths.xgboost_data_dir("long") / "evaluation.parquet"
    XGB_SHORT_EVAL = paths.xgboost_data_dir("short") / "evaluation.parquet"
    HMM_PARAMS = paths.hmm_model_path
    XGB_LONG_MODEL = paths.xgboost_model_path("long")
    XGB_SHORT_MODEL = paths.xgboost_model_path("short")
    BROKER_DATA_DIR = paths.broker_raw_dir
    STOCK_DATA_DIR = paths.stock_dir

# ─── Run flags ─────────────────────────────────────────────────────────────
# Set RUN_INFERENCE = True to include Test 2 (HMM rolling inference, ~6 min).
# It has already been verified; disable it for faster feature-engineering checks.
RUN_INFERENCE = False

# Tolerance for floating-point comparisons
ATOL_FEATURES = 1e-9    # feature arrays: should be bit-for-bit identical
ATOL_PROBA    = 1e-5    # HMM proba: deterministic but may accumulate float drift


# ─── Test runner helpers ──────────────────────────────────────────────────────

class TestResult:
    def __init__(self, name: str):
        self.name   = name
        self.passed = True
        self.lines  = []

    def check(self, condition: bool, msg_pass: str, msg_fail: str):
        if condition:
            self.lines.append(f"  [PASS] {msg_pass}")
        else:
            self.lines.append(f"  [FAIL] {msg_fail}")
            self.passed = False

    def info(self, msg: str):
        self.lines.append(f"  [info] {msg}")

    def print_summary(self):
        status = "PASSED" if self.passed else "FAILED"
        bar    = "=" * 60
        print(f"\n{bar}")
        print(f"  {self.name}  [{status}]")
        print(bar)
        for line in self.lines:
            print(line)


# ─── Test 1: Feature array consistency ────────────────────────────────────────

def test_feature_consistency() -> TestResult:
    """
    Checks that final_vectors_eval.parquet and hmm_data_eval.npz contain the
    same observation arrays.  Both files were written by the same preprocess.py
    run, so they must be exactly equal.
    """
    t = TestResult("Test 1  Feature array consistency  (parquet vs npz)")

    # Load parquet features
    df = pd.read_parquet(FINAL_VECTORS_EVAL)
    df['date'] = pd.to_datetime(df['date'])
    df_sorted = df.sort_values(['sequence_id', 'date'])
    X_from_parquet = df_sorted[FEATURE_COLS].values.astype(float)

    t.info(f"final_vectors_eval rows : {len(df):,}")
    t.info(f"Features checked        : {FEATURE_COLS}")

    # Load npz observations
    npz = np.load(HMM_EVAL_NPZ, allow_pickle=True)
    X_from_npz      = npz['observations'].astype(float)
    lengths         = npz['lengths']
    npz_feature_names = list(npz['feature_names'].astype(str))

    t.info(f"hmm_data_eval lengths sum: {lengths.sum():,}  (sequences: {len(lengths)})")
    t.info(f"NPZ feature names       : {npz_feature_names}")

    # Shape check
    t.check(
        X_from_parquet.shape == X_from_npz.shape,
        f"Shapes match: {X_from_parquet.shape}",
        f"Shape mismatch - parquet {X_from_parquet.shape} vs npz {X_from_npz.shape}",
    )

    if X_from_parquet.shape != X_from_npz.shape:
        return t  # can't diff if shapes differ

    # Value comparison
    max_diff  = np.abs(X_from_parquet - X_from_npz).max()
    mean_diff = np.abs(X_from_parquet - X_from_npz).mean()

    t.info(f"Max  |parquet - npz| : {max_diff:.2e}")
    t.info(f"Mean |parquet - npz| : {mean_diff:.2e}")

    t.check(
        max_diff <= ATOL_FEATURES,
        f"Feature values identical (max diff {max_diff:.2e} <= {ATOL_FEATURES:.0e})",
        f"Feature values differ (max diff {max_diff:.2e} > {ATOL_FEATURES:.0e})",
    )

    # Feature-by-feature breakdown
    for i, fname in enumerate(FEATURE_COLS):
        col_diff = np.abs(X_from_parquet[:, i] - X_from_npz[:, i]).max()
        t.info(f"  {fname:>5}  max diff: {col_diff:.2e}")

    return t


# ─── Test 1b: compute_observation_features replication ───────────────────────

def test_feature_computation() -> TestResult:
    """
    Calls compute_observation_features on the raw broker + stock parquets and
    compares every feature column against final_vectors_eval.parquet row-by-row.

    This is the end-to-end check that verifies the feature engineering code in
    pipeline_functions.py faithfully reproduces what preprocess.py produced.
    """
    t = TestResult("Test 1b  compute_observation_features  (raw data -> features vs parquet)")

    # ── Load raw broker data ──────────────────────────────────────────────────
    broker_files = sorted(BROKER_DATA_DIR.glob("*.parquet"))
    if not broker_files:
        t.check(False, "", f"No broker parquets found in {BROKER_DATA_DIR}")
        return t

    broker_df = pd.concat(
        [pd.read_parquet(f) for f in broker_files], ignore_index=True
    )
    broker_df['date'] = pd.to_datetime(broker_df['date'])
    broker_df = broker_df.drop_duplicates(subset=['date', 'stock_id'])
    if 'securities_trader_id' not in broker_df.columns:
        broker_df['securities_trader_id'] = BROKER_ID

    t.info(f"Broker rows loaded  : {len(broker_df):,}")

    # ── Load raw stock data for every stock that appears in broker data ───────
    stock_ids = broker_df['stock_id'].astype(str).unique().tolist()
    stock_parts = []
    for sid in stock_ids:
        for f in STOCK_DATA_DIR.glob(f"{sid}_*.parquet"):
            stock_parts.append(pd.read_parquet(f))

    if not stock_parts:
        t.check(False, "", f"No stock parquets found under {STOCK_DATA_DIR}")
        return t

    stock_df = pd.concat(stock_parts, ignore_index=True)
    stock_df['date'] = pd.to_datetime(stock_df['date'])
    stock_df = stock_df.drop_duplicates(subset=['stock_id', 'date'])

    t.info(f"Stock rows loaded   : {len(stock_df):,}")

    # ── Compute features (disable_standardize=True matches training data) ─────
    print("  [info] Running compute_observation_features ...")
    computed = compute_observation_features(broker_df, stock_df, disable_standardize=True)
    computed['date'] = pd.to_datetime(computed['date'])
    t.info(f"Computed rows total : {len(computed):,}")

    # ── Load reference and filter computed output to eval dates ──────────────
    ref = pd.read_parquet(FINAL_VECTORS_EVAL)
    ref['date'] = pd.to_datetime(ref['date'])

    eval_dates = set(ref['date'].unique())
    computed_eval = computed[computed['date'].isin(eval_dates)].copy()
    t.info(f"Computed eval rows  : {len(computed_eval):,}")
    t.info(f"Reference eval rows : {len(ref):,}")

    # ── Align on (date, stock_id, securities_trader_id) ──────────────────────
    for df_ in (computed_eval, ref):
        df_['stock_id'] = df_['stock_id'].astype(str)
        df_['securities_trader_id'] = df_['securities_trader_id'].astype(str)

    merge_keys = ['date', 'stock_id', 'securities_trader_id']
    merged = computed_eval.merge(ref, on=merge_keys, suffixes=('_c', '_r'))
    t.info(f"Matched rows        : {len(merged):,}")

    t.check(
        len(merged) > 0,
        f"Merge produced {len(merged):,} matched rows",
        "No rows matched — check stock_id / securities_trader_id types or date range",
    )
    if len(merged) == 0:
        return t

    # ── Compare each feature column ───────────────────────────────────────────
    cols_to_check = FEATURE_COLS + ['bias_60d', 'net_buy_amt_60d']
    for col in cols_to_check:
        c_col, r_col = col + '_c', col + '_r'
        if c_col not in merged.columns or r_col not in merged.columns:
            t.info(f"  {col:>16} : skipped (column absent in one side)")
            continue

        c_vals = merged[c_col].values.astype(float)
        r_vals = merged[r_col].values.astype(float)

        # Check NaN positions agree first
        nan_c = np.isnan(c_vals)
        nan_r = np.isnan(r_vals)
        nan_mismatch = int((nan_c != nan_r).sum())

        # Numeric diff on non-NaN rows only
        both_valid = ~nan_c & ~nan_r
        if both_valid.any():
            diff = np.abs(c_vals[both_valid] - r_vals[both_valid])
            max_diff  = diff.max()
            mean_diff = diff.mean()
        else:
            max_diff = mean_diff = 0.0

        t.info(f"  {col:>16}  max={max_diff:.2e}  mean={mean_diff:.2e}"
               f"  NaN rows={nan_c.sum()} (mismatch={nan_mismatch})")

        if nan_mismatch > 0:
            t.check(False, "",
                    f"{col} NaN positions differ by {nan_mismatch} rows")
        else:
            t.check(
                max_diff <= ATOL_FEATURES,
                f"{col} matches reference (max diff {max_diff:.2e} <= {ATOL_FEATURES:.0e},"
                f" {nan_c.sum()} NaN rows agree)",
                f"{col} DIFFERS from reference (max diff {max_diff:.2e} > {ATOL_FEATURES:.0e})",
            )

    return t


# ─── Test 2: HMM probability replication ─────────────────────────────────────

def _rolling_predict_proba_seq(sequence_features, model, window=120):
    """Internal helper — matches prepare_xgb_data.py rolling_predict_proba exactly."""
    seq_len     = len(sequence_features)
    n_components = model.n_components
    probs = np.zeros((seq_len, n_components))
    for t in range(seq_len):
        start = max(0, t - window + 1)
        X_w   = sequence_features[start : t + 1]
        probs[t] = model.predict_proba(X_w)[-1]
    return probs


def test_hmm_probability_replication() -> TestResult:
    """
    Re-runs the rolling HMM inference and compares against the prob_S columns
    stored in evaluation.parquet (written by prepare_xgb_data.py).

    Two sub-tests:
      2a  groupby sequence_id  — exact match with prepare_xgb_data.py
      2b  groupby stock+trader — pipeline_functions.py grouping in daily_update
    """
    t = TestResult("Test 2  HMM probability replication  (rolling 120-day window)")

    # Load inputs
    df_eval    = pd.read_parquet(FINAL_VECTORS_EVAL)
    df_ref_xgb = pd.read_parquet(XGB_LONG_EVAL)
    df_eval['date'] = pd.to_datetime(df_eval['date'])
    df_ref_xgb['date'] = pd.to_datetime(df_ref_xgb['date'])

    prob_cols = [c for c in df_ref_xgb.columns if c.startswith('prob_S')]
    t.info(f"Eval rows       : {len(df_eval):,}")
    t.info(f"XGB eval rows   : {len(df_ref_xgb):,}")

    # Load HMM model
    if not HMM_PARAMS.exists():
        t.check(False, "", f"HMM params file not found: {HMM_PARAMS}")
        return t
    model = load_hmm_model(str(HMM_PARAMS))
    t.info(f"HMM states      : {model.n_components}")

    # ── Reference probabilities ───────────────────────────────────────────────
    # The reference lives in df_ref_xgb but only for rows that survived
    # the dropna(subset=['high_ret', 'low_ret']) step in prepare_xgb_data.py.
    # We need to join on (date, stock_id) to align.
    ref_probs = df_ref_xgb.set_index(['date', 'stock_id'])[prob_cols]

    # ── Sub-test 2a: groupby sequence_id (exact replication) ─────────────────
    print("  [info] Running 2a: rolling proba groupby sequence_id ...")
    df_s = df_eval.sort_values(['sequence_id', 'date']).dropna(subset=FEATURE_COLS)

    proba_2a_parts = []
    for seq_id, grp in df_s.groupby('sequence_id', sort=True):
        p = _rolling_predict_proba_seq(grp[FEATURE_COLS].values, model, window=120)
        proba_2a_parts.append(
            pd.DataFrame(p, index=grp.index, columns=prob_cols)
        )
    proba_2a = pd.concat(proba_2a_parts)
    df_s = df_s.join(proba_2a.rename(columns={c: c + '_2a' for c in prob_cols}))

    # Align with reference by (date, stock_id)
    df_s_idx = df_s.set_index(['date', 'stock_id'])
    common_idx = df_s_idx.index.intersection(ref_probs.index)
    t.info(f"Rows available for 2a comparison: {len(common_idx):,}")

    if len(common_idx) > 0:
        pred_2a = df_s_idx.loc[common_idx, [c + '_2a' for c in prob_cols]].values
        ref_v   = ref_probs.loc[common_idx].values

        max_diff_2a  = np.abs(pred_2a - ref_v).max()
        mean_diff_2a = np.abs(pred_2a - ref_v).mean()
        t.info(f"2a  max  |computed - reference| : {max_diff_2a:.2e}")
        t.info(f"2a  mean |computed - reference| : {mean_diff_2a:.2e}")
        t.check(
            max_diff_2a <= ATOL_PROBA,
            f"2a sequence_id grouping replicates prepare_xgb_data.py (max diff {max_diff_2a:.2e})",
            f"2a sequence_id grouping DIFFERS from reference (max diff {max_diff_2a:.2e} > {ATOL_PROBA:.0e})",
        )

    # ── Sub-test 2b: groupby (stock_id, securities_trader_id) — pipeline_functions.py ─
    print("  [info] Running 2b: rolling proba groupby stock+trader ...")
    df_t = df_eval.sort_values(['stock_id', 'securities_trader_id', 'date']).dropna(subset=FEATURE_COLS)
    df_t_with_proba = compute_rolling_hmm_proba(
        df_t, model, feature_cols=FEATURE_COLS, window=120, show_progress=True
    )

    df_t_idx = df_t_with_proba.set_index(['date', 'stock_id'])
    common_idx_b = df_t_idx.index.intersection(ref_probs.index)
    t.info(f"Rows available for 2b comparison: {len(common_idx_b):,}")

    if len(common_idx_b) > 0:
        pred_2b = df_t_idx.loc[common_idx_b, prob_cols].values
        ref_v_b = ref_probs.loc[common_idx_b].values

        max_diff_2b  = np.abs(pred_2b - ref_v_b).max()
        mean_diff_2b = np.abs(pred_2b - ref_v_b).mean()
        pct_rows_differ = (np.abs(pred_2b - ref_v_b).max(axis=1) > ATOL_PROBA).mean() * 100

        t.info(f"2b  max  |computed - reference| : {max_diff_2b:.2e}")
        t.info(f"2b  mean |computed - reference| : {mean_diff_2b:.2e}")
        t.info(f"2b  rows with diff > {ATOL_PROBA:.0e} : {pct_rows_differ:.1f}%")
        t.check(
            max_diff_2b <= ATOL_PROBA,
            f"2b stock+trader grouping matches reference (max diff {max_diff_2b:.2e})",
            f"2b stock+trader grouping differs from reference at {pct_rows_differ:.1f}% of rows "
            f"(max diff {max_diff_2b:.2e}) - expected if breaks >7d exist within a sequence",
        )

    return t


# ─── Test 3: Long XGBoost signal quality ─────────────────────────────────────

def _xgb_metrics(t: TestResult, df: pd.DataFrame, direction: str,
                 model_path: Path, threshold: float) -> None:
    """
    Load the XGBoost model, run generate_signals, and report classification
    metrics against the ground-truth target_y column in df.
    """
    if not model_path.exists():
        t.check(False, "", f"{direction} model not found: {model_path}")
        return

    clf = load_xgb_model(str(model_path))
    feature_cols = clf.get_booster().feature_names
    if not feature_cols:
        feature_cols = (
            ['z_t', 'c_t', 'a_t', 's_t', 'm_t', 'bias_60d', 'net_buy_amt_60d']
            + [c for c in df.columns if c.startswith('prob_S')]
        )
    prob_col   = f'pred_prob_{direction}'
    signal_col = f'signal_{direction}'

    # Use only the XGB feature columns present in df
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        t.check(False, "", f"Missing features: {missing}")
        return

    # generate_signals adds pred_prob_long, pred_prob_short, signal_long, signal_short
    # We call it with the correct threshold for the direction we care about
    if direction == 'long':
        df_out = generate_signals(df, clf, clf, feature_cols=feature_cols,
                                  long_threshold=threshold, short_threshold=1.1)
    else:
        df_out = generate_signals(df, clf, clf, feature_cols=feature_cols,
                                  long_threshold=1.1, short_threshold=threshold)

    y_true = df_out['target_y'].values
    y_prob = df_out[prob_col].values
    y_pred = df_out[signal_col].astype(int).values

    n_signals = y_pred.sum()
    base_rate = y_true.mean() * 100
    auc       = roc_auc_score(y_true, y_prob)

    t.info(f"  {direction.upper()} - threshold={threshold}  "
           f"base_rate={base_rate:.2f}%  AUC={auc:.4f}")
    t.info(f"  Signals triggered : {n_signals:,} / {len(y_pred):,} rows")

    if n_signals > 0:
        prec  = precision_score(y_true, y_pred, zero_division=0)
        rec   = recall_score(y_true, y_pred, zero_division=0)
        f1    = f1_score(y_true, y_pred, zero_division=0)
        t.info(f"  Precision : {prec*100:.2f}%")
        t.info(f"  Recall    : {rec*100:.2f}%")
        t.info(f"  F1 score  : {f1:.4f}")

        # Sanity: precision should beat the naive base-rate
        t.check(
            prec > y_true.mean(),
            f"{direction} precision ({prec*100:.2f}%) > base rate ({base_rate:.2f}%) [OK]",
            f"{direction} precision ({prec*100:.2f}%) <= base rate ({base_rate:.2f}%) - model not filtering",
        )
        t.check(
            auc > 0.5,
            f"{direction} AUC={auc:.4f} > 0.5 (better than random) [OK]",
            f"{direction} AUC={auc:.4f} <= 0.5 - model no better than random",
        )
    else:
        t.check(False, "",
                f"{direction} model fired 0 signals at threshold={threshold} - "
                f"check model file or threshold")


def test_long_xgb_signals() -> TestResult:
    """Verify long XGBoost model signals against target_y in evaluation.parquet (long)."""
    t = TestResult("Test 3  Long XGBoost signal quality  (threshold=0.6)")

    df = pd.read_parquet(XGB_LONG_EVAL)
    t.info(f"evaluation.parquet (long) rows : {len(df):,}")
    t.info(f"Positive rate (target_y=1)     : {df['target_y'].mean()*100:.2f}%")

    _xgb_metrics(t, df, 'long', XGB_LONG_MODEL, threshold=0.6)
    return t


def test_short_xgb_signals() -> TestResult:
    """Verify short XGBoost model signals against target_y in evaluation.parquet (short)."""
    t = TestResult("Test 4  Short XGBoost signal quality  (threshold=0.8)")

    df = pd.read_parquet(XGB_SHORT_EVAL)
    t.info(f"evaluation.parquet (short) rows : {len(df):,}")
    t.info(f"Positive rate (target_y=1)      : {df['target_y'].mean()*100:.2f}%")

    _xgb_metrics(t, df, 'short', XGB_SHORT_MODEL, threshold=0.8)
    return t


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run broker-specific pipeline verification tests.")
    add_broker_path_args(parser)
    args = parser.parse_args()
    configure_paths(paths_from_args(args))

    print("\n" + "=" * 60)
    print("  BMEM Pipeline Test Bench")
    print("=" * 60)

    # Verify required files/dirs exist before running
    required = {
        "final_vectors_eval.parquet" : FINAL_VECTORS_EVAL,
        "hmm_data_eval.npz"          : HMM_EVAL_NPZ,
        "evaluation.parquet (long)"  : XGB_LONG_EVAL,
        "evaluation.parquet (short)" : XGB_SHORT_EVAL,
        f"broker data dir ({BROKER_ID})": BROKER_DATA_DIR,
        "stock data dir"             : STOCK_DATA_DIR,
        "xgb long model"             : XGB_LONG_MODEL,
        "xgb short model"            : XGB_SHORT_MODEL,
    }
    if RUN_INFERENCE:
        required["trained_hmm_params.npz"] = HMM_PARAMS
    missing_files = [name for name, path in required.items() if not path.exists()]
    if missing_files:
        print("\n[ERROR] Required files not found:")
        for name in missing_files:
            print(f"  {name}: {required[name]}")
        sys.exit(1)

    results = []

    print("\nRunning Test 1 ...")
    results.append(test_feature_consistency())

    print("\nRunning Test 1b ...")
    results.append(test_feature_computation())

    if RUN_INFERENCE:
        print("\nRunning Test 2 ...")
        results.append(test_hmm_probability_replication())
    else:
        print("\n[SKIP] Test 2  HMM inference (RUN_INFERENCE=False)")

    print("\nRunning Test 3 ...")
    results.append(test_long_xgb_signals())

    print("\nRunning Test 4 ...")
    results.append(test_short_xgb_signals())

    # Print all results
    for r in results:
        r.print_summary()

    # Final summary
    n_pass  = sum(r.passed for r in results)
    n_total = len(results)
    print(f"\n{'='*60}")
    print(f"  FINAL: {n_pass}/{n_total} tests passed")
    print(f"{'='*60}\n")

    sys.exit(0 if n_pass == n_total else 1)


if __name__ == "__main__":
    main()

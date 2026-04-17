"""
pipeline_functions.py

Core reusable functions extracted from the BMEM experimental scripts.
All functions are stateless (no file I/O, no argparse).
Imported by daily_update.py and any future pipeline scripts.
"""

import time
import numpy as np
import pandas as pd
from hmmlearn import hmm
import xgboost as xgb
from tqdm import tqdm


# ─── SHARED CONSTANTS ────────────────────────────────────────────────────────

FEATURE_COLS = ['z_t', 'c_t', 'a_t', 's_t', 'm_t']

# XGBoost expects exactly these 17 columns (must match training order)
XGB_FEATURE_COLS = (
    ['z_t', 'c_t', 'a_t', 's_t', 'm_t', 'bias_60d', 'net_buy_amt_60d']
    + [f'prob_S{i}' for i in range(10)]
)


# ─── DATA FETCHING ───────────────────────────────────────────────────────────

def fetch_broker_activity(
    api,
    start: str,
    end: str,
    trader_id: str = "1440",
    sleep: float = 0.1,
) -> pd.DataFrame:
    """
    Download broker daily trading activity from the FinMind API.

    Aggregates multiple price rows into one row per (date, stock_id) and
    computes: net_buy, net_buy_amount, buy_amount, sell_amount, avg_price.

    Parameters
    ----------
    api        : authenticated FinMind DataLoader instance
    start/end  : date range strings "YYYY-MM-DD" (inclusive)
    trader_id  : broker code, default "1440" (Merrill Lynch)
    sleep      : seconds between requests to avoid rate-limiting

    Returns
    -------
    pd.DataFrame with columns:
        date, stock_id, securities_trader, securities_trader_id,
        buy_amount, sell_amount, buy, sell, net_buy, net_buy_amount, avg_price
    Empty DataFrame on no data.
    """
    dates = pd.date_range(start=start, end=end, freq="B")
    dates = [d.strftime("%Y-%m-%d") for d in dates]

    all_agg = []
    failed_dates = []

    for d in dates:
        try:
            df = api.taiwan_stock_trading_daily_report(
                securities_trader_id=str(trader_id),
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
            # avg_price is undefined when net_buy == 0; leave as NaN
            agg["avg_price"] = agg["net_buy_amount"] / agg["net_buy"].replace(0, np.nan)

            all_agg.append(agg)

        except Exception as e:
            failed_dates.append({"date": d, "error": repr(e)})

        time.sleep(sleep)

    if failed_dates:
        print(f"[fetch_broker_activity] {len(failed_dates)} date(s) failed: "
              f"{[f['date'] for f in failed_dates]}")

    if all_agg:
        return pd.concat(all_agg, ignore_index=True)

    return pd.DataFrame(columns=[
        "date", "stock_id", "securities_trader", "securities_trader_id",
        "buy_amount", "sell_amount", "buy", "sell",
        "net_buy", "net_buy_amount", "avg_price",
    ])


def fetch_stock_prices(
    api,
    stock_ids: list,
    start: str,
    end: str,
    sleep: float = 0.1,
) -> pd.DataFrame:
    """
    Download OHLCV data for a list of stock IDs from the FinMind API.

    Parameters
    ----------
    api       : authenticated FinMind DataLoader instance
    stock_ids : list of stock id strings (e.g. ["2330", "2317"])
    start/end : date range strings "YYYY-MM-DD" (inclusive)
    sleep     : seconds between requests

    Returns
    -------
    Concatenated pd.DataFrame from FinMind taiwan_stock_daily.
    Empty DataFrame if all requests fail.
    """
    all_dfs = []
    failed = []

    for sid in tqdm(stock_ids, desc="Fetching stock prices", unit="stock"):
        try:
            df = api.taiwan_stock_daily(
                stock_id=str(sid),
                start_date=start,
                end_date=end,
            )
            if df is None or df.empty:
                failed.append({"stock_id": sid, "reason": "empty_response"})
                continue
            all_dfs.append(df)
        except Exception as e:
            failed.append({"stock_id": sid, "error": repr(e)})

        time.sleep(sleep)

    if failed:
        print(f"[fetch_stock_prices] {len(failed)} stock(s) failed: "
              f"{[f['stock_id'] for f in failed]}")

    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True)

    return pd.DataFrame()


# ─── FEATURE ENGINEERING ─────────────────────────────────────────────────────

def _rolling_zscore(series: pd.Series, window: int = 60) -> pd.Series:
    """Rolling Z-score normalization with min_periods = window."""
    m = series.rolling(window=window, min_periods=window).mean()
    s = series.rolling(window=window, min_periods=window).std()
    return (series - m) / s.replace(0, np.nan)


def compute_observation_features(
    broker_df: pd.DataFrame,
    stock_df: pd.DataFrame,
    disable_standardize: bool = True,
) -> pd.DataFrame:
    """
    Compute HMM observation features (z_t, c_t, a_t, s_t, m_t) and EDA
    features (bias_60d, net_buy_amt_60d, cost_60d, …) for every row in the
    combined broker × market-day skeleton.

    The caller must supply at least 60 trading-day rows of history per
    (stock_id, securities_trader_id) pair so that rolling normalizations
    produce valid (non-NaN) values on the most recent row.

    NaN rows from the rolling warm-up are NOT dropped here — the caller
    decides which rows to keep.

    Parameters
    ----------
    broker_df           : DataFrame from fetch_broker_activity (or loaded from parquet)
                          Required columns: date, stock_id, securities_trader_id,
                          buy, sell, net_buy, net_buy_amount
    stock_df            : DataFrame from fetch_stock_prices (or loaded from parquet)
                          Required columns: date, stock_id, close, Trading_Volume,
                          max, min, open
    disable_standardize : if True, skip rolling Z-score (use raw ratios).
                          Default True — matches the training data produced by
                          preprocess.py --disable_standardize.

    Returns
    -------
    pd.DataFrame with all input columns plus:
        r_t_raw, r_t,
        net_buy_amt_20d, net_buy_amt_60d, cost_20d, cost_60d,
        bias_20d, bias_60d,
        z_t_raw, z_t, a_t_raw, a_t, m_t_raw, m_t, s_t, c_t
    """
    broker_df = broker_df.copy()
    stock_df = stock_df.copy()

    broker_df['date'] = pd.to_datetime(broker_df['date'])
    stock_df['date'] = pd.to_datetime(stock_df['date'])

    # ── Stock preprocessing ───────────────────────────────────────────────────
    stock_df = stock_df.sort_values(['stock_id', 'date'])

    # Drop stocks with suspiciously low overall volume (illiquid / suspended)
    max_vol = stock_df.groupby('stock_id')['Trading_Volume'].max()
    valid_stocks = max_vol[max_vol >= 10_000_000].index
    stock_df = stock_df[stock_df['stock_id'].isin(valid_stocks)]

    # Drop stocks that ever had a 0 or NaN close (data contamination)
    contaminated = stock_df[stock_df['close'].isin([0, np.nan])]['stock_id'].unique()
    if len(contaminated):
        print(f"[compute_observation_features] Dropping {len(contaminated)} "
              f"contaminated stocks: {list(contaminated)}")
        stock_df = stock_df[~stock_df['stock_id'].isin(contaminated)]

    # Log return: r_t_raw = ln(C_t / C_{t-1})
    stock_df['r_t_raw'] = stock_df.groupby('stock_id')['close'].transform(
        lambda x: np.log(x / x.shift(1))
    )
    if disable_standardize:
        stock_df['r_t'] = stock_df['r_t_raw']
    else:
        stock_df['r_t'] = stock_df.groupby('stock_id')['r_t_raw'].transform(
            lambda x: _rolling_zscore(x, window=60)
        )

    stock_subset = stock_df[['date', 'stock_id', 'close', 'Trading_Volume', 'r_t', 'r_t_raw']]

    # ── Build skeleton: every (trader, stock) pair × every market trading day ─
    trader_stock_pairs = broker_df[['stock_id', 'securities_trader_id']].drop_duplicates()
    skeleton_df = pd.merge(trader_stock_pairs, stock_subset, on='stock_id', how='inner')

    df = pd.merge(
        skeleton_df, broker_df,
        on=['date', 'stock_id', 'securities_trader_id'],
        how='left',
    )

    # Fill missing broker activity days with zeros (broker didn't trade that day)
    df['buy'] = df['buy'].fillna(0)
    df['sell'] = df['sell'].fillna(0)
    df['net_buy'] = df['net_buy'].fillna(0)
    if 'net_buy_amount' in df.columns:
        df['net_buy_amount'] = df['net_buy_amount'].fillna(0)
    else:
        df['net_buy_amount'] = 0.0

    # ── Rolling EDA features ──────────────────────────────────────────────────
    df = df.sort_values(['stock_id', 'securities_trader_id', 'date'])
    g = df.groupby(['stock_id', 'securities_trader_id'])

    df['net_buy_amt_20d'] = g['net_buy_amount'].transform(lambda x: x.rolling(20, min_periods=1).sum())
    df['net_buy_vol_20d'] = g['net_buy'].transform(lambda x: x.rolling(20, min_periods=1).sum())
    df['net_buy_amt_60d'] = g['net_buy_amount'].transform(lambda x: x.rolling(60, min_periods=1).sum())
    df['net_buy_vol_60d'] = g['net_buy'].transform(lambda x: x.rolling(60, min_periods=1).sum())

    df['cost_20d'] = (
        (df['net_buy_amt_20d'] / df['net_buy_vol_20d'])
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )
    df['cost_60d'] = (
        (df['net_buy_amt_60d'] / df['net_buy_vol_60d'])
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    df['bias_20d'] = np.where(df['cost_20d'] > 0, df['close'] / df['cost_20d'] - 1, np.nan)
    df['bias_60d'] = np.where(df['cost_60d'] > 0, df['close'] / df['cost_60d'] - 1, np.nan)

    # ── HMM observation features ──────────────────────────────────────────────

    # z_t: normalized net flow (buy pressure relative to market volume)
    df['z_t_raw'] = df['net_buy'] / df['Trading_Volume']

    # a_t: activity level (total broker participation relative to volume)
    df['a_t_raw'] = (df['buy'].abs() + df['sell'].abs()) / df['Trading_Volume']

    # m_t: 5-day rolling mean of z_t_raw (smoothed flow direction)
    df['m_t_raw'] = g['z_t_raw'].transform(lambda x: x.rolling(5, min_periods=5).mean())

    if disable_standardize:
        df['z_t'] = df['z_t_raw']
        df['a_t'] = df['a_t_raw']
        df['m_t'] = df['m_t_raw']
    else:
        df['z_t'] = g['z_t_raw'].transform(lambda x: _rolling_zscore(x, 60))
        df['a_t'] = g['a_t_raw'].transform(lambda x: _rolling_zscore(x, 60))
        df['m_t'] = g['m_t_raw'].transform(lambda x: _rolling_zscore(x, 60))

    # s_t: directional persistence — 5-day rolling mean of sign(net_buy)
    df['s_t'] = g['net_buy'].transform(
        lambda x: np.sign(x).rolling(5, min_periods=5).mean()
    )

    # c_t: flow-price alignment score ∈ {-2, -1, 0, 1, 2}
    conditions = [
        (df['z_t_raw'] > 0) & (df['r_t_raw'] > 0),   # aggressive long (chasing)
        (df['z_t_raw'] > 0) & (df['r_t_raw'] <= 0),  # mild long (buy dip)
        (df['z_t_raw'] < 0) & (df['r_t_raw'] < 0),   # aggressive short (panic sell)
        (df['z_t_raw'] < 0) & (df['r_t_raw'] >= 0),  # mild short (sell into strength)
    ]
    df['c_t'] = np.select(conditions, [2, 1, -2, -1], default=0)

    return df


# ─── HMM INFERENCE ───────────────────────────────────────────────────────────

def load_hmm_model(params_path: str) -> hmm.GaussianHMM:
    """
    Reconstruct a GaussianHMM from a saved .npz parameter file.

    Applies symmetry and positive-definiteness fixes to the covariance
    matrices to prevent hmmlearn ValueError on floating-point drift.

    Parameters
    ----------
    params_path : path to trained_hmm_params.npz

    Returns
    -------
    A ready-to-use hmmlearn GaussianHMM (no further fitting needed).
    """
    p = np.load(params_path)
    startprob = p['startprob']
    transmat  = p['transmat']
    means     = p['means']
    covars    = p['covars'].copy()   # copy so we can modify in-place

    n_components = means.shape[0]

    if covars.ndim == 3:
        covar_type = "full"
    elif covars.ndim == 2:
        covar_type = "diag"
    else:
        covar_type = "spherical"

    # Fix floating-point precision so hmmlearn doesn't raise ValueError
    if covar_type == "full":
        for i in range(n_components):
            covars[i] = (covars[i] + covars[i].T) / 2
            np.fill_diagonal(covars[i], covars[i].diagonal() + 1e-5)
    elif covar_type == "diag":
        covars = np.maximum(covars, 1e-5)

    model = hmm.GaussianHMM(n_components=n_components, covariance_type=covar_type)
    model.startprob_ = startprob
    model.transmat_  = transmat
    model.means_     = means
    model.covars_    = covars
    model.n_features = means.shape[1]

    return model


def _rolling_predict_proba(
    sequence_features: np.ndarray,
    model: hmm.GaussianHMM,
    window: int = 120,
) -> np.ndarray:
    """
    Rolling state posterior probabilities for a single continuous sequence.

    At time t, only the most recent `window` observations (inclusive) are
    used to infer the state — no future data leaks in.

    Returns
    -------
    np.ndarray of shape (seq_len, n_components)
    """
    seq_len     = len(sequence_features)
    n_components = model.n_components
    probs = np.zeros((seq_len, n_components))

    for t in range(seq_len):
        start_idx = max(0, t - window + 1)
        X_window  = sequence_features[start_idx : t + 1]
        window_probs = model.predict_proba(X_window)
        probs[t] = window_probs[-1]

    return probs


def compute_rolling_hmm_proba(
    df: pd.DataFrame,
    model: hmm.GaussianHMM,
    feature_cols: list = None,
    window: int = 120,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Compute rolling HMM state posterior probabilities for every row in df.

    Groups by (stock_id, securities_trader_id) and applies the rolling
    no-lookahead window independently per sequence.

    Parameters
    ----------
    df           : must be sorted by (stock_id, securities_trader_id, date)
                   and must have no NaN in feature_cols
    model        : loaded GaussianHMM from load_hmm_model()
    feature_cols : list of observation feature column names (default: FEATURE_COLS)
    window       : rolling window size in trading days (default: 120)
    show_progress: show tqdm progress bar

    Returns
    -------
    df with added columns prob_S0 … prob_S{n_components-1}
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS

    n_components = model.n_components
    prob_cols    = [f'prob_S{i}' for i in range(n_components)]

    groups   = df.groupby(['stock_id', 'securities_trader_id'], sort=False)
    iterator = tqdm(groups, desc="HMM inference", total=len(groups)) if show_progress else groups

    result_parts = []
    for _, group in iterator:
        probs   = _rolling_predict_proba(group[feature_cols].values, model, window=window)
        prob_df = pd.DataFrame(probs, index=group.index, columns=prob_cols)
        result_parts.append(prob_df)

    return df.join(pd.concat(result_parts))


# ─── XGBOOST INFERENCE ───────────────────────────────────────────────────────

def load_xgb_model(model_path: str) -> xgb.XGBClassifier:
    """Load a saved XGBoost model from a JSON file."""
    clf = xgb.XGBClassifier()
    clf.load_model(model_path)
    return clf


def generate_signals(
    df: pd.DataFrame,
    clf_long: xgb.XGBClassifier,
    clf_short: xgb.XGBClassifier,
    feature_cols: list = None,
    long_threshold: float = 0.6,
    short_threshold: float = 0.8,
) -> pd.DataFrame:
    """
    Run both long and short XGBoost models and add prediction columns.

    Added columns
    -------------
    pred_prob_long  : model confidence that a +10 % gain will occur
    pred_prob_short : model confidence that a -10 % drop will occur
    signal_long     : True when pred_prob_long  >= long_threshold
    signal_short    : True when pred_prob_short >= short_threshold

    Parameters
    ----------
    df             : DataFrame that contains all columns in feature_cols
    clf_long/short : loaded XGBClassifier from load_xgb_model()
    feature_cols   : list of feature column names (default: XGB_FEATURE_COLS)
    long_threshold : confidence cut-off for long signal (default matches backtest: 0.6)
    short_threshold: confidence cut-off for short signal (default matches backtest: 0.8)
    """
    if feature_cols is None:
        feature_cols = XGB_FEATURE_COLS

    X    = df[feature_cols]
    df   = df.copy()

    df['pred_prob_long']  = clf_long.predict_proba(X)[:, 1]
    df['pred_prob_short'] = clf_short.predict_proba(X)[:, 1]
    df['signal_long']     = df['pred_prob_long']  >= long_threshold
    df['signal_short']    = df['pred_prob_short'] >= short_threshold

    return df

import argparse
import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path

from src.utils.paths import add_broker_path_args, paths_from_args

def time_series_normalize(series, window=20):
    """
    FUNC: 時序面正規化
    input: raw array (pandas Series)
    output: standardized array
    對於input 做20日rolling Z-score, 前19日值設定為NaN
    """
    rolling_mean = series.rolling(window=window, min_periods=window).mean()
    rolling_std = series.rolling(window=window, min_periods=window).std()

    # 避免 std 為 0 導致除以零的錯誤
    z_score = (series - rolling_mean) / rolling_std.replace(0, np.nan)

    return z_score

def main():
    # ARGPARSE Setup
    parser = argparse.ArgumentParser(description="Process broker and stock data to create HMM observation vectors.")
    add_broker_path_args(parser)
    parser.add_argument('--broker-data-path', '--broker_data_path', type=Path, help="Override broker parquet file or directory")
    parser.add_argument('--stock-info-dir', '--stock_info_dir', type=Path, help="Override the shared stock parquet directory")
    parser.add_argument('--output-dir', '--output_dir', type=Path, help="Override the broker-specific HMM data directory")
    parser.add_argument('--split_year', type=int, default=2025, help="Year to split train and eval sets at the first >7 day break.")
    parser.add_argument('--disable_standardize', action='store_true', help="Disable the 20-day rolling Z-score standardization.")

    args = parser.parse_args()
    paths = paths_from_args(args)
    broker_data_path = args.broker_data_path or paths.broker_raw_dir
    stock_info_dir = args.stock_info_dir or paths.stock_dir
    output_dir = args.output_dir or paths.hmm_data_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Read Dataframes
    print(f"Reading broker data from {broker_data_path}...")
    if broker_data_path.is_dir():
        broker_files = sorted(broker_data_path.glob('*.parquet'))
        if not broker_files:
            raise ValueError(f"No parquet files found in broker directory: {broker_data_path}")
        broker_df = pd.concat([pd.read_parquet(f) for f in broker_files], ignore_index=True)
        if 'securities_trader_id' not in broker_df.columns:
            broker_df['securities_trader_id'] = args.broker_id
        broker_df = broker_df.drop_duplicates(subset=['date', 'stock_id', 'securities_trader_id'])
    else:
        broker_df = pd.read_parquet(broker_data_path)
        if 'securities_trader_id' not in broker_df.columns:
            broker_df['securities_trader_id'] = args.broker_id

    print(f"Reading stock data from directory: {stock_info_dir}...")
    stock_files = glob.glob(os.path.join(stock_info_dir, '*.parquet'))
    if not stock_files:
        raise ValueError("No parquet files found in the stock info directory.")
    stock_df = pd.concat([pd.read_parquet(f) for f in stock_files], ignore_index=True)
    stock_df = stock_df.drop_duplicates(subset=['stock_id', 'date'], keep='last')

    broker_df['date'] = pd.to_datetime(broker_df['date'])
    stock_df['date'] = pd.to_datetime(stock_df['date'])

    # 2. Preprocess Stock Data First (Market-Level Processing)
    stock_df = stock_df.sort_values(by=['stock_id', 'date'])

    max_volume = stock_df.groupby('stock_id')['Trading_Volume'].max()
    valid_stocks = max_volume[max_volume >= 10000000].index
    stock_df = stock_df[stock_df['stock_id'].isin(valid_stocks)]

    contaminated_stocks = stock_df[stock_df['close'].isin([0, np.nan])]['stock_id'].unique()
    if len(contaminated_stocks) > 0:
        print(f"WARNING: Dropping {len(contaminated_stocks)} stocks entirely due to 0.0 or NaN close prices.")
        stock_df = stock_df[~stock_df['stock_id'].isin(contaminated_stocks)]

    # Feature 2: Logarithmic return r_t_raw = ln(C_t / C_t-1)
    stock_df['r_t_raw'] = stock_df.groupby('stock_id')['close'].transform(lambda x: np.log(x / x.shift(1)))

    if args.disable_standardize:
        stock_df['r_t'] = stock_df['r_t_raw']
    else:
        stock_df['r_t'] = stock_df.groupby('stock_id')['r_t_raw'].transform(lambda x: time_series_normalize(x, window=60))

    # --- Identify Train/Eval split dates ---
    print(f"Identifying Train/Eval split dates based on the first >7 day break in or after {args.split_year}...")
    market_days = stock_df[['stock_id', 'date']].drop_duplicates()
    market_days['date_diff'] = market_days.groupby('stock_id')['date'].diff().dt.days

    long_breaks = market_days[(market_days['date_diff'] > 7) & (market_days['date'].dt.year >= args.split_year)]
    split_dates = long_breaks.groupby('stock_id')['date'].min()

    stock_subset = stock_df[['date', 'stock_id', 'close', 'Trading_Volume', 'r_t', 'r_t_raw']]

    # 3. Build Skeleton & Merge
    print("Aligning broker data to continuous market trading days...")
    trader_stock_pairs = broker_df[['stock_id', 'securities_trader_id']].drop_duplicates()
    skeleton_df = pd.merge(trader_stock_pairs, stock_subset, on='stock_id', how='inner')

    df = pd.merge(skeleton_df, broker_df, on=['date', 'stock_id', 'securities_trader_id'], how='left')

    df['buy'] = df['buy'].fillna(0)
    df['sell'] = df['sell'].fillna(0)
    df['net_buy'] = df['net_buy'].fillna(0)

    if 'net_buy_amount' in df.columns:
        df['net_buy_amount'] = df['net_buy_amount'].fillna(0)
    
    df = df.merge(split_dates.rename('split_date'), on='stock_id', how='left')
    df['split_date'] = df['split_date'].fillna(pd.Timestamp('2099-12-31'))
    df['is_eval'] = df['date'] >= df['split_date']

    # 4. Define Sequences & Calculate Features
    print("Calculating observation vectors...")
    df = df.sort_values(by=['stock_id', 'securities_trader_id', 'date'])
    grouped_trader = df.groupby(['stock_id', 'securities_trader_id'])

    # === [FOR EDA]: 計算券商 20日/60日 成本價及累積買超金額 ===
    if 'net_buy_amount' in df.columns:
        # 滾動計算累積金額與累積股數
        df['net_buy_amt_20d'] = grouped_trader['net_buy_amount'].transform(lambda x: x.rolling(window=20, min_periods=1).sum())
        df['net_buy_vol_20d'] = grouped_trader['net_buy'].transform(lambda x: x.rolling(window=20, min_periods=1).sum())
        
        df['net_buy_amt_60d'] = grouped_trader['net_buy_amount'].transform(lambda x: x.rolling(window=60, min_periods=1).sum())
        df['net_buy_vol_60d'] = grouped_trader['net_buy'].transform(lambda x: x.rolling(window=60, min_periods=1).sum())
        
        # 成本價 = 累積金額 / 累積股數
        df['cost_20d'] = (df['net_buy_amt_20d'] / df['net_buy_vol_20d']).replace([np.inf, -np.inf], np.nan).fillna(0)
        df['cost_60d'] = (df['net_buy_amt_60d'] / df['net_buy_vol_60d']).replace([np.inf, -np.inf], np.nan).fillna(0)

        # [NEW] 計算乖離率 bias_20d, bias_60d
        df['bias_20d'] = np.where(df['cost_20d'] > 0, df['close'] / df['cost_20d'] - 1, np.nan)
        df['bias_60d'] = np.where(df['cost_60d'] > 0, df['close'] / df['cost_60d'] - 1, np.nan)
    else:
        # 防呆機制
        print("WARNING: 'net_buy_amount' not found in broker data! Filling EDA columns with 0.")
        df['cost_20d'] = 0
        df['cost_60d'] = 0
        df['net_buy_amt_20d'] = 0
        df['net_buy_amt_60d'] = 0
        df['bias_20d'] = np.nan
        df['bias_60d'] = np.nan
    # ==========================================================

    # Feature 1 (z_t), Feature 3 (a_t) & Feature 5 (m_t)
    df['z_t_raw'] = df['net_buy'] / df['Trading_Volume']
    df['a_t_raw'] = (df['buy'].abs() + df['sell'].abs()) / df['Trading_Volume']

    # [NEW] Feature 5: 近 5 日買賣超平均 (m_t_raw)
    df['m_t_raw'] = grouped_trader['z_t_raw'].transform(lambda x: x.rolling(window=5, min_periods=5).mean())    

    if args.disable_standardize:
        df['z_t'] = df['z_t_raw'] 
        df['a_t'] = df['a_t_raw']
        df['m_t'] = df['m_t_raw']
    else:
        df['z_t'] = grouped_trader['z_t_raw'].transform(lambda x: time_series_normalize(x, window=60))
        df['a_t'] = grouped_trader['a_t_raw'].transform(lambda x: time_series_normalize(x, window=60))
        df['m_t'] = grouped_trader['m_t_raw'].transform(lambda x: time_series_normalize(x, window=60))

    # Feature 4: Directional Persistence (s_t) - 永遠使用 5 日 Rolling
    df['s_t'] = grouped_trader['net_buy'].transform(lambda x: np.sign(x).rolling(window=5, min_periods=5).mean())

    # Feature 6: Flow price alignment
    conditions = [
        (df['z_t_raw'] > 0) & (df['r_t_raw'] > 0),   # [If]   買進 且 上漲 -> 激進做多 (追價)
        (df['z_t_raw'] > 0) & (df['r_t_raw'] <= 0),  # [Elif] 買進 且 沒漲 -> 溫和做多 (抄底)
        (df['z_t_raw'] < 0) & (df['r_t_raw'] < 0),   # [Elif] 賣出 且 下跌 -> 激進做空 (殺跌)
        (df['z_t_raw'] < 0) & (df['r_t_raw'] >= 0)   # [Elif] 賣出 且 沒跌 -> 溫和做空 (逢高出)
    ]

    # 對應上面每個條件的分數
    choices = [
        2,  # 對應條件 1 (激進做多)
        1,  # 對應條件 2 (溫和做多)
       -2,  # 對應條件 3 (激進做空)
       -1   # 對應條件 4 (溫和做空)
    ]

    # 使用 np.select 執行判斷，default=0 就是最後的 [Else] (代表 z_t_raw == 0 沒動作)
    df['c_t_raw'] = np.select(conditions, choices, default=0)
    
    # 直接賦值，不需要做 Z-score (保持特徵在 -2, -1, 0, 1, 2 的純淨度)
    df['c_t'] = df['c_t_raw']

    # Calculate sequences
    df['date_diff'] = grouped_trader['date'].diff().dt.days
    df['new_seq_flag'] = df['date_diff'].isnull() | (df['date_diff'] > 7)
    df['sequence_id'] = df['new_seq_flag'].cumsum()


    # 5. Clean up & Split for hmmlearn
    feature_cols = ['z_t', 'c_t', 'a_t', 's_t', 'm_t']

    # Drop rows with NaN (自動根據特徵需求裁切掉無法計算的前導天數)
    cleaned_df = df.dropna(subset=feature_cols).copy()

    train_df = cleaned_df[~cleaned_df['is_eval']].copy()
    eval_df = cleaned_df[cleaned_df['is_eval']].copy()

    lengths_train = train_df.groupby('sequence_id').size().values
    X_train = train_df[feature_cols].values

    lengths_eval = eval_df.groupby('sequence_id').size().values
    X_eval = eval_df[feature_cols].values

    print(f"--- Data Split Summary ---")
    print(f"Standardization: {'DISABLED' if args.disable_standardize else 'ENABLED (20-day Z-score)'}")
    print(f"Features: {feature_cols}")
    print(f"Train Set: {X_train.shape[0]} observations across {len(lengths_train)} continuous sequences.")
    print(f"Eval Set:  {X_eval.shape[0]} observations across {len(lengths_eval)} continuous sequences.")

    # 6. Save outputs into .npz and .parquet files
    print("Packing data into output files...")

    train_npz_path = output_dir / 'hmm_data_train.npz'
    eval_npz_path = output_dir / 'hmm_data_eval.npz'

    train_parquet_path = output_dir / 'final_vectors_train.parquet'
    eval_parquet_path = output_dir / 'final_vectors_eval.parquet'

    np.savez(train_npz_path, lengths=lengths_train, observations=X_train, feature_names=feature_cols)
    np.savez(eval_npz_path, lengths=lengths_eval, observations=X_eval, feature_names=feature_cols)

    # [NEW] 在輸出 parquet 的清單中，確保加入了 net_buy_amt_20d 與兩組 bias
    out_cols = ['date', 'stock_id', 'securities_trader_id', 'sequence_id', 'close', 'net_buy', 
                'net_buy_amt_20d', 'net_buy_amt_60d', 'cost_20d', 'cost_60d', 'bias_20d', 'bias_60d'] + feature_cols
    
    train_df[out_cols].to_parquet(train_parquet_path, index=False)
    eval_df[out_cols].to_parquet(eval_parquet_path, index=False)

    print(f"Saved training data to {train_npz_path} and {train_parquet_path}")
    print(f"Saved evaluation data to {eval_npz_path} and {eval_parquet_path}")
    print("Done! All matrices and dataframes packed and saved successfully.")

if __name__ == '__main__':
    main()

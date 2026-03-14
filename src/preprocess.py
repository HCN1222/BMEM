import argparse
import pandas as pd
import numpy as np
import os
import glob

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
    parser.add_argument('--broker_data_path', type=str, required=True, help="Path to the broker data .parquet file")
    parser.add_argument('--stock_info_dir', type=str, required=True, help="Path to the directory containing stock .parquet files")
    parser.add_argument('--output_dir', type=str, default='./data/preprocessed_data/', help="Directory to save the output files")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Read Dataframes
    print(f"Reading broker data from {args.broker_data_path}...")
    broker_df = pd.read_parquet(args.broker_data_path)
    
    print(f"Reading stock data from directory: {args.stock_info_dir}...")
    stock_files = glob.glob(os.path.join(args.stock_info_dir, '*.parquet'))
    if not stock_files:
        raise ValueError("No parquet files found in the stock info directory.")
    stock_df = pd.concat([pd.read_parquet(f) for f in stock_files], ignore_index=True)
    
    broker_df['date'] = pd.to_datetime(broker_df['date'])
    stock_df['date'] = pd.to_datetime(stock_df['date'])
    
    # 2. Preprocess Stock Data First (Market-Level Processing)
    stock_df = stock_df.sort_values(by=['stock_id', 'date'])
    
    # 過濾從未達到 10,000,000 交易量的股票
    max_volume = stock_df.groupby('stock_id')['Trading_Volume'].max()
    valid_stocks = max_volume[max_volume >= 10000000].index
    stock_df = stock_df[stock_df['stock_id'].isin(valid_stocks)]
    print(f"Volume Filter: Kept {len(valid_stocks)} out of {len(max_volume)} stocks.")
    
    # 移除包含 0.0 或 NaN 收盤價的受污染股票
    contaminated_stocks = stock_df[stock_df['close'].isin([0, np.nan])]['stock_id'].unique()
    if len(contaminated_stocks) > 0:
        print(f"WARNING: Dropping {len(contaminated_stocks)} stocks entirely due to 0.0 or NaN close prices.")
        stock_df = stock_df[~stock_df['stock_id'].isin(contaminated_stocks)]
    
    # Feature 2: Logarithmic return r_t_raw = ln(C_t / C_t-1)
    stock_df['r_t_raw'] = stock_df.groupby('stock_id')['close'].transform(lambda x: np.log(x / x.shift(1)))
    stock_df['r_t'] = stock_df.groupby('stock_id')['r_t_raw'].transform(lambda x: time_series_normalize(x, window=20))
    
    stock_subset = stock_df[['date', 'stock_id', 'Trading_Volume', 'r_t']]
    
    # 3. 建立完整的時間序列骨架 (Skeleton) 並合併資料
    print("Aligning broker data to continuous market trading days...")
    trader_stock_pairs = broker_df[['stock_id', 'securities_trader_id']].drop_duplicates()
    skeleton_df = pd.merge(trader_stock_pairs, stock_subset, on='stock_id', how='inner')
    
    df = pd.merge(skeleton_df, broker_df, on=['date', 'stock_id', 'securities_trader_id'], how='left')
    
    # 填補無交易日的買賣數值為 0
    df['buy'] = df['buy'].fillna(0)
    df['sell'] = df['sell'].fillna(0)
    df['net_buy'] = df['net_buy'].fillna(0)
    
    # 4. Define Sequences & Calculate Features
    print("Calculating observation vectors...")
    df = df.sort_values(by=['stock_id', 'securities_trader_id', 'date'])
    
    # [修正核心] 先以 (卷商, 股票) 為群組計算 Rolling 特徵，避免被長假打斷
    grouped_trader = df.groupby(['stock_id', 'securities_trader_id'])
    
    # Feature 1: Normalized Net Buy (z_t)
    df['z_t_raw'] = df['net_buy'] / df['Trading_Volume']
    df['z_t'] = grouped_trader['z_t_raw'].transform(lambda x: time_series_normalize(x, window=20))
    
    # Feature 3: Activity level (a_t)
    df['a_t_raw'] = (df['buy'].abs() + df['sell'].abs()) / df['Trading_Volume']
    df['a_t'] = grouped_trader['a_t_raw'].transform(lambda x: time_series_normalize(x, window=20))
    
    # Feature 4: Directional Persistence (s_t)
    df['s_t'] = grouped_trader['net_buy'].transform(lambda x: np.sign(x).rolling(window=5, min_periods=5).mean())
    
    # 計算時間差與設定 HMM sequence_id
    df['date_diff'] = grouped_trader['date'].diff().dt.days
    df['new_seq_flag'] = df['date_diff'].isnull() | (df['date_diff'] > 7)
    df['sequence_id'] = df['new_seq_flag'].cumsum()
    
    # 5. Clean up for hmmlearn
    feature_cols = ['z_t', 'r_t', 'a_t', 's_t']
    
    # 刪除含有 NaN 的列 (這會乾淨地移除每個卷商-股票組合最開始那 19 天無法計算完整 Rolling 的資料)
    cleaned_df = df.dropna(subset=feature_cols).copy()
    
    lengths = cleaned_df.groupby('sequence_id').size().values
    X = cleaned_df[feature_cols].values
    
    print(f"Final Data Shape: {X.shape[0]} observations across {len(lengths)} unique continuous sequences.")
    
    # 6. Save outputs
    x_path = os.path.join(args.output_dir, 'hmm_X.npy')
    lengths_path = os.path.join(args.output_dir, 'hmm_lengths.npy')
    df_path = os.path.join(args.output_dir, 'final_vectors.parquet')
    
    np.save(x_path, X)
    np.save(lengths_path, lengths)
    
    cleaned_df[['date', 'stock_id', 'securities_trader_id', 'sequence_id'] + feature_cols].to_parquet(df_path, index=False)
    
    print(f"Saved hmm_X.npy to {x_path}")
    print(f"Saved hmm_lengths.npy to {lengths_path}")
    print("Done!")

if __name__ == '__main__':
    main()
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
    
    # Replace 0 std with NaN to avoid division by zero
    z_score = (series - rolling_mean) / rolling_std.replace(0, np.nan)
    
    return z_score

def main():
    # ARGPARSE Setup
    parser = argparse.ArgumentParser(description="Process broker and stock data to create HMM observation vectors.")
    parser.add_argument('--broker_data_path', type=str, required=True, help="Path to the broker data .parquet file")
    parser.add_argument('--stock_info_dir', type=str, required=True, help="Path to the directory containing stock .parquet files")
    parser.add_argument('--output_dir', type=str, default='./data/preprocessed_data/', help="Directory to save the output files")
    parser.add_argument('--split_year', type=int, default=2025, help="Year to split train and eval sets at the first >7 day break.")
    
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
    
    max_volume = stock_df.groupby('stock_id')['Trading_Volume'].max()
    valid_stocks = max_volume[max_volume >= 10000000].index
    stock_df = stock_df[stock_df['stock_id'].isin(valid_stocks)]
    
    contaminated_stocks = stock_df[stock_df['close'].isin([0, np.nan])]['stock_id'].unique()
    if len(contaminated_stocks) > 0:
        print(f"WARNING: Dropping {len(contaminated_stocks)} stocks entirely due to 0.0 or NaN close prices.")
        stock_df = stock_df[~stock_df['stock_id'].isin(contaminated_stocks)]
    
    # Feature 2: Logarithmic return r_t_raw = ln(C_t / C_t-1)
    stock_df['r_t_raw'] = stock_df.groupby('stock_id')['close'].transform(lambda x: np.log(x / x.shift(1)))
    stock_df['r_t'] = stock_df.groupby('stock_id')['r_t_raw'].transform(lambda x: time_series_normalize(x, window=20))
    
    # --- NEW: Identify Split Dates per Stock ---
    print(f"Identifying Train/Eval split dates based on the first >7 day break in or after {args.split_year}...")
    market_days = stock_df[['stock_id', 'date']].drop_duplicates()
    market_days['date_diff'] = market_days.groupby('stock_id')['date'].diff().dt.days
    
    # Find all breaks > 7 days occurring in or after the target split year
    long_breaks = market_days[(market_days['date_diff'] > 7) & (market_days['date'].dt.year >= args.split_year)]
    # The first break is our watershed date for that stock
    split_dates = long_breaks.groupby('stock_id')['date'].min()
    
    stock_subset = stock_df[['date', 'stock_id', 'Trading_Volume', 'r_t']]
    
    # 3. Build Skeleton & Merge
    print("Aligning broker data to continuous market trading days...")
    trader_stock_pairs = broker_df[['stock_id', 'securities_trader_id']].drop_duplicates()
    skeleton_df = pd.merge(trader_stock_pairs, stock_subset, on='stock_id', how='inner')
    
    df = pd.merge(skeleton_df, broker_df, on=['date', 'stock_id', 'securities_trader_id'], how='left')
    
    df['buy'] = df['buy'].fillna(0)
    df['sell'] = df['sell'].fillna(0)
    df['net_buy'] = df['net_buy'].fillna(0)
    
    # Map the split_date to the main dataframe
    df = df.merge(split_dates.rename('split_date'), on='stock_id', how='left')
    # If a stock has NO long breaks after the split year, assign to a far future date so it all goes to Train
    df['split_date'] = df['split_date'].fillna(pd.Timestamp('2099-12-31'))
    # Label Evaluation rows
    df['is_eval'] = df['date'] >= df['split_date']
    
    # 4. Define Sequences & Calculate Features
    print("Calculating observation vectors...")
    df = df.sort_values(by=['stock_id', 'securities_trader_id', 'date'])
    grouped_trader = df.groupby(['stock_id', 'securities_trader_id'])
    
    # Feature 1: Normalized Net Buy (z_t)
    df['z_t_raw'] = df['net_buy'] / df['Trading_Volume']
    df['z_t'] = grouped_trader['z_t_raw'].transform(lambda x: time_series_normalize(x, window=20))
    
    # Feature 3: Activity level (a_t)
    df['a_t_raw'] = (df['buy'].abs() + df['sell'].abs()) / df['Trading_Volume']
    df['a_t'] = grouped_trader['a_t_raw'].transform(lambda x: time_series_normalize(x, window=20))
    
    # Feature 4: Directional Persistence (s_t)
    df['s_t'] = grouped_trader['net_buy'].transform(lambda x: np.sign(x).rolling(window=5, min_periods=5).mean())
    
    # Calculate sequences
    df['date_diff'] = grouped_trader['date'].diff().dt.days
    df['new_seq_flag'] = df['date_diff'].isnull() | (df['date_diff'] > 7)
    df['sequence_id'] = df['new_seq_flag'].cumsum()
    
    # 5. Clean up & Split for hmmlearn
    feature_cols = ['z_t', 'r_t', 'a_t', 's_t']
    
    # Drop rows with NaN (removes the initial 19 days of rolling window calculations)
    cleaned_df = df.dropna(subset=feature_cols).copy()
    
    # Split the dataset
    train_df = cleaned_df[~cleaned_df['is_eval']].copy()
    eval_df = cleaned_df[cleaned_df['is_eval']].copy()
    
    # Extract lengths and X matrices
    lengths_train = train_df.groupby('sequence_id').size().values
    X_train = train_df[feature_cols].values
    
    lengths_eval = eval_df.groupby('sequence_id').size().values
    X_eval = eval_df[feature_cols].values
    
    print(f"--- Data Split Summary ---")
    print(f"Train Set: {X_train.shape[0]} observations across {len(lengths_train)} continuous sequences.")
    print(f"Eval Set:  {X_eval.shape[0]} observations across {len(lengths_eval)} continuous sequences.")
    
    # 6. Save outputs
    print("Packing data into .npz archives...")
    
    train_path = os.path.join(args.output_dir, 'hmm_data_train.npz')
    eval_path = os.path.join(args.output_dir, 'hmm_data_eval.npz')
    parquet_path = os.path.join(args.output_dir, 'final_vectors.parquet')
    
    # Save Train Set
    np.savez(
        train_path, 
        lengths=lengths_train, 
        observations=X_train, 
        feature_names=feature_cols
    )
    
    # Save Eval Set
    np.savez(
        eval_path, 
        lengths=lengths_eval, 
        observations=X_eval, 
        feature_names=feature_cols
    )
    
    # Save the dataframe for debugging/tracing
    out_cols = ['date', 'stock_id', 'securities_trader_id', 'sequence_id', 'is_eval'] + feature_cols
    cleaned_df[out_cols].to_parquet(parquet_path, index=False)
    
    print(f"Saved training data to {train_path}")
    print(f"Saved evaluation data to {eval_path}")
    print("Done! All matrices packed and saved successfully.")

if __name__ == '__main__':
    main()
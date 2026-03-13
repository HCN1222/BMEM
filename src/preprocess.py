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
    # Calculate rolling mean and standard deviation
    rolling_mean = series.rolling(window=window, min_periods=window).mean()
    rolling_std = series.rolling(window=window, min_periods=window).std()
    
    # Calculate Z-score. Replace 0 std with NaN to avoid division by zero
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
    
    # Filter out stocks that never hit 10,000,000 volume
    max_volume = stock_df.groupby('stock_id')['Trading_Volume'].max()
    valid_stocks = max_volume[max_volume/1000 >= 12600].index
    stock_df = stock_df[stock_df['stock_id'].isin(valid_stocks)]
    print(f"Volume Filter: Kept {len(valid_stocks)} out of {len(max_volume)} stocks.")
    
    # Identify and drop stocks with contaminated close prices (0 or NaN)
    contaminated_stocks = stock_df[stock_df['close'].isin([0, np.nan])]['stock_id'].unique()
    if len(contaminated_stocks) > 0:
        print(f"WARNING: Dropping {len(contaminated_stocks)} stocks entirely due to 0.0 or NaN close prices.")
        stock_df = stock_df[~stock_df['stock_id'].isin(contaminated_stocks)]
    
    # Feature 2: Logarithmic return r_t_raw = ln(C_t / C_t-1)
    # Calculated on continuous market data before merging so gaps in a specific broker's trading don't skew the return
    stock_df['r_t_raw'] = stock_df.groupby('stock_id')['close'].transform(lambda x: np.log(x / x.shift(1)))
    stock_df['r_t'] = stock_df.groupby('stock_id')['r_t_raw'].transform(lambda x: time_series_normalize(x, window=20))
    
    stock_subset = stock_df[['date', 'stock_id', 'Trading_Volume', 'r_t']]
    
    # 3. Merge Data
    print("Merging broker and stock data...")
    df = pd.merge(broker_df, stock_subset, on=['date', 'stock_id'], how='inner')
    
    # 4. Define Sequences & Calculate Features
    print("Calculating sequence-specific observation vectors...")
    df = df.sort_values(by=['stock_id', 'securities_trader_id', 'date'])
    
    # Calculate time difference to find gaps > 7 days
    df['date_diff'] = df.groupby(['stock_id', 'securities_trader_id'])['date'].diff().dt.days
    
    # Flag new sequences (first row of group OR gap > 7 days)
    df['new_seq_flag'] = df['date_diff'].isnull() | (df['date_diff'] > 7)
    df['sequence_id'] = df['new_seq_flag'].cumsum()
    
    grouped = df.groupby('sequence_id')
    
    # Feature 1: Normalized Net Buy (z_t)
    df['z_t_raw'] = df['net_buy'] / df['Trading_Volume']
    df['z_t'] = grouped['z_t_raw'].transform(lambda x: time_series_normalize(x, window=20))
    
    # Feature 3: Activity level (a_t)
    df['a_t_raw'] = (df['buy'].abs() + df['sell'].abs()) / df['Trading_Volume']
    df['a_t'] = grouped['a_t_raw'].transform(lambda x: time_series_normalize(x, window=20))
    
    # Feature 4: Directional Persistence (s_t)
    df['s_t'] = grouped['net_buy'].transform(lambda x: np.sign(x).rolling(window=5, min_periods=5).mean())
    
    # 5. Clean up for hmmlearn
    feature_cols = ['z_t', 'r_t', 'a_t', 's_t']
    
    # Drop rows with NaN (this inherently drops the first 19 days of every sequence)
    cleaned_df = df.dropna(subset=feature_cols).copy()
    
    # Extract the lengths array (count of valid days per sequence)
    # If a sequence was shorter than 20 days, it is completely removed here.
    lengths = cleaned_df.groupby('sequence_id').size().values
    
    # Extract the 2D observation matrix
    X = cleaned_df[feature_cols].values
    
    print(f"Final Data Shape: {X.shape[0]} observations across {len(lengths)} unique continuous sequences.")
    
    # 6. Save outputs
    x_path = os.path.join(args.output_dir, 'hmm_X.npy')
    lengths_path = os.path.join(args.output_dir, 'hmm_lengths.npy')
    df_path = os.path.join(args.output_dir, 'final_vectors.parquet')
    
    np.save(x_path, X)
    np.save(lengths_path, lengths)
    
    # Optional: Save the dataframe if you need to trace sequence IDs back to stock/broker dates later
    cleaned_df[['date', 'stock_id', 'securities_trader_id', 'sequence_id'] + feature_cols].to_parquet(df_path, index=False)
    
    print(f"Saved hmm_X.npy to {x_path}")
    print(f"Saved hmm_lengths.npy to {lengths_path}")
    print("Done!")

if __name__ == '__main__':
    main()
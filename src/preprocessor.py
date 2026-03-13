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
    parser = argparse.ArgumentParser(description="Process broker and stock data to create observation vectors.")
    parser.add_argument('--broker_data_path', type=str, required=True, help="Path to the broker data .parquet file")
    parser.add_argument('--stock_info_dir', type=str, required=True, help="Path to the directory containing stock .parquet files")
    parser.add_argument('--output_path', type=str, default='observation_vectors.parquet', help="Path to save the output .parquet file")
    
    args = parser.parse_args()
    
    # 1. Read Dataframes (.parquet)
    print(f"Reading broker data from {args.broker_data_path}...")
    broker_df = pd.read_parquet(args.broker_data_path)
    
    print(f"Reading stock data from directory: {args.stock_info_dir}...")
    stock_files = glob.glob(os.path.join(args.stock_info_dir, '*.parquet'))
    stock_dfs = [pd.read_parquet(f) for f in stock_files]
    
    if not stock_dfs:
        raise ValueError("No parquet files found in the stock info directory.")
        
    stock_df = pd.concat(stock_dfs, ignore_index=True)
    
    # Ensure date columns are parsed correctly for time-based sorting
    broker_df['date'] = pd.to_datetime(broker_df['date'])
    stock_df['date'] = pd.to_datetime(stock_df['date'])
    
    # 2. Preprocess Stock Data First (Calculate Market Return r_t)
    # Sort by stock and date to ensure correct time sequence
    stock_df = stock_df.sort_values(by=['stock_id', 'date'])
    
    # Feature 2: Logarithmic return r_t_raw = ln(C_t / C_t-1)
    stock_df['r_t_raw'] = stock_df.groupby('stock_id')['close'].transform(lambda x: np.log(x / x.shift(1)))
    
    # Time-series normalization for r_t: 20-day rolling Z-score
    stock_df['r_t'] = stock_df.groupby('stock_id')['r_t_raw'].transform(lambda x: time_series_normalize(x, window=20))
    
    # Keep only required columns from stock data for merging
    stock_subset = stock_df[['date', 'stock_id', 'Trading_Volume', 'close', 'r_t']]
    
    # 3. Merge Broker and Stock Data
    print("Merging broker and stock data...")
    # Merge on date and stock_id to map market volumes to the broker data
    df = pd.merge(broker_df, stock_subset, on=['date', 'stock_id'], how='inner')
    
    # 4. Create Observation Vectors (each broker-stock pair as an individual sequence)
    print("Calculating observation vectors...")
    
    # Sort specifically by stock_id, securities_trader_id, and date to setup sequential calculations
    df = df.sort_values(by=['stock_id', 'securities_trader_id', 'date'])
    grouped = df.groupby(['stock_id', 'securities_trader_id'])
    
    # Feature 1: Normalized Net Buy (z_t)
    # z_t_raw = (buy_t - sell_t) / V_t = net_buy / Trading_Volume
    df['z_t_raw'] = df['net_buy'] / df['Trading_Volume']
    df['z_t'] = grouped['z_t_raw'].transform(lambda x: time_series_normalize(x, window=20))
    
    # Feature 3: Activity level (a_t)
    # a_t_raw = (|buy_t| + |sell_t|) / V_t
    df['a_t_raw'] = (df['buy'].abs() + df['sell'].abs()) / df['Trading_Volume']
    df['a_t'] = grouped['a_t_raw'].transform(lambda x: time_series_normalize(x, window=20))
    
    # Feature 4: Directional Persistence (5 days) (s_t)
    # s_t = 1/5 * sum_{i=t-4}^{t} sign(nb_i)
    # Using rolling mean of the sign directly maps to 1/5 * sum
    df['s_t'] = grouped['net_buy'].transform(lambda x: np.sign(x).rolling(window=5, min_periods=5).mean())
    
    # Select the final columns required for the observation vector
    observation_vectors = df[['date', 'stock_id', 'securities_trader_id', 'z_t', 'r_t', 'a_t', 's_t']]
    
    # 5. Save the file
    print(f"Saving observation vectors to {args.output_path}...")
    observation_vectors.to_parquet(args.output_path, index=False)
    print("Done!")

if __name__ == '__main__':
    main()
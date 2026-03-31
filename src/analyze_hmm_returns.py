import pandas as pd
import numpy as np
import os
import sys
from pandas.api.indexers import FixedForwardWindowIndexer

# 開啟 Pandas 對齊全形中文字的支援
pd.set_option('display.unicode.east_asian_width', True)

print("1. 正在載入主資料與 HMM 狀態...")
# ==========================================
# 1. 載入資料與對齊
# ==========================================
try:
    df = pd.read_parquet('./data/preprocessed_data/exp3/final_vectors_train.parquet')
    hmm_params = np.load('./outputs/exp3/states_10/trained_hmm_params.npz')
    states = hmm_params['decoded_states']
except Exception as e:
    print(f"Error loading files: {e}")
    sys.exit()

if len(df) != len(states):
    min_len = min(len(df), len(states))
    df = df.iloc[-min_len:].reset_index(drop=True)
    states = states[-min_len:]

df['State'] = states
df['date'] = df['date'].astype(str).str[:10]

print("2. 正在執行 EDA 策略條件篩選...")
# ==========================================
# 2. 策略條件篩選 (EDA Filters)
# ==========================================
# 條件 1: 當日收盤價 < 60日成本價的 1.10 倍 (乖離 10% 以內)
# 注意：排除 cost_60d 為 0 的無效數據，避免誤判
cond_cost = (df['cost_60d'] > 0) & (df['close'] < df['cost_60d'] * 1.10)

# 條件 2: 過去 60 天該券商累計買超金額 > 10 億 (1,000,000,000)
cond_amt = df['net_buy_amt_60d'] > 1000000000

# 套用篩選
eda_df = df[cond_cost & cond_amt].copy()

print(f"-> 篩選前總筆數: {len(df):,}")
print(f"-> 篩選後符合條件筆數: {len(eda_df):,}\n")

if len(eda_df) == 0:
    print("沒有任何資料符合此策略條件，請檢查門檻設定（例如 10 億是否過高）。")
    sys.exit()

print("3. 正在計算符合條件樣本的「未來兩週綜合報酬」...")
# ==========================================
# 3. 針對篩選後的樣本計算未來 2 週報酬
# ==========================================
unique_stocks = eda_df['stock_id'].unique()
return_records = []
forward_indexer = FixedForwardWindowIndexer(window_size=10)

for stock_id in unique_stocks:
    file_path = f"./data/stocks/{stock_id}_2021-06-30_to_2026-02-11.parquet"
    if os.path.exists(file_path):
        kdf = pd.read_parquet(file_path)
        kdf['date'] = kdf['date'].astype(str).str[:10]
        kdf = kdf.sort_values('date')
        
        # 計算未來兩週的高、低、均價
        kdf['future_2w_high'] = kdf['max'].shift(-1).rolling(window=forward_indexer, min_periods=1).max()
        kdf['future_2w_low']  = kdf['min'].shift(-1).rolling(window=forward_indexer, min_periods=1).min()
        kdf['future_2w_mean'] = kdf['close'].shift(-1).rolling(window=forward_indexer, min_periods=1).mean()
        
        # 計算相對於 T 日的報酬
        kdf['high_ret'] = kdf['future_2w_high'] / kdf['close'] - 1
        kdf['low_ret']  = kdf['future_2w_low'] / kdf['close'] - 1
        kdf['mean_ret'] = kdf['future_2w_mean'] / kdf['close'] - 1
        
        temp_df = kdf[['date', 'high_ret', 'low_ret', 'mean_ret']].copy()
        temp_df['stock_id'] = str(stock_id)
        return_records.append(temp_df)

all_returns = pd.concat(return_records, ignore_index=True)
eda_df['stock_id'] = eda_df['stock_id'].astype(str)
eda_df = pd.merge(eda_df, all_returns, on=['stock_id', 'date'], how='left')

# 移除無法計算的尾端資料
eda_df = eda_df.dropna(subset=['high_ret', 'low_ret', 'mean_ret'])

print("4. 產出 HMM 狀態策略勝率分析表...\n")
# ==========================================
# 4. 統計分析與報表輸出
# ==========================================
surge_threshold = 0.10  # 高點 >= 10%
drop_threshold = -0.10  # 低點 <= -10%

grouped = eda_df.groupby('State')

stats = pd.DataFrame()
stats['符合條件樣本數'] = grouped.size()

stats['正報酬機率(均值>0)'] = grouped['mean_ret'].apply(lambda x: (x > 0).mean())
stats['大漲機率(高點>10%)'] = grouped['high_ret'].apply(lambda x: (x >= surge_threshold).mean())
stats['大跌機率(低點<-10%)'] = grouped['low_ret'].apply(lambda x: (x <= drop_threshold).mean())

stats['平均高點報酬'] = grouped['high_ret'].mean()
stats['平均低點報酬'] = grouped['low_ret'].mean()

stats['高點中位數'] = grouped['high_ret'].median()
stats['低點中位數'] = grouped['low_ret'].median()

# 格式化為百分比
format_cols = [col for col in stats.columns if '樣本' not in col]
for col in format_cols:
    stats[col] = (stats[col] * 100).round(2).astype(str) + '%'

print("="*100)
print("      EDA 策略測試: 股價 < 60日成本1.1倍 且 60日買超 > 10億 (未來兩週表現)")
print("="*100)
print(stats.to_markdown())

# 將結果存檔，方便後續檢視
# stats.to_csv('./outputs/eda_strategy_results.csv')
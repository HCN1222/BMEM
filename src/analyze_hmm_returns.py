import pandas as pd
import numpy as np
import os
import sys

print("1. 正在載入主資料與 HMM 狀態...")
# ==========================================
# 1. 載入資料與對齊 (與 visualize_output.py 相同)
# ==========================================
try:
    df = pd.read_parquet('./data/preprocessed_data/final_vectors_train.parquet')
    hmm_params = np.load('./outputs/result_20260327_123818/states_10/trained_hmm_params.npz')
    states = hmm_params['decoded_states']
except Exception as e:
    print(f"Error loading files: {e}")
    sys.exit()

if len(df) != len(states):
    min_len = min(len(df), len(states))
    df = df.iloc[-min_len:].reset_index(drop=True)
    states = states[-min_len:]

df['State'] = states
df['date'] = df['date'].astype(str).str[:10] # 確保日期格式一致

print("2. 正在從個別股票檔案提取收盤價並計算未來報酬(T+1)...")
# ==========================================
# 2. 批次計算未來報酬 (T+1)
# ==========================================
unique_stocks = df['stock_id'].unique()
return_records = []

# 遍歷所有出現過的 stock_id
for stock_id in unique_stocks:
    file_path = f"./data/stocks/{stock_id}_2021-06-30_to_2026-02-11.parquet"
    if os.path.exists(file_path):
        kdf = pd.read_parquet(file_path)
        kdf['date'] = kdf['date'].astype(str).str[:10]
        
        # 確保依照日期排序
        kdf = kdf.sort_values('date')
        
        # 計算未來一期報酬率: (隔日收盤 - 今日收盤) / 今日收盤
        kdf['future_return'] = kdf['close'].shift(-1) / kdf['close'] - 1
        
        # 只保留需要的欄位
        temp_df = kdf[['date', 'future_return']].copy()
        temp_df['stock_id'] = str(stock_id)
        return_records.append(temp_df)

# 將所有股票的報酬率串接成一個大型 DataFrame
all_returns = pd.concat(return_records, ignore_index=True)
df['stock_id'] = df['stock_id'].astype(str)

# 將未來報酬率透過 stock_id 與 date 映射回主資料表
df = pd.merge(df, all_returns, on=['stock_id', 'date'], how='left')

# 移除無法計算未來報酬的資料 (例如各檔股票在資料集中的最後一天，缺乏 T+1 的收盤價)
df = df.dropna(subset=['future_return'])

print("3. 開始統計各狀態的報酬特徵...\n")
# ==========================================
# 3. 統計每個 State 的指標
# ==========================================
surge_threshold = 0.05  # 大漲定義: >= 2%
drop_threshold = -0.05  # 大跌定義: <= -2%

grouped = df.groupby('State')['future_return']

stats = pd.DataFrame()
stats['樣本數'] = grouped.count()
stats['平均未來報酬'] = grouped.mean()
stats['中位數未來報酬'] = grouped.median()
stats['標準差'] = grouped.std()

# 比例指標
stats['正報酬比例'] = grouped.apply(lambda x: (x > 0).mean())
stats['大漲比例(>5%)'] = grouped.apply(lambda x: (x >= surge_threshold).mean())
stats['大跌比例(<-5%)'] = grouped.apply(lambda x: (x <= drop_threshold).mean())

# 分位數指標
quantiles = [0.10, 0.25, 0.50, 0.75, 0.90]
for q in quantiles:
    col_name = f'{int(q*100)}%_分位數'
    stats[col_name] = grouped.quantile(q)

# ==========================================
# 4. 輸出結果與格式化
# ==========================================
# 為了方便閱讀，將浮點數轉為百分比 (%) 格式
format_cols = ['平均未來報酬', '中位數未來報酬', '標準差', '正報酬比例', 
               '大漲比例(>5%)', '大跌比例(<-5%)'] + [f'{int(q*100)}%_分位數' for q in quantiles]

# 備份一份原始數值版的 DataFrame 方便後續可能要畫圖或匯出
stats_raw = stats.copy()

for col in format_cols:
    stats[col] = (stats[col] * 100).round(2).astype(str) + '%'

pd.set_option('display.unicode.east_asian_width', True)
print("="*60)
print("                    HMM 狀態未來報酬 (T+1) 統計表")
print("="*60)
print(stats.to_string())

# 如果你想把結果存成 CSV：
# stats_raw.to_csv('./outputs/hmm_states_return_stats.csv')
# print("\n✅ 統計結果已儲存至 ./outputs/hmm_states_return_stats.csv")
import pandas as pd
import numpy as np
import glob
import os
import sys
from hmmlearn import hmm
from pandas.api.indexers import FixedForwardWindowIndexer
from tqdm import tqdm

# 開啟 Pandas 對齊全形中文字的支援
pd.set_option('display.unicode.east_asian_width', True)

print("1. 正在載入資料與重建 HMM 模型...")
# ==========================================
# 1. 載入原始特徵與 HMM 模型
# ==========================================
train_parquet_path = './data/preprocessed_data/exp3/final_vectors_train.parquet'
eval_parquet_path = './data/preprocessed_data/exp3/final_vectors_eval.parquet'
hmm_params_path = './outputs/exp3/states_10/trained_hmm_params.npz'

try:
    df_train = pd.read_parquet(train_parquet_path)
    df_eval = pd.read_parquet(eval_parquet_path)
    hmm_params = np.load(hmm_params_path)
except Exception as e:
    print(f"檔案讀取失敗: {e}")
    sys.exit()

df_train['date'] = df_train['date'].astype(str).str[:10]
df_eval['date'] = df_eval['date'].astype(str).str[:10]

feature_cols = ['z_t', 'c_t', 'a_t', 's_t', 'm_t']

# 解析模型參數並重建模型 (為了 Eval 的滾動機率)
startprob = hmm_params['startprob']
transmat = hmm_params['transmat']
means = hmm_params['means']
covars = hmm_params['covars']
n_components = means.shape[0]

if len(covars.shape) == 3:
    covar_type = "full"
elif len(covars.shape) == 2:
    covar_type = "diag"
else:
    covar_type = "spherical"

# 修復浮點數誤差，確保共變異數矩陣對稱且正定 (避免 ValueError)
if covar_type == "full":
    for i in range(n_components):
        covars[i] = (covars[i] + covars[i].T) / 2
        np.fill_diagonal(covars[i], covars[i].diagonal() + 1e-5)
elif covar_type == "diag":
    covars = np.maximum(covars, 1e-5)

model = hmm.GaussianHMM(n_components=n_components, covariance_type=covar_type)
model.startprob_ = startprob
model.transmat_ = transmat
model.means_ = means
model.covars_ = covars
model.n_features = len(feature_cols)

print("2. 正在提取 Train 的 HMM 狀態機率...")
# ==========================================
# 2. 處理 Train Set 機率 (直接讀取)
# ==========================================
posterior_probs_train = hmm_params['posterior_probabilities']
if len(df_train) != len(posterior_probs_train):
    min_len = min(len(df_train), len(posterior_probs_train))
    df_train = df_train.iloc[-min_len:].reset_index(drop=True)
    posterior_probs_train = posterior_probs_train[-min_len:]

prob_cols = [f'prob_S{i}' for i in range(n_components)]
prob_df_train = pd.DataFrame(posterior_probs_train, columns=prob_cols)
df_train = pd.concat([df_train, prob_df_train], axis=1)

print("3. 正在計算 Eval 的無未來函數滾動狀態機率 (Rolling Proba)...")
# ==========================================
# 3. 處理 Eval Set 機率 (Rolling 防止未來函數)
# ==========================================
def rolling_predict_proba(sequence_features, hmm_model, window=120):
    seq_len = len(sequence_features)
    probs = np.zeros((seq_len, n_components))
    for t in range(seq_len):
        start_idx = max(0, t - window + 1)
        X_window = sequence_features[start_idx : t + 1]
        window_probs = hmm_model.predict_proba(X_window)
        probs[t] = window_probs[-1]
    return probs

eval_probs_list = []
grouped_eval = df_eval.groupby('sequence_id')
for seq_id, group in tqdm(grouped_eval, desc="Eval 滾動機率", total=len(grouped_eval)):
    p = rolling_predict_proba(group[feature_cols].values, model, window=120)
    eval_probs_list.append(pd.DataFrame(p, index=group.index, columns=prob_cols))

prob_df_eval = pd.concat(eval_probs_list)
df_eval = pd.concat([df_eval, prob_df_eval], axis=1)

print("4. 正在計算衍生特徵與實戰標籤 (做多與做空 Target Y)...")
# ==========================================
# 4. 合併資料以批次計算 Target Y (乖離率已在 preprocess 算完)
# ==========================================
df_train['is_eval'] = False
df_eval['is_eval'] = True
df_all = pd.concat([df_train, df_eval], ignore_index=True)

unique_stocks = df_all['stock_id'].unique()
return_records = []
forward_indexer = FixedForwardWindowIndexer(window_size=10)

for stock_id in tqdm(unique_stocks, desc="計算未來價格極值"):
    _matches = glob.glob(f"./data/stocks/{stock_id}_2021-06-30_to_*.parquet")
    file_path = _matches[0] if _matches else ""
    if os.path.exists(file_path):
        kdf = pd.read_parquet(file_path)
        kdf['date'] = kdf['date'].astype(str).str[:10]
        kdf = kdf.sort_values('date')
        
        # 處理 0050 於 6/18 之後的價格調整 (乘以 4)
        if str(stock_id) == '0050':
            split_mask = kdf['date'] >= '2025-06-18' 
            kdf.loc[split_mask, ['close', 'max', 'min']] *= 4

        kdf['future_2w_high'] = kdf['max'].shift(-1).rolling(window=forward_indexer, min_periods=1).max()
        kdf['future_2w_low']  = kdf['min'].shift(-1).rolling(window=forward_indexer, min_periods=1).min()
        
        kdf['high_ret'] = kdf['future_2w_high'] / kdf['close'] - 1
        kdf['low_ret']  = kdf['future_2w_low'] / kdf['close'] - 1
        
        temp_df = kdf[['date', 'high_ret', 'low_ret']].copy()
        temp_df['stock_id'] = str(stock_id)
        return_records.append(temp_df)

all_returns = pd.concat(return_records, ignore_index=True)
df_all['stock_id'] = df_all['stock_id'].astype(str)
df_all = pd.merge(df_all, all_returns, on=['stock_id', 'date'], how='left')

# 移除無法計算標籤的尾端資料
df_all = df_all.dropna(subset=['high_ret', 'low_ret'])

# 目標 1 (做多策略)：獲利達到 10% 且 過程中未觸發 10% 停損 (即跌幅 > -10%)
df_all['target_y_long'] = ((df_all['high_ret'] >= 0.10) & (df_all['low_ret'] > -0.10)).astype(int)

# 目標 2 (做空策略)：最大跌幅達到 10% (low_ret <= -0.10) 且 過程中未觸發 10% 停損 (即最大漲幅 < 10%)
df_all['target_y_short'] = ((df_all['low_ret'] <= -0.10) & (df_all['high_ret'] < 0.10)).astype(int)

print("5. 正在切分並儲存 XGBoost 專用資料集 (分別輸出 Long 與 Short)...")
# ==========================================
# 5. 切分 Train/Eval 並存檔
# ==========================================
# 基礎欄位 (不包含標籤)
base_cols = [ 
    'date', 'stock_id', 'securities_trader_id', 'sequence_id', 
    'net_buy_amt_60d', 'bias_60d'
] + feature_cols + prob_cols

final_train = df_all[~df_all['is_eval']].copy()
final_eval = df_all[df_all['is_eval']].copy()

# 定義輸出路徑
train_long_out_path = './data/preprocessed_data/xgb_dataset_long_train.parquet'
eval_long_out_path = './data/preprocessed_data/xgb_dataset_long_eval.parquet'

train_short_out_path = './data/preprocessed_data/xgb_dataset_short_train.parquet'
eval_short_out_path = './data/preprocessed_data/xgb_dataset_short_eval.parquet'

# === 處理做多資料集 ===
# 將 target_y_long 重新命名為 target_y，方便 XGBoost 直接訓練
df_long_train = final_train[base_cols + ['target_y_long']].rename(columns={'target_y_long': 'target_y'})
df_long_eval = final_eval[base_cols + ['target_y_long']].rename(columns={'target_y_long': 'target_y'})

df_long_train.to_parquet(train_long_out_path, index=False)
df_long_eval.to_parquet(eval_long_out_path, index=False)

# === 處理做空資料集 ===
# 將 target_y_short 重新命名為 target_y
df_short_train = final_train[base_cols + ['target_y_short']].rename(columns={'target_y_short': 'target_y'})
df_short_eval = final_eval[base_cols + ['target_y_short']].rename(columns={'target_y_short': 'target_y'})

df_short_train.to_parquet(train_short_out_path, index=False)
df_short_eval.to_parquet(eval_short_out_path, index=False)

# 輸出結果統計
print(f"✅ XGBoost [做多] 訓練資料集已儲存: {train_long_out_path}")
print(f"✅ XGBoost [做多] 驗證資料集已儲存: {eval_long_out_path}")
print(f"   [做多] 全局基礎勝率 (Base Rate) Train: {df_long_train['target_y'].mean()*100:.2f}% | Eval: {df_long_eval['target_y'].mean()*100:.2f}%\n")

print(f"✅ XGBoost [做空] 訓練資料集已儲存: {train_short_out_path}")
print(f"✅ XGBoost [做空] 驗證資料集已儲存: {eval_short_out_path}")
print(f"   [做空] 全局基礎勝率 (Base Rate) Train: {df_short_train['target_y'].mean()*100:.2f}% | Eval: {df_short_eval['target_y'].mean()*100:.2f}%")
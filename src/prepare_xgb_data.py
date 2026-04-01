import pandas as pd
import numpy as np
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
        # 計算視窗內的所有機率，但只取最後一天 (T日) 的機率
        window_probs = hmm_model.predict_proba(X_window)
        probs[t] = window_probs[-1]
    return probs

# 逐一計算 Eval sequences
eval_probs_list = []
grouped_eval = df_eval.groupby('sequence_id')
for seq_id, group in tqdm(grouped_eval, desc="Eval 滾動機率", total=len(grouped_eval)):
    p = rolling_predict_proba(group[feature_cols].values, model, window=120)
    eval_probs_list.append(pd.DataFrame(p, index=group.index, columns=prob_cols))

prob_df_eval = pd.concat(eval_probs_list)
df_eval = pd.concat([df_eval, prob_df_eval], axis=1)

print("4. 正在計算衍生特徵與實戰標籤 (Target Y)...")
# ==========================================
# 4. 合併資料以批次計算 Target Y 與乖離率
# ==========================================
df_train['is_eval'] = False
df_eval['is_eval'] = True
df_all = pd.concat([df_train, df_eval], ignore_index=True)

# 計算 60 日成本乖離率 (Bias)
df_all['bias_60d'] = np.where(df_all['cost_60d'] > 0, df_all['close'] / df_all['cost_60d'] - 1, np.nan)

unique_stocks = df_all['stock_id'].unique()
return_records = []
forward_indexer = FixedForwardWindowIndexer(window_size=10)

for stock_id in tqdm(unique_stocks, desc="計算未來價格極值"):
    file_path = f"./data/stocks/{stock_id}_2021-06-30_to_2026-02-11.parquet"
    if os.path.exists(file_path):
        kdf = pd.read_parquet(file_path)
        kdf['date'] = kdf['date'].astype(str).str[:10]
        kdf = kdf.sort_values('date')
        
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

# 目標：獲利達到 10% 且 過程中未觸發 10% 停損
df_all['target_y'] = ((df_all['high_ret'] >= 0.10) & (df_all['low_ret'] > -0.10)).astype(int)

print("5. 正在切分並儲存 XGBoost 專用資料集...")
# ==========================================
# 5. 切分 Train/Eval 並存檔
# ==========================================
out_cols = ['date', 'stock_id', 'securities_trader_id', 'sequence_id', 'close', 
            'net_buy', 'net_buy_amt_60d', 'cost_20d', 'cost_60d', 'bias_60d', 
            'high_ret', 'low_ret', 'target_y'] + feature_cols + prob_cols

final_train = df_all[~df_all['is_eval']].copy()
final_eval = df_all[df_all['is_eval']].copy()

train_out_path = './data/preprocessed_data/xgb_dataset_train.parquet'
eval_out_path = './data/preprocessed_data/xgb_dataset_eval.parquet'

final_train[out_cols].to_parquet(train_out_path, index=False)
final_eval[out_cols].to_parquet(eval_out_path, index=False)

print(f"✅ XGBoost 訓練資料集已儲存: {train_out_path} (總筆數: {len(final_train):,})")
print(f"✅ XGBoost 驗證資料集已儲存: {eval_out_path} (總筆數: {len(final_eval):,})")
print(f"全局基礎勝率 (Base Rate) Train: {final_train['target_y'].mean()*100:.2f}% | Eval: {final_eval['target_y'].mean()*100:.2f}%")
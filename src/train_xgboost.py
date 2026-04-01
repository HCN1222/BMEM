import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report, precision_score, recall_score, roc_auc_score
import matplotlib.pyplot as plt
import os
import sys

# 解決 matplotlib 中文顯示問題 (依據你的作業系統可能需要調整字型)
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] # Windows 預設微軟正黑體
plt.rcParams['axes.unicode_minus'] = False

print("1. 載入 XGBoost 訓練與驗證資料...")
# ==========================================
# 1. 載入資料
# ==========================================
train_path = './data/preprocessed_data/xgb_dataset_train.parquet'
eval_path = './data/preprocessed_data/xgb_dataset_eval.parquet'

try:
    df_train = pd.read_parquet(train_path)
    df_eval = pd.read_parquet(eval_path)
except Exception as e:
    print(f"檔案讀取失敗: {e}")
    sys.exit()

# 定義特徵 (確保與準備資料時一致)
prob_cols = [f'prob_S{i}' for i in range(10)]
feature_cols = ['z_t', 'c_t', 'a_t', 's_t', 'm_t', 'bias_60d', 'net_buy_amt_60d'] + prob_cols

X_train = df_train[feature_cols]
y_train = df_train['target_y']

X_eval = df_eval[feature_cols]
y_eval = df_eval['target_y']

# 計算正負樣本比例，幫助 XGBoost 處理不平衡資料 (Imbalanced Data)
# 如果正樣本很少，scale_pos_weight 會放大正樣本的權重
positive_count = y_train.sum()
negative_count = len(y_train) - positive_count
scale_weight = negative_count / positive_count if positive_count > 0 else 1.0

print(f"-> Train 總筆數: {len(df_train):,} | 正樣本: {positive_count:,} | 權重比 (scale_pos_weight): {scale_weight:.2f}")
print(f"-> Eval  總筆數: {len(df_eval):,} | 正樣本: {y_eval.sum():,}")

print("\n2. 初始化並訓練 XGBoost 模型...")
# ==========================================
# 2. 模型設定與訓練
# ==========================================
# 設定 XGBoost 超參數
clf = xgb.XGBClassifier(
    n_estimators=1000,          # 最大樹的數量
    max_depth=5,                # 樹的深度 (設淺一點防止 Overfitting)
    learning_rate=0.05,         # 學習率
    subsample=0.8,              # 每次建樹隨機抽取 80% 樣本 (增加泛化能力)
    colsample_bytree=0.8,       # 每次建樹隨機抽取 80% 特徵
    scale_pos_weight=scale_weight, # 處理樣本不平衡
    eval_metric='auc',          # 評估指標使用 AUC
    early_stopping_rounds=50,   # 如果 eval set 的 AUC 連續 50 輪沒進步就提早停止
    random_state=42,
    n_jobs=-1                   # 使用所有 CPU 核心
)

# 訓練模型，並同時監控 Train 和 Eval 的表現
clf.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_eval, y_eval)],
    verbose=50  # 每 50 輪印一次進度
)

print(f"\n✅ 訓練完成！最佳迭代次數 (Best Iteration): {clf.best_iteration}")

print("\n3. 評估模型實戰勝率 (設定不同信心門檻)...")
# ==========================================
# 3. 預測與門檻分析 (Threshold Tuning)
# ==========================================
# 取得模型對 Eval set 預測為 1 (會大漲) 的「機率值」
y_pred_prob = clf.predict_proba(X_eval)[:, 1]

# 計算 Baseline ROC-AUC
auc = roc_auc_score(y_eval, y_pred_prob)
print(f"-> Eval 區間整體 AUC 表現: {auc:.4f}\n")

print("="*80)
print(f"{'信心門檻':<10} | {'觸發交易次數':<15} | {'實戰勝率 (Precision)':<20} | {'捕捉率 (Recall)':<15}")
print("-"*80)

# 測試不同的預測機率門檻 (模型有多肯定才要買)
thresholds = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90]

for t in thresholds:
    # 當預測機率大於門檻 t 時，才判定為買進訊號 (1)
    y_pred_custom = (y_pred_prob >= t).astype(int)
    
    trades_count = y_pred_custom.sum()
    if trades_count > 0:
        precision = precision_score(y_eval, y_pred_custom, zero_division=0)
        recall = recall_score(y_eval, y_pred_custom, zero_division=0)
        
        print(f" >= {t:.2f}    | {trades_count:<13,} 次 | {precision*100:>15.2f}%    | {recall*100:>12.2f}%")
    else:
        print(f" >= {t:.2f}    | {0:<13,} 次 | {'無交易':>16}    | {'0.00%':>13}")
print("="*80)
print("* 註解：\n實戰勝率 (Precision) = 發出買進訊號後，真的達到 10% 獲利且沒被停損的機率。\n捕捉率 (Recall) = 市場上所有真實大漲的機會中，模型抓到了多少比例。")

print("\n4. 繪製並儲存特徵重要性 (Feature Importance)...")
# ==========================================
# 4. 特徵重要性分析
# ==========================================
# 將特徵重要性提取出來並排序
importance_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': clf.feature_importances_
}).sort_values(by='Importance', ascending=True)

# 畫圖
plt.figure(figsize=(10, 8))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue', edgecolor='black')
plt.title('XGBoost 量化策略特徵重要性 (Feature Importance)', fontsize=16)
plt.xlabel('重要性分數 (Gain)', fontsize=12)
plt.ylabel('特徵名稱', fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()

# 存檔與顯示
os.makedirs('./outputs/models', exist_ok=True)
plt.savefig('./outputs/models/xgboost_feature_importance.png', dpi=300)
print("✅ 特徵重要性圖表已儲存至: ./outputs/models/xgboost_feature_importance.png")

# 儲存 XGBoost 模型
model_path = './outputs/models/xgb_trading_model.json'
clf.save_model(model_path)
print(f"✅ 模型已儲存至: {model_path}")

plt.show()
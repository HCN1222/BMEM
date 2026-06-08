import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score
import matplotlib.pyplot as plt
import os
import sys

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

# 從命令列參數決定 long 或 short
mode = sys.argv[1] if len(sys.argv) > 1 else 'short'
if mode not in ('long', 'short'):
    print(f"用法: python train_xgboost.py [long|short]")
    sys.exit(1)

print(f"1. 載入 XGBoost 資料集 [{mode}] (Train=2021-2023 / Val=2024 / Test=2025)...")
# ==========================================
# 1. 載入三份時間切分資料集
# ==========================================
train_path = f'./data/preprocessed_data/xgb_dataset_{mode}_train.parquet'
val_path   = f'./data/preprocessed_data/xgb_dataset_{mode}_val.parquet'
test_path  = f'./data/preprocessed_data/xgb_dataset_{mode}_test.parquet'

try:
    df_train = pd.read_parquet(train_path)
    df_val   = pd.read_parquet(val_path)
    df_test  = pd.read_parquet(test_path)
except Exception as e:
    print(f"檔案讀取失敗: {e}")
    sys.exit()

# 動態偵測 HMM state 數量
n_hmm_states = sum(1 for c in df_train.columns if c.startswith('prob_S'))
prob_cols = [f'prob_S{i}' for i in range(n_hmm_states)]
feature_cols = ['z_t', 'c_t', 'a_t', 's_t', 'm_t', 'bias_60d', 'net_buy_amt_60d'] + prob_cols

X_train = df_train[feature_cols]
y_train = df_train['target_y']

X_val = df_val[feature_cols]
y_val = df_val['target_y']

X_test = df_test[feature_cols]
y_test = df_test['target_y']

positive_count  = y_train.sum()
negative_count  = len(y_train) - positive_count
scale_weight    = negative_count / positive_count if positive_count > 0 else 1.0

print(f"-> HMM States: {n_hmm_states}")
print(f"-> Train (2021-2023): {len(df_train):,} 筆 | 正樣本: {positive_count:,} | 權重比: {scale_weight:.2f}")
print(f"-> Val   (2024)     : {len(df_val):,} 筆 | 正樣本: {y_val.sum():,}")
print(f"-> Test  (2025+)    : {len(df_test):,} 筆 | 正樣本: {y_test.sum():,} (封存回測集)")


print("\n2. 初始化並訓練 XGBoost 模型 (Val=2024 做 Early Stopping)...")
# ==========================================
# 2. 模型設定與訓練
# ==========================================
clf = xgb.XGBClassifier(
    n_estimators=1000,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_weight,
    eval_metric='auc',
    early_stopping_rounds=50,
    random_state=42,
    n_jobs=-1
)

clf.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    verbose=50
)

print(f"\n訓練完成！最佳迭代次數 (Best Iteration): {clf.best_iteration}")


print("\n3. 使用 Val(2024) 進行暴力窮舉尋找最佳門檻 (Threshold Tuning)...")
# ==========================================
# 3. 用 Val(2024) 做門檻調參
# ==========================================
y_val_pred_prob = clf.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, y_val_pred_prob)
print(f"-> Val(2024) 整體 AUC: {auc:.4f}\n")

thresholds = np.round(np.arange(0.60, 0.91, 0.01), 2)
results = []

for t in thresholds:
    y_pred_custom = (y_val_pred_prob >= t).astype(int)
    trades_count  = y_pred_custom.sum()

    if trades_count > 0:
        precision = precision_score(y_val, y_pred_custom, zero_division=0)
        recall    = recall_score(y_val, y_pred_custom, zero_division=0)
        f1        = f1_score(y_val, y_pred_custom, zero_division=0)
    else:
        precision, recall, f1 = 0.0, 0.0, 0.0

    results.append({"threshold": t, "trades_count": trades_count,
                    "precision": precision, "recall": recall, "f1": f1})

top_7_f1_results  = sorted(results, key=lambda x: x["f1"], reverse=True)[:7]
top_7_sorted_by_t = sorted(top_7_f1_results, key=lambda x: x["threshold"])

print("="*80)
print(f"【Top 7 最佳 F1-score 門檻】(Val=2024，按門檻由小到大排序)")
print(f"{'信心門檻':<10} | {'觸發交易次數':<13} | {'實戰勝率 (Precision)':<18} | {'捕捉率 (Recall)':<13} | {'f1_score'}")
print("-"*80)

for res in top_7_sorted_by_t:
    if res["trades_count"] > 0:
        print(f" >= {res['threshold']:.2f}    | {res['trades_count']:<13,} 次 | {res['precision']*100:>15.2f}%    | {res['recall']*100:>12.2f}%| {res['f1']:>8.4f}")
    else:
        print(f" >= {res['threshold']:.2f}    | {0:<13,} 次 | {'無交易':>16}    | {'0.00%':>13}| {0.0:>8.4f}")

best_result    = top_7_f1_results[0]
best_threshold = best_result["threshold"]
print(f"\n決定最佳信心門檻: {best_threshold:.2f} (Val F1: {best_result['f1']:.4f})")


print("\n4. 終極回測：Test(2025) 完全封存資料最終驗證...")
# ==========================================
# 4. 用 Test(2025) 做最終回測 (完全封存，從不參與訓練/調參)
# ==========================================
y_test_pred_prob   = clf.predict_proba(X_test)[:, 1]
y_test_pred_custom = (y_test_pred_prob >= best_threshold).astype(int)

test_trades_count = y_test_pred_custom.sum()
if test_trades_count > 0:
    test_precision = precision_score(y_test, y_test_pred_custom, zero_division=0)
    test_recall    = recall_score(y_test, y_test_pred_custom, zero_division=0)
    test_f1        = f1_score(y_test, y_test_pred_custom, zero_division=0)
else:
    test_precision, test_recall, test_f1 = 0.0, 0.0, 0.0

print("="*80)
print(f"【最終回測報告 [{mode}] - 套用門檻: {best_threshold:.2f}】(Test=2025，完全封存)")
print(f" 觸發交易次數 : {test_trades_count:,} 次")
print(f" 實戰勝率 (P) : {test_precision*100:.2f}%")
print(f" 捕捉率 (R)   : {test_recall*100:.2f}%")
print(f" F1 Score     : {test_f1:.4f}")
print("="*80)


print("\n5. 繪製並儲存特徵重要性 (Feature Importance)...")
# ==========================================
# 5. 特徵重要性分析與儲存
# ==========================================
importance_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': clf.feature_importances_
}).sort_values(by='Importance', ascending=True)

plt.figure(figsize=(10, 8))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue', edgecolor='black')
plt.title(f'XGBoost 特徵重要性 [{mode}]', fontsize=16)
plt.xlabel('重要性分數 (Gain)', fontsize=12)
plt.ylabel('特徵名稱', fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()

os.makedirs('./outputs/models', exist_ok=True)
importance_plot_path = f'./outputs/models/xgboost_feature_importance_{mode}.png'
plt.savefig(importance_plot_path, dpi=300)
print(f"特徵重要性圖表已儲存至: {importance_plot_path}")

model_path = f'./outputs/models/xgb_trading_model_{mode}.json'
clf.save_model(model_path)
print(f"模型已儲存至: {model_path}")

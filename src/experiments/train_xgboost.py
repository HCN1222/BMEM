import argparse
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report, precision_score, recall_score, roc_auc_score, f1_score
import matplotlib.pyplot as plt
import sys
from pathlib import Path

from src.utils.paths import add_broker_path_args, paths_from_args

# 解決 matplotlib 中文顯示問題
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False

# ARGPARSE Setup
parser = argparse.ArgumentParser(description='Train XGBoost Model')
add_broker_path_args(parser)
parser.add_argument('--side', choices=['long', 'short'], required=True, help='Trading model direction')
parser.add_argument('--train-path', '--train_path', type=Path, help='Override training data')
parser.add_argument('--val-path', '--val_path', type=Path, help='Override validation data')
parser.add_argument('--test-path', '--test_path', '--eval_path', type=Path, help='Override held-out test data')
parser.add_argument('--model-dir', type=Path, help='Override broker/side model output directory')
args = parser.parse_args()
paths = paths_from_args(args)

print("1. 載入 XGBoost 資料集 (Train / Val / Test)...")
# ==========================================
# 1. 載入資料與時間序列切分
# ==========================================
dataset_dir = paths.xgboost_data_dir(args.side)
train_path = args.train_path or dataset_dir / 'train.parquet'
val_path = args.val_path or dataset_dir / 'val.parquet'
test_path = args.test_path or dataset_dir / 'test.parquet'

try:
    df_train = pd.read_parquet(train_path)
    df_val = pd.read_parquet(val_path)
    df_test = pd.read_parquet(test_path)
except Exception as e:
    print(f"檔案讀取失敗: {e}")
    sys.exit()

# 定義特徵 (無標準化，依賴前處理)
prob_cols = [c for c in df_train.columns if c.startswith('prob_S')]
feature_cols = ['z_t', 'c_t', 'a_t', 's_t', 'm_t', 'bias_60d', 'net_buy_amt_60d'] + prob_cols

# 準備模型輸入的 X, y
X_train = df_train[feature_cols]
y_train = df_train['target_y']

X_val = df_val[feature_cols]
y_val = df_val['target_y']

# 最終封存的 Test Set (第5年)
X_test = df_test[feature_cols]
y_test = df_test['target_y']

# 重新計算 scale_pos_weight，只使用 df_train 計算，避免資訊洩漏
positive_count = y_train.sum()
negative_count = len(y_train) - positive_count
scale_weight = negative_count / positive_count if positive_count > 0 else 1.0

print(f"-> Train (前3年) 總筆數: {len(df_train):,} | 正樣本: {positive_count:,} | 權重比: {scale_weight:.2f}")
print(f"-> Val   (第4年) 總筆數: {len(df_val):,} | 正樣本: {y_val.sum():,}")
print(f"-> Test  (第5年) 總筆數: {len(df_test):,} | 正樣本: {y_test.sum():,} (終極封存測試集)")


print("\n2. 初始化並訓練 XGBoost 模型 (使用 Val 進行 Early Stopping)...")
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

# 使用 Train 訓練，並使用 Val 作為 early stopping 的依據
clf.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    verbose=50  
)

print(f"\n[OK] 訓練完成！最佳迭代次數 (Best Iteration): {clf.best_iteration}")


print("\n3. 使用 Validation Set 進行暴力窮舉尋找最佳門檻 (Threshold Tuning)...")
# ==========================================
# 3. 預測與門檻分析 (暴力窮舉區間 0.60 ~ 0.90，精度 0.01)
# ==========================================
# 取得模型對 Val set 預測的機率值
y_val_pred_prob = clf.predict_proba(X_val)[:, 1]

# 計算 Validation Baseline AUC
auc = roc_auc_score(y_val, y_val_pred_prob)
print(f"-> Validation 區間整體 AUC 表現: {auc:.4f}\n")

# 產生 0.60 到 0.90 的所有 threshold，精度 0.01
thresholds = np.round(np.arange(0.60, 0.91, 0.01), 2)
results = []

for t in thresholds:
    y_pred_custom = (y_val_pred_prob >= t).astype(int)
    trades_count = y_pred_custom.sum()

    if trades_count > 0:
        precision = precision_score(y_val, y_pred_custom, zero_division=0)
        recall = recall_score(y_val, y_pred_custom, zero_division=0)
        f1 = f1_score(y_val, y_pred_custom, zero_division=0)
    else:
        precision, recall, f1 = 0.0, 0.0, 0.0

    results.append({
        "threshold": t,
        "trades_count": trades_count,
        "precision": precision,
        "recall": recall,
        "f1": f1
    })

# 找出 F1-score 最大的前 7 筆資料
top_7_f1_results = sorted(results, key=lambda x: x["f1"], reverse=True)[:7]

# 將這 7 筆資料按照 threshold 由小到大重新排序
top_7_sorted_by_t = sorted(top_7_f1_results, key=lambda x: x["threshold"])

print("="*80)
print("【Top 7 最佳 F1-score 門檻紀錄】(已按門檻由小到大排序)")
print(f"{'信心門檻':<10} | {'觸發交易次數':<13} | {'實戰勝率 (Precision)':<18} | {'捕捉率 (Recall)':<13} | {'f1_score'}")
print("-" * 80)

for res in top_7_sorted_by_t:
    if res["trades_count"] > 0:
        print(f" >= {res['threshold']:.2f}    | {res['trades_count']:<13,} 次 | {res['precision']*100:>15.2f}%    | {res['recall']*100:>12.2f}%| {res['f1']:>8.4f}")
    else:
        print(f" >= {res['threshold']:.2f}    | {0:<13,} 次 | {'無交易':>16}    | {'0.00%':>13}| {0.0:>8.4f}")

# 取出 F1-score 真正的最大值，作為最終門檻
best_result = top_7_f1_results[0]
best_threshold = best_result["threshold"]

print(f"\n[OK] 暴力搜索結束，決定最佳信心門檻為: {best_threshold:.2f} (Validation F1: {best_result['f1']:.4f})")


print("\n4. 終極實戰體檢：使用 Test Set (Evaluation) 進行最終驗證...")
# ==========================================
# 4. 一次性解封測試 (最終成績單)
# ==========================================
# 這裡使用完全沒參與訓練與調參的 df_test
y_test_pred_prob = clf.predict_proba(X_test)[:, 1]
y_test_pred_custom = (y_test_pred_prob >= best_threshold).astype(int)

test_trades_count = y_test_pred_custom.sum()
if test_trades_count > 0:
    test_precision = precision_score(y_test, y_test_pred_custom, zero_division=0)
    test_recall = recall_score(y_test, y_test_pred_custom, zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred_custom, zero_division=0)
else:
    test_precision, test_recall, test_f1 = 0.0, 0.0, 0.0

print("="*80)
print(f"【最終測試報告 - 套用門檻: {best_threshold:.2f}】")
print(f" 觸發交易次數 : {test_trades_count:,} 次")
print(f" 實戰勝率 (P) : {test_precision*100:.2f}%")
print(f" 捕捉率 (R)   : {test_recall*100:.2f}%")
print(f" F1 Score     : {test_f1:.4f}")
print("="*80)
print("Note: 這份成績代表了模型未來遇到全新數據時，最客觀的期望表現。")


print("\n5. 繪製並儲存特徵重要性 (Feature Importance)...")
# ==========================================
# 5. 特徵重要性分析與儲存
# ==========================================
importance_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': clf.feature_importances_
}).sort_values(by='Importance', ascending=True)

model_dir = args.model_dir or paths.xgboost_model_dir(args.side)
model_dir.mkdir(parents=True, exist_ok=True)
feature_importance_plot_path = model_dir / 'xgboost_feature_importance.png'
feature_importance_csv_path = model_dir / 'xgboost_feature_importance.csv'

try:
    plt.figure(figsize=(10, 8))
    plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue', edgecolor='black')
    plt.title('XGBoost 量化策略特徵重要性 (Feature Importance)', fontsize=16)
    plt.xlabel('重要性分數 (Gain)', fontsize=12)
    plt.ylabel('特徵名稱', fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(feature_importance_plot_path, dpi=300)
    plt.show()
    print(f"[OK] 特徵重要性圖表已儲存至: {feature_importance_plot_path}")
except Exception as e:
    plt.close()
    importance_df.to_csv(feature_importance_csv_path, index=False, encoding='utf-8-sig')
    print(f"[WARN] 特徵重要性圖表繪製失敗 ({e})，已改存 CSV: {feature_importance_csv_path}")

model_path = model_dir / 'xgb_trading_model.json'
clf.save_model(str(model_path))
print(f"[OK] 模型已儲存至: {model_path}")

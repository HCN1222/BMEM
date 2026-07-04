## 1. 三大法人下載 — `src/experiments/download_institutional_investors.py`

### 資料來源
- **FinMind 資料集**：`TaiwanStockInstitutionalInvestorsBuySell`（台股三大法人買賣超）
- **API 端點**：`https://api.finmindtrade.com/api/v4/data`（REST，直接 `requests` 呼叫，非 SDK）
- **抓取方式**：一次呼叫抓「單日全市場所有股票」（約 4 秒、11 萬列），逐交易日迴圈；**非**逐檔查詢（否則會爆額度）

### 法人別 → 假券商對應
下載時依 FinMind 回傳的 `name` 欄位，拆成獨立的「假券商」資料夾，格式與真實券商 parquet 一致，可直接餵 `preprocess.py`：

| FinMind `name` | 假券商 ID | 說明 |
|---|---|---|
| `Foreign_Investor` | `FOREIGN` | 外資（多為**經紀**，幫客戶下單） |
| `Investment_Trust` | `TRUST` | 投信 |
| `Dealer_self` | `DEALER_SELF` | 自營商（自行買賣）＝**乾淨的自營訊號** |
| `Dealer_Hedging` | `DEALER_HEDGE` | 自營商（避險，多為權證對沖） |

> 研究對照核心：`DEALER_SELF`（自營）vs `FOREIGN`（經紀外商），驗證「自營 vs 經紀」訊號品質。

### ⚠️ 金額為近似值
FinMind 三大法人資料**只有股數（buy/sell），沒有成交金額**。腳本用 `net_buy_amount ≈ 股數 × 當日收盤價` 近似（`build_close_map`；可用 `--no-estimate-amounts` 關閉）。
- **HMM 的 5 個特徵**（z_t/c_t/a_t/s_t/m_t）只需股數÷成交量，**不受影響**。
- **XGBoost 的 2 個特徵**（`bias_60d`、`net_buy_amt_60d`）依賴此近似金額，屬**已知待驗證項**。
- 找不到收盤價的標的（多為權證，不在 `data/stocks`）金額補 0；這些標的會在 preprocess 的成交量門檻被濾掉。

### 用法
```bash
# 全歷史（首次）
python -m src.experiments.download_institutional_investors --start 2021-06-30
# 之後增量更新（自動接續到昨天）
python -m src.experiments.download_institutional_investors
```

---

## 2. 券商 pooling — `src/experiments/pool_broker_data.py`

把多家券商的原始 parquet 合併成單一假券商 `POOLED`，讓現有 pipeline 能用**一個 HMM** 對所有券商**一起訓練**（E2 實驗）。

### 「堆疊」而非「加總」
- 各券商的每一列**垂直堆疊**成一個檔，**保留原本 `securities_trader_id`**（不加總、不混流）。
- `preprocess.py` 以 `(stock_id, securities_trader_id)` 分組切序列，故「美林-台積電」與「摩根-台積電」仍是**各自獨立**的序列。
- 效果 = 單一 HMM 拿到全部券商序列（資料量↑、共變異數估計更穩），但每條序列仍是乾淨的單一券商行為。
- 以 `(date, stock_id, securities_trader_id)` 去重，順帶處理同一家多檔重疊。

### 用法
```bash
# 預設 pool data/brokers 下所有子資料夾
python -m src.experiments.pool_broker_data --broker-ids 1440 1470 1480 8440 1650 1360 8960 7030 1560
python -m src.experiments.preprocess --broker-id POOLED --disable_standardize
```


---

## 3. 關於「失敗日期」（failed_dates）

下載產生的 `*_failed_dates.json` 若列出某些日期，**多為台灣休市日**（春節、228、清明、勞動節、端午等），當天市場無交易資料、本就無資料可抓，**非錯誤**，不影響資料完整性（preprocess 以實際交易日建立骨架，休市日不會納入）。

# BMEM

**Believe Merrill Enrich the Meal**
Step1. 利用HMM分析外資(美林)`建倉/持倉/清倉` 狀態，縮小選股範圍
Step2. 如果順利再利用Gradient Boosted Decision Tree分析多個外資狀態與股價關係

## Data
資料使用Finmind API下載
[Finmind](https://finmindtrade.com/analysis/#/data/api)
籌碼面資料缺失

![籌碼面資料缺失](./img/image.png)

### 目前已下載的資料集

時間皆為 2021/02/30 ~ 2026/02/11

1.  `./data/brokers/卷商分點編號`: 此資料夾中的`.parquet`包含十個外資的分點進出資料(每檔股票淨買賣)
2.  `./data/stocks`: 這裡包含上述分點交易所觸及的所有股票，檔案開頭為股票代碼


## Setup
1. add your **API KEY** in `.env.example` and rename the file as `.env`

### Tools for Downloading data
指令可參考`./script`下面的`.ps1`
- `download_broker_activity.py`: 抓取卷商分點資料
- `download_stock_info.py`: 抓取股票資料

## HMM

### States

There are three states: 
- **Building Position (建倉)**
- **Liquidating Position (清倉)**
- **Holding Position (持倉)**

### Observation Vector

採用 Gaussian Emission

$x_t = [z_t, a_t, s_t, I_t]$

---

1. Normalized Net Buy (Flow Strength) $z_t$
$z_t = \frac{nb_t}{\text{Total Market Volume}_t}= \frac{buy_t - sell_t}{\text{Total Market Volume}_t}$




---

##### 3. Activity Level

$a_t = \frac{|buy_t| + |sell_t|}{\text{Total Market Volume}_t}$

Distinguishes:

* True inactivity
* High-turnover but flat net flow (possible internal matching)

---

##### 4. Directional Persistence

$s_t = \frac{1}{M} \sum_{i=0}^{M-1} \text{sign}(nb_{t-i})$

Captures sustained buying or selling behavior.

---

#### 🟡 Inventory-Based Factors (Strongly Recommended)

##### 5. Cumulative Net Buy (Pseudo Inventory)

$I_t = I_{t-1} + nb_t$

Approximates position accumulation.

---

##### 6. Inventory Change Rate

$\Delta I_t = nb_t$

Captures current position adjustment speed.

---

##### 7. Inventory Acceleration (Optional)

$\Delta^2 I_t = nb_t - nb_{t-1}$

Detects early reversal signals.

---

#### 🟠 Price-Related Factors (Optional Enhancement)

> Not required for state detection, but improves interpretability.

### 8. Log Return

$
r_t = \ln\left(\frac{P_t}{P_{t-1}}\right)
$

Used to contextualize flow behavior.

---

##### 9. Estimated Cost Basis (If Avg Buy Price Available)

$
Cost_t = \frac{\sum nb_i \cdot buy_avg_i}{\sum nb_i}
$

---

##### 10. Cost Distance

$
d_t = \frac{P_t - Cost_t}{Cost_t}
$

Indicates unrealized profit/loss condition.

---

### 3️⃣ Emission Model

Using **Gaussian Emission**:

$
x_t \mid S_t=k \sim \mathcal{N}(\mu_k, \Sigma_k)
$

Each state has:

* Mean vector ( \mu_k )
* Covariance matrix ( \Sigma_k )

---

### 4️⃣ Sticky Transition Structure

Transition probability:

$
P(S_t = k \mid S_{t-1})
$

Sticky bias increases:

$
P(S_t = k \mid S_{t-1}=k)
$

This prevents excessive state switching.

---

### 5️⃣ Minimum Viable Feature Set

For first implementation:

```text
z_t  (Normalized Net Buy)
a_t  (Activity Level)
s_t  (Directional Persistence)
I_t  (Cumulative Net Buy)
```

This is sufficient to separate:

* Building
* Liquidating
* Holding

---

### 6️⃣ Recommended Feature Normalization

Before training:

* Standardize all features (z-score normalization)
* Remove extreme outliers if necessary
* Consider rolling window smoothing (optional)

---

### 7️⃣ Final Model Structure Summary

```
Input:   Flow Features (z, a, s, I)
Model:   Sticky Gaussian HMM
Output:  State Sequence + State Probabilities
```

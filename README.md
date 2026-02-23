# BMEM

## Reference
- [Discrete HMM](https://youtu.be/JRsdt05pMoI?si=wjkP7rq9asWZSGZZ)

## Believe Merrill Enrich the Meal
分析外資的持倉狀態，縮小選股範圍
目標是波段交易

## API
[Finmind](https://finmindtrade.com/analysis/#/data/api)
籌碼面資料缺失

![籌碼面資料缺失](./img/image.png)

## Setup
1. add your **API KEY** in `.env.example` and rename the file as `.env`

### 設想
1. 利用HMM (sticky HMM)去分析外資目的(建倉/持倉)
2. 利用1.分析多加外資
接著再利用 Gradient Boosted Decision Tree

### Instructions

- `download_broker_activity.py`: 抓取卷商分點資料
```
python download_broker_activity.py --start 2025-01-01 --end 2025-12-31 --format parquet
```

### GPT HMM factor

#### 📌 Sticky HMM for Brokerage Branch Flow Regime Detection

##### 🎯 Objective

Infer the hidden trading state of a brokerage branch:

* **Building Position (建倉)**
* **Liquidating Position (清倉)**
* **Holding Position (持倉)**

Using a **Sticky Gaussian Hidden Markov Model (SHMM)**.

---

### 1️⃣ Hidden States

$
S_t \in {\text{Building},\ \text{Liquidating},\ \text{Holding}}
$

Properties:

* States are **persistent** (handled via Sticky HMM)
* State transitions are governed by transition matrix (A)

---

### 2️⃣ Observation Vector (Feature Set)

At each time step (t), define observation vector:

$
x_t = [z_t, a_t, s_t, I_t]
$

---

#### 🟢 Core Flow Factors (Required)

##### 1. Net Buy Volume

$
nb_t = buy_t - sell_t
$

Represents daily position change.

---

##### 2. Normalized Net Buy (Flow Strength)

$
z_t = \frac{nb_t}{\text{Total Market Volume}_t}
$

Measures relative influence.

---

##### 3. Activity Level

$
a_t = \frac{|buy_t| + |sell_t|}{\text{Total Market Volume}_t}
$

Distinguishes:

* True inactivity
* High-turnover but flat net flow (possible internal matching)

---

##### 4. Directional Persistence

$
s_t = \frac{1}{M} \sum_{i=0}^{M-1} \text{sign}(nb_{t-i})
$

Captures sustained buying or selling behavior.

---

#### 🟡 Inventory-Based Factors (Strongly Recommended)

##### 5. Cumulative Net Buy (Pseudo Inventory)

$
I_t = I_{t-1} + nb_t
$

Approximates position accumulation.

---

##### 6. Inventory Change Rate

$
\Delta I_t = nb_t
$

Captures current position adjustment speed.

---

##### 7. Inventory Acceleration (Optional)

$
\Delta^2 I_t = nb_t - nb_{t-1}
$

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

# Adaptive Fintech Fraud Detection System

An end-to-end machine learning pipeline for real-time fraud detection in financial transactions. The system builds behavioural user profiles, engineers deviation-based features, scores transaction risk, and applies a rule-based decision engine to flag, challenge, or block suspicious activity.

---

## Problem Statement

Financial fraud detection requires balancing two competing goals: catching fraudulent transactions while minimising false positives that disrupt legitimate users. Static rule-based systems fail to adapt to individual user behaviour. This system combines user profiling, ML-based risk scoring, and a tiered decision engine to handle both.

---

## Pipeline Architecture
```
Load Transactions (CSV)
↓
Build User Profiles (behavioural baselines per user)
↓
Feature Engineering (deviation-based features)
↓
Risk Scoring (weighted heuristic score)
↓
Model Training (Random Forest Classifier)
↓
Decision Engine (ALLOW / STEP_UP_AUTH / BLOCK)
```
---

## Components

### 1. User Profiling (`user_profiling.py`)
Builds a behavioural baseline for each user from historical transactions:
- Average and standard deviation of transaction amounts
- Most common transaction hour, location, and device

### 2. Feature Engineering (`feature_engineering.py`)
Computes deviation-based features by comparing each transaction against the user's profile:
- `amount_zscore` — how many standard deviations the amount deviates from the user's average
- `unusual_hour` — transaction at an atypical hour for this user
- `unusual_location` — transaction from an atypical location
- `unusual_device` — transaction from an atypical device
- `high_amount_flag` — amount exceeds mean + 3σ threshold

### 3. Risk Scoring (`risk_scoring.py`)
Computes a weighted composite risk score per transaction:
```python
risk_score = (
    2.0 * amount_zscore +
    1.5 * unusual_hour +
    2.0 * unusual_location +
    1.5 * unusual_device +
    1.0 * is_foreign
)
```

### 4. Model Training (`train_model.py`)
Trains a Random Forest Classifier on engineered features:
- 200 estimators, max depth 10
- `class_weight="balanced"` to handle class imbalance (~5% fraud cases)
- Stratified train/test split (75/25)

**Model Performance:**
| Metric | Score |
|--------|-------|
| Accuracy | 92% |
| Precision | 0.90 |
| Recall | 0.88 |
| F1-Score | 0.89 |

### 5. Decision Engine (`decision_engine.py`)
Converts risk scores into real-world actions:
| Risk Score | Decision |
|------------|----------|
| ≥ 6 | BLOCK |
| 3 – 5.9 | STEP_UP_AUTH |
| < 3 | ALLOW |

---

## Project Structure
```
Adaptive-fintech-fraud-detection/
├── data/
│   └── transactions.csv
├── notebooks/
├── src/
│   ├── load_data.py
│   ├── user_profiling.py
│   ├── feature_engineering.py
│   ├── risk_scoring.py
│   ├── train_model.py
│   ├── decision_engine.py
│   └── main.py
├── .gitignore
├── requirements.txt
└── README.md
```
---

## Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/prajna1606/Adaptive-fintech-fraud-detection
cd Adaptive-fintech-fraud-detection
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the pipeline
```bash
python src/main.py
```

---

## Key Design Decisions

**Why Random Forest?**
Random Forest handles class imbalance well with `class_weight="balanced"`, is robust to outliers, and provides feature importance out of the box — useful for explaining why a transaction was flagged.

**Why user profiling instead of global thresholds?**
A transaction of ₹50,000 may be normal for one user and highly suspicious for another. Per-user behavioural baselines make the system adaptive rather than static.

**Why a decision engine on top of ML?**
Raw model predictions are binary. A tiered decision engine (ALLOW / STEP_UP_AUTH / BLOCK) maps risk scores to real-world actions, enabling graduated responses instead of hard blocks.

---

## Requirements
- pandas
- scikit-learn
- numpy
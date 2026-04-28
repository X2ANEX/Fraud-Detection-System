# Fraud-Detection-System
# Fraud Detection System

Binary classifier to detect credit card fraud on a heavily imbalanced dataset (284K transactions, 0.17% fraud rate).

## Problem
Standard ML models fail on imbalanced data — predicting "all legitimate" gives 99.8% accuracy but catches zero frauds. This project addresses that with proper metrics, sampling techniques, and business-driven threshold tuning.

## Tech Stack
Python, Pandas, Scikit-learn, Imbalanced-learn, Matplotlib

## Approach
- Applied SMOTE to handle class imbalance in training data
- Trained Logistic Regression (baseline) and Random Forest (final model)
- Evaluated using PR-AUC instead of accuracy
- Tuned classification threshold using asymmetric business cost model (missed fraud = £500, false alarm = £10)

## Results
| Metric | Value |
|--------|-------|
| PR-AUC | 0.870 |
| Fraud detection rate | 89.8% |
| False alarm rate | 0.063% |
| Cost reduction vs default threshold | 38% |

## Dataset
[Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — 284,807 transactions, 492 frauds.

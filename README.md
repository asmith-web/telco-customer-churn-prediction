# Telco Customer Churn Prediction

Predicts which telecom customers will churn, comparing three models with
cross-validation and hyperparameter tuning.
Dataset: [IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
(7,043 customers, 19 features, about 26.5% churn).

## Key Findings
- Fiber optic customers have a higher churn risk than DSL customers.
- Longer tenure and two-year contracts lower churn risk.
- Raising recall with class weighting catches more churners but lowers precision,
  a tradeoff a retention team can choose based on the cost of a missed churner.

## Approach
1. Cleaning: fixed blank TotalCharges values, dropped customerID
2. EDA: churn by contract, internet service, payment method, and tenure
3. Preprocessing: StandardScaler + OneHotEncoder in a ColumnTransformer (inside a Pipeline to prevent leakage)
4. Models: Logistic Regression, Random Forest, XGBoost, compared with 5-fold stratified cross-validation
5. Tuning: GridSearchCV and RandomizedSearchCV, optimizing ROC-AUC
6. Imbalance: class weighting to improve churner recall
7. Interpretation: model coefficients and permutation importance

## Results (held-out test set, 1,409 customers)
| Model | ROC-AUC | Precision (churn) | Recall (churn) | F1 (churn) |
|---|---|---|---|---|
| Logistic Regression | 0.841 | 0.65 | 0.56 | 0.60 |
| Logistic Regression (balanced) | 0.841 | 0.50 | 0.78 | 0.61 |
| XGBoost (tuned) | 0.848 | 0.68 | 0.51 | 0.59 |

## Tech Stack
Python, pandas, NumPy, scikit-learn, XGBoost, Matplotlib, Seaborn

## Files
- `Telco_Churn_Prediction.ipynb`: full analysis and modeling

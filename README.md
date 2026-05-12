
# Parkinson's Terminology Project

Predicting severe and worsening Parkinson's disease trajectories from vocal acoustic measurements.

## Project Structure

```
Parkinsons_Terminlogy_project/
├── data/                  # Dataset (parkinsons_updrs.data)
├── notebooks/             # Jupyter notebooks
│   └── Parkinsons_Terminlogy_project.ipynb
├── src/                   # Modular Python source code
│   ├── data_cleaning.py   # Load data and feature engineering
│   ├── preprocessing.py   # Split, filtering, SMOTE, scaling
│   ├── eda.py             # Exploratory Data Analysis plots
│   ├── model.py           # Model training (SVM, LR, DT, RF, KNN)
│   └── evaluation.py      # Metrics and comparison table
├── requirements.txt       # Python dependencies
└── README.md
```

## Pipeline

1. **Data Cleaning** - Load UCI Parkinson's dataset, engineer baseline/delta features, create target
2. **Preprocessing** - GroupShuffleSplit (no patient leakage), multicollinearity filtering, SMOTE balancing, scaling + polynomial features
3. **EDA** - Class distribution, baseline severity analysis, voice degradation analysis, correlation heatmaps
4. **Model Training** - SVM (RBF), Logistic Regression, Decision Tree, Random Forest, K-Nearest Neighbors
5. **Evaluation** - Accuracy, F1-Score (macro), ROC-AUC, confusion matrices, optimal threshold tuning

## Usage

```python
from src.data_cleaning import load_data, engineer_features
from src.preprocessing import split_data, remove_highly_correlated, balance_with_smote, scale_and_add_polynomial
from src.model import get_models, train_and_evaluate
from src.evaluation import generate_comparison_table

df = load_data("data/parkinsons_updrs.data")
df, voice_cols = engineer_features(df)
X_train, X_test, y_train, y_test = split_data(df, voice_cols)
X_train, X_test = remove_highly_correlated(X_train, X_test)
X_train_bal, y_train_bal = balance_with_smote(X_train, y_train)
X_tr_s, X_te_s, X_tr_p, X_te_p = scale_and_add_polynomial(X_train_bal, X_test)
models = get_models()
trained = train_and_evaluate(models, X_tr_s, X_tr_p, y_train_bal, X_te_s, X_te_p, y_test)
generate_comparison_table(trained, y_test)
```

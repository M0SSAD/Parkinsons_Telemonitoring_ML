import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from imblearn.over_sampling import SMOTE


def split_data(df, voice_cols):
    drop_cols = (
        ['subject#', 'test_time', 'motor_UPDRS', 'total_UPDRS', 'delta_UPDRS', 'severe_and_worse']
        + voice_cols
    )
    X = df.drop(columns=drop_cols)
    y = df['severe_and_worse']
    groups = df['subject#']

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
    y_train, y_test = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()

    return X_train, X_test, y_train, y_test


def remove_highly_correlated(X_train, X_test, threshold=0.85):
    print(f"Original X_train shape: {X_train.shape}")

    corr_matrix = X_train.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]

    print(f"Identified {len(to_drop)} highly correlated features to drop.")

    X_train = X_train.drop(columns=to_drop)
    X_test = X_test.drop(columns=to_drop)

    print(f"New X_train shape after dropping: {X_train.shape}")
    return X_train, X_test


def balance_with_smote(X_train, y_train):
    print("--- Before SMOTE (Train Set) ---")
    print(f"Healthy (0): {sum(y_train == 0)} | Severe (1): {sum(y_train == 1)}")

    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

    print("--- After SMOTE (Train Set) ---")
    print(f"Healthy (0): {sum(y_train_balanced == 0)} | Severe (1): {sum(y_train_balanced == 1)}\n")

    return X_train_balanced, y_train_balanced


def scale_and_add_polynomial(X_train, X_test, degree=2):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    poly = PolynomialFeatures(degree=degree, interaction_only=True, include_bias=False)
    X_train_poly = poly.fit_transform(X_train_scaled)
    X_test_poly = poly.transform(X_test_scaled)

    return X_train_scaled, X_test_scaled, X_train_poly, X_test_poly

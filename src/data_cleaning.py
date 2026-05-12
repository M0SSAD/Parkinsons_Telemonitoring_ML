import pandas as pd
import numpy as np


def load_data(filepath):
    df = pd.read_csv(filepath)
    return df


def engineer_features(df):
    voice_cols = [col for col in df.columns if col not in
                  ['subject#', 'age', 'sex', 'test_time', 'motor_UPDRS', 'total_UPDRS']]

    baseline_idx = df.groupby('subject#')['test_time'].idxmin()
    baseline_df = df.loc[baseline_idx].copy()

    rename_dict = {col: f"baseline_{col}" for col in voice_cols + ['total_UPDRS']}
    baseline_df = baseline_df[['subject#'] + list(rename_dict.keys())].rename(columns=rename_dict)
    df = df.merge(baseline_df, on='subject#')

    df['delta_UPDRS'] = df['total_UPDRS'] - df['baseline_total_UPDRS']
    for col in voice_cols:
        df[f"delta_{col}"] = df[col] - df[f"baseline_{col}"]

    df['severe_and_worse'] = np.where(
        (df['total_UPDRS'] >= 35) & (df['delta_UPDRS'] > 2), 1, 0
    )

    print(f"Data engineered successfully. Shape: {df.shape}")
    return df, voice_cols

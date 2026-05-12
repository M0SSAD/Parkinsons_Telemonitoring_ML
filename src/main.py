import os
import matplotlib.pyplot as plt
from data_cleaning import load_data, engineer_features
from preprocessing import split_data, remove_highly_correlated, balance_with_smote, scale_and_add_polynomial
from eda import (setup_plotting, plot_class_distribution, plot_baseline_severity,
                 plot_voice_degradation, plot_correlation_heatmap, plot_filtered_correlation_heatmap)
from model import get_models, train_and_evaluate
from evaluation import plot_confusion_matrix, generate_comparison_table


def main():
    data_path = os.path.join(os.path.dirname(__file__), '../data/parkinsons_updrs.data')
    save_dir = None  # Set to a path string to save plots

    print("=== Loading Data ===")
    df = load_data(data_path)

    print("\n=== Engineering Features ===")
    df, voice_cols = engineer_features(df)

    print("\n=== Exploratory Data Analysis ===")
    custom_palette = setup_plotting()
    plot_class_distribution(df, custom_palette,
                            f"{save_dir}1_class_distribution.png" if save_dir else None)
    plot_baseline_severity(df, custom_palette,
                           f"{save_dir}2_baseline_severity.png" if save_dir else None)
    plot_voice_degradation(df, custom_palette,
                           f"{save_dir}3_voice_degradation.png" if save_dir else None)
    plot_correlation_heatmap(df,
                             f"{save_dir}4_full_correlation_heatmap.png" if save_dir else None)

    print("\n=== Splitting Data ===")
    X_train, X_test, y_train, y_test = split_data(df, voice_cols)

    print("\n=== Removing Highly Correlated Features ===")
    X_train, X_test = remove_highly_correlated(X_train, X_test)
    plot_filtered_correlation_heatmap(X_train,
                                      f"{save_dir}filtered_correlation_heatmap.png" if save_dir else None)

    print("\n=== Balancing with SMOTE ===")
    X_train_balanced, y_train_balanced = balance_with_smote(X_train, y_train)

    print("\n=== Scaling and Polynomial Features ===")
    X_train_scaled, X_test_scaled, X_train_poly, X_test_poly = scale_and_add_polynomial(
        X_train_balanced, X_test
    )

    print("\n=== Training Models ===")
    models = get_models()
    trained_models = train_and_evaluate(
        models, X_train_scaled, X_train_poly, y_train_balanced,
        X_test_scaled, X_test_poly, y_test
    )

    print("\n=== Plotting Confusion Matrices ===")
    for name, info in trained_models.items():
        plot_confusion_matrix(y_test, info['y_pred'], name)

    print("\n=== Generating Comparison Table ===")
    generate_comparison_table(trained_models, y_test)

    print("\n=== Done! ===")


if __name__ == "__main__":
    main()

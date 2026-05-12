import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def compute_metrics(y_test, y_pred, y_probs):
    acc = accuracy_score(y_test, y_pred) * 100
    f1 = f1_score(y_test, y_pred, average='macro') * 100
    auc = roc_auc_score(y_test, y_probs)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    return {
        "Accuracy (%)": round(acc, 2),
        "F1-Score (%)": round(f1, 2),
        "ROC-AUC": round(auc, 4),
        "TN": tn, "FP": fp, "FN": fn, "TP": tp
    }


def plot_confusion_matrix(y_test, y_pred, model_name):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No/Mild', 'Severe/Worse'])
    disp.plot(cmap='Blues')
    plt.title(f'Confusion Matrix: {model_name}')
    plt.show()


def generate_comparison_table(trained_models, y_test):
    comparison_results = []

    print("--- Generating Model Comparison Table ---\n")

    for name, info in trained_models.items():
        y_pred = info['y_pred']
        y_probs = info['y_probs']
        metrics = compute_metrics(y_test, y_pred, y_probs)
        metrics["Model"] = name
        comparison_results.append(metrics)

    comparison_df = pd.DataFrame(comparison_results)
    comparison_df = comparison_df.sort_values(by="F1-Score (%)", ascending=False)
    print(comparison_df.to_string(index=False))

    comparison_df.to_csv("model_comparison_results.csv", index=False)
    print("\nResults saved to model_comparison_results.csv")

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def setup_plotting():
    sns.set_theme(style="whitegrid", context="talk")
    return {0: "#4C72B0", 1: "#C44E52"}


def plot_class_distribution(df, custom_palette, save_path=None):
    plt.figure(figsize=(8, 6))
    ax = sns.countplot(data=df, x='severe_and_worse', hue='severe_and_worse', palette=custom_palette, legend=False)
    for container in ax.containers:
        ax.bar_label(container, padding=3, size=12)
    plt.title('Distribution of Patient Trajectories', fontsize=16, pad=15)
    plt.xlabel('Condition (0 = Stable/Mild, 1 = Severe & Worsening)', fontsize=14)
    plt.ylabel('Number of Patient Visits', fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def plot_baseline_severity(df, custom_palette, save_path=None):
    plt.figure(figsize=(10, 7))
    sns.boxplot(
        data=df, x='severe_and_worse', y='baseline_total_UPDRS',
        hue='severe_and_worse', palette=custom_palette, legend=False, width=0.5, linewidth=2
    )
    plt.title('Impact of Initial Baseline Severity on Disease Trajectory', fontsize=16, pad=15)
    plt.xlabel('Condition (0 = Stable/Mild, 1 = Severe & Worsening)', fontsize=14)
    plt.ylabel('Baseline Total UPDRS Score (First Visit)', fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def plot_voice_degradation(df, custom_palette, save_path=None):
    plt.figure(figsize=(10, 7))
    sns.violinplot(
        data=df, x='severe_and_worse', y='delta_Jitter(%)',
        hue='severe_and_worse', palette=custom_palette, legend=False, inner="quartile"
    )
    plt.title('Voice Degradation: Change in Jitter (%) from Baseline', fontsize=16, pad=15)
    plt.xlabel('Condition (0 = Stable/Mild, 1 = Severe & Worsening)', fontsize=14)
    plt.ylabel('Delta Jitter (%) [Current - Baseline]', fontsize=14)
    plt.axhline(0, color='gray', linestyle='--', alpha=0.6)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def plot_correlation_heatmap(df, save_path=None):
    plt.figure(figsize=(20, 16))
    num_df = df.select_dtypes(include=['number']).drop(columns=['subject#', 'test_time'], errors='ignore')
    full_corr_matrix = num_df.corr()
    sns.heatmap(
        full_corr_matrix, annot=False, cmap='coolwarm',
        vmin=-1, vmax=1, center=0, square=True, linewidths=0.5,
        cbar_kws={"shrink": .75, "label": "Pearson Correlation"}
    )
    plt.title('Full Feature Correlation Heatmap (Checking for Multicollinearity)', fontsize=20, pad=20)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def plot_filtered_correlation_heatmap(X_train, save_path=None):
    plt.figure(figsize=(24, 20))
    filtered_corr_matrix = X_train.corr()
    sns.heatmap(filtered_corr_matrix, annot=False, cmap='coolwarm', cbar=True)
    plt.title('5. Filtered Feature Correlation Heatmap', fontsize=18)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Filtered correlation heatmap saved to {save_path}")

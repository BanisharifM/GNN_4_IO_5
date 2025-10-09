import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# 1. Performance Distribution Comparison
def plot_distribution_comparison(full_stats, subsample_df):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # If you have full dataset stats, use them; otherwise use your subsample
    axes[0].hist(subsample_df['tag'], bins=50, alpha=0.7, label='Subsample', color='orange')
    # If full data available: axes[0].hist(full_data['tag'], bins=50, alpha=0.5, label='Full AIIO')
    axes[0].set_xlabel('Performance (MiB/s)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Performance Distribution')
    axes[0].legend()
    
    # Log scale version
    axes[1].hist(np.log10(subsample_df['tag'] + 1), bins=50, alpha=0.7, color='orange')
    axes[1].set_xlabel('Log10(Performance + 1)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Log-Transformed Performance')
    
    plt.tight_layout()
    plt.savefig('performance_distribution.pdf', dpi=300, bbox_inches='tight')

# 2. Feature Correlation Heatmap
def plot_feature_correlation(df):
    # Select top features by variance or importance
    feature_cols = df.columns.drop('tag')
    variances = df[feature_cols].var()
    top_15 = variances.nlargest(15).index
    
    corr_matrix = df[top_15].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('Feature Correlation Matrix (Top 15 Features)')
    plt.tight_layout()
    plt.savefig('feature_correlation.pdf', dpi=300, bbox_inches='tight')

# 3. Stratification Verification
def plot_stratification(df, n_strata=20):
    perf_bins = pd.qcut(df['tag'], q=n_strata, duplicates='drop')
    bin_counts = perf_bins.value_counts().sort_index()
    
    plt.figure(figsize=(12, 4))
    bin_counts.plot(kind='bar', color='steelblue', alpha=0.7)
    plt.xlabel('Performance Stratum')
    plt.ylabel('Sample Count')
    plt.title(f'Stratified Sampling Distribution ({n_strata} Strata)')
    plt.axhline(y=len(df)/n_strata, color='r', linestyle='--', 
                label=f'Expected ({len(df)//n_strata} per stratum)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('stratification_verification.pdf', dpi=300, bbox_inches='tight')
"""
Exploratory Data Analysis (EDA) for FuzzySegment Pro
Run this after preprocessing to understand customer feature distributions.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load preprocessed data
df = pd.read_csv('../data/preprocessed_customers.csv')
print(f"Dataset: {df.shape[0]} customers, {df.shape[1]} features")
print(f"\nColumns: {list(df.columns)}")
print(f"\nFirst 5 rows:")
print(df.head())

# Statistical summary
print(f"\n{'='*60}\nStatistical Summary\n{'='*60}")
print(df.describe())

# Check for missing values
print(f"\n{'='*60}\nMissing Values\n{'='*60}")
print(df.isnull().sum())

# Feature distributions
numeric_cols = [c for c in df.columns if c != 'Customer ID']
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.ravel()

for i, col in enumerate(numeric_cols):
    axes[i].hist(df[col], bins=30, edgecolor='black', alpha=0.7)
    axes[i].set_title(f'{col} Distribution')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('../data/feature_distributions.png', dpi=150)
plt.show()
print("\nSaved: data/feature_distributions.png")

# Correlation matrix
print(f"\n{'='*60}\nCorrelation Matrix\n{'='*60}")
corr = df[numeric_cols].corr()
print(corr)

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('../data/correlation_heatmap.png', dpi=150)
plt.show()
print("\nSaved: data/correlation_heatmap.png")

# Pairplot for category features (if 3 categories exist)
category_cols = [c for c in df.columns if c in ['Furniture', 'Office Supplies', 'Technology']]
if len(category_cols) == 3:
    print(f"\n{'='*60}\nCategory Affinity Pairplot\n{'='*60}")
    sns.pairplot(df[category_cols], diag_kind='kde', plot_kws={'alpha': 0.6})
    plt.savefig('../data/category_pairplot.png', dpi=150)
    plt.show()
    print("\nSaved: data/category_pairplot.png")

print(f"\n{'='*60}\nEDA Complete!\n{'='*60}")

"""
Run this script to preprocess train.csv and generate customer-level features.
Output: data/preprocessed_customers.csv
"""
import pandas as pd
import sys
sys.path.append('..')
from src.feature_engineering import compute_rfm_and_category_features, normalize_features

# Load raw transaction data
df = pd.read_csv('../data/train.csv')
print(f"Loaded {len(df)} transactions for {df['Customer ID'].nunique()} customers")

# Aggregate into customer-level features
customer_features = compute_rfm_and_category_features(df)
print(f"\nCustomer features computed:")
print(customer_features.head())
print(f"\nShape: {customer_features.shape}")
print(f"\nColumns: {list(customer_features.columns)}")

# Normalize features
customer_features_norm, scaler = normalize_features(customer_features)
print(f"\nNormalized features (0-1 scale)")

# Save
output_path = '../data/preprocessed_customers.csv'
customer_features_norm.to_csv(output_path, index=False)
print(f"\nSaved to {output_path}")

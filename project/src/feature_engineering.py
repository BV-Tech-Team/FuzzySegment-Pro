"""
Feature Engineering Module for Customer Segmentation
=====================================================
Transforms raw transaction data into customer-level features for clustering.

This module implements RFM (Recency, Frequency, Monetary) analysis combined
with category affinity percentages to create multi-dimensional customer profiles.

Author: FuzzySegment Pro Team
Date: December 2025
"""

import pandas as pd
import numpy as np
from datetime import datetime


def compute_rfm_and_category_features(df, customer_col='Customer ID', date_col='Order Date',
                                      sales_col='Sales', category_col='Category'):
    """
    Aggregate transaction data into customer-level features:
    - Recency (days since last purchase)
    - Frequency (number of orders)
    - Monetary (total sales)
    - Category affinity percentages (Furniture, Office Supplies, Technology)
    
    Returns: DataFrame with one row per customer
    """
    # Parse dates
    df[date_col] = pd.to_datetime(df[date_col], format='%d/%m/%Y', errors='coerce')
    reference_date = df[date_col].max()
    
    # RFM aggregation
    rfm = df.groupby(customer_col).agg({
        date_col: lambda x: (reference_date - x.max()).days,  # Recency
        'Order ID': 'nunique',  # Frequency (unique orders)
        sales_col: 'sum'  # Monetary
    }).rename(columns={
        date_col: 'Recency',
        'Order ID': 'Frequency',
        sales_col: 'Monetary'
    })
    
    # Category affinity: percentage of spending per category
    category_sales = df.groupby([customer_col, category_col])[sales_col].sum().unstack(fill_value=0)
    category_pct = category_sales.div(category_sales.sum(axis=1), axis=0)
    
    # Merge RFM + category features
    features = rfm.join(category_pct, how='left').fillna(0)
    features = features.reset_index()
    
    return features


def normalize_features(df, exclude_cols=['Customer ID']):
    """
    Normalize all numeric features (except ID columns) to 0-1 scale.
    """
    from sklearn.preprocessing import MinMaxScaler
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    scaler = MinMaxScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    return df, scaler

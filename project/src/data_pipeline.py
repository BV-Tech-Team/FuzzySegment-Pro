import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_and_preprocess(path, features=None):
    """Load CSV, basic cleaning, and return feature matrix (numpy).

    - fills numeric NA with column median
    - scales numeric features with StandardScaler
    """
    df = pd.read_csv(path)
    if features is None:
        features = [c for c in df.columns if c not in ("customer_id",)]
    # fill numeric missing
    for c in features:
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].fillna(df[c].median())
    X = df[features].astype(float)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    return df, Xs, scaler

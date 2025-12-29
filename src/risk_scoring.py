import pandas as pd

def compute_risk_score(df: pd.DataFrame):
    """
    Computes fraud risk score for each transaction
    """

    df["risk_score"] = (
        2.0 * df["amount_zscore"] +
        1.5 * df["unusual_hour"] +
        2.0 * df["unusual_location"] +
        1.5 * df["unusual_device"] +
        1.0 * df["is_foreign"]
    )

    return df

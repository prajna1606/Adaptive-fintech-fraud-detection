import pandas as pd

def add_features(df: pd.DataFrame, profiles: pd.DataFrame):
    """
    Adds behaviour deviation features for fraud detection
    """

    # merge user profiles into transaction-level data
    df = df.merge(
        profiles,
        on="user_id",
        how="left"
    )

    # amount deviation features
    df["amount_deviation"] = abs(df["amount"] - df["avg_amount"])
    df["amount_zscore"] = df["amount_deviation"] / (df["std_amount"] + 1e-5)

    # behavior deviation flags
    df["unusual_hour"] = (df["hour"] != df["most_common_hour"]).astype(int)
    df["unusual_location"] = (df["location"] != df["most_common_location"]).astype(int)
    df["unusual_device"] = (df["device"] != df["most_common_device"]).astype(int)

    # heuristic risk flags
    df["high_amount_flag"] = (
        df["amount"] > df["avg_amount"] + 3 * df["std_amount"]
    ).astype(int)

    return df

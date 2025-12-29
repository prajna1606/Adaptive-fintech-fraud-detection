import pandas as pd

def build_user_profiles(df: pd.DataFrame):
    """
    Builds behavioural profiles for each user
    """
    user_profiles= df.groupby("user_id").agg(
        avg_amount= ("amount","mean"),
        std_amount=("amount","std"),
        min_amount=("amount","min"),
        max_amount=("amount","max"),
        most_common_hour=("hour", lambda x: x.mode()[0]),
        most_common_location=("location", lambda x: x.mode()[0]),
        most_common_device=("device",lambda x: x.mode()[0]),
        txn_amount=("transaction_id","count")
    ).reset_index()

    user_profiles["std_amount"] = user_profiles["std_amount"].fillna(0)


    return user_profiles
import pandas as pd

def apply_decision_layer(df: pd.DataFrame):
    """
    Converts risk scores into real-world actions
    """

    def decide(row):
        if row["risk_score"] >= 6:
            return "BLOCK"
        elif row["risk_score"] >= 3:
            return "STEP_UP_AUTH"
        else:
            return "ALLOW"

    df["decision"] = df.apply(decide, axis=1)

    return df

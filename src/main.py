print("MAIN FILE STARTED")
from risk_scoring import compute_risk_score
from load_data import load_transactions
from user_profiling import build_user_profiles
from feature_engineering import add_features
from train_model import train_fraud_model
from decision_engine import apply_decision_layer


def main():
    print("INSIDE MAIN FUNCTION")

    df = load_transactions("data/transactions.csv")
    print("DATA LOADED")
    print(df.head())

    profiles = build_user_profiles(df)
    print("USER PROFILES")
    print(profiles.head())

    df_feat = add_features(df, profiles)
    print("FEATURE ENGINEERED DATA")
    print(df_feat.head())

    df_feat = add_features(df, profiles)
    df_scored = compute_risk_score(df_feat)

    print("\nRISK SCORED DATA")
    print(df_scored[[
        "transaction_id",
        "user_id",
        "amount",
        "risk_score",
        "label"
    ]].head())

    model = train_fraud_model(df_scored)

    df_decision = apply_decision_layer(df_scored)

    print("\nDECISION ENGINE OUTPUT")
    print(df_decision[[
        "transaction_id",
        "user_id",
        "amount",
        "risk_score",
        "decision",
        "label"
    ]].head())

    import joblib
    joblib.dump(model, "models/fraud_model.pkl")


if __name__ == "__main__":
    print("__main__ TRIGGERED")
    main()

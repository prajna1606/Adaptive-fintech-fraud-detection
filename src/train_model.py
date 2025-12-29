import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def train_fraud_model(df: pd.DataFrame):
    features = [
        "amount_zscore",
        "unusual_hour",
        "unusual_location",
        "unusual_device",
        "high_amount_flag",
        "is_foreign",
        "risk_score"
    ]

    X = df[features]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\nMODEL PERFORMANCE")
    print(classification_report(y_test, y_pred))

    return model

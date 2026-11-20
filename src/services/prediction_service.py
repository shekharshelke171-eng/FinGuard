
import joblib
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "fraud_detection_model.joblib"

model_data = joblib.load(MODEL_PATH)

model = model_data["model"]
FEATURE_COLUMNS = model_data["feature_columns"]
THRESHOLD = float(model_data["threshold"])


def predict_transaction(transaction):

    required_fields = [
        "Transaction_ID", "Customer_ID", "Transaction_Date",
        "Transaction_Amount", "Merchant_Category",
        "Payment_Method", "Device_Type", "Location",
        "Is_International", "Previous_Transactions",
        "Average_Spend", "Account_Age_Days",
        "Suspicious_Keyword"
    ]

    missing = [field for field in required_fields if field not in transaction]

    if missing:
        raise ValueError(f"Missing fields: {missing}")

    df = pd.DataFrame([transaction])

    date = pd.to_datetime(
        df["Transaction_Date"],
        format="%d-%m-%Y %H:%M",
        errors="raise"
    )

    df["Transaction_Hour"] = date.dt.hour
    df["Transaction_Day"] = date.dt.day
    df["Transaction_Month"] = date.dt.month
    df["Transaction_Weekday"] = date.dt.day_of_week

    df.drop(
        columns=["Transaction_ID", "Customer_ID", "Transaction_Date"],
        inplace=True
    )

    df["Suspicious_Keyword"] = df["Suspicious_Keyword"].map(
        {"Yes": 1, "No": 0, True: 1, False: 0}
    )

    categorical_columns = [
        "Merchant_Category", "Payment_Method",
        "Device_Type", "Location"
    ]

    df = pd.get_dummies(
        df,
        columns=categorical_columns,
        dtype=int
    )

    df = df.reindex(columns=FEATURE_COLUMNS, fill_value=0)

    probabilities = model.predict_proba(df)[0]
    fraud_index = list(model.classes_).index(1)

    fraud_probability = float(probabilities[fraud_index])
    predicted_fraud = fraud_probability >= THRESHOLD

    return {
        "transaction_id": transaction["Transaction_ID"],
        "fraud_probability": round(fraud_probability, 4),
        "predicted_fraud": bool(predicted_fraud),
        "prediction": "Fraudulent" if predicted_fraud else "Genuine",
        "threshold": THRESHOLD
    }

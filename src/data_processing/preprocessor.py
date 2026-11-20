
import pandas as pd


def preprocess_data(df):
    """Prepare transaction data for machine learning."""

    df = df.copy()

    # Extract useful date and time features
    df["Transaction_Hour"] = df["Transaction_Date"].dt.hour
    df["Transaction_Day"] = df["Transaction_Date"].dt.day
    df["Transaction_Month"] = df["Transaction_Date"].dt.month
    df["Transaction_Weekday"] = df["Transaction_Date"].dt.day_of_week

    # Remove original date and identifier columns
    df = df.drop(
        columns=["Transaction_ID", "Customer_ID", "Transaction_Date"]
    )

    # Convert Yes/No values into 1/0
    df["Suspicious_Keyword"] = df["Suspicious_Keyword"].map(
        {"Yes": 1, "No": 0}
    )

    # Convert categorical columns into numeric columns
    categorical_columns = [
        "Merchant_Category",
        "Payment_Method",
        "Device_Type",
        "Location"
    ]

    df = pd.get_dummies(
        df,
        columns=categorical_columns,
        dtype=int
    )

    return df



if __name__ == "__main__":
    from data_loader import load_dataset
    from pathlib import Path

    data = load_dataset()
    processed_data = preprocess_data(data)

    # Save processed dataset
    project_root = Path(__file__).resolve().parents[2]
    output_path = project_root / "data" / "processed" / "processed_transactions.csv"

    processed_data.to_csv(output_path, index=False)

    print("FraudShield AI - Preprocessing")
    print("--------------------------------")
    print("Original shape:", data.shape)
    print("Processed shape:", processed_data.shape)
    print("Missing values:", processed_data.isnull().sum().sum())
    print("\nProcessed dataset saved successfully!")
    print("Saved location:", output_path)

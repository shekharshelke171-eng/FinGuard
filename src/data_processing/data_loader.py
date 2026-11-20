
import pandas as pd
from pathlib import Path


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Original dataset path
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "financial_fraud_detection_dataset.csv"


def load_dataset():
    """Load the original financial fraud dataset."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    # Convert transaction date into proper datetime format
    df["Transaction_Date"] = pd.to_datetime(
        df["Transaction_Date"],
        format="%d-%m-%Y %H:%M",
        errors="coerce"
    )

    return df


if __name__ == "__main__":
    data = load_dataset()

    print("FraudShield AI - Data Loader")
    print("--------------------------------")
    print("Dataset loaded successfully!")
    print("Total rows:", data.shape[0])
    print("Total columns:", data.shape[1])
    print("Target column: Fraudulent")
    print("\nTransaction date conversion:")
    print(data["Transaction_Date"].head())

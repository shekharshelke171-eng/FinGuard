
import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT / "data" / "processed"
    / "processed_transactions.csv"
)

MODEL_PATH = (
    PROJECT_ROOT / "models"
    / "fraud_detection_model.joblib"
)


def evaluate_saved_model():

    print("FinGuard - Saved Model Evaluation")
    print("----------------------------------")

    # Load existing dataset
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["Fraudulent"])
    y = df["Fraudulent"]

    # Recreate the original test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Load existing model without retraining
    saved_data = joblib.load(MODEL_PATH)

    model = saved_data["model"]
    threshold = float(saved_data["threshold"])

    # Ensure the feature order matches the saved model
    X_test = X_test.reindex(
        columns=saved_data["feature_columns"],
        fill_value=0
    )

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    print("\nTEST RESULTS")
    print("----------------------------------")
    print(f"Threshold: {threshold:.4f}")
    print(f"Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")
    print(f"Precision: {precision_score(y_test, predictions, zero_division=0):.4f}")
    print(f"Recall: {recall_score(y_test, predictions, zero_division=0):.4f}")
    print(f"F1-Score: {f1_score(y_test, predictions, zero_division=0):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, probabilities):.4f}")

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        target_names=["Genuine", "Fraudulent"],
        zero_division=0
    ))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nEvaluation completed.")
    print("Original model file was not modified.")


if __name__ == "__main__":
    evaluate_saved_model()

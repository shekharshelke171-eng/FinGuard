
import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "processed_transactions.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "fraud_detection_model.joblib"


def train_fraud_model():

    print("FraudShield AI - Improved Model Training")
    print("--------------------------------")

    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["Fraudulent"])
    y = df["Fraudulent"]

    # Separate untouched final test data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Create a validation set from training data
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train,
        test_size=0.25,
        random_state=42,
        stratify=y_train
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    print("Training initial model...")
    model.fit(X_fit, y_fit)

    # Choose threshold using validation data
    val_probabilities = model.predict_proba(X_val)[:, 1]

    precision, recall, thresholds = precision_recall_curve(
        y_val, val_probabilities
    )

    f1_scores = (
        2 * precision[:-1] * recall[:-1]
        / (precision[:-1] + recall[:-1] + 1e-10)
    )

    best_index = f1_scores.argmax()
    best_threshold = float(thresholds[best_index])

    print("\nSelected threshold:", round(best_threshold, 4))
    print("Validation F1-score:", round(f1_scores[best_index], 4))

    # Train final model using all training data
    print("\nTraining final model...")
    model.fit(X_train, y_train)

    # Evaluate on untouched test data
    test_probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (test_probabilities >= best_threshold).astype(int)

    print("\nFINAL TEST RESULTS")
    print("--------------------------------")
    print("Accuracy:", round(accuracy_score(y_test, predictions) * 100, 2), "%")
    print("ROC-AUC:", round(roc_auc_score(y_test, test_probabilities), 4))

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        target_names=["Genuine", "Fraudulent"],
        zero_division=0
    ))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    joblib.dump(
        {
            "model": model,
            "feature_columns": X.columns.tolist(),
            "target_column": "Fraudulent",
            "threshold": best_threshold
        },
        MODEL_PATH
    )

    print("\nImproved model saved successfully!")
    print("Location:", MODEL_PATH)


if __name__ == "__main__":
    train_fraud_model()

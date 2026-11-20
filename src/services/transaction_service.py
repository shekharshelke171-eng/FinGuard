
from datetime import datetime

from database.connection import SessionLocal
from database.models import Transaction
from src.services.prediction_service import predict_transaction


def process_transaction(transaction_data):
    """Predict fraud and save the transaction in the database."""

    db = SessionLocal()

    try:
        transaction_id = transaction_data["Transaction_ID"]

        # Prevent duplicate transaction records.
        existing = (
            db.query(Transaction)
            .filter_by(transaction_id=transaction_id)
            .first()
        )

        if existing:
            raise ValueError(
                f"Transaction already exists: {transaction_id}"
            )

        # Run the existing trained ML model.
        prediction = predict_transaction(transaction_data)

        # Prepare database record.
        record = Transaction(
            transaction_id=transaction_id,
            customer_id=transaction_data["Customer_ID"],
            transaction_date=datetime.strptime(
                transaction_data["Transaction_Date"],
                "%d-%m-%Y %H:%M"
            ),
            transaction_amount=float(
                transaction_data["Transaction_Amount"]
            ),
            merchant_category=transaction_data["Merchant_Category"],
            payment_method=transaction_data["Payment_Method"],
            device_type=transaction_data["Device_Type"],
            location=transaction_data["Location"],
            is_international=bool(
                transaction_data["Is_International"]
            ),
            previous_transactions=int(
                transaction_data["Previous_Transactions"]
            ),
            average_spend=float(
                transaction_data["Average_Spend"]
            ),
            account_age_days=int(
                transaction_data["Account_Age_Days"]
            ),
            suspicious_keyword=(
                transaction_data["Suspicious_Keyword"] == "Yes"
            ),
            actual_fraudulent=int(
                transaction_data["Fraudulent"]
            ) if "Fraudulent" in transaction_data else None,
            fraud_probability=prediction["fraud_probability"],
            predicted_fraud=prediction["predicted_fraud"],
            prediction_threshold=prediction["threshold"]
        )

        db.add(record)
        db.commit()
        db.refresh(record)

        return {
            "status": "success",
            "message": "Transaction saved successfully",
            "transaction_id": record.transaction_id,
            "prediction": prediction["prediction"],
            "fraud_probability": prediction["fraud_probability"],
            "predicted_fraud": prediction["predicted_fraud"]
        }

    except Exception:
        db.rollback()
        raise

    finally:
        db.close()

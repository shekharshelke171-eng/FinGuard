
from datetime import datetime

from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    DateTime,
    Boolean
)

from database.connection import Base


class Transaction(Base):
    """Store transaction details and fraud prediction results."""

    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True)

    transaction_id = Column(String(50), unique=True, nullable=False, index=True)
    customer_id = Column(String(50), nullable=False)

    transaction_date = Column(DateTime, nullable=False)
    transaction_amount = Column(Float, nullable=False)

    merchant_category = Column(String(50), nullable=False)
    payment_method = Column(String(50), nullable=False)
    device_type = Column(String(50), nullable=False)
    location = Column(String(50), nullable=False)

    is_international = Column(Boolean, default=False)
    previous_transactions = Column(Integer)
    average_spend = Column(Float)
    account_age_days = Column(Integer)
    suspicious_keyword = Column(Boolean, default=False)

    actual_fraudulent = Column(Integer, nullable=True)

    fraud_probability = Column(Float, nullable=True)
    predicted_fraud = Column(Boolean, nullable=True)
    prediction_threshold = Column(Float, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

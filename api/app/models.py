"""
SQLAlchemy ORM + Pydantic v2 schemas for the fraud-detection service.

 - User - one row per unique user_id coming from Kafka
 - Transaction - raw transaction data (exact copy of Kafka message)
 - Prediction - model output linked 1-to-1 with Transaction
 - Alert - optional table for analyst workflow

All tables inherit timestamps from Base.
"""
from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, ConfigDict
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Enum as SAEnum,
    Float,
    ForeignKey,
    Integer,
    String,
    Uuid,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# Enum classes 
# class for role 
class Role(str, enum.Enum):
    admin = 'admin' # full system access, can manage users, secrets, configs
    analyst = 'analyst' # reviews alerts, marks transactions as confirmed fraud or false-positive
    customer = 'customer'

# SQLAlchemy base - adds created_at and updated_at for transactions
class Base(DeclarativeBase):
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone = True),
        server_default = text('CURRENT_TIMESTAMP'),
        nullable = False,
    )
    updated_at:Mapped[datetime] = mapped_column(
        DateTime(timezone = True),
        server_default = text('CURRENT_TIMESTAMP'),
        onupdate = datetime.utcnow,
        nullable = False,
    )

# ORM entities to represent tables in database  
class User(Base):
    # User entitiy representing cardholder/owner of account - 1 to many relationship with Transactions
    __tablename__ = 'users'

    id: Mapped[int] = mapped_column(Integer, primary_key = True)
    first_name: Mapped[Optional[str]] = mapped_column(String(64))
    last_name: Mapped[Optional[str]] = mapped_column(String(64))
    role: Mapped[Optional[Role]] = mapped_column(SAEnum(Role))

    transactions: Mapped[List['Transaction']] = relationship(
        back_populates = 'user',
        cascade = 'all, delete-orphan',
    )

# class for transactions 
class Transaction(Base):
    # raw transaction exactly as recieived from Kafka 
    __tablename__ = 'transactions'

    transaction_id: Mapped[str] = mapped_column(
        String(64), primary_key = True
    )
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey('users.id', ondelete = 'CASCADE'), index = True
    )
    amount: Mapped[float] = mapped_column(Float, nullable = False)
    currency: Mapped[Optional[str]] = mapped_column(String(3))
    merchant: Mapped[Optional[str]] = mapped_column(String(128))
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone = True), index = True)
    location: Mapped[Optional[str]] = mapped_column(String(128))
  
    user: Mapped[User] = relationship(back_populates = 'transactions')
    prediction: Mapped[Optional['Prediction']] = relationship(
        back_populates= 'transaction',
        uselist = False,
        cascade = "all, delete-orphan",
        lazy = "joined" 
    )

# class for predictions: Each prediction has a unique id (PK), transaction_id (FK from transactions table), is_fraud flag (model predicted it as fraudulent or not), probability of the prediction, and model version 
class Prediction(Base):
    # output of the model: 1 transaction has 1 prediction 
    __tablename__ = 'predictions'

    id: Mapped[int] = mapped_column(Integer, primary_key = True, autoincrement = True)
    transaction_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey('transactions.transaction_id', ondelete = 'CASCADE'),
        unique = True,
    )
    is_fraud: Mapped[bool] = mapped_column(Boolean, nullable = False)

    # add engineered features 
    transaction_hour: Mapped[int] = mapped_column(Integer)
    is_weekend: Mapped[bool] = mapped_column(Boolean)
    is_night: Mapped[bool] = mapped_column(Boolean)
    transaction_day: Mapped[int] = mapped_column(Integer)
    user_activity_24h: Mapped[int] = mapped_column(Integer)
    amount_to_avg_ratio: Mapped[float] = mapped_column(Float)
    merchant_risk: Mapped[int] = mapped_column(Integer)
    
    transaction: Mapped['Transaction'] = relationship(back_populates = 'prediction')

# Pydantic Schemas 
# class for the base of the schema to config the model 
class _SchemaBase(BaseModel):
    model_config = ConfigDict(from_attributes = True, protected_namespaces = ())

# User schemas 
class UserCreate(_SchemaBase):
    id: int 
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    role: Optional[Role] = None

class UserRead(UserCreate):
    created_at: datetime 
    updated_at: datetime 

# Transaction schemas 
class TxnCreate(_SchemaBase):
    transaction_id: str
    user_id: int
    amount: float
    currency: Optional[str] = None
    merchant: Optional[str] = None
    timestamp: datetime
    location: Optional[str] = None
  
class TxnRead(TxnCreate):
    created_at: datetime
    prediction: Optional['PredictionRead'] = None 

# prediction schemas 
class PredictionRead(_SchemaBase):
    id: int
    transaction_id: str
    is_fraud: bool
    transaction_hour: int
    is_weekend: bool
    is_night: bool
    transaction_day: int
    user_activity_24h: int
    amount_to_avg_ratio: float
    merchant_risk: int
    created_at: datetime

TxnRead.model_rebuild() # resolves PredictionRead inside TxnRead 
from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from app.config import settings


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(30), default="analyst")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class FraudTransaction(Base):
    __tablename__ = "fraud_transactions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    step: Mapped[int] = mapped_column(Integer, nullable=False)
    tx_type: Mapped[str] = mapped_column(String(20), nullable=False)
    amount: Mapped[float] = mapped_column(Float, nullable=False)
    oldbalance_org: Mapped[float] = mapped_column(Float, nullable=False)
    newbalance_orig: Mapped[float] = mapped_column(Float, nullable=False)
    oldbalance_dest: Mapped[float] = mapped_column(Float, nullable=False)
    newbalance_dest: Mapped[float] = mapped_column(Float, nullable=False)
    is_fraud: Mapped[bool] = mapped_column(Boolean, nullable=False)
    fraud_probability: Mapped[float] = mapped_column(Float, default=0.0)
    anomaly_score: Mapped[float] = mapped_column(Float, default=0.0)
    risk_score: Mapped[float] = mapped_column(Float, default=0.0)
    alert_sent: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


def get_engine(use_mysql: bool = False):
    if use_mysql and settings.mysql_url:
        return create_engine(settings.mysql_url, pool_pre_ping=True)
    return create_engine(settings.sqlite_url, future=True)


engine = get_engine(use_mysql=False)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


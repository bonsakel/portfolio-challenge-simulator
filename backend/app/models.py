"""SQLite persistence via SQLAlchemy 2.0 (portable to PostgreSQL by changing the URL)."""

from datetime import datetime

from sqlalchemy import JSON, DateTime, Float, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class PortfolioRow(Base):
    __tablename__ = "portfolios"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(80))
    initial_capital: Mapped[float] = mapped_column(Float)
    max_weight: Mapped[float] = mapped_column(Float)
    weight_mode: Mapped[str] = mapped_column(String(10))
    holdings: Mapped[list[dict]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class BacktestRow(Base):
    __tablename__ = "backtests"

    id: Mapped[int] = mapped_column(primary_key=True)
    portfolio_id: Mapped[int | None] = mapped_column(nullable=True)
    portfolio_name: Mapped[str] = mapped_column(String(80))
    status: Mapped[str] = mapped_column(String(12))  # queued | running | succeeded | failed
    error: Mapped[str | None] = mapped_column(String, nullable=True)
    config: Mapped[dict] = mapped_column(JSON)  # request plus the portfolio snapshot it ran on
    summary: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    result: Mapped[dict | None] = mapped_column(JSON, nullable=True, deferred=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class TrainingJobRow(Base):
    __tablename__ = "training_jobs"

    id: Mapped[int] = mapped_column(primary_key=True)
    algorithm: Mapped[str] = mapped_column(String(8))
    status: Mapped[str] = mapped_column(String(12))  # queued | running | succeeded | failed | cancelled
    error: Mapped[str | None] = mapped_column(String, nullable=True)
    config: Mapped[dict] = mapped_column(JSON)  # confirmed request plus the portfolio snapshot
    model_id: Mapped[str | None] = mapped_column(String(80), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

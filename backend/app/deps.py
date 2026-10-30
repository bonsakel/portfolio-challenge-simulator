from collections.abc import Iterator
from functools import cached_property

from fastapi import Request
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session

from app.assets import SymbolIndex
from app.backtests.jobs import JobManager, mark_interrupted
from app.config import PROJECT_ROOT, Settings
from app.market_data.cache import PriceCache
from app.market_data.provider import PriceProvider, YFinanceProvider
from app.market_data.service import MarketDataService
from app.models import Base
from app.rl.registry import ModelRegistry
from app.rl.training import TrainingManager, mark_interrupted_training


class Services:
    """Lazily created per-app resources, so importing the app has no filesystem side effects."""

    def __init__(self, settings: Settings, provider: PriceProvider | None = None) -> None:
        self.settings = settings
        self._provider = provider

    @cached_property
    def market_data(self) -> MarketDataService:
        provider = self._provider or YFinanceProvider()
        return MarketDataService(provider, PriceCache(self.settings.data_dir))

    @cached_property
    def engine(self) -> Engine:
        self.settings.data_dir.mkdir(parents=True, exist_ok=True)
        engine = create_engine(f"sqlite:///{self.settings.data_dir / 'portfoliolab.sqlite3'}")
        Base.metadata.create_all(engine)
        mark_interrupted(engine)
        mark_interrupted_training(engine)
        return engine

    @cached_property
    def backtests(self) -> JobManager:
        return JobManager(self.engine, self.market_data, self.models)

    @cached_property
    def models(self) -> ModelRegistry:
        return ModelRegistry(
            self.settings.results_dir,
            PROJECT_ROOT / "configs" / "experiment.yaml",
            self.settings.data_dir / "models",
        )

    @cached_property
    def training(self) -> TrainingManager:
        return TrainingManager(self.engine, self.market_data, self.settings.data_dir)

    @cached_property
    def symbols(self) -> SymbolIndex:
        return SymbolIndex(self.settings.data_dir)


def get_services(request: Request) -> Services:
    return request.app.state.services


def get_session(request: Request) -> Iterator[Session]:
    with Session(get_services(request).engine) as session:
        yield session

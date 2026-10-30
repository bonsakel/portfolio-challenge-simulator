from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api import assets, backtests, data, health, portfolios, research, rl
from app.config import Settings, get_settings
from app.deps import Services


from app.market_data.provider import PriceProvider


def create_app(settings: Settings | None = None, provider: PriceProvider | None = None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(
        title="PortfolioLab API",
        version=__version__,
        description="Historical portfolio research simulator. Not investment advice.",
    )
    app.state.services = Services(settings, provider)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["Content-Type"],
    )
    for module in (health, research, assets, portfolios, data, backtests, rl):
        app.include_router(module.router, prefix="/api/v1")
    return app


app = create_app()

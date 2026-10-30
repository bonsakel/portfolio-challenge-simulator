import numpy as np
import pandas as pd
import pytest

from src.strategies.traditional import (
    equal_weight_policy,
    make_mean_variance_policy,
    make_minimum_variance_policy,
)


@pytest.fixture
def synthetic_returns() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2020-01-01", periods=260)
    return pd.DataFrame(
        {
            "A": rng.normal(0.0007, 0.01, len(dates)),
            "B": rng.normal(0.0004, 0.008, len(dates)),
            "C": rng.normal(0.0002, 0.004, len(dates)),
            "D": rng.normal(-0.0001, 0.006, len(dates)),
        },
        index=dates,
    )


def test_equal_weight_policy_uses_current_weight_asset_order() -> None:
    current = pd.Series({"B": 0.25, "A": 0.25, "C": 0.25, "D": 0.25})

    weights = equal_weight_policy(pd.Timestamp("2020-01-01"), pd.DataFrame(), current)

    assert weights.index.tolist() == ["B", "A", "C", "D"]
    assert weights.eq(0.25).all()


def test_minimum_variance_policy_returns_valid_capped_weights(
    synthetic_returns: pd.DataFrame,
) -> None:
    policy = make_minimum_variance_policy(
        synthetic_returns.columns,
        lookback_trading_days=252,
        max_weight=0.30,
    )

    weights = policy(
        synthetic_returns.index[-1],
        synthetic_returns.iloc[:252],
        pd.Series(0.25, index=synthetic_returns.columns),
    )

    assert weights.sum() == pytest.approx(1.0)
    assert (weights >= 0.0).all()
    assert (weights <= 0.30 + 1e-8).all()
    assert weights["C"] > weights["A"]


def test_mean_variance_policy_requires_explicit_daily_risk_free_rate() -> None:
    with pytest.raises(ValueError, match="risk_free_rate_daily"):
        make_mean_variance_policy(["A", "B", "C", "D"])


def test_mean_variance_policy_returns_valid_capped_weights(
    synthetic_returns: pd.DataFrame,
) -> None:
    policy = make_mean_variance_policy(
        synthetic_returns.columns,
        lookback_trading_days=252,
        max_weight=0.30,
        risk_free_rate_daily=0.0,
    )

    weights = policy(
        synthetic_returns.index[-1],
        synthetic_returns.iloc[:252],
        pd.Series(0.25, index=synthetic_returns.columns),
    )

    assert weights.sum() == pytest.approx(1.0)
    assert (weights >= 0.0).all()
    assert (weights <= 0.30 + 1e-8).all()
    assert np.isfinite(weights.to_numpy()).all()


def test_mean_variance_policy_handles_windows_with_all_negative_means(
    synthetic_returns: pd.DataFrame,
) -> None:
    negative_returns = synthetic_returns - 0.01
    policy = make_mean_variance_policy(
        negative_returns.columns,
        lookback_trading_days=252,
        max_weight=0.30,
        risk_free_rate_daily=0.0,
    )

    weights = policy(
        negative_returns.index[-1],
        negative_returns.iloc[:252],
        pd.Series(0.25, index=negative_returns.columns),
    )

    assert weights.sum() == pytest.approx(1.0)
    assert (weights >= 0.0).all()
    assert (weights <= 0.30 + 1e-8).all()


def test_rolling_policies_reject_short_history(synthetic_returns: pd.DataFrame) -> None:
    policy = make_minimum_variance_policy(
        synthetic_returns.columns,
        lookback_trading_days=252,
        max_weight=0.30,
    )

    with pytest.raises(ValueError, match="Need 252 historical returns"):
        policy(
            synthetic_returns.index[-1],
            synthetic_returns.iloc[:251],
            pd.Series(0.25, index=synthetic_returns.columns),
        )


def test_rolling_policies_reject_history_at_or_after_decision_date(
    synthetic_returns: pd.DataFrame,
) -> None:
    policy = make_minimum_variance_policy(
        synthetic_returns.columns,
        lookback_trading_days=252,
        max_weight=0.30,
    )

    with pytest.raises(ValueError, match="strictly before decision date"):
        policy(
            synthetic_returns.index[251],
            synthetic_returns.iloc[:252],
            pd.Series(0.25, index=synthetic_returns.columns),
        )

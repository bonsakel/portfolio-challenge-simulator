from datetime import date

import pytest

from app.config import PROJECT_ROOT
from app.rl.registry import ModelRegistry, ModelSpec, check_compatibility

UNIVERSE = ("XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY")


def spec(**kw) -> ModelSpec:
    base = dict(
        id="user/job-1", algorithm="PPO", label="PPO test", source="user", path=PROJECT_ROOT,
        universe=UNIVERSE, max_weight=0.3, transaction_cost_rate=0.001, observation_size=191,
        action_size=9, training_period=("2016-02-19", "2019-02-18"),
        validation_period=("2019-02-19", "2020-02-18"), seed=42, trained_on="custom", timesteps=1,
    )
    base.update(kw)
    return ModelSpec(**base)


def check(model=None, **kw):
    args = dict(
        assets=list(UNIVERSE), max_weight=0.3, transaction_cost_rate=0.001,
        first_decision=date(2020, 2, 19), last_decision=date(2020, 3, 23),
    )
    args.update(kw)
    return check_compatibility(model or spec(), **args)


def test_matching_out_of_sample_window_is_compatible():
    result = check()
    assert result.compatible and result.issues == [] and result.warnings == []


def test_different_universe_is_rejected_with_the_difference():
    result = check(assets=["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "SPY"])
    assert not result.compatible
    assert "missing XLY" in result.issues[0] and "extra SPY" in result.issues[0]


def test_same_assets_in_different_order_is_rejected():
    result = check(assets=list(reversed(UNIVERSE)))
    assert not result.compatible and "different order" in result.issues[0]


def test_fewer_assets_is_rejected():
    assert not check(assets=["XLB", "XLE"]).compatible


@pytest.mark.parametrize("override", [{"max_weight": 0.5}, {"transaction_cost_rate": 0.002}])
def test_constraint_mismatch_is_rejected(override):
    assert not check(**override).compatible


def test_observation_or_action_shape_mismatch_is_rejected():
    assert not check(spec(observation_size=100)).compatible
    assert not check(spec(action_size=5)).compatible


def test_window_overlapping_training_or_validation_is_in_sample():
    for first, last in [
        (date(2018, 1, 1), date(2018, 6, 1)),  # inside training
        (date(2019, 6, 1), date(2019, 9, 1)),  # inside validation
        (date(2020, 1, 1), date(2020, 3, 1)),  # straddles the validation end
    ]:
        result = check(first_decision=first, last_decision=last)
        assert not result.compatible and "overlaps" in result.issues[-1]


def test_window_before_training_warns_about_look_ahead():
    result = check(first_decision=date(2008, 1, 1), last_decision=date(2009, 1, 1))
    assert result.compatible and "came after this window" in result.warnings[0]


def test_non_equal_initial_weights_are_rejected():
    assert not check(equal_initial_weights=False).compatible


def test_research_models_are_discovered_with_their_recorded_periods():
    registry = ModelRegistry(PROJECT_ROOT / "results", PROJECT_ROOT / "configs/experiment.yaml", PROJECT_ROOT / "nope")
    models = {m.id: m for m in registry.list()}
    if not models:
        pytest.skip("research models are not present")
    assert set(models) == {
        f"research/{event}/{algo}"
        for event in ("global_financial_crisis", "covid_crash", "2022_downturn")
        for algo in ("ppo", "ddpg")
    }
    covid = models["research/covid_crash/ppo"]
    assert covid.universe == UNIVERSE and covid.observation_size == 191 and covid.action_size == 9
    assert covid.training_period == ("2016-02-19", "2019-02-18") and covid.seed == 42
    assert registry.get("research/covid_crash/ppo") == covid and registry.get("missing") is None

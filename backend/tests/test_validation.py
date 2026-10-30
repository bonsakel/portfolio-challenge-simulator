import pytest

from app.validation import validate_portfolio


def run(symbols, weights=None, mode="equal", cap=0.30, capital=100_000.0, listed=lambda s: None):
    return validate_portfolio(
        symbols, weights if weights is not None else [None] * len(symbols), mode, cap, capital, listed
    )


def codes(report, severity="error"):
    return {i.code for i in report.issues if i.severity == severity}


def test_equal_weight_nine_assets_valid_and_sums_to_one():
    report = run(list("ABCDEFGHI"))
    assert report.valid
    assert sum(report.effective_weights.values()) == pytest.approx(1.0)
    assert all(w == pytest.approx(1 / 9) for w in report.effective_weights.values())


def test_cap_infeasible_with_three_assets_at_30pct():
    # 3 * 30% = 90% < 100%: a fully invested portfolio is impossible.
    report = run(["A", "B", "C"])
    assert not report.valid
    assert "cap_infeasible" in codes(report)
    assert report.min_feasible_holdings == 4


def test_cap_boundary_is_feasible_at_exactly_four_assets_of_25pct():
    assert run(list("ABCD"), cap=0.25).valid


def test_min_feasible_holdings_for_30pct_is_four():
    assert run(list("ABCD")).min_feasible_holdings == 4


def test_custom_weights_must_sum_to_100_and_are_not_normalized():
    report = run(list("ABCD"), [0.3, 0.3, 0.3, 0.05], "custom")
    assert "weights_not_100" in codes(report)
    assert report.weight_sum == pytest.approx(0.95)
    assert not report.valid


def test_custom_weight_above_cap_rejected():
    report = run(list("ABCD"), [0.4, 0.2, 0.2, 0.2], "custom")
    assert codes(report) == {"weight_above_cap"}


def test_custom_valid_weights():
    report = run(list("ABCD"), [0.3, 0.3, 0.2, 0.2], "custom")
    assert report.valid
    assert report.effective_weights == {"A": 0.3, "B": 0.3, "C": 0.2, "D": 0.2}


def test_negative_weight_rejected_long_only():
    assert "negative_weight" in codes(run(list("ABCD"), [0.5, 0.3, 0.3, -0.1], "custom"))


def test_missing_weight_in_custom_mode():
    assert "missing_weight" in codes(run(list("ABCD"), [0.25, 0.25, 0.25, None], "custom"))


def test_duplicates_rejected():
    assert "duplicate_symbol" in codes(run(["A", "A", "B", "C"]))


def test_empty_portfolio_and_bad_capital():
    report = run([], capital=0)
    assert {"no_holdings", "capital_not_positive"} <= codes(report)


def test_unknown_symbol_flagged_only_when_index_says_false():
    report = run(list("ABCD"), listed=lambda s: s != "D")
    assert [i.symbol for i in report.issues if i.code == "unknown_symbol"] == ["D"]
    assert "index_unavailable" in codes(run(list("ABCD")), "warning")


def test_cap_of_100pct_allows_single_asset_with_warning():
    report = run(["A"], cap=1.0)
    assert report.valid
    assert "single_asset" in codes(report, "warning")

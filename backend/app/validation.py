"""Pure portfolio validation. No I/O, so every financial rule is unit-testable.

Rules (long-only, fully invested, per-asset cap):
  * weights are never silently normalized; invalid input is reported, not fixed;
  * equal-weight mode assigns 1/n and is feasible only if n * max_weight >= 1;
  * custom weights must each be in [0, max_weight] and sum to 1 within WEIGHT_TOL.
"""

from collections.abc import Callable, Sequence
from typing import Literal

from pydantic import BaseModel

WEIGHT_TOL = 1e-6

Severity = Literal["error", "warning"]


class Issue(BaseModel):
    severity: Severity
    code: str
    message: str
    symbol: str | None = None


class ValidationReport(BaseModel):
    valid: bool
    issues: list[Issue]
    effective_weights: dict[str, float]
    weight_sum: float | None
    min_feasible_holdings: int | None


def validate_portfolio(
    symbols: Sequence[str],
    weights: Sequence[float | None],
    weight_mode: Literal["equal", "custom"],
    max_weight: float,
    initial_capital: float,
    is_listed: Callable[[str], bool | None] = lambda s: None,
) -> ValidationReport:
    """Validate a normalized configuration. `is_listed` returns None if the index is unavailable."""
    issues: list[Issue] = []

    def err(code: str, message: str, symbol: str | None = None) -> None:
        issues.append(Issue(severity="error", code=code, message=message, symbol=symbol))

    def warn(code: str, message: str, symbol: str | None = None) -> None:
        issues.append(Issue(severity="warning", code=code, message=message, symbol=symbol))

    n = len(symbols)
    if initial_capital <= 0:
        err("capital_not_positive", "Initial capital must be greater than zero.")
    if not 0 < max_weight <= 1:
        err("max_weight_out_of_range", "Maximum weight must be above 0% and at most 100%.")
    if n == 0:
        err("no_holdings", "Add at least one asset to the portfolio.")
    elif n == 1:
        warn("single_asset", "A single-asset portfolio cannot be compared meaningfully across strategies.")

    for sym in sorted({s for s in symbols if list(symbols).count(s) > 1}):
        err("duplicate_symbol", f"{sym} appears more than once.", sym)

    for sym in dict.fromkeys(symbols):
        listed = is_listed(sym)
        if listed is False:
            err("unknown_symbol", f"{sym} is not in the supported symbol index.", sym)
    if n and is_listed(symbols[0]) is None:
        warn("index_unavailable", "Symbol index not built; symbols could not be checked against it.")

    min_holdings = None
    if 0 < max_weight <= 1:
        # ceil(1 / cap) without float error: smallest k with k * cap >= 1 - tol.
        min_holdings = 1
        while min_holdings * max_weight < 1 - WEIGHT_TOL:
            min_holdings += 1
        if 0 < n < min_holdings:
            err(
                "cap_infeasible",
                f"A {max_weight:.0%} per-asset cap requires at least {min_holdings} assets to be "
                f"fully invested; you have {n}. Add assets or raise the cap.",
            )

    effective: dict[str, float] = {}
    weight_sum: float | None = None
    unique = list(dict.fromkeys(symbols))
    if weight_mode == "equal":
        if unique:
            effective = {s: 1.0 / len(unique) for s in unique}
            weight_sum = 1.0
    else:
        if any(w is None for w in weights):
            err("missing_weight", "Every holding needs a weight in custom mode.")
        else:
            given = [float(w) for w in weights if w is not None]
            weight_sum = sum(given)
            for sym, w in zip(symbols, given):
                if w < 0:
                    err("negative_weight", f"{sym} has a negative weight; only long positions are supported.", sym)
                elif 0 < max_weight <= 1 and w > max_weight + WEIGHT_TOL:
                    err("weight_above_cap", f"{sym} weight {w:.2%} exceeds the {max_weight:.0%} cap.", sym)
            if abs(weight_sum - 1.0) > WEIGHT_TOL:
                err("weights_not_100", f"Weights sum to {weight_sum:.4%}, not 100%.")
            if not any(i.code == "duplicate_symbol" for i in issues):
                effective = dict(zip(symbols, given))

    return ValidationReport(
        valid=not any(i.severity == "error" for i in issues),
        issues=issues,
        effective_weights=effective,
        weight_sum=weight_sum,
        min_feasible_holdings=min_holdings,
    )

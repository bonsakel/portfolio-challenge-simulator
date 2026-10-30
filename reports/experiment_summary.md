# Experiment Results Summary

This report summarizes one seeded PPO/DDPG training run per event and
the three configured traditional strategies. It is descriptive, not
a claim of statistical significance or evidence that one strategy is
generally superior.

## global_financial_crisis

- Highest downturn total return: **minimum_variance** (-41.03%).
- Highest downturn Sharpe ratio: **minimum_variance** (-1.297).
- Recovered within 24 months: equal_weight (no), minimum_variance (no), mean_variance (no), PPO (no), DDPG (no).

## covid_crash

- Highest downturn total return: **minimum_variance** (-29.73%).
- Highest downturn Sharpe ratio: **minimum_variance** (-4.788).
- Recovered within 24 months: equal_weight (yes), minimum_variance (yes), mean_variance (yes), PPO (yes), DDPG (yes).

## 2022_downturn

- Highest downturn total return: **mean_variance** (-2.56%).
- Highest downturn Sharpe ratio: **mean_variance** (-0.083).
- Recovered within 24 months: equal_weight (yes), minimum_variance (yes), mean_variance (yes), PPO (no), DDPG (no).

## Interpretation

Results reflect the specified historical windows, one random seed,
one validation episode per checkpoint, the chosen transaction-cost
model, and a zero daily risk-free-rate assumption. Interpret strategy
rankings accordingly; PPO and DDPG performance may vary with seeds
and hyperparameters. No result should be generalized beyond the
tested assets and periods without additional robustness analysis.

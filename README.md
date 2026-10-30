# Reinforcement Learning vs. Traditional Portfolio Strategies During Market Downturns

## PortfolioLab (interactive app, in development)

PortfolioLab is a local web app (Next.js frontend + FastAPI backend) built on this research code. The research code in `src/` is unchanged and reused. Status: **Milestone 7** complete (scaffold, health check, read-only research results, symbol search, portfolio builder with saved portfolios, market-data provider/cache and Data Status page, backtests of the traditional strategies, risk analytics and interactive charts, PPO/DDPG backtests and model training in the Strategy Lab, and a locked Research Mode). Results are historical simulations, not investment advice.

Requirements: Python 3.11, Node.js 20.9+ (tested on 22.16), npm.

```bash
# one-time setup
python -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt
(cd frontend && npm install)
(cd backend && python -m app.update_symbol_index)   # needs internet; builds the local symbol index

# terminal 1: backend (http://127.0.0.1:8000, docs at /docs)
cd backend && python -m uvicorn app.main:app --host 127.0.0.1 --port 8000

# terminal 2: frontend (http://127.0.0.1:3000)
cd frontend && npm run dev

# tests
python -m pytest            # research + backend
(cd frontend && npm test && npm run lint)
```

Troubleshooting: if the sidebar shows "Backend offline", start the backend and check port 8000 is free (`lsof -i :8000`). If search returns "Symbol index not built", run the `update_symbol_index` command above. Optional settings use the `PORTFOLIOLAB_` prefix (see `.env.example`); no API key is required.

**Symbol search and portfolios.** Search uses a local index built from the [NASDAQ Trader symbol directory](https://www.nasdaqtrader.com/dynamic/SymDir/) (currently listed US stocks and ETFs; test issues, warrants, units, rights and preferreds are excluded; `.` class separators become `-`, e.g. `BRK-B`). Delisted symbols are absent, so custom backtests are selected-universe studies. Being listed does not guarantee price history exists; use "Check data coverage" in the Portfolio Builder to verify it. Portfolios are stored in local SQLite (`data/portfoliolab/`, git-ignored). Validation is long-only and fully invested: weights are never silently normalized, and a per-asset cap `c` needs at least ⌈1/c⌉ assets (30% → 4).

**Market data.** Prices come from [yfinance](https://github.com/ranaroussos/yfinance) (Yahoo Finance; unofficial, for research and education, not real-time) using the **Adj Close** field, the same total-return-oriented field the research used (a live check of XLK and XLE matched the research data to ~2e-6 in daily returns). The full history of each symbol is downloaded once (needs internet) and cached as CSV under `data/portfoliolab/prices/`; later checks use the cache, and re-downloading is a deliberate action (Settings / Data Status → Re-download, or Clear cache, which asks for confirmation). Missing values are never forward-filled: dates missing for any asset are dropped for all assets, and gaps longer than 5 calendar days, unavailable symbols, stale history and a requested range outside the common history are reported. Failed downloads are recorded and shown, not hidden. API: `GET /data/status`, `POST /data/coverage`, `POST /data/refresh`, `DELETE /data/cache?confirm=true`. Yahoo may rate-limit or change its responses without notice.

**Backtests.** The Backtests page runs Equal Weight (monthly rebalance), Minimum Variance and Mean Variance (daily, 2-point threshold) on a saved portfolio and date range, reusing the research engine in `src/` unchanged. Fee: 0.10% of traded value (sum of absolute weight changes × pre-trade value), charged once before the day's return. Decisions on day t use only returns before t. Each optimizer needs a warm-up of `lookback_days` returns before the first decision (the app says which start date to use). Weights are long-only, fully invested, with the position cap. Initial holdings are equal weight at no cost (research convention), or optionally the portfolio's own weights. All strategies share the same dates. Frequency can be daily or monthly only. Runs are background jobs polled for progress, and results are stored in SQLite. API: `POST /backtests`, `GET /backtests`, `GET /backtests/{id}`, `GET /backtests/{id}/results`, `GET /backtests/{id}/export.csv`, `DELETE /backtests/{id}`. Live check: the Global Financial Crisis window on the research 9 ETFs matched `results/tables/summary_metrics.csv` to ~4 decimals (e.g. Minimum Variance −41.03%). RL strategies come in a later milestone. Results are historical simulations, not investment advice.

**Benchmark, risk analytics and charts.** A backtest can include an optional benchmark ticker (default SPY): a buy-and-hold of that symbol over exactly the same dates and starting capital, with no trading costs. It uses the same cached Adj Close prices and is never forward-filled; if the benchmark has no price on one of the dates used, the run fails with a clear message instead of guessing. The Risk Analytics page (`/risk-analytics`) shows, for a completed run: total and annualized return, volatility, Sharpe, Sortino, maximum drawdown, historical 1-day VaR and CVaR at 95%, worst day, costs, turnover, and drawdown/recovery. Definitions (also shown on the page): 252 trading days per year; Sortino uses the risk-free rate as target and the downside deviation over all days; VaR is the 5th percentile of daily net returns (needs at least 20 days) and CVaR the mean of returns at or below it. Recovery is measured from the maximum-drawdown trough back to the peak that preceded it (the starting capital counts as a peak): recovery time is in trading days, recovery return is trough to end of window, and a series that has not recovered is reported as "not recovered" rather than extrapolated. Charts (Plotly, loaded in the browser only) are built directly from the stored daily results: portfolio value with benchmark, drawdown, risk versus return, total return and drawdown comparison, recovery trajectory, cumulative costs, monthly turnover and stacked allocation. Every chart has a Table view for accessibility, and the CSV export includes drawdown and the benchmark. The analytics are in `backend/app/backtests/analytics.py` and are tested against hand calculations and the research `period_metrics`. Live check: Global Financial Crisis (strategies unchanged from the research table) and COVID windows with SPY; the COVID trough (2020-03-23) and the SPY drawdown (about −34%) agree with the historical record.

**RL integration and Strategy Lab (Milestone 6).** PPO and DDPG can be selected in the Backtests page and run as frozen, pretrained policies (no learning during a backtest) through the unchanged research `PortfolioEnv` and the `evaluate_agent` procedure: frozen `VecNormalize`, deterministic actions, CPU, daily decisions, no trade threshold. The six research models (`results/models/<event>/<algo>`) are listed, and a model is only usable if it is compatible with the run: the same ordered 9-asset universe, the same position cap and cost rate, equal initial weights, the 191-value observation, and a date window that does not overlap the model's training or validation period (a warning appears if the window ends before the training start, which would be look-ahead). Incompatible models are shown with the reason and are never selected silently; compatibility is re-checked when the job runs. The VIX input comes from the research file `data/processed/daily_returns/vix.csv` (CBOE), so RL runs and training are limited to 2003-03-03 → 2024-10-11, and the 21 prior trading days must exist. The Strategy Lab (`/strategy-lab`) explains each strategy, lists models with their compatibility, and can train a new PPO or DDPG model on a 9-asset portfolio with the research `train_agent` in a subprocess. Training needs an explicit confirmation (`confirm=true` in the API, a checkbox dialog in the UI), shows progress from validation checkpoints, can be cancelled (partial output is removed), and allows one job at a time. Trained models are saved under `data/portfoliolab/models/job-<id>/` with their environment metadata and appear as `user/job-<id>`. Time estimates are rough (about 1500 steps/s for PPO and 300 for DDPG on the development machine). API: `GET /models`, `POST /training-jobs`, `GET /training-jobs`, `GET /training-jobs/{id}`, `POST /training-jobs/{id}/cancel`. Verification: a test reproduces the research COVID PPO and DDPG final values (93,707.11 and 120,737.85) to a relative 1e-9; a live run over 2020-02-19 → 2022-03-23 gave 93,706.8 and 120,737.6 (the few dollars of difference come from how the research table splits downturn and recovery), and a short training job completed and registered a model. Limitations: no CSV upload, training only for 9-asset portfolios, no VIX after 2024-10-11, and a toy `user/job-1` model from a 5,000-step live check may exist locally.

**Research Mode (Milestone 7).** `/research-mode` is a locked, read-only view of the CSC-475 experiment. The settings come from `configs/experiment.yaml` and every number from `results/tables/summary_metrics.csv`; custom portfolios, backtests and trained models are stored elsewhere and cannot change it (the page and API only read). It shows the fixed assets and assumptions, each event's training, validation, downturn and recovery windows, per-strategy downturn metrics (total and annualized return, volatility, Sharpe, maximum drawdown, costs) and recovery results ("Not recovered within 24 months" when the artifact says so). The "What the stored results show" list is computed from those rows at request time (best strategy per event, how many runs lost money, how many regained $100,000), not stored text; with the current artifacts it agrees with the earlier project summary (Minimum Variance best in the GFC and COVID, Mean Variance best in 2022, PPO/DDPG never best, all 15 downturn runs negative). "Recompute" re-runs an event's five strategies with PortfolioLab's engine and the frozen research models and compares the six downturn metrics with the stored table (tolerance 1e-6); it saves nothing. Live check: COVID, 30 of 30 metrics match. A "recovered" strategy means it regained $100,000 at some point (the artifact's definition), so a strategy can be "recovered" yet end below $100,000 (PPO in COVID: $93,707); the page shows both. API: `GET /research/overview`, `POST /research/verify/{event}`. The Overview page also lists the five most recent custom backtests, labeled exploratory. Only the downturn windows are recomputed, not the recovery windows.

## Project Overview

This project compares reinforcement learning (RL) portfolio strategies
with traditional portfolio strategies during major U.S. stock-market
downturns. It measures return, risk, transaction costs, and recovery
using the same nine sector ETFs and comparable test conditions. The
experiment does **not** assume that RL will outperform traditional
strategies.

### Research question

**How do reinforcement learning portfolio strategies compare with
traditional portfolio optimization strategies during major market
downturns?**

### Supporting question

**How do PPO and DDPG differ in their risk and return performance across
different market downturns?**

### Working argument

Reinforcement learning may provide a useful alternative to traditional
portfolio optimization during market downturns, but its performance will
depend on market conditions and the specific RL strategy being used.

## Strategies

The core comparison contains five strategies.

### Traditional

1.  **Equal Weight (1/N):** Each ETF receives approximately 11.11% of
    the portfolio. Rebalance monthly without optimization.
2.  **Mean Variance Optimization (MVO):** Use a rolling 252-trading-day
    window, historical mean daily returns, and the sample covariance
    matrix. Maximize the Sharpe ratio subject to long-only weights and a
    30% maximum per ETF. Rebalance daily when the target allocation
    passes the 2-percentage-point turnover threshold.
3.  **Minimum Variance:** Use the same rolling 252-trading-day window
    and sample covariance matrix to minimize variance. Use long-only
    weights, a 30% maximum per ETF, and the same turnover threshold.

### Reinforcement learning

4.  **PPO (Proximal Policy Optimization)**
5.  **DDPG (Deep Deterministic Policy Gradient)**

PPO and DDPG must use the same environment, assets, observation fields,
constraints, cost assumptions, training/validation/test periods, and
metrics. Use Stable-Baselines3 for the RL algorithms. FinRL can be used
as a reference for portfolio-management examples.

## Assets and Data

The portfolio contains these nine U.S. sector ETFs:

  Ticker   Sector
  -------- ------------------------
  `XLB`    Materials
  `XLE`    Energy
  `XLF`    Financials
  `XLI`    Industrials
  `XLK`    Technology
  `XLP`    Consumer Staples
  `XLU`    Utilities
  `XLV`    Health Care
  `XLY`    Consumer Discretionary

-   **ETF data:** Yahoo Finance through `yfinance`.
-   **Volatility feature:** Historical VIX data from CBOE.
-   Use adjusted close prices to calculate returns. Retain daily volume
    and OHLC data when available.
-   VIX is an RL state feature, **not** a portfolio asset.
-   Save downloaded raw data separately from cleaned data. Record the
    source, retrieval date, date coverage, missing-data treatment, and
    any adjustments.
-   Check that all assets have adequate history and that aligned dates
    contain valid observations.

Run the downloader from the repository root after installing the
dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m src.data.download
```

The script saves individual ETF downloads and the CBOE VIX history under
`data/raw/`, then writes aligned adjusted closes, daily returns, VIX
observations, and a retrieval/coverage manifest under
`data/processed/`. It does not forward-fill missing values. Existing
output files are preserved unless `--force` is supplied. Inspect the
manifest and date coverage before using the data in experiments.

Suggested data layout:

``` text
data/
  raw/
    etf_prices/
    vix/
  processed/
    daily_returns/
    features/
```

Large data files can be excluded from Git if the download and processing
steps are documented.

## Market Downturns and Periods

The planned date ranges are:

  --------------------------------------------------------------------------
  Event          Training       Validation     Downturn test  Recovery
                                                              observation
  -------------- -------------- -------------- -------------- --------------
  Global         2004-04-01 to  2006-10-09 to  2007-10-09 to  2009-03-10 to
  Financial      2006-10-06     2007-10-08     2009-03-09     2011-03-09
  Crisis                                                      

  COVID-19 crash 2016-02-19 to  2019-02-19 to  2020-02-19 to  2020-03-24 to
                 2019-02-18     2020-02-18     2020-03-23     2022-03-23

  2022 downturn  2018-01-03 to  2021-01-04 to  2022-01-03 to  2022-10-13 to
                 2021-01-01     2021-12-31     2022-10-12     2024-10-12
  --------------------------------------------------------------------------

These are planned calendar ranges. Verify exact usable endpoints against
downloaded trading-day data before final runs.

For each event: - Train RL models on three years of historical data. -
Use one year for validation and model selection. - Test on the specified
downturn. - Freeze the selected RL model before the test period. -
Observe recovery for up to 24 months after the trough. - Preserve
chronological order; do not randomly shuffle time-series data.

Because the downturn periods differ in length, do not compare raw total
return alone. Include annualized return and risk measures.

## Shared Portfolio Environment

The PPO and DDPG environment should implement the following locked
assumptions:

-   Starting capital: **\$100,000**
-   Assets: nine sector ETFs
-   Long-only; no borrowing or leverage
-   Maximum weight per ETF: **30%**
-   Rebalancing: daily
-   Transaction cost: **0.10% of dollar value traded**
-   Fully invested portfolio across the nine ETFs unless the design is
    explicitly revised and documented.

### State / observation

The observation contains: - Previous 20 trading days of daily returns
for nine ETFs: `20 × 9 = 180` values. - Current nine portfolio weights:
`9` values. - Current VIX value: `1` value. - 20-trading-day change in
VIX: `1` value.

**Expected observation size: 191 values.**

Every observation must use only information available at the decision
time. Do not use a day's closing return to choose weights that are
assumed to have earned that same day's return.

### Action

The agent outputs continuous actions for the nine desired ETF weights.
Convert actions into valid portfolio weights consistently for PPO and
DDPG. Enforce nonnegative weights, a maximum of 30% per ETF, and weights
summing to 1.0.

### Reward and costs

The planned reward is daily log portfolio return minus transaction
costs. Document the exact formula and units in code. Transaction costs
must reflect the value traded.

A planned turnover measure is:

``` text
turnover = sum(abs(target_weight - current_weight))
```

State whether turnover is one-way or the sum of absolute weight changes.
Apply the 0.10% cost consistently and avoid double-counting costs in
both portfolio accounting and reward.

`src/environment/portfolio_env.py` implements the shared Gymnasium
environment. At the decision for trading day `t`, it exposes the previous
20 ETF return rows, pre-trade portfolio weights, VIX from the previous
trading day's close, and the absolute VIX point change from 20 trading
days earlier. This gives `20 * 9 + 9 + 2 = 191` float32 observation
values. VIX is lagged so no same-day closing information enters the
decision.

PPO and DDPG share a normalized nine-value `[-1, 1]` action space. The
environment maps actions affinely to desired weights in `[0, 1]`, then
projects them onto the nonnegative, fully invested simplex with the 30%
per-ETF cap. It calculates turnover as the sum of absolute
weight changes and transaction costs as `0.001 * pre_trade_portfolio_value
* turnover`. The fee is deducted once before the target weights earn the
current day's return. Reward is exactly
`ln(portfolio_value_after_return_and_cost / portfolio_value_before_trade)`.

Training resets sample a contiguous 252-trading-day window within the
configured training period; observations inside the selected window stay
chronological. Set `episode_length=None` for a full validation or test
period. The environment returns diagnostics including dates, weights,
turnover, dollar traded, costs, and net and gross daily returns.

### Training and normalization

-   Use rolling 252-trading-day episodes within the three-year training
    period.
-   Do not randomly shuffle time-series observations.
-   Fit normalization using training data only.
-   Freeze normalization for validation and test.
-   Evaluate one complete validation period after each configured number
    of training steps. Select the checkpoint with the highest summed
    validation daily log reward (equivalent to validation cumulative
    log net return); validation reward normalization is disabled.
-   Freeze the final model before the downturn test.

Run one event/algorithm training job from the repository root:

```bash
python -m src.experiments.train_agents --event global_financial_crisis --algorithm PPO
python -m src.experiments.train_agents --event global_financial_crisis --algorithm DDPG
```

Training settings, evaluation frequency, and the random seed are in
`configs/experiment.yaml`. Each run writes the selected model and its
paired training-only observation-normalization statistics under
`results/models/<event>/<algorithm>/selected/`, along with the last
training checkpoint, metadata, and evaluation history. The trainer refuses
to write into a non-empty output directory so existing runs are not
silently replaced. Use shorter `--total-timesteps` and `--eval-freq`
values for a smoke run; do not use smoke-run models for final results.

## Traditional Strategy Implementation

### Equal Weight

Assign `1 / 9` of the portfolio to each ETF and rebalance monthly. Apply
the same transaction-cost assumption as the other strategies.

### Mean Variance Optimization

At each rebalance: 1. Use only the previous 252 trading days. 2.
Estimate expected returns using the historical mean daily returns. 3.
Estimate the sample covariance matrix over the same window. 4. Maximize
the Sharpe ratio under the portfolio constraints. 5. Enforce long-only
weights and the 30% per-ETF cap. 6. Apply the 2-percentage-point
turnover threshold and transaction costs.

Document the risk-free-rate assumption and the precise Sharpe-ratio
calculation.

### Minimum Variance

At each rebalance: 1. Use only the previous 252 trading days. 2.
Estimate the sample covariance matrix. 3. Minimize portfolio variance.
4. Enforce long-only weights and the 30% per-ETF cap. 5. Apply the
2-percentage-point turnover threshold and transaction costs.

Use consistent timing and accounting across all strategies.

The policies are implemented in `src/strategies/traditional.py` and plug
into the common backtest engine. Both rolling optimizers use the preceding
252 daily returns, arithmetic mean daily returns, and sample daily
covariance. Minimum Variance minimizes daily portfolio variance.
Mean Variance maximizes the daily Sharpe ratio,
`(mean_daily_return - risk_free_rate_daily) / daily_volatility`; since all
inputs are daily, no annualization factor is applied during optimization.
The MVO policy requires an explicit daily risk-free rate and does not
silently assume one. The final experiment config specifies a zero daily
risk-free rate for MVO and performance Sharpe calculations; this is an
explicit simplifying assumption because no risk-free asset return series is
included in the data. Its constrained Sharpe objective is solved
directly with SLSQP; deterministic feasible starts are used when every
estimated return is below the risk-free rate. The minimum-variance objective
uses PyPortfolioOpt.

### Backtest timing and accounting

The backtest engine makes decisions after the previous trading day's close
and before the current day's return. A strategy receives only returns with
dates strictly earlier than the decision date. The selected allocation
earns the current day's return, avoiding same-day close lookahead.

The portfolio starts with equal weights. Turnover is the **sum of absolute
weight changes**, including both buys and sells; it is not divided by two.
For a trade, dollar turnover is pre-trade portfolio value multiplied by
turnover. Transaction cost is 0.10% of that dollar turnover, deducted once
before applying the day's asset returns. The resulting net portfolio return
already includes costs. The RL log reward is `ln(1 + net_daily_return)`; do
not subtract transaction costs from that reward a second time.

For monthly rebalancing, the strategy is evaluated on the first available
trading day of each month. A turnover threshold suppresses trades below the
configured threshold. The engine records daily portfolio value, gross and
net returns, log reward, turnover, traded value, transaction costs,
allocation before each day's return, and end-of-day drifted weights.

## Preventing Lookahead Bias

A fair backtest must not use future information.

-   A decision on day `t` may use only information available before that
    decision.
-   Do not use the current day's closing return to select weights for
    that same return.
-   Use rolling historical windows for MVO and Minimum Variance.
-   Build RL observations only from information available at decision
    time.
-   Fit scalers and preprocessing on training data only.
-   Do not train or update RL models during the test period.
-   Do not tune strategies using downturn or recovery outcomes.
-   Use the same assets, starting capital, test periods, cost
    assumptions, and metric definitions across all strategies.
-   Add checks for date alignment, missing values, invalid weights, and
    possible future-data leakage.

## Evaluation Metrics

Report results separately for each strategy and each downturn.

### During the downturn

1.  **Total return**
2.  **Annualized return**
3.  **Volatility**
4.  **Sharpe ratio**
5.  **Maximum drawdown**
6.  **Transaction costs**

### During recovery

7.  **Recovery return**
8.  **Recovery time**
9.  **Recovered within 24 months** --- yes/no. If it does not recover in
    the observation window, report **"not recovered within 24 months."**

Define the recovery reference peak consistently. Use the same
definitions for every strategy and save daily portfolio values and
returns so summary metrics can be reproduced.

## Planned Software

-   `pandas` and `numpy` --- data processing and numerical calculations
-   `yfinance` --- ETF market data
-   `PyPortfolioOpt` --- MVO and Minimum Variance
-   `gymnasium` --- RL environment interface
-   `stable-baselines3` --- PPO and DDPG
-   `finrl` --- reference examples where useful
-   `matplotlib` --- plots and charts
-   `pytest` --- tests, if adopted

Use a Python virtual environment. Once the environment is working,
record tested package versions in `requirements.txt`. Check
compatibility between packages and install incrementally.

## Suggested Repository Structure

``` text
portfolio-downturn-study/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   └── experiment.yaml
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── exploratory_analysis.ipynb
├── src/
│   ├── data/
│   │   ├── download.py
│   │   ├── clean.py
│   │   └── features.py
│   ├── environment/
│   │   └── portfolio_env.py
│   ├── strategies/
│   │   ├── equal_weight.py
│   │   ├── mean_variance.py
│   │   ├── minimum_variance.py
│   │   ├── ppo_agent.py
│   │   └── ddpg_agent.py
│   ├── backtest/
│   │   ├── engine.py
│   │   └── metrics.py
│   ├── experiments/
│   │   └── run_experiment.py
│   └── visualization/
│       └── plots.py
├── tests/
│   ├── test_data.py
│   ├── test_portfolio_env.py
│   ├── test_strategies.py
│   └── test_metrics.py
├── results/
│   ├── tables/
│   └── figures/
└── reports/
```

This is a suggested layout. Create it gradually and keep modules small
and testable.

## Recommended Build Order

### 1. Set up the project

Create the repository, virtual environment, dependency file,
`.gitignore`, folder structure, and `configs/experiment.yaml`.

### 2. Download and inspect data

Download ETF and VIX data. Check date coverage, duplicates, missing
values, adjusted prices, and alignment. Create daily returns without
future leakage.

### 3. Build the backtest engine

Define when observations are known, when trades occur, and which returns
the portfolio earns. Implement portfolio accounting and transaction
costs. Start with Equal Weight and verify it on a small sample.

### 4. Implement traditional strategies

Implement MVO and Minimum Variance with rolling windows, constraints,
turnover thresholds, and transaction costs. Test them on a short period
before running full experiments.

### 5. Implement and test the RL environment

Implement the 191-value state, continuous action conversion, portfolio
constraints, costs, and reward. Test with random actions before
training.

### 6. Train PPO and DDPG

Use the same splits and environment. Fit normalization on training data
only. Use validation for model selection. Save configuration, random
seeds, logs, and models. Freeze the models before testing.

### 7. Run all experiments

Run each of the five strategies on all three downturns. Save daily
portfolio values, returns, weights, turnover, and transaction costs.

### 8. Calculate metrics and make charts

Create comparison tables and plots. Check that all strategies use the
same test dates and metric definitions.

### 9. Document findings

Record implementation choices and any deviations from the plan. Use the
observed results to answer the research questions rather than deciding
in advance which strategy should win.

Run the complete experiment from the repository root after downloading
data:

```bash
python -m src.experiments.run_experiment
```

By default this trains validation-selected PPO and DDPG models for each
event using the configured full timestep budget, then evaluates all five
strategies over each downturn and its following recovery period. It writes
the summary/recovery tables, daily strategy files, comparison charts,
configuration snapshot, run metadata, and findings report. Use
`--skip-training` only when all six selected model checkpoints already
exist. For a development smoke run, pass small `--total-timesteps` and
`--eval-freq` values; smoke results are not final research results.

## Reproducibility

For each run, save: - Experiment configuration and date ranges - Package
versions and random seeds - Training and validation settings - Portfolio
weights through time - Daily portfolio values and returns - Turnover and
transaction costs - Summary metrics - Tables and figures

Example output paths:

``` text
results/tables/summary_metrics.csv
results/tables/recovery_metrics.csv
results/figures/portfolio_values_covid_crash.png
results/figures/drawdowns_all_events.png
```

Keep a record of earlier configurations instead of silently overwriting
results.

The downturn annualized return compounds the observed net daily returns
over 252 trading days per year; annualized volatility and Sharpe use the
same 252-day convention. Sharpe uses the configured zero daily risk-free
rate. Maximum drawdown includes starting capital as the pre-first-day
wealth peak. Recovery return is measured from the strategy's lowest daily
closing value during the downturn through the final available recovery
date. Recovery is achieved when portfolio value first reaches the starting
capital reference after that trough; recovery time is reported in trading
days. Configured end dates are inclusive, and reports record the last
actual trading date in each period.

## Finalized Experiment Choices

These decisions are implemented in `configs/experiment.yaml` and
documented above: prior-close decision timing, capped-simplex RL action
projection, fee accounting and reward formula, zero daily risk-free-rate
assumption, initial-capital recovery reference, trading-day recovery time,
inclusive configured date ranges with actual aligned endpoints, and
validation-selected hyperparameters. If a practical issue requires
changing one, document why and update the configuration and this README.

## Scope and Expected Artifact

Decision-Focused Learning is outside the core experiment because of the
project's scope. It may be discussed as future work but should not be
added to the core comparison unless the project plan changes.

The final artifact should include a reproducible backtesting pipeline,
saved results, comparison tables, and charts for the Global Financial
Crisis, COVID-19 crash, and 2022 downturn, including the following
recovery periods. The final conclusion must follow from the experiment's
results. RL should not be declared better simply because it is more
complex or because prior research found positive results.

# PortfolioLab — Project Specification for Claude Code

## 1. Project Overview

Build a polished, locally runnable portfolio research and backtesting platform called **PortfolioLab**. The application should let users search for individual stocks and ETFs, create custom portfolios, select portfolio strategies, backtest them on historical market data, compare performance and risk, and export results.

This is a serious software engineering and quantitative finance project, not just a static dashboard. The interface should look polished enough to demonstrate to computer science faculty, quantitative researchers, data analysts, and finance professionals. Financial correctness, transparency, reproducibility, and usability are more important than decorative effects.

The application is for research, education, and historical simulation. It must **not** place real trades or present backtested results as guaranteed future performance.

### Main goals

1. Run locally on the developer's computer.
2. Provide a modern web frontend and a Python API backend.
3. Search for and add individual stocks and ETFs.
4. Let users configure portfolios, constraints, strategies, and test periods.
5. Compare three traditional strategies and two reinforcement learning strategies.
6. Show performance, risk, allocation, trading costs, and recovery through interactive charts and tables.
7. Support reproducible research runs and a dedicated mode for the existing CSC-475 research experiment.
8. Be designed so the application can be deployed later, without requiring cloud deployment for local use.

## 2. How to Work on This Project

Act as a careful senior full-stack engineer and quantitative developer. Work incrementally. Do not attempt to generate the entire application in one giant step.

### First steps — required

1. Inspect the existing project directory before making changes. Read the existing README, source files, dependency files, tests, and any existing experiment code.
2. Preserve working code and existing project decisions. Do not overwrite or remove existing work without explaining why.
3. Identify the current operating system, Python version, Node.js version, package manager, and available files where possible.
4. Write a short implementation plan and identify technical risks before coding.
5. Build one working vertical slice at a time and run it before moving on.
6. Add tests as financial calculations and API endpoints are implemented.
7. Keep the project runnable at each major milestone.
8. Do not claim a feature works unless you have run it or clearly label it as untested.
9. If something is ambiguous, make a reasonable, documented default and continue instead of stopping for a long questionnaire. Ask only when a decision materially affects correctness or requires the user's choice.
10. Do not invent market data, API responses, financial results, or test results.

### Suggested milestones

- Milestone 1: inspect repository, write plan, scaffold frontend and backend, confirm both run locally.
- Milestone 2: asset search and portfolio builder.
- Milestone 3: data provider interface and historical data retrieval/cache.
- Milestone 4: validated backtesting engine and traditional strategies.
- Milestone 5: metrics, interactive visualizations, and exports.
- Milestone 6: RL environment and PPO/DDPG integration.
- Milestone 7: research reproduction mode, tests, documentation, and polish.

At each milestone, summarize changed files, commands run, test outcomes, known limitations, and the next step.

## 3. Proposed Technology Stack

Use this stack unless repository inspection reveals a strong reason to adapt it:

### Frontend
- Next.js with React and TypeScript
- Tailwind CSS
- A consistent accessible component system (shadcn/ui is acceptable if installed and compatible)
- Plotly.js or react-plotly.js for interactive charts
- TanStack Table or a comparable table component if useful
- Zod for client-side form validation where appropriate

### Backend
- Python with FastAPI
- Pydantic models for request and response validation
- pandas and NumPy for financial data processing
- PyPortfolioOpt for traditional portfolio optimization, if compatible with the environment
- Gymnasium for the RL environment
- Stable-Baselines3 for PPO and DDPG
- pytest for testing
- SQLite initially for local persistence; use SQLAlchemy if persistence is needed. Keep the design easy to migrate to PostgreSQL later.

### Data and configuration
- `yfinance` may be used as an initial historical market data adapter, but do not treat it as a guaranteed or official production data API. Handle outages, rate limits, missing values, symbol changes, and provider limitations clearly.
- Use a provider interface so another licensed market data provider can be added later.
- Do not require a paid API key for the application to start.
- Store optional secrets in `.env`, read only on the backend, and provide `.env.example` with placeholder names. Never expose API secrets to the frontend or commit a real `.env` file.
- Cache downloaded historical data locally, record provider and retrieval timestamp, and provide a way to refresh data deliberately.
- Do not invent or silently fill missing market data. Show data availability and gaps to users.

## 4. User Experience and Visual Design

Create a refined, professional quantitative-finance interface. Avoid a generic template look, excessive gradients, excessive rounded cards, distracting animations, or decorative charts that do not convey information.

### Design direction
- Dark charcoal/navy base with restrained accent colors and strong contrast.
- Clean typography, consistent spacing, concise labels, and information-dense layouts.
- Use color consistently for positive and negative values, but never communicate meaning through color alone.
- Support readable tables, loading states, empty states, error states, and responsive layout.
- Include units, dates, frequency, benchmark, and assumptions near every chart/table where needed.
- Make all major controls keyboard accessible and label inputs clearly.
- Use tooltips or short help text to explain finance metrics and assumptions.

### Main navigation
1. **Overview**
2. **Portfolio Builder**
3. **Strategy Lab**
4. **Backtests**
5. **Risk Analytics**
6. **Research Mode**
7. **Settings / Data Status**

### Overview page
Show recent backtests, saved portfolios, latest local data status, and an entry point to start a new analysis. Do not show fabricated account balances or live investment positions. This is a research simulator, not a brokerage account.

### Portfolio Builder page
- Search by ticker or company name.
- Show ticker, company name, exchange/asset type when provider data supports them.
- Add/remove holdings and show the current asset list.
- Allow users to set an initial capital amount.
- Provide configurable target weights when relevant.
- Show whether weights sum to 100% and whether all constraints are feasible.
- Support long-only portfolios in the first release.
- Support CSV upload of ticker symbols or holdings only if it can be validated safely.
- Clearly distinguish supported symbols from symbols for which historical data is unavailable.
- Let users save and reopen configurations locally.

### Strategy Lab page
Allow users to select one or multiple strategies:
- Equal Weight (1/N)
- Mean Variance Optimization (MVO)
- Minimum Variance
- PPO
- DDPG

Explain each strategy briefly. Let users configure supported parameters, but do not expose misleading controls that are not implemented.

### Backtests page
- Choose a start and end date or a named historical event.
- Choose a benchmark, such as SPY, when sufficient data is available.
- Configure initial capital, rebalancing frequency, transaction cost, and portfolio constraints.
- Show a pre-run summary of the chosen assets, data coverage, strategies, and assumptions.
- Require a deliberate action to start a simulation.
- Display progress for longer runs, with the ability to report failure and retry.
- Do not rerun expensive calculations on every UI change.
- Make it clear whether data is cached, freshly downloaded, or unavailable.

### Risk Analytics page
Display clearly defined metrics and charts, including:
- Total return
- Annualized return
- Annualized volatility
- Sharpe ratio
- Maximum drawdown
- Transaction costs
- Turnover
- Recovery return and recovery time where relevant
- Optional downside metrics such as Sortino ratio, VaR, and CVaR only when their calculation assumptions are implemented and documented

### Research Mode page
Provide a locked, reproducible configuration for the CSC-475 project described in Section 10. Users should be able to run this baseline separately from custom experiments. Custom portfolios must never overwrite the research configuration or its outputs.

### Settings / Data Status page
Show the data provider, most recent download, cached date range, unavailable symbols, and any provider errors. Never show secret API keys. Provide cache refresh and clear-cache controls with confirmation where appropriate.

## 5. Core User Workflow

1. User opens PortfolioLab locally in a browser.
2. User searches for tickers by symbol or company name.
3. User adds multiple assets to a portfolio.
4. User chooses a starting balance, dates, benchmark, transaction costs, rebalancing frequency, and portfolio constraints.
5. User chooses strategies to compare.
6. The system validates the configuration, asset history, date coverage, and feasibility of constraints.
7. The user clicks **Run Backtest**.
8. The backend retrieves cached data or downloads missing data, then executes each selected strategy under the same configured assumptions.
9. The UI displays portfolio value over time, returns, drawdowns, allocations, costs, and metrics.
10. The user can compare strategies and export the results and configuration.

If the selected strategy includes PPO or DDPG, the UI must explain whether it will use a compatible trained model or start a new training run. Do not quietly use a model trained for a different asset universe, observation shape, or action space.

## 6. Asset Search and Market Data

### Requirements
- Users can search for individual stocks and ETFs, not only the nine research ETFs.
- Design an `AssetDataProvider` abstraction, so symbol search, asset metadata, and historical prices are separate operations.
- Implement a clearly documented initial provider adapter. A symbol search provider may differ from the historical price provider.
- Normalize ticker symbols and validate input; reject blank or malformed symbols and handle delisted or unsupported tickers gracefully.
- Retrieve adjusted historical prices for total-return-oriented analyses where the data provider supports them. Document the exact price field used and avoid double-counting distributions.
- Align asset price series by date using a documented rule. Do not forward-fill prices across long gaps without explicit justification.
- Display available history for each asset and reject or explain date ranges where data coverage is insufficient.
- Include provider attribution and a short disclaimer that data may be delayed, incomplete, adjusted, or revised.
- Never imply data is real-time unless the provider and plan genuinely support it.
- Do not put a provider API key in frontend code.
- Provide local caching and a deliberate refresh action.

### Search implementation
Search should be debounced and should show loading, no-results, error, and success states. If the initial provider does not offer reliable symbol search, use a provider with a documented search endpoint or a locally maintained supported-symbol index; explain the source and update process. Do not fake search results.

## 7. Backtesting Engine and Financial Correctness

Create a reusable backtesting engine used by all strategies. Keep strategy decisions separate from portfolio accounting.

### Shared behavior
- Start from a user-configured initial capital amount.
- Support long-only positions in the first version.
- Enforce portfolio weights that sum to 1 and respect per-asset constraints.
- Track portfolio value, holdings/weights, returns, turnover, trades, and transaction costs at each step.
- Apply transaction costs consistently to all strategies.
- Define precisely whether costs are calculated on one-way or two-way turnover, then use that convention everywhere and document it.
- Account for the timing of weight decisions and subsequent returns. A decision at time `t` must not earn the return that occurred before that decision.
- Prevent lookahead bias in data preparation, feature scaling, optimization, training, validation, and testing.
- Avoid survivorship-bias claims unless point-in-time universe data is actually available. Custom current ticker selection tested historically should be labeled as a selected-universe historical backtest, not a complete historical-universe study.
- Document rebalancing timing, execution-price assumptions, treatment of cash, missing observations, and corporate actions.
- Use the same rules when comparing strategies.
- Handle infeasible optimization problems with a clear error or documented fallback. Do not silently change the objective or constraints.
- Use deterministic settings where possible and record seeds for stochastic methods.
- Do not compare strategies using different date ranges without labeling that difference.

### Portfolio constraints
- Long only in the first release.
- Maximum per-asset weight is configurable.
- If minimum/maximum constraints make a fully invested portfolio impossible, show a validation error before running.
- If users choose a weight cap that cannot accommodate all selected assets, explain why.
- Do not silently normalize invalid weights in a way that changes user intent.

### Traditional strategies
**Equal Weight**
- Allocate equal weight to each selected asset.
- Respect configured rebalancing frequency.
- Clearly define the initial allocation and subsequent rebalance dates.

**Mean Variance Optimization**
- Use a rolling lookback window based only on historical data available before the rebalance decision.
- Use a documented expected-return estimator and covariance estimator.
- Support a documented objective, initially maximum Sharpe ratio if constraints allow.
- Expose a configurable lookback period.
- Include long-only and per-asset maximum constraints.
- Handle unstable or infeasible solutions transparently.

**Minimum Variance**
- Use a rolling historical covariance estimate based only on prior data.
- Minimize estimated portfolio variance under the same long-only and per-asset constraints.
- Share the same lookback and rebalancing conventions as MVO where applicable.

### Transaction costs
- Default research assumption: 0.10% of dollar value traded.
- Define turnover and cost calculation explicitly. A common convention is `turnover = sum(abs(target_weight - current_weight))`; the implementation must specify whether the cost rate is applied to this one-way measure or another clearly defined traded-notional convention.
- Record total and daily costs.
- Explain that this is a simplified cost model and does not fully model spreads, market impact, taxes, or liquidity.

## 8. Reinforcement Learning

Use Stable-Baselines3 with Gymnasium where compatible. Keep RL training and inference separate from the UI request handlers.

### Shared RL environment
For the CSC-475 baseline:
- Nine research ETFs.
- Initial capital of $100,000.
- Long-only allocation.
- Maximum 30% weight per ETF.
- Daily rebalancing.
- Observation: previous 20 trading days of daily returns for nine assets (180 values), current nine portfolio weights, current VIX value, and 20-trading-day VIX change (191 values total).
- Action: target allocation across nine assets, transformed into valid nonnegative weights that sum to one while respecting the 30% cap.
- Reward: daily log portfolio return after transaction costs.
- Transaction cost assumption: 0.10% of dollar value traded.
- Use VIX as a feature, never as a portfolio asset.
- Normalize using training data only and freeze normalization for validation/test.
- Training episodes use rolling 252-trading-day windows inside the specified three-year training period, without random shuffling of temporal order.
- Validation period: one year.
- Freeze model weights before test.
- Evaluate the downturn test period without further learning.
- PPO and DDPG use the same information, environment, dates, constraints, cost assumptions, and evaluation rules.

### Custom portfolios and RL compatibility
The nine-ETF research model cannot be assumed to support arbitrary stocks. For custom portfolios:
- Check the model's recorded asset universe, observation dimensions, action dimensions, feature definitions, scaling parameters, and training period.
- If the universe differs, require a new training run or a specifically compatible model.
- Clearly show estimated training time or progress if possible.
- Save model metadata, random seed, asset universe, training dates, environment configuration, and normalization parameters.
- Never label a model as validated for a portfolio or date range it was not trained and evaluated for.
- Avoid starting a long training job automatically just because the user changes a form control. Training should begin only after explicit confirmation.
- Support cancellation or clear status for long-running jobs where feasible.
- Use a task/background-job pattern for long training jobs rather than blocking the main API request.
- Explain that RL results can vary with random seeds and that one run is not conclusive.

### Action-space design warning
The 30% cap and fully invested constraint must be implemented mathematically correctly. A plain softmax guarantees weights sum to one but does not guarantee each weight is at most 30%. Implement and test a valid capped-simplex projection or another method that enforces both constraints. Do not merely clip softmax weights and leave the portfolio weights summing to less than one.

## 9. Metrics and Visualizations

Implement metrics with tests and clear definitions.

### Required metrics
- Total return: ending portfolio value divided by starting value minus one.
- Annualized return: use a documented convention appropriate to the number of trading observations and date span.
- Annualized volatility: standard deviation of periodic returns multiplied by the square root of the chosen annualization factor.
- Sharpe ratio: use documented return frequency, annualization factor, and risk-free-rate assumption. Research baseline risk-free rate is 0% daily, as specified in Section 10.
- Maximum drawdown: minimum percentage decline from a prior running peak.
- Transaction costs: total currency cost over the period.
- Turnover: calculate using the documented convention.
- Recovery time: elapsed trading days from a trough or end-of-downturn reference point to regain the relevant previous peak, with the reference definition stated explicitly.
- Recovery return: return during the specified recovery period.
- Recovered within window: boolean or clearly labeled status.

### Useful charts
- Cumulative portfolio value with benchmark.
- Drawdown over time.
- Strategy performance comparison.
- Risk/return scatter plot.
- Allocation weights over time.
- Transaction costs or turnover over time.
- Recovery trajectory after a downturn.

Charts must use real calculation outputs, correct date alignment, labels, legends, and hover details. Do not fabricate or hardcode result values. Use consistent metric definitions across all strategies. Include a table view for accessibility and export.

## 10. CSC-475 Research Reproduction Mode

This application supports the research project **“Reinforcement Learning vs. Traditional Portfolio Strategies During Market Downturns.”**

Research question:
> How do reinforcement learning portfolio strategies compare with traditional portfolio optimization strategies during major market downturns?

Supporting question:
> How do PPO and DDPG differ in their risk and return performance across different market downturns?

Argument/hypothesis:
> Reinforcement learning may provide a useful alternative to traditional portfolio optimization during market downturns, but its performance will depend on market conditions and the specific RL strategy being used.

Do not assume RL outperforms traditional methods. The application should report actual results.

### Research assets
- XLB — Materials
- XLE — Energy
- XLF — Financials
- XLI — Industrials
- XLK — Technology
- XLP — Consumer Staples
- XLU — Utilities
- XLV — Health Care
- XLY — Consumer Discretionary

Historical VIX is a model input, not a portfolio asset.

### Five strategies
1. Equal Weight (1/N)
2. Mean Variance Optimization
3. Minimum Variance
4. PPO
5. DDPG

### Shared baseline
- Starting capital: $100,000.
- Transaction cost: 0.10% of dollar value traded, using the documented turnover convention.
- All strategies use the same research assets, event test dates, cost assumptions, and evaluation metrics.
- All portfolios are long-only and fully invested.
- Maximum weight: 30% per asset for optimization and RL strategies. Equal Weight naturally has approximately 11.11% per asset.
- Equal Weight rebalances monthly.
- MVO uses a rolling 252-trading-day lookback, historical mean daily returns, sample covariance, maximum Sharpe objective, long-only constraints, and a 30% per-asset cap. A 2-percentage-point turnover threshold was specified to avoid tiny trades; preserve and document the exact implementation if existing experiment code clarifies how this threshold works.
- Minimum Variance uses the same 252-trading-day lookback and sample covariance, minimizes variance, is long-only, and has a 30% per-asset cap. Apply the same 2-percentage-point turnover threshold where the existing experiment design specifies it.
- Optimization strategies rebalance daily.
- RL observations use the prior 20 trading days of asset returns (180 features), current nine portfolio weights, current VIX and the 20-trading-day VIX change (191 features total).
- RL actions select target portfolio weights.
- RL reward is daily log return minus transaction costs.
- Daily risk-free-rate assumption for Sharpe calculations: 0%.
- Normalize features using training data only; freeze scaling for validation and test.
- Train with rolling 252-trading-day episodes inside the three-year training window; do not shuffle temporal observations randomly.
- One-year validation period.
- Freeze the RL model before the test window.

### Historical event periods and windows
Use these as the starting configuration, but validate exact trading-day endpoints against the available dataset. Do not silently shift dates or invent observations.

**Global Financial Crisis**
- Training: 2004-04-01 through 2006-10-06
- Validation: 2006-10-09 through 2007-10-08
- Downturn test: 2007-10-09 through 2009-03-09
- Recovery observation: 2009-03-10 through 2011-03-09
- Approximate S&P 500 peak-to-trough decline: 56.8–57%

**COVID-19 crash**
- Training: 2016-02-19 through 2019-02-18
- Validation: 2019-02-19 through 2020-02-18
- Downturn test: 2020-02-19 through 2020-03-23
- Recovery observation: 2020-03-24 through 2022-03-23
- Approximate S&P 500 peak-to-trough decline: 34%

**2022 downturn**
- Training: 2018-01-03 through 2021-01-01
- Validation: 2021-01-04 through 2021-12-31
- Downturn test: 2022-01-03 through 2022-10-12
- Recovery observation: 2022-10-13 through 2024-10-12
- Approximate S&P 500 peak-to-trough decline: 25.4%

The exact date ranges and reported returns should be checked against the experiment's actual source data and existing code. Some end dates fall on non-trading days; use documented trading-day inclusion rules.

### Research metrics
During each downturn:
- Total return
- Annualized return
- Volatility
- Sharpe ratio
- Maximum drawdown
- Transaction costs

During recovery:
- Recovery return
- Recovery time
- Whether the portfolio recovered within 24 months
- If not recovered, display “Not recovered within 24 months.”

Do not compare only raw total returns across downturns of different lengths. Include annualized measures and risk metrics.

### Current reported findings to verify against actual artifacts
A previous project summary states:
- No strategy was consistently best across all three downturns.
- PPO and DDPG did not have the strongest downturn total return in any event.
- Minimum Variance performed best by downturn total return and Sharpe ratio in the Global Financial Crisis (return −41.0%, Sharpe −1.297) and COVID-19 crash (return −29.7%, Sharpe −4.788).
- Mean Variance performed best during the 2022 downturn (return −2.6%, Sharpe −0.083).
- All 15 strategy/event combinations lost money during their downturn tests.
- During the 24-month GFC recovery window, none returned to the initial $100,000.
- During the COVID recovery, all five crossed $100,000, although PPO ended the observation period below $100,000.
- During the 2022 recovery, three traditional strategies crossed $100,000, while PPO and DDPG did not.
- The experiment used one random seed per RL model, one validation episode per checkpoint, nine sector ETFs, and a 0% daily risk-free-rate assumption. Rankings should be treated as exploratory, not statistically conclusive.

These findings are not to be hardcoded as application output. Verify them against the user's actual experiment results/code if available. The application must compute displayed numbers from its own result artifacts. If source outputs are unavailable, label the findings as prior reported results awaiting verification.

### Separation between research and custom mode
Research Mode must preserve the fixed experiment settings. Custom portfolios may use other tickers and dates, but their results must be stored separately and labeled exploratory. Never overwrite research results with custom results.

## 11. API Design

Use versioned, validated FastAPI endpoints. Adapt names as needed, but keep responsibilities clear.

Potential endpoints:
- `GET /api/v1/health` — application health
- `GET /api/v1/assets/search?q=...` — search supported symbols
- `GET /api/v1/assets/{symbol}` — metadata and data availability
- `POST /api/v1/portfolios` — create/save a portfolio
- `GET /api/v1/portfolios` — list saved local portfolios
- `GET /api/v1/portfolios/{id}` — retrieve a portfolio
- `PUT /api/v1/portfolios/{id}` — update a portfolio
- `POST /api/v1/backtests` — validate and start a backtest
- `GET /api/v1/backtests/{id}` — status and summary
- `GET /api/v1/backtests/{id}/results` — metrics and time series
- `GET /api/v1/backtests/{id}/export.csv` — CSV export
- `POST /api/v1/training-jobs` — explicitly start an RL training job
- `GET /api/v1/training-jobs/{id}` — training status
- `POST /api/v1/training-jobs/{id}/cancel` — cancel if supported
- `GET /api/v1/data/status` — provider/cache status

Use proper HTTP status codes, useful error messages, request timeouts, and validation. Do not allow arbitrary Python execution or shell commands through API inputs. Keep the API safe for localhost use and avoid binding publicly by default.

## 12. Local Development and Run Instructions

The project must run locally without cloud services.

Provide:
- `README.md` with setup and run instructions.
- `.env.example` with placeholders only.
- `.gitignore` covering `.env`, Python environments, Node dependencies, caches, downloaded data, model checkpoints, local databases, and generated results where appropriate.
- Pinned or bounded dependencies with documented Python and Node versions.
- One command to start the backend and one command to start the frontend, or a documented root script to start both.
- A clear explanation of which features require internet access, which API keys are optional, and which calculations run locally.
- Clear troubleshooting steps for dependency installation, data-provider failures, and port conflicts.
- A health endpoint and a visible frontend connection status.
- No mandatory paid API key for the app to start. If a provider key is needed for a specific feature, show a clear configuration message and retain a usable alternative where possible.

Keep API keys server-side. Do not commit secrets, real user data, or large model files.

## 13. Testing and Quality Standards

Write unit and integration tests as features are built.

### Minimum financial tests
- Weights sum to 1 within tolerance.
- Maximum asset weights are respected.
- Infeasible constraints are rejected.
- Transaction costs are calculated correctly.
- Portfolio value accounting is correct on a small hand-calculated example.
- Return calculations handle edge cases.
- Maximum drawdown is correct for a known series.
- Annualization is consistent and documented.
- Sharpe ratio uses the configured risk-free-rate assumption.
- Recovery detection behaves correctly for recovered and unrecovered portfolios.
- Date alignment and missing-data handling are tested.
- Strategy decisions do not access future returns.
- RL state dimension is 191 for the nine-ETF research configuration.
- RL action projection obeys the long-only, fully invested, 30%-maximum constraints.
- Train/validation/test separation is enforced.
- Custom runs cannot alter the locked research configuration.

### Minimum frontend tests
- Search loading, success, no-results, and error states.
- Portfolio add/remove behavior.
- Invalid configuration displays helpful errors.
- Backtest progress, success, and failure states.
- Charts and tables display actual API outputs.
- CSV export is accessible and contains expected columns.

### General quality
- Use type hints and clear module boundaries.
- Avoid one giant file or duplicated calculations.
- Add comments where financial assumptions need explanation, not for every line.
- Log errors without exposing secrets.
- Do not silently swallow exceptions.
- Use formatting and linting tools if practical.
- Do not fabricate sample results in production views. If demo data is added, label it clearly as simulated example data.

## 14. Security, Privacy, and Limitations

- Keep the server bound to localhost by default during development.
- Never send API secrets to the browser.
- Do not include real trading or brokerage order execution.
- Validate all incoming symbols, dates, weights, and numeric inputs.
- Avoid arbitrary file path access through user-supplied inputs.
- Treat uploaded CSVs as untrusted input and validate file size and structure.
- Make clear that results are historical simulations and not investment advice.
- Explain that backtests omit or simplify some real-world effects, such as market impact, taxes, liquidity constraints, and execution slippage unless explicitly modeled.
- Respect the selected data provider's licensing and usage terms.
- Avoid claiming institutional-grade accuracy without adequate testing and validated data.

## 15. Definition of Done

The first complete local version is ready when:

1. The frontend and backend both start locally from documented commands.
2. The user can search for supported individual stocks and ETFs using real provider data or a clearly sourced symbol index.
3. The user can create a custom portfolio with validated constraints.
4. The user can run Equal Weight, Mean Variance, and Minimum Variance backtests on selected assets.
5. The user can view actual calculated returns, risk metrics, allocations, transaction costs, and interactive charts.
6. The user can compare strategies under the same assumptions.
7. CSV export works.
8. PPO and DDPG can run for the fixed research universe, with compatibility checks and documented training/test separation.
9. Research Mode reproduces the defined nine-ETF experiment without being changed by custom portfolios.
10. Unit tests cover the main financial calculations and constraints.
11. The README explains setup, API configuration, local operation, limitations, and troubleshooting.
12. No displayed result is hardcoded or falsely represented as a completed calculation.

## 16. Final Instruction to Claude

Begin by inspecting the existing project folder and reading its current README and source code. Do not immediately generate every file.

First, report:
- What files and code already exist.
- Which runtime versions are available.
- What can be reused.
- Any important technical risks or conflicting existing decisions.
- A phased implementation plan.

Then start Milestone 1 and implement a small working slice. Verify it by running the relevant commands and tests before moving forward.

Prioritize correctness, maintainability, clear assumptions, and a professional user experience. The goal is a genuinely useful interactive portfolio research platform that runs locally and can later be deployed—not just a visually impressive mockup.

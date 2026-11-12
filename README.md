
# AI-Powered Hedging

## Training Agents on Synthetic Financial Markets to Replicate Derivatives and Optimize Dynamic Portfolios Using Deep Reinforcement Learning

An experimental Deep Reinforcement Learning framework for financial prediction, derivative replication and dynamic portfolio hedging, combining real market data, synthetic financial simulations and agent-based decision-making.

## Project Overview

### Objective: Intelligent Financial Decision-Making Through Deep Reinforcement Learning

The project explores how **Deep Reinforcement Learning (DRL)** can be applied to financial prediction, derivative replication and dynamic portfolio optimization in complex and uncertain market environments.

Reinforcement Learning is based on the continuous interaction between an agent and its environment. The agent learns a decision-making policy by observing market conditions, selecting actions and evaluating their consequences through rewards or penalties.

By integrating neural networks, Deep Reinforcement Learning enables agents to process complex financial information and progressively improve their decisions.

The central objective is to investigate whether agents trained in **synthetic financial environments** can learn effective strategies and adapt them to real market conditions and increasingly complex portfolio management problems.

The project addresses three complementary challenges:

- **Financial prediction:** training an RL agent on synthetic financial series and evaluating its ability to generalize to real market data.
- **Derivative replication:** constructing a dynamically rebalanced portfolio capable of replicating the theoretical value of a European call option.
- **Dynamic portfolio hedging:** developing an intelligent agent capable of processing multiple financial signals and adapting portfolio allocations to changing market conditions.

The experimental methodology progresses from predictive analysis to numerical derivative replication and finally to autonomous portfolio management in a stochastic, multidimensional environment.

## Experimental Architecture and Methodology

The project is organized into three successive experimental stages.

### Part 1 — Predictive Finance with Reinforcement Learning on Real and Synthetic Data

#### 1. Real Financial Data and Trading Environment

A custom `Finance` environment is developed using historical market data retrieved through **Yahoo Finance (`yfinance`)**.

The analysis considers 12 financial assets belonging to four investment categories:

| Category | Financial Assets |
|---|---|
| Established US companies | AAPL, MSFT, JNJ, PG |
| Technology and innovation | TSLA, NVDA, ARKK |
| Commodity and defensive assets | IAU, USO, SLV |
| Broad-market ETFs | SPY, VTI |

Historical observations are processed through:

- Calculation of logarithmic returns.
- Weekly aggregation to reduce daily market noise.
- Z-score normalization to improve numerical stability.

The resulting financial environment provides the historical information required for agent training and evaluation.

#### 2. Synthetic Financial Data Generation

A major challenge in financial Reinforcement Learning is the limited availability of historical observations relative to the amount of data required for robust agent training.

Training exclusively on a single historical series can lead to **overfitting**, reducing the agent's ability to generalize to different market conditions.

To address this limitation, the project generates additional financial time series using **Monte Carlo simulations based on the Vasicek stochastic model (1977)**.

The simulation framework controls three fundamental parameters:

- **Theta (θ):** long-term mean level.
- **Kappa (κ):** speed of mean reversion.
- **Sigma (σ):** volatility of the stochastic process.

By adjusting these parameters, the system generates alternative financial trajectories with different statistical characteristics and levels of uncertainty.

The purpose is to create more varied training environments while preserving meaningful relationships with observed financial market behavior.

#### 3. Statistical Validation of Synthetic Data

Before using the generated series for agent training, the project evaluates their similarity to real financial returns.

The validation process includes:

- **Kolmogorov-Smirnov (KS) test:** comparison of synthetic and real return distributions.
- **Histogram analysis:** visual assessment of distributional similarities.
- **Cumulative Distribution Function (CDF) comparison:** examination of differences between empirical return distributions.

The results show reasonable distributional similarities for most assets, while **ARKK and SLV** exhibit differences highlighted by the KS test.

These findings demonstrate the importance of validating synthetic financial data before using them to train predictive models.

#### 4. Advantage Actor-Critic Agent

The predictive model uses an **Advantage Actor-Critic (A2C)** architecture.

A2C combines two neural-network components:

- **Actor:** selects actions according to the learned policy.
- **Critic:** estimates the value of the current state and evaluates the quality of the agent's decisions.

The implementation incorporates:

- Entropy regularization to encourage exploration.
- Reward normalization.
- Advantage-based gradient calculations.
- A simplified form of Generalized Advantage Estimation (GAE).

The agent is trained on synthetic financial series and subsequently evaluated on real market observations.

#### 5. Training, Testing and Performance Evaluation

The trained agent is tested across **12 real financial assets**, using five evaluation episodes per asset.

Two principal metrics are considered:

- **Reward:** measures the quality of the learned decision-making policy.
- **Accuracy:** measures the proportion of actions consistent with observed market direction.

Reward is treated as the primary evaluation metric, since classification accuracy alone does not fully capture the effectiveness of a Reinforcement Learning strategy.

The experimental results reveal differences across assets:

- **Higher average rewards:** NVDA (49.8), VTI (24.0), JNJ (22.0) and AAPL (19.6).
- **Intermediate results:** USO (17.4) and SLV (17.2).
- **Lower rewards:** MSFT, PG, IAU and TSLA.
- **Notable underperformance:** ARKK, with an average reward of 1.0 and accuracy of approximately 17%.

ARKK also exhibits discrepancies between real and synthetic return distributions, reinforcing the relevance of statistical validation for transfer from simulated to real financial environments.

This first experimental stage investigates the ability of an RL agent to learn from artificial market scenarios and apply its policy to historical financial data.

### Part 2 — Dynamic Replication of a European Call

#### 1. Objective and Financial Framework

The second experimental stage investigates the **numerical replication of a European call option** through a dynamically rebalanced investment portfolio.

The methodology is based on the **Black-Scholes-Merton (1973)** option pricing framework.

The replicating portfolio combines:

- **Microsoft (MSFT):** the risky underlying asset.
- **iShares Silver Trust (SLV):** used as a simplified lower-risk asset or bond proxy within the simulated environment.

The objective is to construct a portfolio whose value tracks the theoretical value of the European call over time.

#### 2. Synthetic Market Simulation

The underlying MSFT price trajectory is generated using **Geometric Brownian Motion (GBM)**.

The simulation incorporates the stochastic evolution of the risky asset, while the second portfolio component approximates the lower-risk allocation.

The theoretical European call value and its delta are calculated using the Black-Scholes-Merton model.

#### 3. Discrete Dynamic Replication

The portfolio is rebalanced at successive time steps according to the theoretical option delta.

The replication process involves:

- Calculating the theoretical call value.
- Estimating the option delta.
- Adjusting the portfolio allocation between the risky asset and the lower-risk component.
- Comparing the replicating portfolio value with the theoretical option value.

This experiment provides a numerical benchmark for evaluating portfolio replication accuracy before introducing a learning-based hedging strategy.

#### 4. Replication Error Analysis

Replication quality is evaluated using:

- **Mean Absolute Error (MAE).**
- **Mean Squared Error (MSE).**
- Visual comparison of theoretical call values and replicating portfolio values.
- Distribution analysis of replication errors.

The reported results are:

| Metric | Reported Result |
|---|---:|
| MAE | 0.00000 |
| MSE | 0.00000 |

Within the simulated framework, the numerical replication produces virtually zero reported error.

These results establish a reference case against which more complex and stochastic hedging environments can be examined.

### Part 3 — Multi-Feature Dynamic Hedging Environment

#### 1. Objective: Adaptive Portfolio Hedging

The third experimental stage extends the analysis from numerical option replication to **dynamic hedging through Deep Reinforcement Learning**.

The objective is to develop an agent capable of adapting portfolio allocations over time in an uncertain, interactive financial environment.

Unlike the previous replication experiment, which relies on theoretical option-pricing relationships, this stage introduces a decision-making framework in which the agent learns from sequential observations and rewards.

#### 2. Multi-Dimensional Financial Environment

A custom stochastic environment, `HedgingMock`, is developed to simulate market trajectories and portfolio evolution.

The environment provides **eight financial input features**:

1. Index price.
2. Bond price.
3. Underlying logarithmic return.
4. Five-period moving average.
5. Historical volatility.
6. Remaining time to maturity.
7. Cumulative average price.
8. Momentum.

These variables allow the agent to process information about prices, trends, volatility and time-dependent market conditions.

#### 3. Portfolio Allocation and Decision-Making

At each simulation step, the agent determines the proportion of portfolio capital allocated to the risky asset.

The allocation parameter, **phi (φ)**, represents the risky-asset weight, while the remaining capital is assigned to the bond component.

The environment updates portfolio values and records:

- Market prices.
- Portfolio allocation decisions.
- Portfolio value.
- Profit and loss.
- Replication errors.

Each simulated episode represents a different market trajectory, introducing variability and uncertainty into the decision-making process.

#### 4. Random Portfolio Baseline

The first experiment evaluates a portfolio constructed using random actions, without a learned investment strategy.

The analysis compares the portfolio trajectory with the theoretical European call value.

The results show:

- A significant mismatch between the random portfolio and the theoretical option value.
- A broad distribution of replication errors.
- Substantially less accurate replication than the theoretical benchmark developed in Part 2.

This baseline demonstrates the limitations of unmanaged or randomly determined portfolio allocations in a dynamic hedging context.

#### 5. Deep Reinforcement Learning Hedging Agent

The second experiment introduces a specialized **Deep Reinforcement Learning hedging agent**.

The implementation includes:

- A `DQLAgent` base architecture.
- A specialized `HedgingAgent`.
- Continuous portfolio allocation optimization.
- Experience replay for learning from previous interactions.
- Training and testing across simulated market episodes.

The agent uses financial observations to select portfolio allocations and update its decision-making strategy through interaction with the environment.

The objective is to learn an adaptive hedging policy capable of improving portfolio management relative to the random baseline.

#### 6. Training Results and Interpretation

The reported training results show an initial progression in rewards, followed by fluctuations across episodes.

Most rewards are positive, suggesting that the agent is developing potentially useful decision-making behavior.

However, performance is not consistently positive across all episodes, indicating that further training and evaluation would be necessary to establish the robustness of the learned strategy.

Compared with the random baseline, the learning-based approach provides a framework for progressively improving hedging decisions through experience and multidimensional financial information.

## Key Findings

The three experimental stages highlight complementary aspects of applying Deep Reinforcement Learning to financial markets.

**1. Synthetic data can support financial RL training.**

Monte Carlo simulations based on the Vasicek model provide additional financial trajectories, reducing dependence on limited historical observations.

Statistical comparisons with real returns remain essential to assess the suitability of generated data for training.

**2. Learned policies can be evaluated across real financial assets.**

The A2C agent demonstrates heterogeneous performance across 12 assets, with stronger results for certain instruments and weaker generalization for others.

The findings underline the importance of evaluating both reward and predictive accuracy.

**3. Numerical derivative replication provides a benchmark for hedging accuracy.**

The Black-Scholes-Merton-based replication experiment produces virtually zero reported MAE and MSE within the simulated setting.

**4. Dynamic hedging requires adaptive decision-making.**

The stochastic multi-feature environment demonstrates the limitations of random portfolio allocation and provides a framework for training agents that learn from financial signals and sequential experience.

## Technologies

Python · Google Colab · TensorFlow / Keras · Deep Reinforcement Learning · Advantage Actor-Critic (A2C) · Deep Q-Learning · NumPy · Pandas · SciPy · yfinance · Matplotlib · Monte Carlo Simulation

## Project Files

- [RL_in_Synthetic_Financial_Markets.ipynb](RL_in_Synthetic_Financial_Markets.ipynb) — Main project notebook, including financial simulations, RL agent training, derivative replication and dynamic hedging experiments.

Historical market data are retrieved programmatically through Yahoo Finance. Synthetic financial series and simulated market environments are generated within the notebook; no separate dataset is required.

## Methodology

The project combines stochastic financial modeling, synthetic data augmentation, statistical validation and Deep Reinforcement Learning.

Its experimental architecture follows a progressive sequence:

**Real market data → Synthetic financial simulations → Statistical validation → RL agent training and testing → Derivative replication → Multi-feature dynamic hedging.**

The approach enables financial decision-making strategies to be developed and evaluated under increasingly complex market assumptions.

## Project Scope

Independent experimental prototype developed to investigate the application of Deep Reinforcement Learning to financial prediction, derivative replication and dynamic portfolio hedging.

The project provides a foundation for further research into intelligent financial agents capable of processing multidimensional market information and adapting portfolio decisions to uncertain and changing conditions.

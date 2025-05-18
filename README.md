# Ising Model for Financial Markets

This project implements an Ising Model simulation to analyze financial market behavior, based on the paper "Phase Transitions in Financial Markets Using the Ising Model: A Statistical Mechanics Perspective" (arXiv:2504.19050).

## Features

- Implementation of 2D Ising Model with Monte Carlo simulation
- Financial data analysis and comparison with Ising Model results
- Analysis of key financial statistics:
  - Volatility clustering
  - Negative skewness
  - Heavy tails
  - Autocorrelation in returns and absolute returns
- Phase transition analysis

## Update Mechanism
The model combines two competing forces that drive the evolution of market agents' states:

### 1. Nearest Neighbor Coupling
- Each agent (spin) interacts with its immediate neighbors (up, down, left, right)
- Periodic boundary conditions are used (agents on the edge interact with those on the opposite edge)
- The coupling strength (J) determines how strongly agents influence each other
- This force tends to make agents align with their neighbors, representing local market sentiment

### 2. Market Sentiment Force (α)
- A global force that makes agents more likely to do the opposite of the majority
- Controlled by the alpha parameter (default: 7.0)
- The force is proportional to the absolute value of the average magnetization
- This represents the "contrarian" behavior in markets where agents tend to go against the crowd

### Combined Update Rule
The total local field acting on an agent at position (x,y) is:
```
H(x,y) = J * (sum of neighbor spins) - α * S(x,y) * |average magnetization|
```

Where:
- J is the coupling strength
- α is the market sentiment parameter
- S(x,y) is the spin at position (x,y)
- |average magnetization| is the absolute value of the average spin

### Update Rules
The model supports two different update rules:

1. **Metropolis Algorithm**:
```
P(flip) = min(1, exp(-ΔE/T))
```
Where:
- ΔE is the change in energy
- T is the temperature parameter

2. **Logistic (Bornholdt) Rule**:
```
P(flip) = 1 / (1 + exp(2H(x,y)/T))
```
Where:
- H(x,y) is the local field
- T is the temperature parameter

The logistic rule is often preferred for financial market modeling as it better captures the decision-making process of market participants.

## Installation

1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the main analysis script:
```bash
python main.py
```

This will:
1. Download S&P 500 data
2. Run Ising Model simulations
3. Compare statistical properties
4. Generate visualization plots

## Project Structure

- `ising_model.py`: Core Ising Model implementation
- `financial_analysis.py`: Financial data analysis and comparison
- `main.py`: Main script to run the analysis
- `requirements.txt`: Project dependencies

## Results

The simulation reproduces several key features of financial markets:
- Volatility clustering
- Heavy-tailed return distributions
- Autocorrelation in absolute returns
- Phase transitions at critical temperatures

## References

- Giorgio, B. (2025). Phase Transitions in Financial Markets Using the Ising Model: A Statistical Mechanics Perspective. arXiv:2504.19050 
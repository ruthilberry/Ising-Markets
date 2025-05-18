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
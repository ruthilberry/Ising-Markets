"""Calibrate the Bornholdt‑Ising model so that its synthetic
log‑returns reproduce selected stylised facts of the S&P 500 lately

Dependencies
------------
* numpy, pandas, matplotlib  – common SciPy stack
* yfinance                   – download historical S&P 500 prices
* statsmodels                – ACF and Ljung‑Box
* optuna                     – Bayesian/TPE hyper‑parameter search
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import statsmodels.api as sm
import optuna
import yfinance as yf
from datetime import datetime

# -----------------------------------------------------------------------------
# 0 · Helper functions for empirical targets
# -----------------------------------------------------------------------------

START_DATE = "2020-01-01"
END_DATE   = datetime.now().strftime("%Y-%m-%d")
TICKER     = "^GSPC"          # S&P 500 index (yfinance symbol)


def get_sp500_returns() -> pd.Series:
    """Download daily closes and return log‑returns as a Pandas Series."""
    spy = yf.download(TICKER, start=START_DATE, end=END_DATE, progress=False)
    close = spy["Close"].dropna()
    r = np.log(close).diff().dropna()
    r.name = "log_return"
    return r


def daily_vol(series: pd.Series) -> float:
    return series.std(ddof=0)


def tail_exponent(series: np.ndarray, tail_frac: float = 0.05) -> float:
    """Hill estimator on the absolute returns."""
    x = np.sort(np.abs(series))
    k = int(len(x) * tail_frac)
    x_tail = x[-k:]
    x_k = x_tail[0]
    hill = k / np.sum(np.log(x_tail / x_k))
    return hill


def acf_abs(series: np.ndarray, nlags: int = 30) -> np.ndarray:
    return sm.tsa.acf(np.abs(series), nlags=nlags, fft=True)[1:]  # drop lag 0


# -----------------------------------------------------------------------------
# 1 · Objective function for Optuna
# -----------------------------------------------------------------------------

from ising_model import IsingModel  # adjust import to your file name

EMP_RET = get_sp500_returns()
SIG_TARGET   = daily_vol(EMP_RET)
ALPHA_TARGET = tail_exponent(EMP_RET.values)
ACF_TARGET   = acf_abs(EMP_RET.values, nlags=30)

print("Empirical targets → σ=%.4f  α(tail exponent)=%.2f" % (SIG_TARGET, ALPHA_TARGET))

LATTICE_SIZE   = 50               # speed/good statistics trade‑off
THERMAL_SWEEPS = 1000            # burn‑in
PROD_SWEEPS    = 400_000           # > 3 × S&P sample length
SAMPLE_EVERY   = 100               # sample every 100 sweeps to reduce memory usage


def simulate_returns(T: float, alpha: float, seed: int) -> np.ndarray:
    """Run the model and convert magnetisation changes to log‑returns."""
    model = IsingModel(
        size=LATTICE_SIZE,
        temperature=T,
        alpha=alpha,
        use_metropolis=False,  # Bornholdt
        seed=seed,
    )
    m_hist, _ = model.simulate(
        sweeps=PROD_SWEEPS,
        thermal_sweeps=THERMAL_SWEEPS,
        sample_interval=SAMPLE_EVERY,
        show_progress=False,
    )
    # Ensure we're working with numpy arrays
    m_hist = np.asarray(m_hist, dtype=np.float64)
    dm = np.diff(m_hist) / (LATTICE_SIZE * LATTICE_SIZE)
    # scale λ so that model volatility matches empirical
    lam = float(SIG_TARGET / np.std(dm, ddof=0))  # Ensure lam is a scalar
    return np.multiply(lam, dm)  # Use numpy's multiply to ensure array output


# Weights for composite loss
W_SIG = 1.0
W_TAIL = 10.0
W_ACF = 50.0 / len(ACF_TARGET)


def loss_function(trial: optuna.Trial) -> float:
    T     = trial.suggest_float("T", 2.1, 2.4)
    alpha = trial.suggest_float("alpha", 3.0, 10.0)

    r_sim = simulate_returns(T, alpha, seed=trial.number)

    # statistics
    sig  = np.std(r_sim, ddof=0)
    alp  = tail_exponent(r_sim)
    acf  = acf_abs(r_sim, nlags=30)

    l_sig  = (sig - SIG_TARGET) ** 2
    l_tail = (alp - ALPHA_TARGET) ** 2
    l_acf  = np.mean((acf - ACF_TARGET) ** 2)

    loss = W_SIG * l_sig + W_TAIL * l_tail + W_ACF * l_acf

    trial.report(loss, step=0)
    return loss


# -----------------------------------------------------------------------------
# 2 · Run optimisation
# -----------------------------------------------------------------------------

def main():
    study = optuna.create_study(
        study_name="SP500_Calibration",
        direction="minimize",
        sampler=optuna.samplers.TPESampler(),
    )
    study.optimize(loss_function, n_trials=100, timeout=3 * 3600)

    print("\n====  Best parameters  ====")
    print(study.best_params)
    print("Loss:", study.best_value)

    # run once more with best parameters and plot
    r_best = simulate_returns(
        T=study.best_params["T"],
        alpha=study.best_params["alpha"],
        seed=12345,
    )

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 4))
    plt.plot(r_best[:2000], label="synthetic")
    plt.title("Synthetic log‑returns (first 2000 sweeps)")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

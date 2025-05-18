import numpy as np
import pandas as pd
import yfinance as yf
from typing import Tuple, List
import matplotlib.pyplot as plt
from scipy import stats
from ising_model import IsingModel

class FinancialAnalysis:
    def __init__(self, symbol: str = "^GSPC", start_date: str = "2020-01-01"):
        """
        Initialize financial analysis for a given symbol.
        
        Args:
            symbol: Stock symbol (default: S&P 500)
            start_date: Start date for analysis
        """
        self.symbol = symbol
        self.data = yf.download(symbol, start=start_date)
        close = self.data['Close']
        if isinstance(close, pd.DataFrame):
            close = close.squeeze()
        # Calculate log returns
        self.returns = np.log(close / close.shift(1)).dropna()
        
    def calculate_statistics(self) -> dict:
        """Calculate key financial statistics."""
        stats_dict = {
            'volatility': self.returns.std(),
            'skewness': stats.skew(self.returns),
            'kurtosis': stats.kurtosis(self.returns),
            'autocorr_returns': self.returns.autocorr(),
            'autocorr_abs_returns': self.returns.abs().autocorr()
        }
        return stats_dict
    
    def analyze_volatility_clustering(self, returns: np.ndarray, window: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """
        Analyze volatility clustering using rolling standard deviation.
        
        Args:
            returns: Array of returns
            window: Rolling window size
            
        Returns:
            Tuple of (rolling volatility, autocorrelation of absolute returns)
        """
        # Calculate rolling volatility
        rolling_vol = pd.Series(returns).rolling(window=window).std()
        
        # Calculate autocorrelation of absolute returns
        abs_returns = np.abs(returns)
        autocorr = pd.Series(abs_returns).autocorr(lag=1)
        
        return rolling_vol, autocorr
    
    def simulate_ising_returns(self, size: int, temperature: float, 
                             steps: int = 1000) -> np.ndarray:
        """
        Simulate returns using Ising Model.
        
        Args:
            size: Size of Ising lattice
            temperature: Temperature parameter
            steps: Number of simulation steps
            
        Returns:
            Simulated returns
        """
        model = IsingModel(size, temperature)
        magnetization_history, _ = model.simulate(steps)
        
        # Plot magnetization history
        plt.figure(figsize=(12, 4))
        plt.plot(magnetization_history)
        plt.title('Magnetization History')
        plt.xlabel('Step')
        plt.ylabel('Magnetization')
        plt.grid(True)
        plt.show()
        
        # Convert magnetization to log returns
        returns = np.log(magnetization_history[1:] / magnetization_history[:-1])
        
        # Plot log returns
        plt.figure(figsize=(12, 4))
        plt.plot(returns)
        plt.title('Ising Model Log Returns')
        plt.xlabel('Step')
        plt.ylabel('Log Returns')
        plt.grid(True)
        plt.show()
        
        return returns
    
    def compare_statistics(self, ising_returns: np.ndarray) -> pd.DataFrame:
        """
        Compare statistics between real and Ising-simulated returns.
        
        Args:
            ising_returns: Returns simulated from Ising model
            
        Returns:
            DataFrame with comparison
        """
        real_stats = self.calculate_statistics()
        ising_series = pd.Series(ising_returns)
        ising_stats = {
            'volatility': np.std(ising_returns),
            'skewness': stats.skew(ising_returns),
            'kurtosis': stats.kurtosis(ising_returns),
            'autocorr_returns': ising_series.autocorr(),
            'autocorr_abs_returns': ising_series.abs().autocorr()
        }
        
        comparison = pd.DataFrame({
            'Real Market': real_stats,
            'Ising Model': ising_stats
        })
        
        return comparison
    
    def plot_comparison(self, ising_returns: np.ndarray) -> None:
        """
        Plot comparison between real and simulated returns.
        
        Args:
            ising_returns: Returns simulated from Ising model
        """
        # Create a 3x1 subplot layout
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
        
        # Returns comparison
        ax1.plot(self.returns.values[:len(ising_returns)], 
                label='Real Returns', alpha=0.7)
        ax1.plot(ising_returns, label='Ising Returns', alpha=0.7)
        ax1.set_title('Log Returns Comparison')
        ax1.legend()
        
        # Distribution comparison
        ax2.hist(self.returns, bins=50, density=True, alpha=0.5, 
                label='Real Returns')
        ax2.hist(ising_returns, bins=50, density=True, alpha=0.5, 
                label='Ising Returns')
        ax2.set_title('Log Returns Distribution')
        ax2.legend()
        
        # Volatility clustering comparison
        real_vol, real_autocorr = self.analyze_volatility_clustering(self.returns.values)
        ising_vol, ising_autocorr = self.analyze_volatility_clustering(ising_returns)
        
        ax3.plot(real_vol.values[:len(ising_vol)], 
                label='Real Volatility', alpha=0.7)
        ax3.plot(ising_vol.values, label='Ising Volatility', alpha=0.7)
        ax3.set_title('Volatility Clustering Comparison')
        ax3.legend()
        
        # Add autocorrelation information to the plot
        ax3.text(0.02, 0.95, 
                f'Real Autocorr: {real_autocorr:.3f}\nIsing Autocorr: {ising_autocorr:.3f}',
                transform=ax3.transAxes, 
                bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.show() 
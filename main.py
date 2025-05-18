from financial_analysis import FinancialAnalysis
import numpy as np
import matplotlib.pyplot as plt
from ising_model import IsingModel
import pandas as pd
from tqdm import tqdm

size = 50
temperature = 100000000000000
steps = 1000
alpha = 0

def main():
    print("Starting financial analysis...")
    # Initialize financial analysis
    analysis = FinancialAnalysis(symbol="^GSPC", start_date="2010-01-01")
    print("S&P 500 data downloaded.")
    
    print("Running Ising simulation for returns...")
    # Run simulation
    ising_returns = analysis.simulate_ising_returns(size, temperature, steps)
    print("Ising simulation completed.")
    
    # Compare statistics
    print("Computing statistical comparison...")
    comparison = analysis.compare_statistics(ising_returns)
    print("\nStatistical Comparison:")
    print(comparison)
    
    # Plot results
    print("Plotting results...")
    analysis.plot_comparison(ising_returns)
    print("Plots generated.")
    
    # Additional analysis: Phase transition
    print("Starting phase transition analysis...")
    temperatures = np.linspace(1.0, 3.0, 20)
    magnetizations = []
    
    for T in tqdm(temperatures, desc="Phase Transition"):
        model = IsingModel(size, T)
        mag_history, _ = model.simulate(1000)
        magnetizations.append(np.mean(np.abs(mag_history)))
    
    plt.figure(figsize=(10, 6))
    plt.plot(temperatures, magnetizations, 'o-')
    plt.xlabel('Temperature')
    plt.ylabel('Average Magnetization')
    plt.title('Phase Transition in Ising Model')
    plt.grid(True)
    plt.show()
    print("Phase transition analysis completed.")

def make_ising_gif():
    print("Starting GIF generation...")
    steps = 2000  # More steps = longer GIF
    snapshot_interval = 5  # Save every 5 steps

    model = IsingModel(size, temperature)
    model.simulate_with_snapshots(steps, snapshot_dir='snapshots', snapshot_interval=snapshot_interval)
    IsingModel.create_gif_from_snapshots(snapshot_dir='snapshots', gif_name='ising_evolution.gif', duration=0.1)
    print("GIF saved as ising_evolution.gif")

if __name__ == "__main__":
    main()
    make_ising_gif() 
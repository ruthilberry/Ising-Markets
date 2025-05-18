from financial_analysis import FinancialAnalysis
import numpy as np
import matplotlib.pyplot as plt
from ising_model import IsingModel
import pandas as pd
from tqdm import tqdm
import os
from datetime import datetime

def create_run_folder(temperature: float, alpha: float) -> str:
    """Create a folder for this run with parameters in the name."""
    # Create results directory if it doesn't exist
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"run_T{temperature:.2f}_alpha{alpha:.2f}_{timestamp}"
    full_path = os.path.join(results_dir, folder_name)
    os.makedirs(full_path, exist_ok=True)
    return full_path

def save_statistics(comparison: pd.DataFrame, folder: str):
    """Save statistical comparison to a text file."""
    with open(os.path.join(folder, 'statistics.txt'), 'w') as f:
        f.write("Statistical Comparison:\n")
        f.write("=====================\n\n")
        f.write(comparison.to_string())
        f.write("\n\nParameters:\n")
        f.write(f"Temperature: {temperature}\n")
        f.write(f"Alpha: {alpha}\n")
        f.write(f"Size: {size}\n")
        f.write(f"Steps: {steps}\n")

def main():
    print("Starting financial analysis...")
    # Create run-specific folder
    run_folder = create_run_folder(temperature, alpha)
    print(f"Created output folder: {run_folder}")
    
    # Initialize financial analysis with the run folder
    analysis = FinancialAnalysis(symbol="^GSPC", start_date="2020-01-01", output_folder=run_folder)
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
    
    # Save statistics to file
    save_statistics(comparison, run_folder)
    
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
    plt.savefig(os.path.join(run_folder, 'phase_transition.png'))
    plt.close()
    print("Phase transition analysis completed.")
    
    return run_folder

def make_ising_gif(run_folder: str):
    print("Starting GIF generation...")
    steps = 2000  # More steps = longer GIF
    snapshot_interval = 5  # Save every 5 steps

    # Create snapshots directory inside the run folder
    snapshot_dir = os.path.join(run_folder, 'snapshots')
    os.makedirs(snapshot_dir, exist_ok=True)

    model = IsingModel(size=size, temperature=temperature, alpha=alpha)
    model.simulate_with_snapshots(steps, snapshot_dir=snapshot_dir, snapshot_interval=snapshot_interval)
    gif_name = f'ising_evolution_T{temperature}_alpha{alpha}.gif'
    IsingModel.create_gif_from_snapshots(snapshot_dir=snapshot_dir, gif_name=os.path.join(run_folder, gif_name), duration=0.1)
    print(f"GIF saved as '{gif_name}'")

if __name__ == "__main__":
    # Global parameters
    size = 50
    temperature = 1
    steps = 10000
    alpha = 0.
    
    run_folder = main()
    make_ising_gif(run_folder) 
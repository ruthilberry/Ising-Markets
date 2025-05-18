from financial_analysis import FinancialAnalysis
import numpy as np
import matplotlib.pyplot as plt
from ising_model import IsingModel
from tqdm import tqdm
from utils import create_run_folder, save_statistics, create_snapshot_dir
import os
import pandas as pd
from datetime import datetime

def create_run_folder(temperature: float, alpha: float, metropolis: bool) -> str:
    """Create a folder for this run with parameters in the name."""
    # Create results directory if it doesn't exist
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"run_T{temperature:.2f}_alpha{alpha:.2f}_metropolis_{metropolis}_{timestamp}"
    full_path = os.path.join(results_dir, folder_name)
    os.makedirs(full_path, exist_ok=True)
    return full_path

def save_statistics(comparison: pd.DataFrame, folder: str, params: dict):
    """Save statistical comparison to a text file."""
    with open(os.path.join(folder, 'statistics.txt'), 'w') as f:
        f.write("Statistical Comparison:\n")
        f.write("=====================\n\n")
        f.write(comparison.to_string())
        f.write("\n\nParameters:\n")
        f.write(f"Temperature: {params['temperature']}\n")
        f.write(f"Alpha: {params['alpha']}\n")
        f.write(f"Size: {params['size']}\n")
        f.write(f"Steps: {params['steps']}\n")
        f.write(f"Update Mechanism: {'Metropolis' if params['use_metropolis'] else 'Bornholdt'}\n")

def run_simulation(params: dict):
    """Run the Ising model simulation with given parameters."""
    # Create run folder
    run_folder = create_run_folder(
        params['temperature'], 
        params['alpha'],
        params['use_metropolis']
    )
    print(f"Created output folder: {run_folder}")
    
    # Initialize financial analysis
    analysis = FinancialAnalysis(symbol="^GSPC", start_date="2020-01-01")
    print("S&P 500 data downloaded.")
    
    # Run Ising simulation
    print("Running Ising simulation for returns...")
    ising_returns = analysis.simulate_ising_returns(
        params['size'], 
        params['temperature'], 
        params['steps']
    )
    print("Ising simulation completed.")
    
    # Compare statistics
    print("Computing statistical comparison...")
    comparison = analysis.compare_statistics(ising_returns)
    print("\nStatistical Comparison:")
    print(comparison)
    
    # Save statistics
    save_statistics(comparison, run_folder, params)
    
    # Plot results
    print("Plotting results...")
    analysis.plot_comparison(ising_returns)
    analysis.plot_volatility_analysis(ising_returns)
    print("Plots generated.")
    
    # Phase transition analysis
    print("Starting phase transition analysis...")
    temperatures = np.linspace(1.0, 3.0, 20)
    magnetizations = []
    
    for T in tqdm(temperatures, desc="Phase Transition"):
        model = IsingModel(
            params['size'], 
            T,
            use_metropolis=params['use_metropolis']
        )
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
    
    # Generate GIF
    print("Starting GIF generation...")
    snapshot_dir = create_snapshot_dir(run_folder)
    
    model = IsingModel(
        size=params['size'], 
        temperature=params['temperature'], 
        alpha=params['alpha'],
        use_metropolis=params['use_metropolis']
    )
    model.simulate_with_snapshots(
        params['gif_steps'], 
        snapshot_dir=snapshot_dir, 
        snapshot_interval=params['snapshot_interval']
    )
    
    metropolis_str = "metropolis" if params['use_metropolis'] else "Bornholdt"
    gif_name = f'ising_evolution_T{params["temperature"]}_alpha{params["alpha"]}_{metropolis_str}.gif'
    IsingModel.create_gif_from_snapshots(
        snapshot_dir=snapshot_dir, 
        gif_name=os.path.join(run_folder, gif_name), 
        duration=0.1
    )
    print(f"Simulation complete! All results saved in: {run_folder}")
    print(f"  - GIF: {gif_name} and snapshots")
    print(f"  - Statistics: statistics.txt and plots")
    print(f"  - Phase transition plot: phase_transition.png")

if __name__ == "__main__":
    # Simulation parameters
    params = {
        'size': 50,              # Size of the Ising lattice
        'temperature': 2.2,      # Temperature parameter
        'steps': 10000,         # Number of simulation steps
        'alpha': 5,           # Alpha parameter
        'gif_steps': 2000,      # Number of steps for GIF generation
        'snapshot_interval': 5,  # Interval between snapshots
        'use_metropolis': False      # Whether to use Metropolis algorithm (True) or Bornholdt dynamics (False)
    }
    
    run_simulation(params) 
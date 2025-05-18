import os
from datetime import datetime
import pandas as pd

def create_run_folder(temperature: float, alpha: float, use_metropolis: bool) -> str:
    """Create a folder for this run with parameters in the name."""
    # Create results directory if it doesn't exist
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metropolis_str = "metropolis" if use_metropolis else "glauber"
    folder_name = f"run_T{temperature:.2f}_alpha{alpha:.2f}_{metropolis_str}_{timestamp}"
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
        for key, value in params.items():
            f.write(f"{key}: {value}\n")

def create_snapshot_dir(run_folder: str) -> str:
    """Create and return path to snapshot directory."""
    snapshot_dir = os.path.join(run_folder, 'snapshots')
    os.makedirs(snapshot_dir, exist_ok=True)
    return snapshot_dir 
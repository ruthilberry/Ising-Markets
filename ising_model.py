import numpy as np
from typing import Tuple, Optional
import matplotlib.pyplot as plt
import os
import imageio
from tqdm import tqdm

class IsingModel: 
    def __init__(self, size: int, temperature: float, coupling: float = 1.0, alpha: float = 7.0, lattice: np.ndarray = None, seed = None):
        """
        Initialize the Ising Model (with periodic boundary conditions).
        
        Args:
            size: Size of the square lattice (size x size)
            temperature: Temperature parameter (T)
            coupling: Coupling constant (J)
            alpha: Minority feedback parameter
            lattice: initial lattice configuration
        """
        self.size = size
        self.temperature = temperature
        self.coupling = coupling
        self.alpha = alpha
        self.lattice = lattice.copy() if lattice is not None else np.random.choice([-1, 1], size=(size, size)) # lattice randomly initialized
        self.energy = self._calculate_energy()
        self.magnetization = np.sum(self.lattice)
        self.rng     = np.random.default_rng(seed)
        i = np.arange(size)[:, None]
        j = np.arange(size)[None, :]
        self._parity = (i + j) & 1 # parity of each site (trick for latter vectorization)

        
    def _calculate_energy(self) -> float:
        """Calculate the total energy of the system using vectorized operations."""
        S = self.lattice
        E = -self.coupling * (
            (S * np.roll(S, -1, axis=0)).sum() +   # down neighbours
            (S * np.roll(S, -1, axis=1)).sum()     # right neighbours
        )
        return E
    
    def _calculate_delta_energy(self, i, j): # unused in the current vectorized implementation
        S      = self.lattice
        nn_sum = (
            S[(i+1)%self.size, j] + S[(i-1)%self.size, j] +
            S[i, (j+1)%self.size] + S[i, (j-1)%self.size]
        )
        m = abs(S.mean())
        local_field = nn_sum - self.alpha * S[i, j] * m   # ← α term
        return -2.0 * self.coupling * S[i, j] * local_field
    
    def step(self, *, metropolis: bool = True) -> None:
        """
        One checker-board sweep.

        Parameters
        ----------
        metropolis  True  → Metropolis acceptance rule
                    False → Bornholdt / heat-bath (Glauber-type) rule
        """
        L     = self.size
        beta  = 1.0 / self.temperature
        rng   = self.rng 

        # --- two half-lattices: black (0) then red (1) --------------------
        for parity in (0, 1):
            mask = (self._parity == parity)          # Boolean L×L

            # 1. nearest-neighbour sum S_{nn}(x,y) – vectorised
            nn_sum = (
                np.roll(self.lattice,  1, 0) + np.roll(self.lattice, -1, 0) +
                np.roll(self.lattice,  1, 1) + np.roll(self.lattice, -1, 1)
            )

            # ----------------------------------------------------------------
            # 2. choose update rule
            # ----------------------------------------------------------------
            if metropolis:
                # ----- Metropolis -------------------------------------------
                deltaE = 2.0 * self.coupling * self.lattice * nn_sum   # L×L
                boltz  = np.exp(-beta * deltaE)

                accept = (deltaE <= 0) | (rng.random(deltaE.shape) < boltz)
                flip   = mask & accept
                self.lattice[flip] *= -1

            else:
                # ----- Bornholdt heat-bath ----------------------------------
                m_abs = abs(self.magnetization) / (L * L)              # |m|
                local_field = (
                    self.coupling * nn_sum -
                    self.alpha * self.lattice * m_abs                  # −α s_i |m|
                )

                p_up    = 1.0 / (1.0 + np.exp(-2.0 * beta * local_field))
                rand    = rng.random(self.lattice.shape)
                new_s   = np.where(rand < p_up, 1, -1)                 # L×L
                self.lattice[mask] = new_s[mask]                       # only this parity
                self.magnetization = self.lattice.sum(dtype=np.int32) # magnetization needs to be updated between both parities

        # --- update extensive observables (vectorised) ---------------------
        self.magnetization = self.lattice.sum(dtype=np.int32)
        self.energy = -self.coupling * (
            (self.lattice * np.roll(self.lattice, 1, 0)).sum(dtype=np.int32) +
            (self.lattice * np.roll(self.lattice, 1, 1)).sum(dtype=np.int32)
        )
    
    def simulate(self, sweeps: int, thermal_sweeps: int = 10_000, sample_interval: int = 10, show_progress: bool = False,):
    # --- warm-up ---
        for _ in range(thermal_sweeps):
            self.step()

        n_samples = sweeps // sample_interval
        m_hist = np.empty(n_samples, dtype=np.int32)
        e_hist = np.empty(n_samples, dtype=np.int32)

        iterator = (tqdm(range(sweeps), desc="MC") if show_progress else range(sweeps))
        k = 0
        for s in iterator:
            self.step()
            if (s + 1) % sample_interval == 0:
                m_hist[k] = self.magnetization   # still extensive
                e_hist[k] = self.energy
                k += 1
        return m_hist, e_hist
    
    # def step(self) -> None:
       #  """Perform one Monte Carlo step."""
        # for _ in range(self.size * self.size):
            # i, j = np.random.randint(0, self.size, 2)
            # delta_energy = self._calculate_delta_energy(i, j)
            
            # Metropolis algorithm
            # if delta_energy <= 0 or np.random.random() < np.exp(-delta_energy / self.temperature):
                # self.lattice[i, j] *= -1
                # self.energy += delta_energy
                # self.magnetization += 2 * self.lattice[i, j]
    
    def plot_lattice(self, title: Optional[str] = None) -> None:
        """Plot the current state of the lattice."""
        plt.figure(figsize=(8, 8))
        plt.imshow(self.lattice, cmap='binary')
        plt.colorbar(label='Spin')
        if title:
            plt.title(title)
        plt.show()

    def save_lattice_heatmap(self, filename: str, title: Optional[str] = None) -> None:
        plt.figure(figsize=(5, 5))
        plt.imshow(self.lattice, cmap='coolwarm', vmin=-1, vmax=1)
        plt.axis('off')
        if title:
            plt.title(title)
        plt.tight_layout()
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        plt.close()

    def simulate_with_snapshots(self, steps: int, snapshot_dir: str = 'snapshots', snapshot_interval: int = 10, thermalization_steps: int = 1000) -> None:
        """
        Simulate and save lattice heatmaps at intervals for GIF creation.
        """
        if not os.path.exists(snapshot_dir):
            os.makedirs(snapshot_dir)
        # Thermalization
        for _ in tqdm(range(thermalization_steps), desc="Thermalization"):
            self.step()
        # Main simulation with snapshots
        for step in tqdm(range(steps), desc="Simulation"):
            self.step()
            if step % snapshot_interval == 0:
                filename = os.path.join(snapshot_dir, f'lattice_{step:04d}.png')
                self.save_lattice_heatmap(filename)

    @staticmethod
    def create_gif_from_snapshots(snapshot_dir: str = 'snapshots', gif_name: str = 'ising_evolution.gif', duration: float = 0.1) -> None:
        images = []
        files = sorted([f for f in os.listdir(snapshot_dir) if f.endswith('.png')])
        for file in files:
            images.append(imageio.imread(os.path.join(snapshot_dir, file)))
        imageio.mimsave(gif_name, images, duration=duration)
from Metropolis import Metropolis
import numpy as np
import pickle
import logging
from tqdm import tqdm
import os
from multiprocessing import Pool, cpu_count

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s %(asctime)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Parameters
Tc = 2.0 / np.log(1.0 + np.sqrt(2.0))
T_list = np.linspace(1.0, 3.5, 1000)
L_list = [10, 20, 30, 40, 60]
J = 1.0
total_steps = 3 * (10**6)
sampling_steps = 2 * (10**3)

def process_task(params):
    """Worker function for parallel processing."""
    L, T = params
    save_dir = os.path.join("Data_new", str(L))
    os.makedirs(save_dir, exist_ok=True)
    save_file = os.path.join(save_dir, f"L{L}_T{T:.4f}.npy")
    if os.path.exists(save_file):
        return  # Skip if file exists
    Metropolis(L, T, J, total_steps, sampling_steps, save_file, 'cold')

if __name__ == "__main__":
    # Create a list of tasks for parallel processing
    tasks = [(L, T) for L in L_list for T in T_list]

    # Use multiprocessing Pool for parallel execution
    num_workers = min(cpu_count(), len(tasks))  # Use available CPUs
    with Pool(num_workers) as pool:
        list(tqdm(pool.imap_unordered(process_task, tasks), total=len(tasks)))

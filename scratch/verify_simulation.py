import sys
import os
import shutil
import time

# Add src to path
sys.path.append(os.path.abspath("src"))

from tests.test_hongloumeng_simulation_v3 import run_simulation

if __name__ == "__main__":
    # Clean old db if exists to starts fresh
    db_path = "rems_sim.db"
    if os.path.exists(db_path):
        os.remove(db_path)
    
    print("Starting 1-round simulation for Verification...")
    run_simulation(1)
    print("Simulation finished.")

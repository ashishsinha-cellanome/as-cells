import os
import sys
from unittest.mock import MagicMock

# Mock HydraConfig before importing setup_cluster_env
sys.modules['hydra'] = MagicMock()
sys.modules['hydra.core.hydra_config'] = MagicMock()

from utils.distributed_utils import setup_cluster_env

def test_setup():
    print("--- Test 1: Standard Slurm Job (No Hydra) ---")
    os.environ["SLURM_JOB_ID"] = "123456"
    os.environ["SLURM_NTASKS"] = "4"
    if "MASTER_PORT" in os.environ: del os.environ["MASTER_PORT"]
    
    setup_cluster_env()
    print(f"MASTER_PORT (expected ~23456): {os.environ.get('MASTER_PORT')}")
    
    print("\n--- Test 2: Hydra Multirun (Job 5) ---")
    import hydra.core.hydra_config
    mock_hydra = MagicMock()
    mock_hydra.job.num = 5
    hydra.core.hydra_config.HydraConfig.get.return_value = mock_hydra
    
    if "MASTER_PORT" in os.environ: del os.environ["MASTER_PORT"]
    setup_cluster_env()
    print(f"MASTER_PORT (expected ~23461): {os.environ.get('MASTER_PORT')}")

    print("\n--- Test 3: Warning Check (SLURM_NTASKS > 1 but no PROCID) ---")
    # This should trigger the new warning print
    setup_cluster_env()

if __name__ == "__main__":
    test_setup()

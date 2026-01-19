import os
import torch
import torch.distributed as dist
import socket

def main():
    print(f"[{socket.gethostname()}] Starting DDP check...")
    
    # Get environment variables
    rank = int(os.environ.get("SLURM_PROCID", "0"))
    world_size = int(os.environ.get("SLURM_NTASKS", "1"))
    local_rank = int(os.environ.get("SLURM_LOCALID", "0"))
    
    print(f"[{socket.gethostname()}] Rank: {rank}/{world_size}, Local: {local_rank}")
    
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        print(f"[{socket.gethostname()}] CUDA Device: {torch.cuda.current_device()}")
    
    # Initialize DDP
    try:
        # Compute Canada / SLURM specific: ensure MASTER_ADDR is set
        if "MASTER_ADDR" not in os.environ:
            print("WARNING: MASTER_ADDR not set. Attempting to deduce from SLURM_NODELIST...")
        
        print(f"[{socket.gethostname()}] Initializing process group...")
        dist.init_process_group(backend="nccl")
        print(f"[{socket.gethostname()}] Success! DDP Initialized.")
    except Exception as e:
        print(f"[{socket.gethostname()}] FAILED: {e}")
        raise

    dist.destroy_process_group()

if __name__ == "__main__":
    main()

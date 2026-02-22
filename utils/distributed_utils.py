import os
import socket
import subprocess
import torch.distributed as dist


def setup_cluster_env():
    """Apply cluster-friendly defaults for distributed training."""
    # Hydra-specific context (if available)
    try:
        from hydra.core.hydra_config import HydraConfig
        hydra_cfg = HydraConfig.get()
        job_num = hydra_cfg.job.num
    except (ImportError, ValueError, AttributeError):
        job_num = 0

    if "SLURM_PROCID" in os.environ:
        os.environ["RANK"] = os.environ["SLURM_PROCID"]
        os.environ["LOCAL_RANK"] = os.environ.get("SLURM_LOCALID", "0")
        os.environ["WORLD_SIZE"] = os.environ.get("SLURM_NTASKS", "1")

    # Potential timeout warning: If SLURM_NTASKS > 1 but we aren't using srun (check for LOCALID/PROCID)
    if int(os.environ.get("SLURM_NTASKS", "1")) > 1:
        if "SLURM_PROCID" not in os.environ:
            print(
                "\n" + "!" * 80 + "\n"
                "WARNING: SLURM_NTASKS > 1 but SLURM_PROCID is missing.\n"
                "This usually happens when you run 'uv run ...' instead of 'srun uv run ...'.\n"
                "Lightning will hang waiting for other ranks that were never spawned.\n"
                "!" * 80 + "\n"
            )

    if "MASTER_PORT" not in os.environ:
        job_id = os.environ.get("SLURM_JOB_ID")
        if job_id:
            try:
                # Add job_num to avoid collisions in Hydra multirun sweeps
                base_port = 20000 + (int(job_id) % 10000)
                os.environ["MASTER_PORT"] = str(base_port + job_num)
            except ValueError:
                os.environ["MASTER_PORT"] = str(29505 + job_num)
        else:
            os.environ["MASTER_PORT"] = str(29505 + job_num)

    if "MASTER_ADDR" not in os.environ and "SLURM_NODELIST" in os.environ:
        try:
            node_list = os.environ["SLURM_NODELIST"]
            # Get hostnames from slurm nodelist
            host_output = subprocess.check_output(
                ["scontrol", "show", "hostnames", node_list],
                stderr=subprocess.DEVNULL,
            ).decode().splitlines()
            if host_output:
                os.environ["MASTER_ADDR"] = host_output[0]
        except Exception:
            if os.environ.get("RANK", "0") == "0":
                os.environ["MASTER_ADDR"] = socket.gethostname()

    # NCCL P2P/IB defaults (only set if not explicitly provided)
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")


def get_rank():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    if "SLURM_PROCID" in os.environ:
        if "SLURM_NTASKS" not in os.environ:
            os.environ["SLURM_NTASKS"] = "1"
        return int(os.environ["SLURM_PROCID"])
    if "RANK" in os.environ:
        return int(os.environ["RANK"])
    for var in ("OMPI_COMM_WORLD_RANK", "PMI_RANK", "LSB_RANK"):
        if var in os.environ:
            return int(os.environ[var])
    return 0


def rank_zero_print(*args, **kwargs):
    if get_rank() == 0:
        print(*args, **kwargs)

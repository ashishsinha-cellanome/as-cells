import os
import socket
import subprocess
import torch.distributed as dist


def setup_cluster_env():
    """Apply cluster-friendly defaults for distributed training."""
    if "SLURM_PROCID" in os.environ:
        os.environ["RANK"] = os.environ["SLURM_PROCID"]
        os.environ["LOCAL_RANK"] = os.environ.get("SLURM_LOCALID", "0")
        os.environ["WORLD_SIZE"] = os.environ.get("SLURM_NTASKS", "1")

    if "MASTER_PORT" not in os.environ:
        job_id = os.environ.get("SLURM_JOB_ID")
        if job_id:
            try:
                os.environ["MASTER_PORT"] = str(20000 + (int(job_id) % 10000))
            except ValueError:
                os.environ["MASTER_PORT"] = "29505"
        else:
            os.environ["MASTER_PORT"] = "29505"

    if "MASTER_ADDR" not in os.environ and "SLURM_NODELIST" in os.environ:
        try:
            node_list = os.environ["SLURM_NODELIST"]
            host = subprocess.check_output(
                ["scontrol", "show", "hostnames", node_list],
                stderr=subprocess.DEVNULL,
            ).decode().splitlines()[0]
            os.environ["MASTER_ADDR"] = host
        except Exception:
            if os.environ.get("RANK", "0") == "0":
                os.environ["MASTER_ADDR"] = socket.gethostname()

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

#!/usr/bin/env python3
"""
Training Monitor Agent

This agent:
1. Monitors running SLURM and Python processes
2. If no training jobs are running, requests a SLURM allocation
3. Launches training sessions focused on improving small/tiny object detection (beads)
4. Supports interactive shell or tmux session management
"""

import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# Configuration
PROJECT_ROOT = Path(__file__).parent.resolve()
LOGS_DIR = PROJECT_ROOT / "agent_logs"
LOGS_DIR.mkdir(exist_ok=True)

# SLURM configuration for small object detection focus
SLURM_SCRIPT_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --account=aip-robsc
#SBATCH --nodes=1
#SBATCH --mail-user=ashish.sinha@amii.ca
#SBATCH --mail-type=END,FAIL,BEGIN
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-gpu=32G
#SBATCH --time={time_limit}
#SBATCH --output=logs/%x-%A.out
#SBATCH --error=logs/%x-%A.err

ACCOUNT="aip-robsc"

if [ -z "$SLURM_JOB_ID" ]; then
    echo "Submitting job with account: $ACCOUNT"
    sbatch --account=$ACCOUNT "$0"
    exit
fi

set -e
export PATH="$HOME/.cargo/bin:$PATH"

echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"

cd {project_root}

# Launch training command
{training_command}
"""

# Small object detection focused configurations
# Beads are typically tiny objects that benefit from:
# - Higher resolution inputs
# - More queries
# - Focused data augmentation
# - Lower NMS thresholds

# MAX 50 EPOCHS - All configurations limited to 50 epochs for vulcan cluster
BEAD_FOCUS_CONFIGS = {
    "rf_detr_nano_beads": {
        "script": "train_rf_detr.py",
        "args": [
            "model=rfdetr",
            "model.rfdetr.size=nano",
            "model.rfdetr.num_queries=600",
            "model.rfdetr.num_select=400",
            "data=vulcan",
            "trainer.max_epochs=50",
            "optimizer.optimizer.lr=5e-4",
            "scheduler=cosine_warmup",
        ],
        "description": "RF-DETR Nano with increased queries for small objects (50 epochs)",
    },
    "rf_detr_small_beads": {
        "script": "train_rf_detr.py",
        "args": [
            "model=rfdetr",
            "model.rfdetr.size=small",
            "model.rfdetr.num_queries=600",
            "model.rfdetr.num_select=400",
            "data=vulcan",
            "trainer.max_epochs=50",
            "optimizer.optimizer.lr=5e-4",
            "scheduler=cosine_warmup",
        ],
        "description": "RF-DETR Small with increased queries for small objects (50 epochs)",
    },
    "rf_detr_medium_beads": {
        "script": "train_rf_detr.py",
        "args": [
            "model=rfdetr",
            "model.rfdetr.size=medium",
            "model.rfdetr.num_queries=600",
            "data=vulcan",
            "trainer.max_epochs=50",
            "optimizer.optimizer.lr=5e-4",
            "scheduler=step",
        ],
        "description": "RF-DETR Medium balanced for small object detection (50 epochs)",
    },
    "rt_detr_v2_beads": {
        "script": "train_rt_detr_v2.py",
        "args": [
            "model=rtdetr_v2",
            "model.rtdetr.num_queries=600",
            "model/backbone=resnet50",
            "model.backbone.train_backbone=True",
            "model.backbone.freeze_at_stage=2",
            "data=vulcan",
            "trainer.max_epochs=50",
            "optimizer.optimizer.lr=5e-4",
            "scheduler=cosine_warmup",
        ],
        "description": "RT-DETR v2 with more queries for tiny object detection (50 epochs)",
    },
    "rt_detr_dinov2_beads": {
        "script": "train_rt_detr_v2.py",
        "args": [
            "model=rtdetr_v2",
            "model.rtdetr.num_queries=600",
            "model/backbone=dinov2",
            "data=vulcan",
            "trainer.max_epochs=50",
            "optimizer.optimizer.lr=5e-5",
            "scheduler=cosine_warmup",
        ],
        "description": "RT-DETR v2 DINOv2 backbone for better small object features (50 epochs)",
    },
    "rf_detr_no300_beads": {
        "script": "train_rf_detr.py",
        "args": [
            "model=rfdetr",
            "model.rfdetr.size=small",
            "model.rfdetr.num_queries=600",
            "data=vulcan_no300_eval_train_plus_valgt300",
            "trainer.max_epochs=50",
            "optimizer.optimizer.lr=5e-4",
            "scheduler=cosine_warmup",
        ],
        "description": "RF-DETR with filtered dataset + promoted high-bbox images (50 epochs)",
    },
}


class ProcessMonitor:
    """Monitor running SLURM and Python processes."""

    def __init__(self):
        self.process_patterns = [
            "train_rf_detr.py",
            "train_rt_detr_v2.py",
            "train_yolov5.py",
            "python.*srun",
        ]

    def get_running_slurm_jobs(self) -> list[dict]:
        """Get currently running SLURM jobs for this user."""
        try:
            result = subprocess.run(
                [
                    "squeue",
                    "-u",
                    os.environ.get("USER", "unknown"),
                    "--format=%j,%i,%T,%N",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            jobs = []
            lines = result.stdout.strip().split("\n")[1:]  # Skip header
            for line in lines:
                if len(line.split(",")) >= 4:
                    parts = line.split(",")
                    jobs.append(
                        {
                            "name": parts[0],
                            "job_id": parts[1],
                            "state": parts[2],
                            "nodes": parts[3],
                        }
                    )
            return jobs
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            print(f"Warning: Could not query SLURM: {e}")
            return []

    def get_training_processes(self) -> list[dict]:
        """Get running Python training processes."""
        try:
            result = subprocess.run(
                ["ps", "aux"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            processes = []
            for line in result.stdout.split("\n"):
                if any(pattern in line for pattern in self.process_patterns):
                    parts = line.split()
                    if len(parts) >= 11:
                        processes.append(
                            {
                                "pid": parts[1],
                                "command": " ".join(parts[10:]),
                                "cpu": parts[2],
                                "mem": parts[3],
                            }
                        )
            return processes
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            print(f"Warning: Could not query processes: {e}")
            return []

    def has_active_training(self) -> bool:
        """Check if there are active training jobs running."""
        slurm_jobs = self.get_running_slurm_jobs()
        training_processes = self.get_training_processes()

        # Check for running SLURM jobs (not pending or completed)
        active_slurm = [
            j
            for j in slurm_jobs
            if j["state"] in ["RUNNING", "CONFIG", "COMPLETING", "PENDING"]
        ]

        return len(active_slurm) > 0 or len(training_processes) > 0

    def print_status(self):
        """Print current training status."""
        print("\n" + "=" * 60)
        print("TRAINING MONITOR STATUS")
        print("=" * 60)

        slurm_jobs = self.get_running_slurm_jobs()
        if slurm_jobs:
            print(f"\nSLURM Jobs ({len(slurm_jobs)}):")
            for job in slurm_jobs:
                print(
                    f"  [{job['job_id']}] {job['name']:20s} State: {job['state']:10s} Nodes: {job['nodes']}"
                )
        else:
            print("\nNo SLURM jobs running")

        processes = self.get_training_processes()
        if processes:
            print(f"\nLocal Training Processes ({len(processes)}):")
            for proc in processes:
                print(
                    f"  PID {proc['pid']:>6} CPU: {proc['cpu']:>5}% MEM: {proc['mem']:>5}% {proc['command'][:50]}"
                )
        else:
            print("\nNo local training processes")

        print("=" * 60 + "\n")


class TmuxSessionManager:
    """Manage tmux sessions for training."""

    @staticmethod
    def list_sessions() -> list[str]:
        """List all tmux sessions."""
        try:
            result = subprocess.run(
                ["tmux", "list-sessions", "-F", "#S"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.stdout.strip().split("\n")
        except (subprocess.SubprocessError, FileNotFoundError):
            return []

    @staticmethod
    def session_exists(session_name: str) -> bool:
        """Check if a tmux session exists."""
        sessions = TmuxSessionManager.list_sessions()
        return session_name in sessions

    @staticmethod
    def create_session(session_name: str, command: str) -> bool:
        """Create a new tmux session with the given command."""
        try:
            # Detach if already attached
            subprocess.run(
                ["tmux", "kill-session", "-t", session_name],
                capture_output=True,
                timeout=5,
            )
        except subprocess.SubprocessError:
            pass

        try:
            subprocess.Popen(
                ["tmux", "new-session", "-d", "-s", session_name, command],
                cwd=PROJECT_ROOT,
            )
            return True
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            print(f"Error creating tmux session: {e}")
            return False

    @staticmethod
    def attach_session(session_name: str):
        """Attach to an existing tmux session."""
        os.execvp("tmux", ["tmux", "attach-session", "-t", session_name])

    @staticmethod
    def send_command(session_name: str, command: str):
        """Send a command to a tmux session."""
        try:
            subprocess.run(
                ["tmux", "send-keys", "-t", session_name, command, "Enter"],
                cwd=PROJECT_ROOT,
                timeout=5,
            )
        except subprocess.SubprocessError as e:
            print(f"Error sending command to tmux session: {e}")


class TrainingAgent:
    """Main training monitor agent."""

    def __init__(self, use_tmux: bool = True):
        self.monitor = ProcessMonitor()
        self.tmux_manager = TmuxSessionManager() if use_tmux else None
        self.use_tmux = use_tmux

    def select_training_config(self, interactive: bool = True) -> Optional[dict]:
        """Select a training configuration focused on small object detection."""
        print("\n" + "=" * 60)
        print("AVAILABLE SMALL OBJECT DETECTION CONFIGURATIONS")
        print("=" * 60)

        configs = list(BEAD_FOCUS_CONFIGS.items())
        for i, (name, cfg) in enumerate(configs, 1):
            print(f"\n[{i}] {name}")
            print(f"    Script: {cfg['script']}")
            print(f"    Description: {cfg['description']}")
            print(f"    Args: {' '.join(cfg['args'][:3])}...")

        print("\n" + "=" * 60)

        if interactive:
            try:
                choice = input(
                    f"\nSelect configuration (1-{len(configs)}, or 'q' to quit): "
                )
                if choice.lower() == "q":
                    return None
                idx = int(choice) - 1
                if 0 <= idx < len(configs):
                    return configs[idx]
                else:
                    print("Invalid selection")
                    return None
            except (ValueError, KeyboardInterrupt):
                return None
        else:
            # Auto-select first config
            return configs[0]

    def build_training_command(self, config: tuple) -> str:
        """Build the full training command."""
        name, cfg = config
        cmd = f"uv run {cfg['script']} " + " ".join(cfg["args"])
        return cmd

    def submit_slurm_job(self, config: tuple) -> Optional[str]:
        """Submit a SLURM job for the given configuration."""
        name, cfg = config
        job_name = f"bead_{name}"
        training_cmd = self.build_training_command(config)

        script_content = SLURM_SCRIPT_TEMPLATE.format(
            job_name=job_name,
            time_limit="1-12:00:00",  # Reduced time for 50 epochs max
            project_root=PROJECT_ROOT,
            training_command=training_cmd,
        )

        script_path = LOGS_DIR / f"{job_name}.sh"
        script_path.write_text(script_content)
        script_path.chmod(0o755)

        # Create logs directory if needed
        (PROJECT_ROOT / "logs").mkdir(exist_ok=True)

        try:
            result = subprocess.run(
                ["sbatch", str(script_path)],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                job_id = result.stdout.split()[-1]
                print(f"\nSLURM job submitted: {job_id}")
                return job_id
            else:
                print(f"SLURM submission failed: {result.stderr}")
                return None
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            print(f"Error submitting SLURM job: {e}")
            return None

    def start_tmux_training(self, config: tuple) -> Optional[str]:
        """Start training in a tmux session."""
        name, cfg = config
        session_name = f"train_{name}"
        training_cmd = self.build_training_command(config)

        if self.tmux_manager.session_exists(session_name):
            print(f"Session {session_name} already exists")
            return session_name

        # Create log file
        log_file = LOGS_DIR / f"{session_name}.log"

        # Change to project directory and run
        full_cmd = f"cd {PROJECT_ROOT} && {training_cmd} 2>&1 | tee {log_file}"

        if self.tmux_manager.create_session(session_name, f"bash -c '{full_cmd}'"):
            print(f"\nTraining started in tmux session: {session_name}")
            print(f"Log file: {log_file}")
            print(f"Attach with: tmux attach -t {session_name}")
            return session_name
        else:
            print("Failed to create tmux session")
            return None

    def run(self, interactive: bool = True, auto_start: bool = False):
        """Main agent loop."""
        print("\n" + "=" * 60)
        print("TRAINING MONITOR AGENT - Small Object Detection Focus")
        print("=" * 60)
        print(f"Project root: {PROJECT_ROOT}")
        print(f"Logs directory: {LOGS_DIR}")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)

        while True:
            # Check current status
            self.monitor.print_status()

            has_training = self.monitor.has_active_training()

            if has_training:
                print("Active training detected. Monitoring...")
                time.sleep(60)  # Check again in 1 minute
                continue

            print("\nNo active training jobs detected.")

            if auto_start or interactive:
                config = self.select_training_config(interactive=interactive)
                if config is None:
                    if auto_start:
                        print("No config selected, exiting.")
                        break
                    time.sleep(30)
                    continue

                # Choose launch method
                if self.use_tmux:
                    print("\nStarting training in tmux session...")
                    session = self.start_tmux_training(config)
                    if session:
                        if interactive:
                            try:
                                response = input("\nAttach to session now? (y/n): ")
                                if response.lower() == "y":
                                    self.tmux_manager.attach_session(session)
                            except KeyboardInterrupt:
                                pass
                else:
                    print("\nSubmitting SLURM job...")
                    job_id = self.submit_slurm_job(config)
                    if job_id:
                        print(f"Job {job_id} submitted successfully")

                if auto_start:
                    print("Auto-start mode: launched one training, exiting.")
                    break
            else:
                print("Waiting for manual training launch...")
                time.sleep(300)  # Check again in 5 minutes

    def run_once(self, config_name: str, use_tmux: bool = True):
        """Run a specific configuration once."""
        if config_name not in BEAD_FOCUS_CONFIGS:
            print(f"Unknown configuration: {config_name}")
            print(f"Available: {list(BEAD_FOCUS_CONFIGS.keys())}")
            return

        config = (config_name, BEAD_FOCUS_CONFIGS[config_name])

        if use_tmux and self.use_tmux:
            self.start_tmux_training(config)
        else:
            self.submit_slurm_job(config)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Training Monitor Agent for Small Object Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Interactive mode
  %(prog)s --auto                   # Auto-start first available config
  %(prog)s --config rf_detr_small   # Run specific config
  %(prog)s --no-tmux                # Use SLURM instead of tmux
  %(prog)s --list                   # List available configs
        """,
    )

    parser.add_argument(
        "--auto",
        action="store_true",
        help="Auto-start training without prompting",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Run specific configuration by name",
    )
    parser.add_argument(
        "--no-tmux",
        action="store_true",
        help="Use SLURM submission instead of tmux",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available configurations and exit",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show current training status and exit",
    )

    args = parser.parse_args()

    if args.list:
        print("\nAvailable Small Object Detection Configurations:")
        print("=" * 60)
        for name, cfg in BEAD_FOCUS_CONFIGS.items():
            print(f"\n{name}:")
            print(f"  Description: {cfg['description']}")
            print(f"  Script: {cfg['script']}")
            print(f"  Args: {' '.join(cfg['args'])}")
        return

    if args.status:
        monitor = ProcessMonitor()
        monitor.print_status()
        return

    if args.config:
        agent = TrainingAgent(use_tmux=not args.no_tmux)
        agent.run_once(args.config, use_tmux=not args.no_tmux)
        return

    agent = TrainingAgent(use_tmux=not args.no_tmux)
    agent.run(interactive=not args.auto, auto_start=args.auto)


if __name__ == "__main__":
    main()

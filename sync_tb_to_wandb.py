import os
import glob
import wandb
import yaml
import argparse

PROJECT = "cell-detection-motifs"
CONFIGS_BASE = "configs/data"

def main():
    parser = argparse.ArgumentParser(description="Sync TensorBoard logs to WandB")
    parser.add_argument("--checkpoints_dir", type=str, default=None, help="Path to phase2 checkpoints directory")
    args = parser.parse_args()

    # Auto-detect checkpoints dir if not provided
    if args.checkpoints_dir is None:
        common_ckpt_paths = [
            "checkpoints/phase2",
            "/mnt/direct-attached/as-cells/checkpoints/phase2",
            "/mnt/direct-attached/checkpoints/phase2",
            "/project/aip-robsc/asinha/cellanome/DATA/checkpoints/phase2"
        ]
        for p in common_ckpt_paths:
            if os.path.isdir(p):
                args.checkpoints_dir = p
                break
        if args.checkpoints_dir is None:
            args.checkpoints_dir = "checkpoints/phase2" # fallback

    print(f"Scanning for TensorBoard logs in {args.checkpoints_dir}")
    
    # Find all tfevents files recursively
    tfevent_files = glob.glob(os.path.join(args.checkpoints_dir, "**", "events.out.tfevents*"), recursive=True)
    
    # Extract unique directories containing these files
    tb_dirs = set(os.path.dirname(f) for f in tfevent_files)
    
    if not tb_dirs:
        print("No TensorBoard logs found.")
        return

    print(f"Found {len(tb_dirs)} runs to sync. Starting upload...\n")
    
    for tb_dir in tb_dirs:
        run_name = os.path.basename(tb_dir)
        print(f"==================================================")
        print(f"Syncing: {run_name}")
        print(f"==================================================")
        
        # Try to extract some config metadata from the folder name
        # Format is usually: {motif_config_name}_{timestamp}
        # Example: motif_02_mc38_to_mc38_upward_2026-07-03_05-49-39
        motif_name = run_name
        timestamp = "unknown"
        if "_202" in run_name:
            parts = run_name.split("_202")
            motif_name = parts[0]
            timestamp = "202" + parts[1]

        # Parse config for train_datasets
        train_datasets = []
        split_motif = "Unknown"
        coverage = None
        
        yaml_paths = [
            os.path.join(CONFIGS_BASE, f"{motif_name}.yaml"),
            os.path.join(CONFIGS_BASE, "coverage_splits", f"{motif_name}.yaml")
        ]
        
        for y_path in yaml_paths:
            if os.path.exists(y_path):
                try:
                    with open(y_path, 'r') as f:
                        cfg = yaml.safe_load(f)
                        if cfg and "data" in cfg:
                            train_datasets = cfg["data"].get("train_datasets", [])
                            split_motif = cfg["data"].get("split_motif", "Unknown")
                            coverage = cfg["data"].get("coverage", None)
                    break
                except Exception as e:
                    print(f"Failed to parse {y_path}: {e}")

        # Using lora as default for phase2
        finetune_mode = "lora"

        # Initialize WandB, forcing it to sync TensorBoard logs from this directory
        run = wandb.init(
            project=PROJECT,
            name=run_name,
            sync_tensorboard=True,
            dir=tb_dir, # Store wandb local metadata in the same directory
            config={
                "motif_config": motif_name,
                "timestamp": timestamp,
                "local_dir": tb_dir,
                "note": "Retroactively synced from TensorBoard",
                "model.rfdetr.finetune_mode": finetune_mode,
                "data.train_datasets": train_datasets,
                "data.split_motif": split_motif,
                "data.coverage": coverage
            },
            reinit=True # Allow multiple runs in the same script
        )
        
        # Finish the run to flush logs to the cloud
        run.finish()

if __name__ == "__main__":
    main()

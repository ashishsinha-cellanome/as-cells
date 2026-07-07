import os
import glob
import wandb

PROJECT = "cell-detection-motifs"
BASE_DIR = "/mnt/direct-attached/as-cells/checkpoints/phase2/"

def main():
    print(f"Scanning for TensorBoard logs in {BASE_DIR}")
    
    # Find all tfevents files recursively
    tfevent_files = glob.glob(os.path.join(BASE_DIR, "**", "events.out.tfevents*"), recursive=True)
    
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
                "note": "Retroactively synced from TensorBoard"
            },
            reinit=True # Allow multiple runs in the same script
        )
        
        # Finish the run to flush logs to the cloud
        run.finish()

if __name__ == "__main__":
    main()

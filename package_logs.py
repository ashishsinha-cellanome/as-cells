import os
import glob
import shutil
import tarfile
import argparse

def main():
    parser = argparse.ArgumentParser(description="Package TensorBoard events and Hydra logs")
    parser.add_argument("--checkpoints_dir", type=str, default=None, help="Path to phase2 checkpoints directory")
    parser.add_argument("--logs_dir", type=str, default=None, help="Path to hydra logs/outputs directory")
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

    # Auto-detect logs dir if not provided
    if args.logs_dir is None:
        common_log_paths = [
            "outputs",
            "logs",
            "/mnt/direct-attached/logs",
            "/project/aip-robsc/asinha/cellanome/logs"
        ]
        for p in common_log_paths:
            if os.path.isdir(p):
                args.logs_dir = p
                break
        if args.logs_dir is None:
            args.logs_dir = "outputs" # fallback

    print(f"Using checkpoints directory: {args.checkpoints_dir}")
    print(f"Using logs directory: {args.logs_dir}")

    staging_dir = "tar_staging"
    if os.path.exists(staging_dir):
        shutil.rmtree(staging_dir)
    os.makedirs(staging_dir)

    run_dirs = glob.glob(os.path.join(args.checkpoints_dir, "*"))
    
    for run_dir in run_dirs:
        if not os.path.isdir(run_dir):
            continue
            
        run_name = os.path.basename(run_dir)
        events_files = glob.glob(os.path.join(run_dir, "events.out.tfevents*"))
        
        if not events_files:
            continue
            
        # Create run folder in staging
        staging_run_dir = os.path.join(staging_dir, run_name)
        os.makedirs(staging_run_dir, exist_ok=True)
        
        # Copy events files
        for ev_file in events_files:
            shutil.copy2(ev_file, staging_run_dir)
            
        # Try to find corresponding log file in outputs/logs dir
        if "_202" in run_name:
            parts = run_name.split("_202")
            date_time_str = "202" + parts[1]
            try:
                date_str, time_str = date_time_str.split("_")
                
                # Check logs_dir/YYYY-MM-DD/HH-MM-SS/
                output_dir = os.path.join(args.logs_dir, date_str, time_str)
                log_files = glob.glob(os.path.join(output_dir, "*.log"))
                
                for log_file in log_files:
                    log_name = os.path.basename(log_file)
                    shutil.copy2(log_file, os.path.join(staging_run_dir, log_name))
            except Exception as e:
                print(f"Error parsing date/time for {run_name}: {e}")

    print("Creating combined_logs.tar.gz...")
    with tarfile.open("combined_logs.tar.gz", "w:gz") as tar:
        tar.add(staging_dir, arcname=".")
        
    print("Done! Cleaning up staging dir...")
    shutil.rmtree(staging_dir)

if __name__ == "__main__":
    main()

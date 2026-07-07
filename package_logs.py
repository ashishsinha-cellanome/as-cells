import os
import glob
import shutil
import tarfile

def main():
    base_checkpoints = "checkpoints/phase2"
    base_outputs = "outputs"
    staging_dir = "tar_staging"

    if os.path.exists(staging_dir):
        shutil.rmtree(staging_dir)
    os.makedirs(staging_dir)

    run_dirs = glob.glob(os.path.join(base_checkpoints, "*"))
    
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
            
        # Try to find corresponding log file in outputs/
        # Format: motif_name_YYYY-MM-DD_HH-MM-SS
        if "_202" in run_name:
            parts = run_name.split("_202")
            date_time_str = "202" + parts[1]
            try:
                date_str, time_str = date_time_str.split("_")
                
                # Check outputs/YYYY-MM-DD/HH-MM-SS/
                output_dir = os.path.join(base_outputs, date_str, time_str)
                log_files = glob.glob(os.path.join(output_dir, "*.log"))
                
                for log_file in log_files:
                    log_name = os.path.basename(log_file)
                    shutil.copy2(log_file, os.path.join(staging_run_dir, log_name))
            except Exception as e:
                print(f"Error parsing date/time for {run_name}: {e}")

    # Create tar archive
    print("Creating combined_logs.tar.gz...")
    with tarfile.open("combined_logs.tar.gz", "w:gz") as tar:
        tar.add(staging_dir, arcname=".")
        
    print("Done! Cleaning up staging dir...")
    shutil.rmtree(staging_dir)

if __name__ == "__main__":
    main()

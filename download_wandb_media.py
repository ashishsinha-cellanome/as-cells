import wandb
import os
import argparse
from pathlib import Path

def get_dataset_type(targets):
    targets_str = str(targets).lower()
    has_u87 = 'u87' in targets_str
    has_20250108_neuron = '20250108_neuron' in targets_str or 'uncaged-202501' in targets_str
    has_20250305_neuron = '20250305_neuron' in targets_str
    
    if has_u87 and has_20250108_neuron:
        return 'u87-2025-neuron-adhered'
    elif has_u87 and not has_20250108_neuron:
        return '2024-u87'
    elif not has_u87 and has_20250108_neuron and has_20250305_neuron:
        return '2025-neuron-adhered'
    elif not has_u87 and has_20250108_neuron and not has_20250305_neuron:
        return '20250108_neuron-adhered'
    
    return 'unknown'

def download_wandb_files(output_dir):
    api = wandb.Api(timeout=120)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Connecting to W&B project: sinashish/cell-detection-motifs")
    
    filters = {
        "$and": [
            {"config.model.value.rfdetr.finetune_mode": "lora"},
            {"state": {"$nin": ["crashed", "failed", "killed"]}},
        ]
    }
    
    runs = api.runs(
        "sinashish/cell-detection-motifs",
        filters=filters,
        per_page=50,
        include_sweeps=False,
    )
    
    print(f"Found {len(runs)} total successful LoRA runs in the project.")
    
    downloaded_count = 0
    for run in runs:
        # Fetch fully-hydrated config
        full_run = api.run(f"sinashish/cell-detection-motifs/{run.id}")
        config = full_run.config
        
        data_cfg = config.get("data", {})
        
        # Pull targets
        targets = data_cfg.get("target_datasets", [])
        train_datasets = data_cfg.get("train_datasets", [])
        all_targets = targets + train_datasets
        
        target_str = get_dataset_type(all_targets)
        
        if target_str == 'unknown':
            continue
            
        # Determine the rank (defaulting to 64 if not explicitly saved)
        rank = config.get("model", {}).get("rfdetr", {}).get("lora", {}).get("r", 64)
        
        # Determine the fraction
        frac = data_cfg.get("lora_frac", data_cfg.get("target_data_frac"))
        if frac is None:
            continue
            
        expected_filename = f"lora_r{rank}_{frac}_{target_str}.html"
        dest_path = out_path / expected_filename
        
        if dest_path.exists():
            print(f"  -> Already exists, skipping: {expected_filename}")
            continue
            
        # Download the HTML file
        found_html = False
        try:
            for file in full_run.files(pattern="media/html/%"):
                if file.name.endswith(".html"):
                    print(f"  -> Downloading {file.name} to {expected_filename} (Run: {run.name})")
                    file.download(root=str(out_path), replace=True)
                    
                    # Move and rename the downloaded nested file out of the media/html/... folder
                    actual_path = out_path / file.name
                    actual_path.rename(dest_path)
                    
                    downloaded_count += 1
                    found_html = True
                    break
        except Exception as e:
            print(f"  -> Failed to fetch files for {run.name}: {e}")
            
        if not found_html:
            pass # No media found

    print(f"\nDone! Downloaded {downloaded_count} new files to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download W&B media specific to LoRA tracking.")
    parser.add_argument("--output-dir", type=str, default="coverage_exp/lora_exp", help="Directory to save downloaded files")
    args = parser.parse_args()
    download_wandb_files(args.output_dir)

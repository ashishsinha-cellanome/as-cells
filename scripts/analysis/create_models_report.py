import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import wandb
import socket
import json
import csv
import os
import argparse
from concurrent.futures import ThreadPoolExecutor

parser = argparse.ArgumentParser(description="Generate a CSV report of wandb runs for a specific machine.")
parser.add_argument("--host", type=str, default=None, help="The hostname to filter by (defaults to the current machine's hostname).")
args = parser.parse_args()

api = wandb.Api()
projects = ["cell-detection", "rt-detr-cell-detection", "cellanome"]

# Get the target host. Use exact string matching but handle simple domain suffixes
target_host = args.host if args.host else socket.gethostname()
target_host_short = target_host.split('.')[0].lower()

results = []

def process_run(run, proj):
    try:
        host = getattr(run, "host", None)
        cmd = ""
        
        # Pull metadata to find hostname if missing, and to capture the exact CLI command
        try:
            tmp_root = f"/tmp/wandb_meta_{run.id}"
            meta_file = run.file("wandb-metadata.json")
            if meta_file:
                downloaded_file = meta_file.download(root=tmp_root, replace=True)
                with open(downloaded_file.name) as f:
                    meta = json.load(f)
                    if not host:
                        host = meta.get("host")
                    program = meta.get("program", "")
                    cli_args = meta.get("args", [])
                    if program:
                        cmd = f"python {program} " + " ".join(cli_args)
        except Exception:
            pass
            
        if not host:
            return None
            
        # Smart hostname comparison
        run_host_short = str(host).split('.')[0].lower()
        if run_host_short != target_host_short:
            return None
            
        summary = run.summary
        
        # Overall mAP 50-95
        best_val_map = summary.get("val/mAP_50_95") or summary.get("val/detailed_bbox_map") or summary.get("val/map") or summary.get("val/detailed_segm_map") or summary.get("test/map") or ""
        best_val_map_ema = summary.get("val/ema_mAP_50_95") or summary.get("val/detailed_bbox_map_ema") or summary.get("val/map_ema") or summary.get("val/detailed_segm_map_ema") or summary.get("test/map_ema") or ""
        
        # Overall mAP 50
        best_val_map_50 = summary.get("val/mAP_50") or summary.get("val/detailed_bbox_map_50") or summary.get("val/map_50") or summary.get("val/detailed_segm_map_50") or summary.get("test/map_50") or ""
        best_val_map_50_ema = summary.get("val/ema_mAP_50") or summary.get("val/detailed_bbox_map_50_ema") or summary.get("val/map_50_ema") or summary.get("val/detailed_segm_map_50_ema") or summary.get("test/map_50_ema") or ""

        # Per class avg mAP (50-95)
        map_cell = summary.get("val/detailed_bbox_map_cell") or summary.get("test/map_cell") or ""
        map_bead = summary.get("val/detailed_bbox_map_bead") or summary.get("test/map_bead") or ""
        map_cell_adhered = summary.get("val/detailed_bbox_map_cell-adhered") or summary.get("test/map_cell-adhered") or ""
        map_soma = summary.get("val/detailed_bbox_map_soma") or summary.get("test/map_soma") or ""

        # Per class mAP 50
        map50_cell = summary.get("val/detailed_bbox_map_50_cell") or summary.get("test/map_50_cell") or ""
        map50_bead = summary.get("val/detailed_bbox_map_50_bead") or summary.get("test/map_50_bead") or ""
        map50_cell_adhered = summary.get("val/detailed_bbox_map_50_cell-adhered") or summary.get("test/map_50_cell-adhered") or ""
        map50_soma = summary.get("val/detailed_bbox_map_50_soma") or summary.get("test/map_50_soma") or ""
        
        # Skip runs that crashed before logging metrics
        if not best_val_map and not best_val_map_ema:
            return None

        config = run.config
        
        # Parse or infer the model type
        model_type = "unknown"
        if "model" in config and isinstance(config["model"], dict):
            model_type = config["model"].get("_target_", config["model"].get("name", "unknown"))
        elif "model_type" in config:
            model_type = config["model_type"]
        else:
            if "rf_detr" in run.name.lower() or "rfdetr" in run.name.lower() or "rfedert" in run.name.lower():
                model_type = "RF-DETR"
                if "seg" in run.name.lower(): model_type += "-Seg"
            elif "rt_detr" in run.name.lower() or "rtdetr" in run.name.lower():
                model_type = "RT-DETR"
            elif "yolo" in run.name.lower():
                model_type = "YOLO"
            elif "mask2former" in run.name.lower():
                model_type = "Mask2Former"
            else:
                model_type = run.name

        # Calculate checkpoint path
        ckpt_dir = ""
        if "checkpointing" in config and isinstance(config["checkpointing"], dict):
            ckpt_dir = config["checkpointing"].get("save_dir", "")
        elif "data" in config and isinstance(config["data"], dict):
            ckpt_dir = config["data"].get("save_dir", "")
        
        if not ckpt_dir:
            ckpt_dir = "checkpoints" # Default fallback for legacy runs

        ckpt_path = os.path.join(ckpt_dir, config.get("run_name", run.name))

        return {
            "Run Name": run.name,
            "Project": proj,
            "Model Type": model_type,
            "Overall avg.mAP (50-95)": best_val_map,
            "Overall mAP@0.5": best_val_map_50,
            "Overall EMA avg.mAP (50-95)": best_val_map_ema,
            "Overall EMA mAP@0.5": best_val_map_50_ema,
            "avg.mAP_cell": map_cell,
            "avg.mAP_bead": map_bead,
            "avg.mAP_cell-adhered": map_cell_adhered,
            "avg.mAP_soma": map_soma,
            "mAP@0.5_cell": map50_cell,
            "mAP@0.5_bead": map50_bead,
            "mAP@0.5_cell-adhered": map50_cell_adhered,
            "mAP@0.5_soma": map50_soma,
            "Checkpoint Path": ckpt_path,
            "Training Command": cmd
        }
    except Exception:
        return None

for proj in projects:
    try:
        print(f"Fetching runs for project: {proj} for host: {target_host_short}...")
        runs = list(api.runs(proj))
        
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(process_run, run, proj) for run in runs]
            for future in futures:
                res = future.result()
                if res:
                    results.append(res)
                    print(f"  Added {res['Run Name']}")
    except Exception as e:
        print(f"Error on project {proj}: {e}")

csv_filename = f"{target_host_short}_models_report.csv"
fieldnames = [
    "Run Name", "Project", "Model Type", 
    "Overall avg.mAP (50-95)", "Overall mAP@0.5", 
    "Overall EMA avg.mAP (50-95)", "Overall EMA mAP@0.5",
    "avg.mAP_cell", "avg.mAP_bead", "avg.mAP_cell-adhered", "avg.mAP_soma",
    "mAP@0.5_cell", "mAP@0.5_bead", "mAP@0.5_cell-adhered", "mAP@0.5_soma",
    "Checkpoint Path", "Training Command"
]

with open(csv_filename, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

print(f"\nReport generated: {csv_filename}")

import os
import torch
import numpy as np
import wandb
import csv
import tabulate
from omegaconf import OmegaConf

def run_benchmark(inner_model, dummy_input, batch_size, num_runs=50, label=""):
    print(f"--- Benchmarking {label} ---")
    print("Warming up...")
    with torch.no_grad():
        for _ in range(10):
            try:
                _ = inner_model(dummy_input)
            except TypeError:
                _ = inner_model(*dummy_input) if isinstance(dummy_input, tuple) else inner_model(**dummy_input) if isinstance(dummy_input, dict) else inner_model(dummy_input)
            
    torch.cuda.synchronize()
    print("Benchmarking Forward Pass...")
    
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_runs)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_runs)]
    
    with torch.no_grad():
        for i in range(num_runs):
            start_events[i].record()
            try:
                _ = inner_model(dummy_input)
            except TypeError:
                _ = inner_model(*dummy_input) if isinstance(dummy_input, tuple) else inner_model(**dummy_input) if isinstance(dummy_input, dict) else inner_model(dummy_input)
            end_events[i].record()
            
    torch.cuda.synchronize()
    
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)] # in milliseconds
    avg_time_ms = np.mean(times)
    time_per_image_ms = avg_time_ms / batch_size
    fps = 1000.0 / time_per_image_ms
    
    return avg_time_ms, time_per_image_ms, fps

def get_model_and_input(config, device, batch_size):
    model_name = config.get("model", {}).get("name", "")
    if not model_name:
        model_name = config.get("model_type", "")
        
    input_size = config.get("model", {}).get("input_size", 640)
    dummy_input = torch.randn(batch_size, 3, input_size, input_size, device=device)
    
    inner_model = None

    if "rtdetr" in model_name.lower():
        from transformers import RTDetrForObjectDetection, AutoConfig
        from models.custom_rt_detr_with_dinov2_backbone import RTDetrV2ForObjectDetectionWithCustomBackbone, RTDetrV2ConfigWithCustomBackBone
        OmegaConf.register_new_resolver("extract_name", lambda path: path.split("/")[-1], replace=True)
        
        pretrained_path = config.get("model", {}).get("rtdetr", {}).get("pretrained_name_or_path", "PekingU/rtdetr_v2_r50vd")
        rtdetr_name = config.get("model", {}).get("rtdetr", {}).get("model_name", "rtdetr_v2_r50vd")
        
        if "rtdetr_v2" in rtdetr_name:
            model_cls = RTDetrV2ForObjectDetectionWithCustomBackbone
            try:
                model_config = RTDetrV2ConfigWithCustomBackBone.from_pretrained(pretrained_path)
            except:
                model_config = RTDetrV2ConfigWithCustomBackBone.from_pretrained("PekingU/rtdetr_v2_r50vd")
        else:
            model_cls = RTDetrForObjectDetection
            try:
                model_config = AutoConfig.from_pretrained(pretrained_path)
            except:
                model_config = AutoConfig.from_pretrained("PekingU/rtdetr_r50vd")
                
        model_config.num_labels = 4
        try:
            inner_model = model_cls.from_pretrained(pretrained_path, config=model_config, ignore_mismatched_sizes=True)
        except:
            inner_model = model_cls(config=model_config)
            
    elif "rf_detr" in model_name.lower() or "rfdetr" in model_name.lower():
        if "seg" in model_name.lower():
            from rfdetr import RFDETRSegLarge, RFDETRSegMedium, RFDETRSegSmall
            if "small" in model_name.lower(): m = RFDETRSegSmall(num_classes=4)
            elif "medium" in model_name.lower(): m = RFDETRSegMedium(num_classes=4)
            else: m = RFDETRSegLarge(num_classes=4)
        else:
            from rfdetr import RFDETR
            m = RFDETR(num_classes=4)
        inner_model = m.model.model
        
    elif "yolo" in model_name.lower():
        from models.yolov5_lightning_module import YOLOv5LightningModule
        repo_path = config.get("model", {}).get("yolov5", {}).get("repo_path", "")
        # fallback if repo path isn't local
        if not os.path.exists(repo_path): repo_path = os.path.join(os.getcwd(), "models", "yolov5")
        try:
            m = YOLOv5LightningModule(config=OmegaConf.create(config), yolo_repo_path=repo_path)
            inner_model = m.model
        except Exception as e:
            print(f"Failed to load YOLOv5: {e}")
            
    elif "mask2former" in model_name.lower():
        from models.mask2former_model import build_original_mask2former, build_mask2former_with_dinov2_backbone
        bb_name = config.get("model", {}).get("backbone", {}).get("model_name", "")
        if "dinov2" in bb_name:
            inner_model = build_mask2former_with_dinov2_backbone(id2label={0:'cell', 1:'bead', 2:'cell-adhered', 3:'soma'}, mask2former_pretrained_name_or_path="facebook/mask2former-swin-base-coco-panoptic")
        else:
            inner_model = build_original_mask2former(id2label={0:'cell', 1:'bead', 2:'cell-adhered', 3:'soma'}, mask2former_pretrained_name_or_path="facebook/mask2former-swin-base-coco-panoptic")
            
    elif "deim" in model_name.lower():
        from models.deim_v2_lightning_module import DeimV2LightningModule
        try:
            m = DeimV2LightningModule(config=OmegaConf.create(config))
            inner_model = m.model
        except Exception as e:
            print(f"Failed to load DEIMv2: {e}")

    if inner_model is not None:
        inner_model.to(device)
        inner_model.eval()
        if torch.cuda.is_available():
            inner_model = inner_model.bfloat16()
            dummy_input = dummy_input.bfloat16()
            
    return inner_model, dummy_input

def get_checkpoint_path(run):
    config = run.config
    ckpt_dir = ""
    if "checkpointing" in config and isinstance(config["checkpointing"], dict):
        ckpt_dir = config["checkpointing"].get("save_dir", "")
    elif "data" in config and isinstance(config["data"], dict):
        ckpt_dir = config["data"].get("save_dir", "")
    
    if not ckpt_dir:
        ckpt_dir = "checkpoints"
        
    run_name = config.get("run_name", run.name)
    ckpt_path = os.path.join(ckpt_dir, run_name)
    
    # Try different combinations of paths that might exist locally
    possible_paths = [
        ckpt_path,
        ckpt_path + ".pt",
        ckpt_path + ".pth",
        os.path.join(ckpt_path, "last.ckpt"),
        os.path.join(ckpt_path, "checkpoint_best_ema.pth"),
        os.path.join(ckpt_path, "checkpoint_best_regular.pth"),
    ]
    
    for p in possible_paths:
        if os.path.exists(p):
            return p
            
    return None

def benchmark_all_models(batch_size=1):
    api = wandb.Api()
    projects = ["cell-detection", "rt-detr-cell-detection", "cellanome"]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    results = []
    
    for proj in projects:
        try:
            print(f"\nFetching runs for project: {proj}")
            runs = list(api.runs(proj))
            print(f"Found {len(runs)} runs. Processing...")
            
            for run in runs:
                config = run.config
                summary = run.summary
                model_name = config.get("model", {}).get("name", config.get("model_type", "unknown"))
                
                ckpt_path = get_checkpoint_path(run)
                
                if not ckpt_path:
                    print(f"Skipping {run.name} ({model_name}): Checkpoint not found locally.")
                    continue
                    
                print(f"\n---> Benchmarking {run.name} ({model_name}) from {ckpt_path}")
                
                inner_model, dummy_input = get_model_and_input(config, device, batch_size)
                if inner_model is None:
                    print(f"Failed to instantiate model architecture for {model_name}. Skipping.")
                    continue
                    
                # 1. Calculate FLOPS
                gflops = 0.0
                try:
                    from fvcore.nn import FlopCountAnalysis
                    flops = FlopCountAnalysis(inner_model, dummy_input)
                    gflops = (flops.total() / 1e9) / batch_size
                    print(f"GFLOPS per image: {gflops:.2f}")
                except Exception as e:
                    print(f"Could not compute FLOPS: {e}")
                    
                # 2. Baseline Inference Speed
                avg_ms, img_ms, fps = run_benchmark(inner_model, dummy_input, batch_size, label="Baseline")
                
                # 3. Torch Compile Inference Speed
                compile_fps = 0.0
                if torch.cuda.is_available():
                    try:
                        print("Compiling model...")
                        compiled_model = torch.compile(inner_model, mode="reduce-overhead")
                        _, _, compile_fps = run_benchmark(compiled_model, dummy_input, batch_size, label="torch.compile")
                    except Exception as e:
                        print(f"Failed to compile model: {e}")
                
                # 4. Extract validation metrics
                is_rfdetr_seg = "seg" in model_name.lower() and "rf" in model_name.lower()
                is_mask2former = "mask2former" in model_name.lower()

                if is_rfdetr_seg:
                    map_val = summary.get("val/detailed_bbox_map", "")
                    map_ema = summary.get("val/detailed_bbox_map_ema", "")
                elif is_mask2former:
                    map_val = summary.get("val/segm_map", "")
                    map_ema = summary.get("val/segm_map_ema", "")
                else:
                    map_val = summary.get("val/map", "")
                    map_ema = summary.get("val/map_ema", "")

                res = {
                    "Project": proj,
                    "Run Name": run.name,
                    "Model": model_name,
                    "mAP (50-95)": f"{map_val:.4f}" if isinstance(map_val, float) else map_val,
                    "EMA mAP": f"{map_ema:.4f}" if isinstance(map_ema, float) else map_ema,
                    "GFLOPS": f"{gflops:.2f}",
                    "Baseline FPS": f"{fps:.2f}",
                    "Compiled FPS": f"{compile_fps:.2f}"
                }
                results.append(res)
                print(f"Completed {run.name}.")
                
                # We can break early for local testing to avoid doing 800 runs if they all hit the fallback path.
                # But since we use the fallback `output/checkpoint_best_ema.pth`, it might benchmark many!
                # Let's break after 1 successful benchmark per model type to avoid an infinite loop of testing on the same local fallback.
                
        except Exception as e:
            print(f"Error on project {proj}: {e}")

    # Deduplicate results if they fallback to the same local output checkpoint
    # We will just write all of them to CSV.
    if results:
        print("\n" + "="*80)
        print("FINAL REPORT")
        print("="*80)
        print(tabulate.tabulate(results, headers="keys", tablefmt="pretty"))
        
        with open("benchmark_all_models_report.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print("\nReport saved to benchmark_all_models_report.csv")
    else:
        print("\nNo checkpoints found locally to benchmark.")

if __name__ == "__main__":
    benchmark_all_models(batch_size=1)

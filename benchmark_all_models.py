import os
import torch
import numpy as np
import wandb
import csv
import tabulate
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("extract_name", lambda path: path.split("/")[-1], replace=True)

def run_benchmark(inner_model, postprocess_fn, dummy_input, batch_size, num_runs=50, label=""):
    orig_sizes = torch.tensor([[dummy_input.shape[2], dummy_input.shape[3]] for _ in range(batch_size)], device=dummy_input.device) if hasattr(dummy_input, "device") else None
    print(f"--- Benchmarking {label} ---")
    print("Warming up...")
    with torch.no_grad():
        for _ in range(10):
            try:
                outputs = inner_model(dummy_input)
            except TypeError:
                outputs = inner_model(*dummy_input) if isinstance(dummy_input, tuple) else inner_model(**dummy_input) if isinstance(dummy_input, dict) else inner_model(dummy_input)
            if postprocess_fn is not None:
                try:
                    _ = postprocess_fn(outputs, orig_sizes)
                except Exception:
                    pass
            
    torch.cuda.synchronize()
    print("Benchmarking Forward Pass + Post-processing...")
    
    start_events_fwd = [torch.cuda.Event(enable_timing=True) for _ in range(num_runs)]
    end_events_fwd = [torch.cuda.Event(enable_timing=True) for _ in range(num_runs)]
    start_events_post = [torch.cuda.Event(enable_timing=True) for _ in range(num_runs)]
    end_events_post = [torch.cuda.Event(enable_timing=True) for _ in range(num_runs)]
    
    with torch.no_grad():
        for i in range(num_runs):
            start_events_fwd[i].record()
            try:
                outputs = inner_model(dummy_input)
            except TypeError:
                outputs = inner_model(*dummy_input) if isinstance(dummy_input, tuple) else inner_model(**dummy_input) if isinstance(dummy_input, dict) else inner_model(dummy_input)
            end_events_fwd[i].record()
            
            start_events_post[i].record()
            if postprocess_fn is not None:
                try:
                    _ = postprocess_fn(outputs, orig_sizes)
                except Exception:
                    pass
            end_events_post[i].record()
            
    torch.cuda.synchronize()
    
    fwd_times = [s.elapsed_time(e) for s, e in zip(start_events_fwd, end_events_fwd)]
    post_times = [s.elapsed_time(e) for s, e in zip(start_events_post, end_events_post)]
    
    avg_fwd_ms = np.mean(fwd_times) / batch_size
    avg_post_ms = np.mean(post_times) / batch_size
    
    total_ms = avg_fwd_ms + avg_post_ms
    fps = 1000.0 / total_ms if total_ms > 0 else 0.0
    
    return avg_fwd_ms, avg_post_ms, total_ms, fps

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
            
        if config.get("model", {}).get("backbone", {}).get("type", "") == "dinov2":
            from models.backbone_factory import build_backbone
            # For dinov2 with rtdetr_v2, the custom backbone builder expects the rtdetr_model_name string 
            # to determine the intermediate feature sizes (e.g. [128, 256, 512] for r18vd vs [512, 1024, 2048] for r50vd).
            # If the user is evaluating rtdetr_v2_r50vd but we want to test dinov2, we need to pass the correct model string.
            backbone_model, backbone_config_obj, _ = build_backbone(config.model.backbone, rtdetr_name)
            inner_model.model.backbone = backbone_model
            inner_model.config.backbone_config = backbone_config_obj
            inner_model.config.encoder_in_channels = backbone_config_obj.intermediate_channel_sizes
            inner_model.config.use_timm_backbone = False
            
    elif "rf_detr" in model_name.lower() or "rfdetr" in model_name.lower():
        group_detr = config.get("model", {}).get("rfdetr", {}).get("group_detr", 11)
        if "seg" in model_name.lower():
            from rfdetr import RFDETRSegLarge, RFDETRSegMedium, RFDETRSegSmall
            if "small" in model_name.lower(): m = RFDETRSegSmall(num_classes=4, group_detr=group_detr)
            elif "medium" in model_name.lower(): m = RFDETRSegMedium(num_classes=4, group_detr=group_detr)
            else: m = RFDETRSegLarge(num_classes=4, group_detr=group_detr)
        else:
            from rfdetr import RFDETR
            m = RFDETR(num_classes=4, group_detr=group_detr)
        inner_model = m.model.model
        
    elif "yolo" in model_name.lower():
        import sys, os
        import yaml
        model_key = "yolov26" if "yolov26" in config.get("model", {}).get("name", "") else "yolov5"
        
        if model_key == "yolov5":
            repo_path = os.path.join(os.getcwd(), "models", "yolov5")
            if repo_path not in sys.path: 
                sys.path.insert(0, repo_path)
            try:
                # Avoid Hydra interpolation error by reading directly
                model_cfg_path = "models/yolov5m.yaml"
                yaml_cfg_path = os.path.join(repo_path, model_cfg_path)
                with open(yaml_cfg_path) as f:
                    yaml_cfg = yaml.safe_load(f)
                yaml_cfg['nc'] = 4
                
                from models.yolo import Model
                inner_model = Model(cfg=yaml_cfg, ch=3, nc=4)
                
                def yolo_postprocess(outputs, orig_sizes=None):
                    from utils.general import non_max_suppression
                    preds = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
                    return non_max_suppression(preds, 0.05, 0.5)
                postprocess_fn = yolo_postprocess
            except Exception as e:
                print(f"Failed to load YOLOv5: {e}")
            finally:
                if repo_path in sys.path:
                    sys.path.remove(repo_path)
                for k in list(sys.modules.keys()):
                    if k.startswith(("models", "utils", "detect", "export")):
                        del sys.modules[k]
        else:
            # YOLOv26 uses the ultralytics package
            try:
                from ultralytics import YOLO
                m = YOLO("yolo26m.pt")
                inner_model = m.model
                def yolo_v8_postprocess(outputs, orig_sizes=None):
                    from ultralytics.utils.ops import non_max_suppression
                    # Ultralytics v8 outputs tuple (preds, ...). non_max_suppression expects preds
                    preds = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
                    return non_max_suppression(preds, 0.05, 0.5)
                postprocess_fn = yolo_v8_postprocess
            except Exception as e:
                print(f"Failed to load YOLOv26: {e}")
            
    elif "mask2former" in model_name.lower():
        import sys, os
        if os.getcwd() not in sys.path:
            sys.path.insert(0, os.getcwd())
        elif sys.path[0] != os.getcwd():
            sys.path.remove(os.getcwd())
            sys.path.insert(0, os.getcwd())
        from models.mask2former_model import build_original_mask2former, build_mask2former_with_dinov2_backbone
        
        m2f_pretrained = config.get("model", {}).get("mask2former", {}).get("pretrained_name_or_path", "")
        if not m2f_pretrained:
            if "large" in model_name.lower() or "large" in config.get("model", {}).get("backbone", {}).get("name", "").lower():
                m2f_pretrained = "facebook/mask2former-swin-large-mapillary-vistas-panoptic"
            else:
                m2f_pretrained = "facebook/mask2former-swin-base-coco-panoptic"
                
        bb_name = config.get("model", {}).get("backbone", {}).get("model_name", "")
        if "dinov2" in bb_name:
            inner_model = build_mask2former_with_dinov2_backbone(
                id2label={0:'cell', 1:'bead', 2:'cell-adhered', 3:'soma'}, 
                mask2former_pretrained_name_or_path=m2f_pretrained,
                backbone_pretrained_name_or_path=config.get("model", {}).get("backbone", {}).get("pretrained_name_or_path", "facebook/dinov2-base")
            )
        else:
            inner_model = build_original_mask2former(id2label={0:'cell', 1:'bead', 2:'cell-adhered', 3:'soma'}, mask2former_pretrained_name_or_path=m2f_pretrained)
            
    elif "deim" in model_name.lower():
        import sys, os
        if os.getcwd() not in sys.path:
            sys.path.insert(0, os.getcwd())
        elif sys.path[0] != os.getcwd():
            sys.path.remove(os.getcwd())
            sys.path.insert(0, os.getcwd())
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
            
    return inner_model, postprocess_fn, dummy_input

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
                
                inner_model, postprocess_fn, dummy_input = get_model_and_input(config, device, batch_size)
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
                avg_fwd_native, avg_post_native, total_ms_native, fps_native = run_benchmark(inner_model, postprocess_fn, dummy_input, batch_size, label="Baseline")
                
                # 3. Torch Compile Inference Speed
                compile_fps = 0.0
                if torch.cuda.is_available():
                    try:
                        print("Compiling model...")
                        compiled_model = torch.compile(inner_model, mode="reduce-overhead")
                        _, _, _, compile_fps = run_benchmark(compiled_model, postprocess_fn, dummy_input, batch_size, label="torch.compile")
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
                    "Batch Size": batch_size,
                    "mAP (50-95)": f"{map_val:.4f}" if isinstance(map_val, float) else map_val,
                    "EMA mAP": f"{map_ema:.4f}" if isinstance(map_ema, float) else map_ema,
                    "GFLOPS/img": f"{gflops:.2f}",
                    "Forward Time/img (ms)": f"{avg_fwd_native:.2f}",
                    "Postproc Time/img (ms)": f"{avg_post_native:.2f}",
                    "Total Time/img (ms)": f"{total_ms_native:.2f}",
                    "Baseline FPS": f"{fps_native:.2f}",
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

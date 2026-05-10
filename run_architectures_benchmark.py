import torch
import numpy as np
import tabulate
import csv
from omegaconf import OmegaConf
from hydra import compose, initialize
from fvcore.nn import FlopCountAnalysis

import logging
logging.getLogger("fvcore").setLevel(logging.ERROR)

def run_benchmark(inner_model, dummy_input, batch_size=1, num_runs=50):
    with torch.no_grad():
        for _ in range(5):
            try:
                _ = inner_model(dummy_input)
            except TypeError:
                _ = inner_model(*dummy_input) if isinstance(dummy_input, tuple) else inner_model(**dummy_input) if isinstance(dummy_input, dict) else inner_model(dummy_input)
            
    torch.cuda.synchronize()
    
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
    
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    avg_time_ms = np.mean(times)
    fps = 1000.0 / avg_time_ms
    
    return fps

def get_model(config):
    model_name = config.get("model", {}).get("name", "")
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
            
            # Apply overrides from our config
            if "backbone" in config.get("model", {}):
                if hasattr(model_config, "backbone_name"):
                    model_config.backbone_name = config.model.backbone.get("name", model_config.backbone_name)
                if hasattr(model_config, "backbone_kwargs"):
                    model_config.backbone_kwargs = {"out_indices": config.model.backbone.get("out_indices", [2, 3, 4])}
                if hasattr(config.model.backbone, "fpn_type"):
                    if hasattr(model_config, "fpn_type"):
                        model_config.fpn_type = config.model.backbone.fpn_type
                    elif hasattr(model_config, "backbone_config") and hasattr(model_config.backbone_config, "fpn_type"):
                        model_config.backbone_config.fpn_type = config.model.backbone.fpn_type
            if config.model.rtdetr.get("num_queries"):
                model_config.num_queries = config.model.rtdetr.num_queries
        else:
            model_cls = RTDetrForObjectDetection
            try:
                model_config = AutoConfig.from_pretrained(pretrained_path)
            except:
                model_config = AutoConfig.from_pretrained("PekingU/rtdetr_r50vd")
                
        model_config.num_labels = 4
        try:
            inner_model = model_cls(config=model_config)
        except Exception as e:
            print(f"Failed RTDETR: {e}")
            
    elif "rf_detr" in model_name.lower() or "rfdetr" in model_name.lower():
        if "seg" in model_name.lower():
            from rfdetr import RFDETRSegLarge, RFDETRSegMedium, RFDETRSegSmall
            if "small" in model_name.lower(): m = RFDETRSegSmall(num_classes=4)
            elif "medium" in model_name.lower(): m = RFDETRSegMedium(num_classes=4)
            else: m = RFDETRSegLarge(num_classes=4)
        else:
            from rfdetr import RFDETRLarge, RFDETRMedium, RFDETRSmall
            if "small" in model_name.lower(): m = RFDETRSmall(num_classes=4)
            elif "medium" in model_name.lower(): m = RFDETRMedium(num_classes=4)
            else: m = RFDETRLarge(num_classes=4)
        inner_model = m.model.model
        
    elif "yolo" in model_name.lower():
        import os
        from models.yolov5_lightning_module import YOLOv5LightningModule
        try:
            repo_path = config.get("model", {}).get("yolov5", {}).get("repo_path", "")
        except Exception:
            repo_path = ""
        if not repo_path or not os.path.exists(repo_path): 
            repo_path = os.path.join(os.getcwd(), "models", "yolov5")
        try:
            m = YOLOv5LightningModule(config=OmegaConf.create(config), yolo_repo_path=repo_path)
            inner_model = m.model
        except Exception as e:
            print(f"Failed to load YOLOv5: {e}")
            
    elif "mask2former" in model_name.lower():
        from models.mask2former_model import build_original_mask2former, build_mask2former_with_dinov2_backbone
        bb_name = config.get("model", {}).get("backbone", {}).get("model_name", "")
        if "dinov2" in bb_name:
            inner_model = build_mask2former_with_dinov2_backbone(
                id2label={0:'cell', 1:'bead', 2:'cell-adhered', 3:'soma'}, 
                mask2former_pretrained_name_or_path="facebook/mask2former-swin-base-coco-panoptic",
                backbone_pretrained_name_or_path=config.get("model", {}).get("backbone", {}).get("pretrained_name_or_path", "facebook/dinov2-base")
            )
            # Override FPN if specified
            fpn_type = config.get("model", {}).get("backbone", {}).get("fpn_type", "adapter")
            if hasattr(inner_model, "config"):
                inner_model.config.fpn_type = fpn_type
        else:
            inner_model = build_original_mask2former(id2label={0:'cell', 1:'bead', 2:'cell-adhered', 3:'soma'}, mask2former_pretrained_name_or_path="facebook/mask2former-swin-base-coco-panoptic")
            
    elif "deim" in model_name.lower():
        from models.deim_v2_lightning_module import DeimV2LightningModule
        try:
            m = DeimV2LightningModule(config=OmegaConf.create(config))
            inner_model = m.model
        except Exception as e:
            print(f"Failed to load DEIMv2: {e}")

    return inner_model

def get_input_size(config):
    if "rfdetr_seg" in config.model.name:
        return 504
    return config.get("model", {}).get("input_size", 640)

MODELS_TO_TEST = [
    ("yolov5 PL", ['model=yolov5']),
    ("yolov26 Medium", ['model=yolov5']), # Yolov26 is just yolov5 but trained more
    ("RF-DETR base", ['model=rfdetr', 'model.rfdetr.size=small']),
    ("RF-DETR medium", ['model=rfdetr', 'model.rfdetr.size=medium']),
    ("RT-DETR-v1 resnet50", ['model=rtdetr_v1']),
    ("RT-DETR-v2 w/ ResNet50 300 queries, stage 4", ['model=rtdetr_v2', '++model.rtdetr.num_queries=300']),
    ("RT-DETR-v2 w/ ResNet50 600 queries, stage 4", ['model=rtdetr_v2', '++model.rtdetr.num_queries=600']),
    ("RT-DETR-v2 w/ ResNet50 300 queries, stage 0", ['model=rtdetr_v2', '++model.rtdetr.num_queries=300']),
    ("RT-DETR-v2 w/ ResNet50 600 queries, stage 0", ['model=rtdetr_v2', '++model.rtdetr.num_queries=600']),
    ("RT-DETR-v2 w/ ResNet50 300 queries, stage 2", ['model=rtdetr_v2', '++model.rtdetr.num_queries=300']),
    ("RT-DETR-v2 w/ ResNet50 600 queries, stage 2", ['model=rtdetr_v2', '++model.rtdetr.num_queries=600']),
    ("RT-DETR-v2 w/ Dinov2 fused fpn, 300q, [3,7,11]", ['model=rtdetr_dinov2', '++model.rtdetr.num_queries=300', '++model.backbone.fpn_type=fused']),
    ("RT-DETRv2 w/ Dinov2 simple fpn, 300 Q, [3, 7, 11]", ['model=rtdetr_dinov2', '++model.rtdetr.num_queries=300', '++model.backbone.fpn_type=adapter']),
    ("Mask2Former dinov2 (same spatial res)", ['model=mask2former', 'model/backbone=dinov2_mask2former']),
    ("Mask2Former swin-base", ['model=mask2former', 'model/backbone=swin_base_mask2former']),
    ("Mask2Former swin-large", ['model=mask2former', 'model/backbone=swin_large_mask2former']),
    ("Mask2Former dinov2 + fusedFPN", ['model=mask2former', 'model/backbone=dinov2_mask2former', '++model.backbone.fpn_type=fused']),
    ("Mask2Former dinov2 + tinyFPN", ['model=mask2former', 'model/backbone=dinov2_mask2former', '++model.backbone.fpn_type=tiny']),
    ("Mask2Former dinvo2 + SFP", ['model=mask2former', 'model/backbone=dinov2_mask2former', '++model.backbone.fpn_type=sfp']),
    ("Mask2Former dinov2 + adapter", ['model=mask2former', 'model/backbone=dinov2_mask2former', '++model.backbone.fpn_type=adapter']),
    ("DEIMv2-X num_denoising=100", ['model=deimv2_x', '++model.deimv2.decoder.num_denoising=100']),
    ("DEIMv2-X num_denoising=0", ['model=deimv2_x', '++model.deimv2.decoder.num_denoising=0']),
    ("DEIMv2-M num_denoising=100", ['model=deimv2_m', '++model.deimv2.decoder.num_denoising=100']),
    ("DEIMv2-M num_denoising=0", ['model=deimv2_m', '++model.deimv2.decoder.num_denoising=0']),
    ("RF-DETR-Seg group_detr=1 + 13 epochs", ['model=rfdetr_seg', 'model.rfdetr.size=large']),
]

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = []
    
    with initialize(version_base=None, config_path="configs"):
        for name, overrides in MODELS_TO_TEST:
            print(f"\nEvaluating: {name}")
            try:
                cfg = compose(config_name="config", overrides=overrides)
                inner_model = get_model(cfg)
                
                if inner_model is None:
                    print("Failed to initialize model")
                    continue
                    
                inner_model.eval()
                inner_model.to(device)
                
                # Input
                input_size = get_input_size(cfg)
                dummy_input = torch.randn(1, 3, input_size, input_size, device=device)
                
                # Special cases for models that crash in bfloat16
                if "DEIMv2" in name:
                    inner_model = inner_model.float()
                    dummy_input = dummy_input.float()
                elif "Mask2Former" in name:
                    # Mask2Former FLOP computation crashes occasionally in mixed dtypes
                    inner_model = inner_model.float()
                    dummy_input = dummy_input.float()
                elif torch.cuda.is_available():
                    inner_model = inner_model.bfloat16()
                    dummy_input = dummy_input.bfloat16()
                
                # Parameters
                params = sum(p.numel() for p in inner_model.parameters()) / 1e6
                    
                # GFLOPS
                try:
                    flops = FlopCountAnalysis(inner_model, dummy_input)
                    gflops = flops.total() / 1e9
                except Exception as e:
                    print(f"GFLOPS error: {e}")
                    gflops = 0.0
                    
                # FPS Native
                fps_native = run_benchmark(inner_model, dummy_input, num_runs=50)
                
                # FPS Compiled
                fps_compiled = 0.0
                try:
                    compiled_model = torch.compile(inner_model, mode="reduce-overhead")
                    fps_compiled = run_benchmark(compiled_model, dummy_input, num_runs=50)
                except Exception as e:
                    print(f"Compile error: {e}")
                
                results.append({
                    "Model / Config": name,
                    "Params (M)": f"{params:.1f}",
                    "Input Size": input_size,
                    "GFLOPS": f"{gflops:.1f}" if gflops > 0 else "Error",
                    "Native FPS": f"{fps_native:.1f}",
                    "Compiled FPS": f"{fps_compiled:.1f}"
                })
                
                # free memory
                del inner_model
                if 'compiled_model' in locals(): del compiled_model
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Error evaluating {name}: {e}")

    print("\n\n" + "="*80)
    print("FINAL BENCHMARK REPORT")
    print("="*80)
    print(tabulate.tabulate(results, headers="keys", tablefmt="github"))
    
    if results:
        with open("run_architectures_benchmark_report.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print("\nReport saved to run_architectures_benchmark_report.csv")

if __name__ == "__main__":
    main()

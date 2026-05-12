import torch
import numpy as np
import tabulate
import csv
from omegaconf import OmegaConf
from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig
from fvcore.nn import FlopCountAnalysis

# Register custom OmegaConf resolvers before composing configurations
OmegaConf.register_new_resolver("extract_name", lambda path: path.split("/")[-1], replace=True)

import logging
logging.getLogger("fvcore").setLevel(logging.ERROR)

def run_benchmark(inner_model, postprocess_fn, dummy_input, batch_size=1, num_runs=50):
    orig_sizes = torch.tensor([[dummy_input.shape[2], dummy_input.shape[3]] for _ in range(batch_size)], device=dummy_input.device) if hasattr(dummy_input, "device") else None
    with torch.no_grad():
        for _ in range(5):
            try:
                outputs = inner_model(dummy_input)
            except TypeError:
                outputs = inner_model(*dummy_input) if isinstance(dummy_input, tuple) else inner_model(**dummy_input) if isinstance(dummy_input, dict) else inner_model(dummy_input)
            if postprocess_fn is not None:
                try:
                    _ = postprocess_fn(outputs, orig_sizes)
                except Exception as e:
                    pass
            
    torch.cuda.synchronize()
    
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
    fps = 1000.0 / total_ms if total_ms > 0 else 0
    
    return avg_fwd_ms, avg_post_ms, total_ms, fps

def get_model(config):
    model_name = config.get("model", {}).get("name", "")
    inner_model = None
    postprocess_fn = None

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
            
            if config.get("model", {}).get("backbone", {}).get("type", "") == "dinov2":
                from models.backbone_factory import build_backbone
                backbone_model, backbone_config_obj, _ = build_backbone(config.model.backbone, rtdetr_name)
                inner_model.model.backbone = backbone_model
                inner_model.config.backbone_config = backbone_config_obj
                inner_model.config.encoder_in_channels = backbone_config_obj.intermediate_channel_sizes
                inner_model.config.use_timm_backbone = False
                
        except Exception as e:
            print(f"Failed RTDETR: {e}")
            
        if hasattr(inner_model, "post_process_object_detection"):
            def rtdetr_postprocess(outputs, orig_sizes):
                return inner_model.post_process_object_detection(outputs, 0.05, orig_sizes)
            postprocess_fn = rtdetr_postprocess
            
    elif "rf_detr" in model_name.lower() or "rfdetr" in model_name.lower():
        group_detr = config.get("model", {}).get("rfdetr", {}).get("group_detr", 11)
        if "seg" in model_name.lower():
            from rfdetr import RFDETRSegLarge, RFDETRSegMedium, RFDETRSegSmall
            if "small" in model_name.lower(): m = RFDETRSegSmall(num_classes=4, group_detr=group_detr)
            elif "medium" in model_name.lower(): m = RFDETRSegMedium(num_classes=4, group_detr=group_detr)
            else: m = RFDETRSegLarge(num_classes=4, group_detr=group_detr)
        else:
            from rfdetr import RFDETRLarge, RFDETRMedium, RFDETRSmall
            if "small" in model_name.lower(): m = RFDETRSmall(num_classes=4, group_detr=group_detr)
            elif "medium" in model_name.lower(): m = RFDETRMedium(num_classes=4, group_detr=group_detr)
            else: m = RFDETRLarge(num_classes=4, group_detr=group_detr)
        inner_model = m.model.model
        if hasattr(m.model, "postprocess"):
            def rfdetr_postproc(outputs, orig_sizes):
                return m.model.postprocess(outputs, target_sizes=orig_sizes)
            postprocess_fn = rfdetr_postproc
        else:
            postprocess_fn = None
        
    elif "yolo" in model_name.lower():
        import os, sys
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
                
                def yolo_postprocess(outputs, orig_sizes):
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
                def yolo_v8_postprocess(outputs, orig_sizes):
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
        if not m2f_pretrained or "${" in m2f_pretrained:
            # Fallback if hydra didn't interpolate or if it's missing
            bb_name_check = config.get("model", {}).get("backbone", {}).get("name", "")
            if "large" in bb_name_check.lower() or "large" in model_name.lower():
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
            # Override FPN if specified
            fpn_type = config.get("model", {}).get("backbone", {}).get("fpn_type", "adapter")
            if hasattr(inner_model, "config"):
                inner_model.config.fpn_type = fpn_type
        else:
            inner_model = build_original_mask2former(id2label={0:'cell', 1:'bead', 2:'cell-adhered', 3:'soma'}, mask2former_pretrained_name_or_path=m2f_pretrained)
            
        def m2f_postprocess(outputs, orig_sizes):
            return outputs.class_queries_logits.argmax(dim=-1)
        postprocess_fn = m2f_postprocess
            
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
            postprocess_fn = getattr(m.model, "postprocess", None)
        except Exception as e:
            print(f"Failed to load DEIMv2: {e}")

    return inner_model, postprocess_fn

def get_input_size(config):
    if "rfdetr_seg" in config.model.name:
        return 504
    return config.get("model", {}).get("input_size", 640)

MODELS_TO_TEST = [
    ("yolov5 PL", ['model=yolov5']),
    ("yolov26 Medium", ['model=yolov26']), # Yolov26 is just yolov5 but trained more
    ("RF-DETR base", ['model=rfdetr', 'model.rfdetr.size=small']),
    ("RF-DETR medium", ['model=rfdetr', 'model.rfdetr.size=medium']),
    ("RT-DETR-v1 resnet50", ['model=rtdetr_v1', 'model/backbone=resnet50']),
    ("RT-DETR-v2 w/ ResNet50 300 queries, stage 4", ['model=rtdetr_v2', '++model.rtdetr.num_queries=300', 'model/backbone=resnet50']),
    ("RT-DETR-v2 w/ ResNet50 600 queries, stage 4", ['model=rtdetr_v2', '++model.rtdetr.num_queries=600', 'model/backbone=resnet50']),
    ("RT-DETR-v2 w/ ResNet50 300 queries, stage 0", ['model=rtdetr_v2', '++model.rtdetr.num_queries=300', 'model/backbone=resnet50', '++model.backbone.freeze_at_stage=0']),
    ("RT-DETR-v2 w/ ResNet50 600 queries, stage 0", ['model=rtdetr_v2', '++model.rtdetr.num_queries=600', 'model/backbone=resnet50', '++model.backbone.freeze_at_stage=0']),
    ("RT-DETR-v2 w/ ResNet50 300 queries, stage 2", ['model=rtdetr_v2', '++model.rtdetr.num_queries=300', 'model/backbone=resnet50', '++model.backbone.freeze_at_stage=2']),
    ("RT-DETR-v2 w/ ResNet50 600 queries, stage 2", ['model=rtdetr_v2', '++model.rtdetr.num_queries=600', 'model/backbone=resnet50', '++model.backbone.freeze_at_stage=2']),
    ("RT-DETR-v2 w/ Dinov2 fused fpn, 300q, [3,7,11]", ['model=rtdetr_dinov2', '++model.rtdetr.num_queries=300', '++model.backbone.fpn_type=fused']),
    ("RT-DETRv2 w/ Dinov2 simple fpn, 300 Q, [3, 7, 11]", ['model=rtdetr_dinov2', '++model.rtdetr.num_queries=300', '++model.backbone.fpn_type=simple']),
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
    ("RF-DETR-Seg group_detr=1 + 13 epochs", ['model=rfdetr_seg', 'model.rfdetr.size=large', '++model.rfdetr.group_detr=1']),
]

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = []
    
    with initialize(version_base=None, config_path="configs"):
        for name, overrides in MODELS_TO_TEST:
            print(f"\nEvaluating: {name}")
            try:
                cfg = compose(config_name="config", overrides=overrides, return_hydra_config=True)
                HydraConfig.instance().set_config(cfg)
                inner_model, postprocess_fn = get_model(cfg)
                
                if inner_model is None:
                    print("Failed to initialize model")
                    continue
                    
                inner_model.eval()
                inner_model.to(device)
                
                for batch_size in [1, 8]:
                    # Input
                    input_size = get_input_size(cfg)
                    dummy_input = torch.randn(batch_size, 3, input_size, input_size, device=device)
                    
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
                        gflops = (flops.total() / 1e9) / batch_size
                    except Exception as e:
                        print(f"GFLOPS error: {e}")
                        gflops = 0.0
                        
                    # FPS Native
                    avg_fwd_native, avg_post_native, total_ms_native, fps_native = run_benchmark(inner_model, postprocess_fn, dummy_input, batch_size=batch_size, num_runs=50)
                    
                    # FPS Compiled
                    avg_fwd_compiled, avg_post_compiled, total_ms_compiled, fps_compiled = 0.0, 0.0, 0.0, 0.0
                    try:
                        # Torch compile is sensitive to batch size changes if dynamic=False, 
                        # so we might need to re-compile or clear cache, but `compiled_model` is scoped inside.
                        # Actually torch compile caches based on input shape so it's fine.
                        compiled_model = torch.compile(inner_model, mode="reduce-overhead")
                        avg_fwd_compiled, avg_post_compiled, total_ms_compiled, fps_compiled = run_benchmark(compiled_model, postprocess_fn, dummy_input, batch_size=batch_size, num_runs=50)
                    except Exception as e:
                        print(f"Compile error: {e}")
                    
                    results.append({
                        "Model / Config": name,
                        "Batch Size": batch_size,
                        "Params (M)": f"{params:.1f}",
                        "Input Size": input_size,
                        "GFLOPS/img": f"{gflops:.1f}" if gflops > 0 else "Error",
                        "Forward Time/img (ms)": f"{avg_fwd_native:.2f}",
                        "Postproc Time/img (ms)": f"{avg_post_native:.2f}",
                        "Total Time/img (ms)": f"{total_ms_native:.2f}",
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

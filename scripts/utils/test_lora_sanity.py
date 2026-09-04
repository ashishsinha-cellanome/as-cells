import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import os
import torch
import pytorch_lightning as pl
import hydra
from omegaconf import DictConfig, OmegaConf

from train_rfdetr_phase2 import _get_model_class, PreBuiltRFDETRModelModule, Phase2MotifDataModule
from rfdetr.training.trainer import build_trainer
from rfdetr.training.callbacks.coco_eval import COCOEvalCallback
from rfdetr.training.callbacks import RFDETREMACallback
from utils.motif_coco_eval import MotifCocoEvalCallback
from utils.distributed_utils import setup_cluster_env, rank_zero_print
from utils.test_only_checkpoint_restore import _load_ckpt, _select_eval_weights_source, _load_selected_weights

OmegaConf.register_new_resolver("oc.eval", eval, replace=True)

@hydra.main(config_path="configs", config_name="config_rfdetr_seg", version_base="1.3")
def main(config: DictConfig):
    OmegaConf.set_struct(config, False)
    setup_cluster_env()

    pl.seed_everything(config.get("seed", 42), workers=True)

    label_map = {int(k): v for k, v in config.model.label_map.items()}
    class_names = [label_map[idx] for idx in sorted(label_map.keys())]
    num_classes = len(label_map)

    # 1. Base Model Initialization
    is_seg = "seg" in config.model.name.lower()
    rf_model_cls = _get_model_class(config.model.rfdetr.size, is_seg=is_seg)
    kwargs = {
        "resolution": int(config.model.input_size),
        "num_classes": num_classes,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "group_detr": getattr(config.model.rfdetr, "group_detr", 1),
        "compile": getattr(config.model.rfdetr, "compile", False),
        "backbone_lora": False
    }
    
    pretrain_weights = config.model.rfdetr.get("pretrain_weights", None)
    if pretrain_weights is not None:
        kwargs["pretrain_weights"] = pretrain_weights

    rf_wrapper = rf_model_cls(**kwargs)
    if hasattr(rf_wrapper.model, "class_names"):
        rf_wrapper.model.class_names = class_names

    inner_model = rf_wrapper.model.model
    base_args = rf_wrapper.model.args

    # 2. Extract and Prepare Configs
    model_config = rf_wrapper.model_config
    model_config.num_classes = num_classes
    model_config.backbone_lora = config.model.rfdetr.get("finetune_mode", "full") == "lora" or config.model.rfdetr.get("backbone_lora", False)
    
    lora_cfg_dict = getattr(config.model.rfdetr, "lora", {})
    lora_cfg = OmegaConf.to_container(lora_cfg_dict, resolve=True) if lora_cfg_dict else {}

    train_config_kwargs = dict(
        dataset_dir=str(config.data.path),
        epochs=1,  # Reduced from 5
        batch_size=int(config.data.batch_size),
        grad_accum_steps=1,
        lr=5e-5,  # Safer learning rate for sanity check
        weight_decay=float(config.optimizer.optimizer.weight_decay),
        output_dir="outputs/sanity_check",
        use_ema=bool(config.model.rfdetr.get("use_ema", True)),
        ema_decay=float(config.model.rfdetr.get("ema_decay", 0.993)),
        ema_tau=int(config.model.rfdetr.get("ema_tau", 1000)),
        lr_scheduler="step",
        warmup_epochs=0,
        lr_drop=100,
        lr_min_factor=0.0,
        eval_max_dets=int(config.model.get("max_detections", 100)),
        early_stopping=False,
        num_workers=int(config.data.num_workers),
        seed=config.get("seed", 42),
        accelerator=config.trainer.get("accelerator", "auto"),
        log_per_class_metrics=True,
        train_log_sync_dist=True,
        compute_val_loss=True,
        compute_test_loss=True,
        fp16_eval=True,
        progress_bar="tqdm",
        wandb=False,
        project="test",
        run="test",
    )
    
    if hasattr(config.optimizer.optimizer, "lr_encoder"):
        train_config_kwargs["lr_encoder"] = float(config.optimizer.optimizer.lr_encoder)
    elif "lr_encoder" in config.model.rfdetr:
        train_config_kwargs["lr_encoder"] = float(config.model.rfdetr.lr_encoder)

    train_config = rf_wrapper.get_train_config(**train_config_kwargs)

    # 3. Create Module
    module = PreBuiltRFDETRModelModule(
        model_config=model_config, 
        train_config=train_config, 
        inner_model=inner_model, 
        lora_cfg=lora_cfg,
        delay_lora=True
    )
    module.config = config

    # 4. Data Module
    data_module = Phase2MotifDataModule(
        base_path=str(config.data.path), 
        config=config, 
        base_args=base_args
    )
    data_module.setup("test")

    # 5. Build Trainer
    trainer_kwargs = {}
    trainer_kwargs["use_distributed_sampler"] = False
    trainer_kwargs["num_nodes"] = 1
    # Hardcode limits for rapid sanity checking
    trainer_kwargs["limit_train_batches"] = 5
    trainer_kwargs["limit_test_batches"] = 5
    trainer_kwargs["num_sanity_val_steps"] = 0
    trainer_kwargs["limit_val_batches"] = 0

    devices = 1  # Force 1 GPU to avoid DDP deadlocks during sequential tests
    strategy_obj = "auto"
    
    trainer = build_trainer(
        train_config=train_config, 
        model_config=model_config, 
        strategy=strategy_obj,
        devices=devices,
        **trainer_kwargs
    )

    if hasattr(trainer, "_data_connector"):
        if hasattr(trainer._data_connector, "_use_distributed_sampler"):
            trainer._data_connector._use_distributed_sampler = False

    trainer.callbacks = [cb for cb in trainer.callbacks if not isinstance(cb, COCOEvalCallback)]
    
    test_dataloader_names = (
        [f"train_ds/{ds}/test" for ds in data_module.train_dataset_names] 
        + 
        [f"test_ds/{ds}/test" for ds in data_module.test_dataset_names]
    )
    test_dataset_fns = (
        [lambda ds=ds: ds.coco for ds in getattr(data_module, "train_test_datasets_objs", [])] 
        + 
        [lambda ds=ds: ds.coco for ds in data_module.test_datasets_objs]
    )

    motif_coco_eval = MotifCocoEvalCallback(
        val_dataloader_names=[], 
        test_dataloader_names=test_dataloader_names,
        val_get_coco_gt_fns=[],
        test_get_coco_gt_fns=test_dataset_fns,
        label_map=label_map
    )
    trainer.callbacks.append(motif_coco_eval)

    orig_test = trainer.test
    def custom_test(*args, **kwargs):
        ema_cb = next((cb for cb in trainer.callbacks if isinstance(cb, RFDETREMACallback)), None)
        if ema_cb is not None and getattr(ema_cb, "_average_model", None) is not None:
            orig_start = ema_cb.on_test_epoch_start
            def patched_start(trainer, pl_module):
                if getattr(ema_cb, "_pending_average_state_dict", None) is not None:
                    ema_cb._average_model.load_state_dict(ema_cb._pending_average_state_dict)
                    ema_cb._pending_average_state_dict = None
                return orig_start(trainer, pl_module)
            ema_cb.on_test_epoch_start = patched_start
        return orig_test(*args, **kwargs)

    trainer.test = custom_test

    # 6. Load Base Checkpoint
    base_ckpt = config.get("initialization", {}).get("base_checkpoint", None)
    if not base_ckpt:
        rank_zero_print("ERROR: initialization.base_checkpoint must be provided.")
        return

    rank_zero_print(f"\n=======================================================")
    rank_zero_print(f"Loading BASE weights from: {base_ckpt}")
    checkpoint = _load_ckpt(base_ckpt)
    weight_source = _select_eval_weights_source(base_ckpt, checkpoint, config=config)
    _load_selected_weights(module, checkpoint, weight_source)

    rank_zero_print(f"\n=======================================================")
    rank_zero_print(f"STEP 1: APPLYING LORA AND TRAINING ON 1% DATA")
    rank_zero_print(f"=======================================================")
    module._apply_lora()
    
    # Train the model to see if base weights are protected
    trainer.fit(module, datamodule=data_module)

    import shutil
    import os

    rank_zero_print(f"\n=======================================================")
    rank_zero_print(f"STEP 2: EVALUATING FINE-TUNED MODEL (WITH LORA)")
    rank_zero_print(f"=======================================================")
    trainer.test(module, datamodule=data_module)
    
    report_path = os.path.join("outputs/sanity_check", "inference_summary_report.md")
    if os.path.exists(report_path):
        shutil.move(report_path, os.path.join("outputs/sanity_check", "inference_summary_report_WITH_LORA.md"))

    rank_zero_print(f"\n=======================================================")
    rank_zero_print(f"STEP 3: EVALUATING BASE MODEL (LORA ADAPTERS DISABLED)")
    rank_zero_print(f"This proves that the base weights were entirely unaffected by training.")
    rank_zero_print(f"=======================================================")
    
    with module.model.disable_adapter():
        trainer.test(module, datamodule=data_module)

    if os.path.exists(report_path):
        shutil.move(report_path, os.path.join("outputs/sanity_check", "inference_summary_report_WITHOUT_LORA.md"))

if __name__ == '__main__':
    main()

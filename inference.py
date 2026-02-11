
import os
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning import Trainer

from models.rt_detr_lightning_module import RTDETRLightningModule
from data.coco_data_module import COCODataModule
from transformers import RTDetrImageProcessor

@hydra.main(config_path="configs", config_name="config.yaml", version_base=None)
def main(config: DictConfig):
    # Unlock config to allow modifications
    OmegaConf.set_struct(config, False)
    
    # Checkpoint path
    ckpt_path = config.initialization.load_from_checkpoint
    if not ckpt_path:
        raise ValueError("Please provide a checkpoint path via 'initialization.load_from_checkpoint=/path/to/ckpt'")
        
    ckpt_path = hydra.utils.to_absolute_path(ckpt_path)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    print(f"Loading model from: {ckpt_path}")
    
    # Load model from checkpoint
    # Note: We load directly using the LightningModule class
    # The actual model architecture is inferred/restored from the checkpoint or config
    model = RTDETRLightningModule.load_from_checkpoint(
        ckpt_path,
        config=config, # Override config in module with current run config? Or use checkpoint's?
        # Usually providing config here overrides/updates the module's stored config
        # But we need to ensure data parameters match
    )
    model.eval()
    
    # Setup data module
    # Ensure input size matches model config (or checkpoint config)
    # The loaded model might have its own config, but let's assume hydra config is correct for data path
    
    # Processor setup
    processor = RTDetrImageProcessor.from_pretrained(config.model.rtdetr.pretrained_name_or_path)
    processor.do_normalize = True
    processor.resample = 3
    processor.size = {
        "height": config.data.model_input_size,
        "width": config.data.model_input_size
    }
    # Reset image processor in model just in case (though init should handle it)
    model.image_processor = processor

    data_module = COCODataModule(
        dataset_path=hydra.utils.to_absolute_path(config.data.path),
        processor=processor,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        model_input_size=config.data.model_input_size,
        # ... other params ...
        min_random_scale=config.data.min_random_scale,
        max_random_scale=config.data.max_random_scale,
        p_noise=config.data.p_noise,
        org_images_in_model_input_size=config.data.org_images_in_model_input_size,
        config=config
    )
    
    # Trainer for testing
    trainer = Trainer(
        accelerator=config.trainer.accelerator,
        devices=1, # Inference usually on 1 device
        logger=None, # Disable wandb for inference script usually? Or keep it? User might want logs
        enable_checkpointing=False,
    )
    
    print("Starting Inference (Test Mode)...")
    trainer.test(model, datamodule=data_module)
    
    print("Inference Complete.")

if __name__ == "__main__":
    main()

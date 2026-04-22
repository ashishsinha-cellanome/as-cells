import torch
from omegaconf import OmegaConf
from models.mask2former_lightning_module import Mask2FormerLightningModule
from models.mask2former_model import build_original_mask2former

def test_optimizer_groups():
    config = OmegaConf.create({
        "optimizer": {"optimizer": {"lr": 1e-4, "weight_decay": 0.05}},
        "scheduler": {"type": "step", "step_size": 10, "gamma": 0.1},
        "model": {"ema": {"enabled": False}}
    })

    label_map = {0: "cell", 1: "bead", 2: "cell-adhered", 3: "soma"}
    
    print("Testing Swin-Large with frozen backbone...")
    model = build_original_mask2former(
        id2label=label_map,
        mask2former_pretrained_name_or_path="facebook/mask2former-swin-large-coco-instance",
        training_mode="frozen"
    )

    # Mock lightning module to use configure_optimizers
    pl_module = Mask2FormerLightningModule(
        model=model,
        image_processor=None,
        config=config,
        model_to_coco={},
        val_coco_gt=None,
        test_coco_gt=None,
        val_segm_coco_gt=None,
        test_segm_coco_gt=None,
        val_image_root="",
        test_image_root=""
    )

    opt_dict = pl_module.configure_optimizers()
    optimizer = opt_dict["optimizer"]

    print(f"Total parameter groups: {len(optimizer.param_groups)}")
    for i, group in enumerate(optimizer.param_groups):
        print(f"Group {i}: {len(group['params'])} params, lr: {group['lr']}")
        # Print a few parameter names for this group
        # Since param_groups only contains param tensors, we need to map them back to names
        param_names = []
        for name, param in model.named_parameters():
            if any(p is param for p in group['params']):
                param_names.append(name)
        if param_names:
            print(f"  Sample names: {param_names[:3]} ... {param_names[-1:]}")

if __name__ == '__main__':
    test_optimizer_groups()

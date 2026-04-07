import re

with open("train_deim_v2.py", "r") as f:
    content = f.read()

# Replace the imports
content = re.sub(
    r"from data.deim_v2_data_module import DeimV2DataModule",
    "from data.coco_data_module import COCODataModule\nfrom torchvision.datasets import CocoDetection",
    content,
)

# Replace the instantiation block
replacement = """    dataset_path = to_absolute_path(config.data.path)
    rank_zero_print(f"[Startup] Using DEIMv2 dataset path: {dataset_path}")

    # Initialize data module (without processor, since DEIM doesn't need HuggingFace processor)
    data_module = COCODataModule(
        dataset_path=dataset_path,
        processor=None,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        model_input_size=config.data.model_input_size,
        min_random_scale=config.data.min_random_scale,
        max_random_scale=config.data.max_random_scale,
        p_noise=config.data.p_noise,
        org_images_in_model_input_size=config.data.org_images_in_model_input_size,
        config=config,
    )
    data_module.setup()

    val_annot_path = os.path.join(dataset_path, "images", config.val_name)
    val_json_path = os.path.join(dataset_path, f"{config.val_name}_annotations.json")
    val_coco_dataset = CocoDetection(root=val_annot_path, annFile=val_json_path, transforms=None)
    val_coco_gt = val_coco_dataset.coco

    test_annot_path = os.path.join(dataset_path, "images", config.test_name)
    test_json_path = os.path.join(dataset_path, f"{config.test_name}_annotations.json")
    test_coco_dataset = CocoDetection(root=test_annot_path, annFile=test_json_path, transforms=None)
    test_coco_gt = test_coco_dataset.coco

    lightning_model = DeimV2LightningModule(
        config=config,
        val_coco_gt=val_coco_gt,
        test_coco_gt=val_coco_gt if config.debug else test_coco_gt,
        val_image_root=val_annot_path,
        test_image_root=val_annot_path if config.debug else test_annot_path,
    )

    logger = _setup_logger(config)"""

# Find the start of dataset_path and end of lightning_model instantiation
content = re.sub(
    r"    dataset_path = to_absolute_path\(config.data.path\).*?logger = _setup_logger\(config\)",
    replacement,
    content,
    flags=re.DOTALL,
)

with open("train_deim_v2.py", "w") as f:
    f.write(content)

print("train_deim_v2.py updated successfully!")

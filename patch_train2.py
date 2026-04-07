import re

with open("train_deim_v2.py", "r") as f:
    content = f.read()

# Replace the imports back
content = re.sub(
    r"from data.coco_data_module import COCODataModule\nfrom torchvision.datasets import CocoDetection",
    "from data.deim_v2_data_module import DeimV2DataModule\nfrom torchvision.datasets import CocoDetection",
    content,
)

# Replace the instantiation block back
replacement = """    dataset_path = to_absolute_path(config.data.path)
    rank_zero_print(f"[Startup] Using DEIMv2 dataset path: {dataset_path}")

    data_module = DeimV2DataModule(
        dataset_path=dataset_path,
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

# Find the start of dataset_path and end of logger instantiation
content = re.sub(
    r"    dataset_path = to_absolute_path\(config.data.path\).*?logger = _setup_logger\(config\)",
    replacement,
    content,
    flags=re.DOTALL,
)

with open("train_deim_v2.py", "w") as f:
    f.write(content)

print("train_deim_v2.py reverted to DeimV2DataModule successfully!")

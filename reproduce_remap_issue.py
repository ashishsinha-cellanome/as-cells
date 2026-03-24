import hydra
from omegaconf import DictConfig
from data.coco_data_module import COCODataModule


@hydra.main(config_path="configs", config_name="config.yaml", version_base=None)
def verify_on_real_data(cfg: DictConfig):
    print("\n" + "=" * 50)
    print("VERIFYING WITH REAL DATASET")
    print("=" * 50)

    # 1. Print Config Settings
    print(f"Data Path: {cfg.data.path}")
    print(f"Remap Labels: {cfg.get('remap_labels', 'Not Found')}")
    print(f"Class Remapping: {cfg.data.class_remapping}")

    # 2. Initialize Data Module
    # We can pass None for processor as we just want to inspect the dataset
    data_module = COCODataModule(dataset_path=cfg.data.path, processor=None, config=cfg)

    # 3. Setup (loads the COCO json)
    print("\nLoading Datasets... (this may take a moment)")
    data_module.setup(stage="fit")

    # 4. Inspect Validation Dataset
    val_dataset = data_module.val_dataset

    print("\n--- Validation Dataset Inspection ---")
    if hasattr(val_dataset, "remap_dict"):
        print(f"Remap Dict Present: {val_dataset.remap_dict}")
    else:
        print("Remap Dict: None")

    # Get the underlying COCO object
    coco = val_dataset.dataset_coco.coco

    print("\n--- Source Categories (from JSON) ---")
    print(coco.cats)

    # Check a few annotations to see if they are remapped
    print("\n--- Checking first 5 annotations ---")
    # We access the internal dataset wrapper to see what __getitem__ returns
    # But __getitem__ does remapping on the fly if remap_dict is set.

    # Let's check what __getitem__ returns for the first few items
    for i in range(min(5, len(val_dataset))):
        # The dataset returns (image, target)
        # target is {'image_id': ..., 'annotations': [...]}
        _, target = val_dataset[i]
        anns = target["annotations"]
        if anns:
            print(
                f"Image {target['image_id']}: Cat IDs -> {[a['category_id'] for a in anns]}"
            )
        else:
            print(f"Image {target['image_id']}: No annotations")


if __name__ == "__main__":
    verify_on_real_data()

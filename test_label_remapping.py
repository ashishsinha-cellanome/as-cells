from unittest.mock import MagicMock
from data.coco_data_module import COCODataModule
from omegaconf import OmegaConf


def test_remapping():
    # 1. Simulate the config you just showed
    conf_yaml = """
    remap_labels: true
    model:
      label_map:
        0: 'cell'
        1: 'bead'
        2: 'cell-adhered'
        3: 'soma'
    data:
      class_remapping:
        'cell-adhered': 'cell'
        'soma': 'cell'
    """
    cfg = OmegaConf.create(conf_yaml)

    # 2. Mock a COCO dataset with the source IDs
    # Suppose the JSON has: 1: 'cell', 2: 'bead', 3: 'cell-adhered', 4: 'soma'
    mock_coco = MagicMock()
    mock_coco.coco.cats = {
        1: {"name": "cell"},
        2: {"name": "bead"},
        3: {"name": "cell-adhered"},
        4: {"name": "soma"},
    }

    # 3. Initialize DataModule
    dm = COCODataModule(dataset_path=".", processor=None, config=cfg)

    # 4. Run the remapping logic
    remap_dict = dm._get_remap_dict(mock_coco)

    print("\n--- Remapping Test ---")
    print(
        f"Source COCO Cats: {[(k, v['name']) for k, v in mock_coco.coco.cats.items()]}"
    )
    print(f"Target Label Map: {cfg.model.label_map}")
    print(f"Remapping Rules:  {cfg.data.class_remapping}")
    print("RESULTING REMAP DICT (Source ID -> Target ID):")
    for src_id, tgt_id in remap_dict.items():
        src_name = mock_coco.coco.cats[src_id]["name"]
        tgt_name = cfg.model.label_map[tgt_id]
        print(f"  {src_id} ({src_name}) -> {tgt_id} ({tgt_name})")


if __name__ == "__main__":
    test_remapping()

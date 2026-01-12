
import torch
import hydra
from unittest.mock import MagicMock
from omegaconf import OmegaConf, DictConfig

class MockCOCO:
    def __init__(self):
        # Simulating a dataset with the classes likely found in the source
        # (Assuming standard IDs, but the key is the NAME matching)
        self.dataset = {
            'categories': [
                {'id': 1, 'name': 'cell'},
                {'id': 2, 'name': 'bead'},
                {'id': 3, 'name': 'cell-adhered'},
                {'id': 4, 'name': 'soma'}
            ],
            'annotations': [
                {'id': 101, 'image_id': 1, 'category_id': 1}, # cell
                {'id': 102, 'image_id': 1, 'category_id': 3}, # cell-adhered
                {'id': 103, 'image_id': 1, 'category_id': 4}  # soma
            ]
        }
        self.cats = {c['id']: c for c in self.dataset['categories']}
        self.imgToAnns = {}
        self.catToImgs = {}
        
    def createIndex(self):
        print("MockCOCO: createIndex called")
        self.cats = {c['id']: c for c in self.dataset['categories']}

@hydra.main(config_path="configs", config_name="config.yaml", version_base=None)
def test_remap_logic(cfg: DictConfig):
    print("--- Loaded Config ---")
    # Ensure remap_labels is True (it was added in previous turn)
    print(f"remap_labels: {cfg.get('remap_labels', 'Not Found')}")
    print(f"Target Label Map: {cfg.model.label_map}")
    print(f"Remapping Rules: {cfg.data.class_remapping}")

    # 2. Setup Mock COCO
    coco_gt = MockCOCO()
    print("\n--- Before Remap ---")
    print(f"Categories: {[c['name'] for c in coco_gt.dataset['categories']]}")
    print(f"Annotations Cat IDs: {[a['category_id'] for a in coco_gt.dataset['annotations']]}")
    
    # 3. Define the function logic (copied from RTDETRLightningModule)
    def _remap_coco_gt(coco_gt, config):
        if not coco_gt or hasattr(coco_gt, '_remapped'):
            return
        
        # ADDED CHECK: Only run if enabled
        if hasattr(config, 'remap_labels') and not config.remap_labels:
            print("Skipping remapping (remap_labels=False)")
            return

        target_label_map = config.model.label_map
        name_to_target_id = {v: int(k) for k, v in target_label_map.items()}
        
        remapping_rules = {}
        if hasattr(config, 'data') and config.data and 'class_remapping' in config.data:
            remapping_rules = config.data.class_remapping
        elif 'class_remapping' in config:
            remapping_rules = config.class_remapping
            
        remap_dict = {}
        # NOTE: Using coco_gt.cats (source of truth for existing IDs)
        for cat_id, cat_info in coco_gt.cats.items():
            src_name = cat_info['name']
            effective_name = remapping_rules.get(src_name, src_name)
            if effective_name in name_to_target_id:
                remap_dict[cat_id] = name_to_target_id[effective_name]
        
        print(f"DEBUG: remap_dict: {remap_dict}")

        # Apply to annotations
        for ann in coco_gt.dataset.get('annotations', []):
            if ann['category_id'] in remap_dict:
                ann['category_id'] = remap_dict[ann['category_id']]
        
        # Update categories in GT to match target
        new_categories = []
        for target_id, name in target_label_map.items():
            new_categories.append({'id': int(target_id), 'name': name})
        coco_gt.dataset['categories'] = new_categories
        
        # Re-index
        coco_gt.createIndex()
        coco_gt._remapped = True
        
    # 4. Run Remap
    print(f"Running _remap_coco_gt with remap_labels={cfg.get('remap_labels')}")
    _remap_coco_gt(coco_gt, cfg)
    
    print("\n--- After Remap ---")
    print(f"Categories: {coco_gt.dataset['categories']}")
    print(f"Annotations Cat IDs: {[a['category_id'] for a in coco_gt.dataset['annotations']]}")

    # 5. Test Disabled Case
    print("\n--- Test Disabled Case ---")
    cfg_disabled = cfg.copy()
    cfg_disabled.remap_labels = False
    
    coco_gt_2 = MockCOCO()
    print(f"Running _remap_coco_gt with remap_labels={cfg_disabled.remap_labels}")
    _remap_coco_gt(coco_gt_2, cfg_disabled)
    
    print("Categories (Should be original):", [c['name'] for c in coco_gt_2.dataset['categories']])
    print("Annotations (Should be original):", [a['category_id'] for a in coco_gt_2.dataset['annotations']])

if __name__ == "__main__":
    test_remap_logic()



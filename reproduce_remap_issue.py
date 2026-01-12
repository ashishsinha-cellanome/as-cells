
import torch
from unittest.mock import MagicMock
from omegaconf import OmegaConf

class MockCOCO:
    def __init__(self):
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
        # (Simplified index creation for test)

def test_remap_logic():
    # 1. Setup Config
    # SCENARIO: User wants to map 'cell-adhered' and 'soma' to 'cell'.
    # BUT, the model config still lists all 4 classes? Or does it list only 2?
    # Based on user's previous output, the model config lists ALL 4 classes.
    
    conf_yaml = """
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
    
    # 2. Setup Mock COCO
    coco_gt = MockCOCO()
    print("--- Before Remap ---")
    print(f"Categories: {[c['name'] for c in coco_gt.dataset['categories']]}")
    print(f"Annotations Cat IDs: {[a['category_id'] for a in coco_gt.dataset['annotations']]}")
    
    # 3. Define the function logic (copied from RTDETRLightningModule)
    def _remap_coco_gt(coco_gt, config):
        if not coco_gt or hasattr(coco_gt, '_remapped'):
            return
        
        target_label_map = config.model.label_map
        name_to_target_id = {v: int(k) for k, v in target_label_map.items()}
        
        remapping_rules = config.data.class_remapping
            
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
    _remap_coco_gt(coco_gt, cfg)
    
    print("\n--- After Remap ---")
    print(f"Categories: {coco_gt.dataset['categories']}")
    print(f"Annotations Cat IDs: {[a['category_id'] for a in coco_gt.dataset['annotations']]}")

if __name__ == "__main__":
    test_remap_logic()

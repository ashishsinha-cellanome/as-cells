import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
import pycocotools.mask as coco_mask
from PIL import Image
from typing import Dict, Any, Tuple

class CustomConvertCoco:
    def __init__(self, include_masks: bool = False, cat2label=None) -> None:
        self.include_masks = include_masks
        self.cat2label = cat2label

    def __call__(self, image: Image.Image, target: Dict[str, Any]) -> Tuple[Image.Image, Dict[str, Any]]:
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if "iscrowd" not in obj or obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        if boxes.numel() > 0:
            boxes[:, 2:] += boxes[:, :2]
            boxes[:, 0::2].clamp_(min=0, max=w)
            boxes[:, 1::2].clamp_(min=0, max=h)

        classes = []
        for obj in anno:
            category_id = obj["category_id"]
            if getattr(self, "cat2label", None) is not None:
                if category_id not in self.cat2label:
                    raise KeyError(f"Unknown category_id {category_id}")
                classes.append(self.cat2label[category_id])
            else:
                classes.append(category_id)
        classes = torch.tensor(classes, dtype=torch.int64)

        if boxes.numel() > 0:
            keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
            boxes = boxes[keep]
            classes = classes[keep]
        else:
            keep = torch.zeros(0, dtype=torch.bool)

        new_target = {}
        new_target["boxes"] = boxes
        new_target["labels"] = classes
        new_target["image_id"] = image_id

        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        if boxes.numel() > 0:
            new_target["area"] = area[keep]
            new_target["iscrowd"] = iscrowd[keep]
        else:
            new_target["area"] = area
            new_target["iscrowd"] = iscrowd

        if self.include_masks:
            masks = []
            for i, obj in enumerate(anno):
                if not keep[i]:
                    continue
                seg = obj.get("segmentation", [])
                if not seg:
                    masks.append(torch.zeros((h, w), dtype=torch.uint8))
                    continue
                    
                if isinstance(seg, dict):
                    rles = seg
                    mask = coco_mask.decode(rles)
                    if mask.ndim < 3:
                        mask = mask[..., None]
                    mask = torch.as_tensor(mask, dtype=torch.uint8).any(dim=2)
                    
                    if mask.shape[0] != h or mask.shape[1] != w:
                        bbox = obj["bbox"]
                        x1, y1, bw, bh = [int(v) for v in bbox]
                        full_mask = torch.zeros((h, w), dtype=torch.uint8)
                        x1_c = max(0, x1)
                        y1_c = max(0, y1)
                        x2_c = min(w, x1 + mask.shape[1])
                        y2_c = min(h, y1 + mask.shape[0])
                        
                        mw = x2_c - x1_c
                        mh = y2_c - y1_c
                        if mw > 0 and mh > 0:
                            full_mask[y1_c:y2_c, x1_c:x2_c] = mask[:mh, :mw]
                        masks.append(full_mask)
                    else:
                        masks.append(mask)
                else:
                    try:
                        rles = coco_mask.frPyObjects(seg, h, w)
                    except:
                        rles = seg
                    mask = coco_mask.decode(rles)
                    if mask.ndim < 3:
                        mask = mask[..., None]
                    mask = torch.as_tensor(mask, dtype=torch.uint8).any(dim=2)
                    masks.append(mask)
                    
            if len(masks) > 0:
                new_target["masks"] = torch.stack(masks, dim=0).bool()
            else:
                new_target["masks"] = torch.zeros((0, h, w), dtype=torch.bool)
        
        new_target["orig_size"] = torch.as_tensor([int(h), int(w)])
        new_target["size"] = torch.as_tensor([int(h), int(w)])

        return image, new_target

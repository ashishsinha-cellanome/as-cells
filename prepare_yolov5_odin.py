import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath('.'))

from utils.yolo_utils import convert_coco_to_yolo, create_data_yaml

label_map = {
    0: "cell",
    1: "bead",
    2: "cell-adhered",
    3: "soma"
}

data_path = "/mnt/direct-attached/TRAINING_DATA"
cache_dir = ".cache/datasets/yolov5_train_valid_no300_test_no300"

splits = [
    ("train", "train"),
    ("valid_no300", "valid_no300"),
    ("test_no300", "test_no300")
]

for split_name, json_name in splits:
    json_path = os.path.join(data_path, f"{json_name}_annotations.json")
    image_dir = os.path.join(data_path, "images", json_name)
    convert_coco_to_yolo(split_name, image_dir, json_path, cache_dir, label_map)

yaml_path = create_data_yaml(cache_dir, "train", "valid_no300", "test_no300", label_map)
print(f"Dataset generated at {yaml_path}")

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import os
import json
import pandas as pd

base_dir = "/mnt/direct-attached/PHASE2_meet"
results = []

for dataset in os.listdir(base_dir):
    dataset_path = os.path.join(base_dir, dataset)
    if not os.path.isdir(dataset_path):
        continue
        
    for model in os.listdir(dataset_path):
        model_path = os.path.join(dataset_path, model)
        if not os.path.isdir(model_path):
            continue
            
        metrics_file = os.path.join(model_path, "coco_metrics.json")
        if os.path.exists(metrics_file):
            with open(metrics_file, "r") as f:
                data = json.load(f)
                
            # Overall
            results.append({
                "Model": model,
                "Dataset": dataset,
                "Class": "all",
                "mAP@50": data["overall"].get("AP50", -1.0),
                "mAP@50-95": data["overall"].get("AP", -1.0)
            })
            
            # Class-wise
            if "class_wise" in data:
                for cls, cls_data in data["class_wise"].items():
                    results.append({
                        "Model": model,
                        "Dataset": dataset,
                        "Class": cls,
                        "mAP@50": cls_data.get("AP50", -1.0),
                        "mAP@50-95": cls_data.get("AP", -1.0),
                        "TP": cls_data.get("TP", 0),
                        "FP": cls_data.get("FP", 0),
                        "FN": cls_data.get("FN", 0),
                        "Precision": cls_data.get("Precision", 0),
                        "Recall": cls_data.get("Recall", 0),
                        "F1": cls_data.get("F1", 0)
                    })

df = pd.DataFrame(results)
output_file = os.path.join(base_dir, "all_metrics.csv")
df.to_csv(output_file, index=False)
print(f"Saved aggregated metrics to {output_file}")
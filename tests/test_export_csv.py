import os
import numpy as np
import pytest
from unittest.mock import patch
from custom_dinov2_embedding_pipeline import process_level

def test_process_level_exports_csv(tmp_path):
    embs_dict = {
        "Dataset_A": np.random.rand(5, 128),
        "Dataset_B": np.random.rand(5, 128)
    }
    
    with patch("custom_dinov2_embedding_pipeline.generate_clustermap"), \
         patch("custom_dinov2_embedding_pipeline.generate_heatmap"), \
         patch("custom_dinov2_embedding_pipeline.generate_scatter_plots"):
        
        process_level("test_level", embs_dict, str(tmp_path), "ModelName", "ClassName", coverage_subsample=5, scatter_subsample=5)
        
    assert os.path.exists(tmp_path / "matrix_coverage_distance_k5.csv"), "CSV file should be created"

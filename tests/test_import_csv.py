import os
import pandas as pd
import numpy as np
import subprocess

def test_topology_from_csv(tmp_path):
    names = ["Dataset_A", "Dataset_B", "Dataset_C"]
    matrix = np.array([
        [0.0, 0.1, 0.9],
        [0.8, 0.0, 0.2],
        [0.1, 0.9, 0.0]
    ])
    df = pd.DataFrame(matrix, index=names, columns=names)
    csv_path = tmp_path / "test_matrix.csv"
    df.to_csv(csv_path)
    
    out_dir = tmp_path / "out"
    
    cmd = ["uv", "run", "python", "tree_topology_analysis.py", "--csv", str(csv_path), "--out-dir", str(out_dir)]
    subprocess.run(cmd, check=True)
    
    assert os.path.exists(out_dir / "tree_topology_report.md"), "Report should be generated"
    assert os.path.exists(out_dir / "component_evolution.csv"), "Evolution CSV should be generated"
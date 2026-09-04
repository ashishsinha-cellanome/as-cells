#!/usr/bin/env python3
import os
import re
import glob
import json
import math
import argparse
import pandas as pd

def clean_val(v):
    if pd.isna(v):
        return None
    if isinstance(v, (int, float)):
        if math.isnan(v):
            return None
        if v == -1.0:
            return None
        return v
    if isinstance(v, str):
        v_stripped = v.strip()
        if v_stripped in ('-1.0', '-1', 'NaN', 'nan', 'null', ''):
            return None
        try:
            val = float(v_stripped)
            if val == -1.0:
                return None
            return val
        except ValueError:
            return v_stripped
    return v

def parse_arborescence_report(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Report file not found: {file_path}")
        
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    # 1. Parse Root Node
    root_match = re.search(r"Root Node \(Broadest Dataset\):\s*(.+?)\s*\(Mean Out-Coverage:\s*(\d+\.\d+)\)", content)
    if root_match:
        root_node = root_match.group(1).strip()
        root_mean_out_coverage = float(root_match.group(2))
    else:
        root_node = None
        root_mean_out_coverage = None
        
    ranked_nodes = []
    edges = []
    
    # 2. Parse Ranked Nodes and Edges line by line
    lines = content.splitlines()
    in_ranked_section = False
    in_edges_section = False
    
    # Refined rank regex to prevent greedy eating of trailing digits
    rank_re = re.compile(r"^\s*(\d+)\s+(.*?)\s*(0\.\d+|1\.0*|0)\s*$")
    edge_re = re.compile(r"^\s*(.+?)\s*->\s*(.+?)\s*\|\s*(\d+\.\d+)\s*\|\s*(\w+)\s*$")
    
    for line in lines:
        stripped = line.strip()
        if "Ranked Nodes (by Mean Out-Coverage):" in line:
            in_ranked_section = True
            in_edges_section = False
            continue
        elif "Tree Edges:" in line:
            in_ranked_section = False
            in_edges_section = True
            continue
        elif stripped.startswith("==="):
            continue
            
        if in_ranked_section:
            if stripped.startswith("Rank") or stripped.startswith("---") or not stripped:
                continue
            match = rank_re.match(line)
            if match:
                rank = int(match.group(1))
                dataset = match.group(2).strip()
                mean_out_coverage = float(match.group(3))
                ranked_nodes.append({
                    "rank": rank,
                    "dataset": dataset,
                    "mean_out_coverage": mean_out_coverage
                })
        elif in_edges_section:
            if stripped.startswith("Source") or stripped.startswith("---") or not stripped:
                continue
            match = edge_re.match(line)
            if match:
                source = match.group(1).strip()
                target = match.group(2).strip()
                coverage = float(match.group(3))
                status = match.group(4).strip()
                edges.append({
                    "source": source,
                    "target": target,
                    "coverage": coverage,
                    "status": status
                })
                
    return {
        "root_node": root_node,
        "root_mean_out_coverage": root_mean_out_coverage,
        "ranked_nodes": ranked_nodes,
        "edges": edges
    }

def parse_metrics_csv(file_path):
    if not os.path.exists(file_path):
        return []
    try:
        df = pd.read_csv(file_path)
        if df.empty:
            return []
        records = df.to_dict(orient="records")
        return [
            {col: clean_val(val) for col, val in row.items()}
            for row in records
        ]
    except Exception as e:
        print(f"Error parsing metrics CSV {file_path}: {e}")
        return []

def parse_generalization_csv(file_path):
    if not os.path.exists(file_path):
        return []
    try:
        return parse_metrics_csv(file_path)
    except Exception as e:
        print(f"Error parsing generalization CSV {file_path}: {e}")
        return []

def parse_benchmark_csv(file_path):
    if not os.path.exists(file_path):
        return []
    try:
        return parse_metrics_csv(file_path)
    except Exception as e:
        print(f"Error parsing benchmark CSV {file_path}: {e}")
        return []

def compile_data():
    # Find all coverage_arborescence_report_*.txt in full_arboresnce_plots/ and in root
    report_paths = glob.glob("full_arboresnce_plots/coverage_arborescence_report_*.txt")
    report_paths.extend(glob.glob("coverage_arborescence_report_*.txt"))
    
    seen = set()
    unique_report_paths = []
    for p in report_paths:
        abs_p = os.path.abspath(p)
        if abs_p not in seen:
            seen.add(abs_p)
            unique_report_paths.append(p)
            
    parsed_reports = {}
    for path in unique_report_paths:
        filename = os.path.basename(path)
        key = filename.replace("coverage_arborescence_report_", "").replace(".txt", "")
        try:
            parsed_reports[key] = parse_arborescence_report(path)
        except Exception as e:
            print(f"Error parsing {path}: {e}")
            
    # Find active metrics csv files
    metrics_path = "PHASE2_meet_all_metrics.csv"
    if not os.path.exists(metrics_path):
        metrics_path = "phase2_per_cellline_metrics.csv"
        
    generalization_path = "generalization_tracking.csv"
    benchmark_path = "run_architectures_benchmark_report.csv"
    
    metrics_data = parse_metrics_csv(metrics_path)
    generalization_data = parse_generalization_csv(generalization_path)
    benchmark_data = parse_benchmark_csv(benchmark_path)
    
    compiled = {
        "arborescence_reports": parsed_reports,
        "metrics_data": metrics_data,
        "generalization_data": generalization_data,
        "benchmark_data": benchmark_data
    }
    
    return json.dumps(compiled, indent=2)

def compile_to_html(template_path="dashboard_template.html", output_path="coverage_dashboard.html"):
    compiled_json = compile_data()
    
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Template file not found: {template_path}")
        
    with open(template_path, "r", encoding="utf-8") as f:
        template_content = f.read()
        
    html_content = template_content.replace("{{COMPILED_DATA_JSON}}", compiled_json)
    
    warning_header = "<!-- WARNING: DO NOT EDIT THIS FILE DIRECTLY. IT IS AUTO-GENERATED BY compile_dashboard.py FROM dashboard_template.html -->\n"
    html_content = warning_header + html_content
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    print(f"Successfully compiled dashboard to {output_path}")

def get_source_files_state(template_path):
    state = {}
    paths = glob.glob("full_arboresnce_plots/coverage_arborescence_report_*.txt")
    paths.extend(glob.glob("coverage_arborescence_report_*.txt"))
    specific_files = [
        "PHASE2_meet_all_metrics.csv",
        "phase2_per_cellline_metrics.csv",
        "generalization_tracking.csv",
        "run_architectures_benchmark_report.csv"
    ]
    if template_path:
        specific_files.append(template_path)
        
    for p in paths + specific_files:
        if os.path.exists(p):
            state[os.path.abspath(p)] = os.path.getmtime(p)
    return state

def run_tdd_assertions():
    print("Running inline TDD assertions...")
    
    # Assertions for parse_arborescence_report on 'coverage_arborescence_report_dinov2_large.txt'
    large_report = "coverage_arborescence_report_dinov2_large.txt"
    if os.path.exists(large_report):
        data = parse_arborescence_report(large_report)
        
        # Check Root Node and Out-Coverage
        assert data["root_node"] == "20240625_mc38_10x_caged_4_class", f"Unexpected root: {data['root_node']}"
        assert abs(data["root_mean_out_coverage"] - 0.567) < 1e-5, f"Unexpected root mean out-coverage: {data['root_mean_out_coverage']}"
        
        # Check Ranked Nodes
        assert len(data["ranked_nodes"]) == 16, f"Expected 16 ranked nodes, got {len(data['ranked_nodes'])}"
        assert data["ranked_nodes"][0]["rank"] == 1, "Expected Rank 1"
        assert data["ranked_nodes"][0]["dataset"] == "20240625_mc38_10x_caged_4_class", f"Unexpected rank 1 node: {data['ranked_nodes'][0]['dataset']}"
        assert abs(data["ranked_nodes"][0]["mean_out_coverage"] - 0.567) < 1e-5, f"Unexpected rank 1 value: {data['ranked_nodes'][0]['mean_out_coverage']}"
        
        assert data["ranked_nodes"][-1]["rank"] == 16, "Expected Rank 16"
        assert data["ranked_nodes"][-1]["dataset"] == "20240924_enteric-glia-adhered_10x_uncaged_4_class", f"Unexpected rank 16 node: {data['ranked_nodes'][-1]['dataset']}"
        assert abs(data["ranked_nodes"][-1]["mean_out_coverage"] - 0.242) < 1e-5, f"Unexpected rank 16 value: {data['ranked_nodes'][-1]['mean_out_coverage']}"
        
        # Check Edges
        assert len(data["edges"]) == 15, f"Expected 15 edges, got {len(data['edges'])}"
        first_edge = data["edges"][0]
        assert first_edge["source"] == "20240625_mc38_10x_caged_4_class"
        assert first_edge["target"] == "20240905_u87-adhered_10x_caged_4_class"
        assert abs(first_edge["coverage"] - 0.859) < 1e-5
        assert first_edge["status"] == "STRONG"
        
        print("coverage_arborescence_report_dinov2_large.txt assertions PASSED.")
    else:
        print(f"Skipping {large_report} assertions because file was not found.")
        
    # Assertions for parse_arborescence_report on 'coverage_arborescence_report_dinov2_large_t0.5.txt'
    large_t05_report = "coverage_arborescence_report_dinov2_large_t0.5.txt"
    if os.path.exists(large_t05_report):
        data = parse_arborescence_report(large_t05_report)
        
        # Check Root Node and Out-Coverage
        assert data["root_node"] == "20240624_mc38_10x_caged_4_class", f"Unexpected root: {data['root_node']}"
        assert abs(data["root_mean_out_coverage"] - 0.512) < 1e-5, f"Unexpected root mean out-coverage: {data['root_mean_out_coverage']}"
        
        # Check Ranked Nodes
        assert len(data["ranked_nodes"]) == 21, f"Expected 21 ranked nodes, got {len(data['ranked_nodes'])}"
        assert data["ranked_nodes"][0]["rank"] == 1
        assert data["ranked_nodes"][0]["dataset"] == "20240624_mc38_10x_caged_4_class"
        assert abs(data["ranked_nodes"][0]["mean_out_coverage"] - 0.512) < 1e-5
        
        assert data["ranked_nodes"][-1]["rank"] == 21
        assert data["ranked_nodes"][-1]["dataset"] == "20240924_enteric-glia-adhered_10x_uncaged_4_class"
        assert abs(data["ranked_nodes"][-1]["mean_out_coverage"] - 0.203) < 1e-5
        
        # Check Edges
        assert len(data["edges"]) == 20, f"Expected 20 edges, got {len(data['edges'])}"
        last_edge = data["edges"][-1]
        assert last_edge["source"] == "20250108_neuron-adhered_10x_uncaged_4_class"
        assert last_edge["target"] == "20250305_neuron-adhered_10x_uncaged_4_class"
        assert abs(last_edge["coverage"] - 0.850) < 1e-5
        assert last_edge["status"] == "STRONG"
        
        print("coverage_arborescence_report_dinov2_large_t0.5.txt assertions PASSED.")
    else:
        print(f"Skipping {large_t05_report} assertions because file was not found.")
        
    # Verify compile_data() works and handles missing files gracefully
    print("Verifying compile_data()...")
    compiled_json_str = compile_data()
    assert compiled_json_str is not None, "compile_data() returned None"
    
    # Load and parse compiled JSON to verify structure
    compiled_data = json.loads(compiled_json_str)
    assert "arborescence_reports" in compiled_data, "Missing 'arborescence_reports' in compiled data"
    assert "metrics_data" in compiled_data, "Missing 'metrics_data' in compiled data"
    assert "generalization_data" in compiled_data, "Missing 'generalization_data' in compiled data"
    assert "benchmark_data" in compiled_data, "Missing 'benchmark_data' in compiled data"
    
    # Check arborescence reports was parsed
    reports = compiled_data["arborescence_reports"]
    assert len(reports) > 0, "No arborescence reports compiled"
    if "dinov2_large" in reports:
        assert reports["dinov2_large"]["root_node"] == "20240625_mc38_10x_caged_4_class"
        
    # Check metrics data was parsed (PHASE2_meet_all_metrics.csv or phase2_per_cellline_metrics.csv)
    metrics = compiled_data["metrics_data"]
    assert len(metrics) > 0, "No metrics data compiled"
    assert metrics[0]["Model"] is not None, "Model should not be None"
    
    # Check benchmark data was parsed (run_architectures_benchmark_report.csv)
    benchmark = compiled_data["benchmark_data"]
    assert len(benchmark) > 0, "No benchmark data compiled"
    assert "Model / Config" in benchmark[0] or "Model" in benchmark[0], "Benchmark should have Model or Model / Config"
    
    # Check generalization data is gracefully handled (should dynamically check file presence)
    generalization = compiled_data["generalization_data"]
    assert isinstance(generalization, list), "generalization_data must be a list"
    if os.path.exists("generalization_tracking.csv"):
        assert len(generalization) == 117 or len(generalization) > 0, f"Expected non-empty generalization because the file exists, got {len(generalization)}"
    else:
        assert len(generalization) == 0, f"Expected 0 generalization items because the file does not exist, got {len(generalization)}"
    
    print("compile_data() verification PASSED.")
    print("All inline TDD assertions PASSED successfully!")

def main():
    parser = argparse.ArgumentParser(description="Compile Coverage Dashboard HTML")
    parser.add_argument("output_path", nargs="?", default="coverage_dashboard.html",
                        help="Optional output path for compilation (default: 'coverage_dashboard.html')")
    parser.add_argument("--template", default="dashboard_template.html",
                        help="Template HTML file (default: 'dashboard_template.html')")
    parser.add_argument("--watch", action="store_true",
                        help="Enter a background polling loop monitoring raw reports/CSVs for changes every 1 second")
    args = parser.parse_args()

    if args.watch:
        print(f"Watching for changes in reports, CSVs, and template '{args.template}'...")
        last_state = get_source_files_state(args.template)
        # Compile once initially if template exists
        if os.path.exists(args.template):
            try:
                compile_to_html(template_path=args.template, output_path=args.output_path)
            except Exception as e:
                print(f"Initial compilation failed: {e}")
        else:
            print(f"Warning: Template '{args.template}' does not exist. Compilation will occur once template is created/updated.")
            
        import time
        try:
            while True:
                time.sleep(1)
                current_state = get_source_files_state(args.template)
                if current_state != last_state:
                    print("Detected changes in source files. Recompiling dashboard...")
                    if os.path.exists(args.template):
                        try:
                            compile_to_html(template_path=args.template, output_path=args.output_path)
                        except Exception as e:
                            print(f"Compilation failed: {e}")
                    else:
                        print(f"Cannot compile: Template '{args.template}' does not exist.")
                    last_state = current_state
        except KeyboardInterrupt:
            print("\nWatch mode stopped.")
    else:
        if os.path.exists(args.template):
            compile_to_html(template_path=args.template, output_path=args.output_path)
        else:
            print(f"Template file '{args.template}' does not exist. Running inline TDD assertions to verify parsing...")
            run_tdd_assertions()

if __name__ == "__main__":
    main()

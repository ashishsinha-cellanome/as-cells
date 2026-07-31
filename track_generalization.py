import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
from html.parser import HTMLParser
import colorsys

def generate_dynamic_colors(n):
    if n <= 0:
        return []
        
    distinct_colors = [
        "#0072B2",  # Deep Blue
        "#E69F00",  # Warm Orange
        "#009E73",  # Bluish Green
        "#D55E00",  # Vermillion/Red-Orange
        "#CC79A7",  # Reddish Purple/Pink
        "#56B4E9",  # Sky Blue
        "#332288",  # Indigo/Dark Purple
        "#CC6677",  # Rose/Dusty Red
        "#999933",  # Olive/Yellow-Green
        "#117733",  # Dark Green
        "#882255",  # Wine/Burgundy
        "#44AA99",  # Teal
        "#AA4499",  # Purple
        "#DDCC77",  # Sand/Tan
        "#661100"   # Dark Brown/Maroon
    ]
    
    if n <= len(distinct_colors):
        return distinct_colors[:n]
        
    colors = list(distinct_colors)
    remaining = n - len(distinct_colors)
    inv_phi = 0.618033988749895
    
    for i in range(remaining):
        idx = i + len(distinct_colors)
        h = (idx * inv_phi) % 1.0
        lightness = 0.45 + 0.25 * (idx % 2)
        s = 0.75 + 0.1 * (idx % 3)
        
        r, g, b = colorsys.hls_to_rgb(h, lightness, s)
        hex_color = f"#{int(round(r*255)):02x}{int(round(g*255)):02x}{int(round(b*255)):02x}"
        colors.append(hex_color)
        
    return colors

class ReportHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows = []
        self.in_h2 = False
        self.in_p = False
        self.in_tbody = False
        self.in_tr = False
        self.in_td = False
        self.cur_split_type = None
        self.cur_dataset = None
        self.cur_metric_type = None
        self.temp_text = []
        self.current_row = []

    def handle_starttag(self, tag, attrs):
        if tag == 'h2':
            self.in_h2 = True
            self.temp_text = []
        elif tag == 'p':
            self.in_p = True
            self.temp_text = []
        elif tag == 'tbody':
            self.in_tbody = True
        elif tag == 'tr':
            self.in_tr = True
            self.current_row = []
        elif tag == 'td':
            self.in_td = True
            self.temp_text = []

    def handle_endtag(self, tag):
        if tag == 'h2':
            self.in_h2 = False
            text = "".join(self.temp_text).strip()
            if text.startswith("Dataset:"):
                self.cur_split_type = None
                self.cur_dataset = None
                full = text.replace("Dataset:", "").strip()
                segments = [s for s in full.split("/") if s]
                if len(segments) >= 2:
                    self.cur_split_type, self.cur_dataset = segments[0], segments[1]
                elif len(segments) == 1:
                    self.cur_dataset = segments[0]
        elif tag == 'p':
            self.in_p = False
            text = "".join(self.temp_text).strip()
            if text.startswith("Metric Type:"):
                self.cur_metric_type = "BBOX" if "BBOX" in text.upper() else None
        elif tag == 'tbody':
            self.in_tbody = False
        elif tag == 'tr':
            self.in_tr = False
            if self.in_tbody and self.current_row and len(self.current_row) >= 6:
                if self.cur_metric_type == "BBOX" and self.current_row[0] == "all":
                    try:
                        self.rows.append({
                            "dataset": self.cur_dataset,
                            "split_type": self.cur_split_type,
                            "mAP50_95": float(self.current_row[5])
                        })
                    except ValueError:
                        pass
        elif tag == 'td':
            self.in_td = False
            self.current_row.append("".join(self.temp_text).strip())

    def handle_data(self, data):
        if self.in_h2 or self.in_p or self.in_td:
            self.temp_text.append(data)

def parse_metrics_file(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    if file_path.endswith('.html'):
        with open(file_path, "r", encoding="utf-8") as f:
            parser = ReportHTMLParser()
            parser.feed(f.read())
            return pd.DataFrame(parser.rows, columns=['dataset', 'split_type', 'mAP50_95'])
    elif file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
        required_cols = {'metric_type', 'class', 'dataset', 'split_type', 'mAP50_95'}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"CSV file must contain columns: {', '.join(required_cols)}")
        return df[(df['metric_type'] == 'BBOX') & (df['class'] == 'all')][['dataset', 'split_type', 'mAP50_95']]
    else:
        raise ValueError("Unsupported file format. Use .html or .csv")

def update_master_csv(master_csv_path, exp_name, new_df):
    if exp_name == "Baseline":
        train_ds = ""
    else:
        train_ds = ",".join(new_df[new_df['split_type'] == 'train_ds']['dataset'].unique())
    
    new_df = new_df.copy()
    new_df['experiment'] = exp_name
    new_df['train_datasets'] = train_ds
    
    if os.path.exists(master_csv_path):
        master_df = pd.read_csv(master_csv_path)
        master_df = master_df[master_df['experiment'] != exp_name]
        updated_df = pd.concat([master_df, new_df], ignore_index=True)
    else:
        updated_df = new_df
        
    updated_df.to_csv(master_csv_path, index=False)
    return updated_df

def generate_plot(master_df, baseline_name):
    master_df = master_df[master_df['split_type'] == 'test_ds'].copy()
    
    baseline_df = master_df[master_df['experiment'] == baseline_name]
    if baseline_df.empty:
        print("Baseline not found")
        return
        
    import re
    def short_name(d):
        m = re.match(r"^(\d{6,8})_(.*)$", str(d))
        date_part, rest = (m.group(1), m.group(2)) if m else ("", str(d))
        rest = rest.replace("_4_class", "").replace("_10x", "")
        date_short = date_part[2:] if len(date_part) == 8 else date_part
        return f"{rest} ({date_short})" if date_short else rest
        
    master_df['label'] = master_df['dataset'].apply(short_name)
    
    baseline_df = master_df[master_df['experiment'] == baseline_name][['label', 'mAP50_95']].set_index('label').sort_index()
    
    non_baseline_exps = sorted([e for e in master_df['experiment'].unique() if e != baseline_name])
    colors = generate_dynamic_colors(len(non_baseline_exps))
    
    # 1. Matplotlib Scatter Plot
    plt.figure(figsize=(12, 7))
    
    # Ensure x-axis is populated with all baseline datasets in consistent order
    plt.plot(baseline_df.index, [100]*len(baseline_df), alpha=0.0)
    
    for i, exp in enumerate(non_baseline_exps):
        exp_df = master_df[master_df['experiment'] == exp]
        valid_datasets = exp_df[exp_df['label'].isin(baseline_df.index)]
        if valid_datasets.empty:
            continue
            
        rel_perf = valid_datasets.set_index('label')['mAP50_95'] / baseline_df['mAP50_95'] * 100
        rel_perf = rel_perf.replace([float('inf'), float('-inf')], float('nan'))
        rel_perf = rel_perf.reindex(baseline_df.index)
        
        plt.scatter(rel_perf.index, rel_perf.values, label=exp, color=colors[i], s=60)

    plt.axhline(y=100, color='black', linestyle='-', label='Baseline (100%)')
    plt.axhline(y=90, color='gray', linestyle='--', label='90% Threshold')
    plt.axhline(y=50, color='red', linestyle='--', label='50% Threshold')
    
    plt.xticks(rotation=90, ha='center', fontsize=9)
    plt.yticks(fontsize=9)
    plt.ylabel("Relative Performance (% of Baseline)", fontsize=11)
    plt.xlabel("Dataset", fontsize=11)
    plt.title("Generalization Performance relative to Baseline", fontsize=14)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10, title="Experiment Type")
    plt.tight_layout()
    plt.savefig("generalization_relative_performance_scatter.png", dpi=180, bbox_inches="tight")
    plt.close()
    
    # 2. Matplotlib Scatter + Lines Plot
    plt.figure(figsize=(12, 7))
    
    # Ensure x-axis is populated with all baseline datasets in consistent order
    plt.plot(baseline_df.index, [100]*len(baseline_df), alpha=0.0)
    
    for i, exp in enumerate(non_baseline_exps):
        exp_df = master_df[master_df['experiment'] == exp]
        valid_datasets = exp_df[exp_df['label'].isin(baseline_df.index)]
        if valid_datasets.empty:
            continue
            
        rel_perf = valid_datasets.set_index('label')['mAP50_95'] / baseline_df['mAP50_95'] * 100
        rel_perf = rel_perf.replace([float('inf'), float('-inf')], float('nan'))
        rel_perf = rel_perf.reindex(baseline_df.index)
        
        plt.plot(rel_perf.index, rel_perf.values, label=exp, color=colors[i], marker='o', markersize=8, linewidth=1, alpha=0.8)

    plt.axhline(y=100, color='black', linestyle='-', label='Baseline (100%)')
    plt.axhline(y=90, color='gray', linestyle='--', label='90% Threshold')
    plt.axhline(y=50, color='red', linestyle='--', label='50% Threshold')
    
    plt.xticks(rotation=90, ha='center', fontsize=9)
    plt.yticks(fontsize=9)
    plt.ylabel("Relative Performance (% of Baseline)", fontsize=11)
    plt.xlabel("Dataset", fontsize=11)
    plt.title("Generalization Performance relative to Baseline", fontsize=14)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10, title="Experiment Type")
    plt.tight_layout()
    plt.savefig("generalization_relative_performance_lines.png", dpi=180, bbox_inches="tight")
    plt.close()
    
    # 3. Plotly Interactive Plot
    try:
        import plotly.graph_objects as go
        
        fig = go.Figure()
        
        for i, exp in enumerate(non_baseline_exps):
            exp_df = master_df[master_df['experiment'] == exp]
            valid_datasets = exp_df[exp_df['label'].isin(baseline_df.index)]
            if valid_datasets.empty:
                continue
                
            rel_perf = valid_datasets.set_index('label')['mAP50_95'] / baseline_df['mAP50_95'] * 100
            rel_perf = rel_perf.replace([float('inf'), float('-inf')], float('nan'))
            rel_perf = rel_perf.reindex(baseline_df.index)
            
            # Extract train datasets for hover
            train_ds_str = exp_df['train_datasets'].iloc[0] if 'train_datasets' in exp_df.columns and len(exp_df) > 0 else "N/A"
            if pd.isna(train_ds_str) or train_ds_str == "":
                train_ds_str = "None/Baseline"
            else:
                ds_list = str(train_ds_str).split(',')
                train_ds_str = "<br>".join([", ".join(ds_list[j:j+3]) for j in range(0, len(ds_list), 3)])
            
            hover_text = [
                f"<b>Dataset:</b> {ds}<br>"
                f"<b>Relative Perf:</b> {val:.1f}%<br>"
                f"<b>Experiment Type:</b> {exp}<br>"
                f"<b>Train Datasets:</b><br>{train_ds_str}"
                for ds, val in zip(rel_perf.index, rel_perf.values)
            ]
            
            fig.add_trace(go.Scatter(
                x=rel_perf.index,
                y=rel_perf.values,
                mode='lines+markers',
                name=exp,
                text=hover_text,
                hoverinfo='text',
                line=dict(color=colors[i], width=1),
                marker=dict(size=8)
            ))
            
        fig.add_hline(y=100, line_dash="solid", line_color="black", annotation_text="Baseline (100%)", annotation_position="top right")
        fig.add_hline(y=90, line_dash="dash", line_color="gray", annotation_text="90% Threshold", annotation_position="top right")
        fig.add_hline(y=50, line_dash="dash", line_color="red", annotation_text="50% Threshold", annotation_position="top right")
        
        fig.update_layout(
            title="Generalization Performance relative to Baseline",
            xaxis_title="Dataset",
            xaxis=dict(
                categoryorder='array',
                categoryarray=baseline_df.index.tolist()
            ),
            yaxis_title="Relative Performance (% of Baseline)",
            legend_title="Experiment Type",
            xaxis_tickangle=-90,
            hovermode="closest",
            template="plotly_white",
            margin=dict(b=150)
        )
        
        fig.write_html("generalization_relative_performance.html")
    except ImportError:
        print("Plotly is not installed. Skipping interactive HTML plot. Run `uv add plotly` to enable.")

def parse_args():
    parser = argparse.ArgumentParser(description="Track Generalization Performance")
    parser.add_argument("--baseline", help="Path to baseline HTML or CSV report")
    parser.add_argument("--add-exp", help="Path to new experiment HTML or CSV report")
    parser.add_argument("--exp-name", help="Name of the new experiment")
    parser.add_argument("--master-csv", default="generalization_tracking.csv", help="Path to master tracking CSV")
    return parser.parse_args()

def main():
    args = parse_args()
    baseline_name = "Baseline"
    
    if bool(args.add_exp) != bool(args.exp_name):
        raise SystemExit("Error: Both --add-exp and --exp-name must be provided together.")
    
    if args.baseline:
        try:
            base_df = parse_metrics_file(args.baseline)
            update_master_csv(args.master_csv, baseline_name, base_df)
            print(f"Updated baseline using {args.baseline}")
        except Exception as e:
            raise SystemExit(f"Error parsing baseline file: {e}")
        
    if args.add_exp and args.exp_name:
        try:
            exp_df = parse_metrics_file(args.add_exp)
            update_master_csv(args.master_csv, args.exp_name, exp_df)
            print(f"Added experiment {args.exp_name} from {args.add_exp}")
        except Exception as e:
            raise SystemExit(f"Error parsing experiment file: {e}")
        
    if not os.path.exists(args.master_csv):
        raise SystemExit("Error: Master CSV does not exist. Please provide a baseline or experiment.")
        
    master_df = pd.read_csv(args.master_csv)
    if baseline_name not in master_df['experiment'].values:
        raise SystemExit("Error: No baseline found in master CSV. Please provide one with --baseline.")
        
    generate_plot(master_df, baseline_name)
    print("Plot saved to generalization_relative_performance.png")

if __name__ == "__main__":
    main()

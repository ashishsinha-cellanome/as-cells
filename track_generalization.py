import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
from html.parser import HTMLParser
import colorsys

def generate_dynamic_colors(n):
    if n <= 0:
        return []
        
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    cmap = plt.get_cmap('tab20')
    # Use even indices first (darker, more distinct colors), then odd (lighter)
    indices = list(range(0, 20, 2)) + list(range(1, 20, 2))
    colors = [mcolors.to_hex(cmap(i)) for i in indices]
    
    if n <= len(colors):
        return colors[:n]
        
    remaining = n - len(colors)
    inv_phi = 0.618033988749895
    
    for i in range(remaining):
        idx = i + len(colors)
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
                if "BBOX" in text.upper():
                    self.cur_metric_type = "BBOX"
                elif "SEGM" in text.upper():
                    self.cur_metric_type = "SEGM"
                else:
                    self.cur_metric_type = None
        elif tag == 'tbody':
            self.in_tbody = False
        elif tag == 'tr':
            self.in_tr = False
            if self.in_tbody and self.current_row and len(self.current_row) >= 6:
                if self.cur_metric_type is not None and self.current_row[0] == "all":
                    try:
                        self.rows.append({
                            "dataset": self.cur_dataset,
                            "split_type": self.cur_split_type,
                            "metric_type": self.cur_metric_type,
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

def parse_metrics_file(file_path, metric_type="BBOX"):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    if file_path.endswith('.html'):
        with open(file_path, "r", encoding="utf-8") as f:
            parser = ReportHTMLParser()
            parser.feed(f.read())
            df = pd.DataFrame(parser.rows, columns=['dataset', 'split_type', 'metric_type', 'mAP50_95'])
            return df[df['metric_type'] == metric_type][['dataset', 'split_type', 'mAP50_95']]
    elif file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
        # Filter out commented lines if present in the loaded file
        if 'dataset' in df.columns:
            df = df[~df['dataset'].astype(str).str.startswith('#')]
        required_cols = {'metric_type', 'class', 'dataset', 'split_type', 'mAP50_95'}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"CSV file must contain columns: {', '.join(required_cols)}")
        return df[(df['metric_type'] == metric_type) & (df['class'] == 'all')][['dataset', 'split_type', 'mAP50_95']]
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

def generate_plot(master_df, baseline_name, metric_type="BBOX"):
    # Filter out commented out datasets
    master_df = master_df[~master_df['dataset'].astype(str).str.startswith('#')]
    
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
    
    # 1. Matplotlib Scatter + Lines Plot (All Experiments - NO LORA)
    plt.figure(figsize=(12, 7))
    
    # Ensure x-axis is populated with all baseline datasets in consistent order
    plt.plot(baseline_df.index, [100]*len(baseline_df), alpha=0.0)
    
    for i, exp in enumerate(non_baseline_exps):
        if 'lora' in exp.lower():
            continue
            
        exp_df = master_df[master_df['experiment'] == exp]
        valid_datasets = exp_df[exp_df['label'].isin(baseline_df.index)]
        if valid_datasets.empty:
            continue
            
        rel_perf = valid_datasets.set_index('label')['mAP50_95'] / baseline_df['mAP50_95'] * 100
        rel_perf = rel_perf.replace([float('inf'), float('-inf')], float('nan'))
        rel_perf = rel_perf.reindex(baseline_df.index)
        
        plt.plot(rel_perf.index, rel_perf.values, label=exp, color=colors[i], marker='o', linestyle='-', markersize=8, linewidth=1, alpha=0.8)

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
    suffix = "_segm" if metric_type == "SEGM" else ""
    plt.savefig(f"generalization_relative_performance_lines{suffix}.png", dpi=180, bbox_inches="tight")
    plt.close()
    
    # 2. Separate Plot for 8-Node & LoRA grouped by inferred target
    def is_8_node(e):
        e_low = e.lower()
        has_all_3 = 'a549' in e_low and 'mc38' in e_low and 'hs675' in e_low
        has_other = 'moc22' in e_low or 'astro' in e_low
        return has_all_3 and not has_other

    # Pre-parse LoRA exps and their inferred targets
    lora_exps_by_target = {}
    for exp in non_baseline_exps:
        if not ('lora' in exp.lower()):
            continue
            
        rank_match = re.search(r'(?:_|-|^)(?:r|rank)(\d+)(?:_|-|$)', exp.lower())
        rank = int(rank_match.group(1)) if rank_match else 64
        if rank != 64:
            continue
            
        t_name = re.sub(r'lora', '', exp, flags=re.IGNORECASE)
        t_name = re.sub(r'(?:_|-|^)([0-1]?\.[0-9]+|[0-9]+(?:pct|%))(?:_|-|$)', '_', t_name, count=1, flags=re.IGNORECASE)
        t_name = re.sub(r'(?:_|-|^)(?:r|rank)(\d+)(?:_|-|$)', '_', t_name, count=1, flags=re.IGNORECASE)
        t_name = re.sub(r'[_\-]+', '_', t_name).strip('_-')
        target_group = t_name if t_name else "unknown_target"
        
        match = re.search(r'(?:_|-|^)([0-1]?\.[0-9]+|[0-9]+(?:pct|%))(?:_|-|$)', exp.lower())
        if match:
            val_str = match.group(1)
            frac_val = float(val_str.replace('pct', '').replace('%', '')) / 100.0 if 'pct' in val_str or '%' in val_str else float(val_str)
            frac_pct = int(round(frac_val * 100))
        else:
            frac_pct = 0
            
        if target_group not in lora_exps_by_target:
            lora_exps_by_target[target_group] = []
        lora_exps_by_target[target_group].append((exp, frac_pct))

    base_8node_exps = [e for e in non_baseline_exps if is_8_node(e)]

    # Generate one plot per target dataset group
    for target_group, lora_exps in lora_exps_by_target.items():
        plt.figure(figsize=(12, 7))
        plt.plot(baseline_df.index, [100]*len(baseline_df), alpha=0.0)
        
        bright_colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe']
        color_idx = 0
        
        # Sort lora exps by fraction
        lora_exps = sorted(lora_exps, key=lambda x: x[1])
        
        # Plot 8-node baseline first
        for exp in base_8node_exps:
            exp_df = master_df[master_df['experiment'] == exp]
            valid_datasets = exp_df[exp_df['label'].isin(baseline_df.index)]
            if valid_datasets.empty: continue
            rel_perf = valid_datasets.set_index('label')['mAP50_95'] / baseline_df['mAP50_95'] * 100
            rel_perf = rel_perf.replace([float('inf'), float('-inf')], float('nan')).reindex(baseline_df.index)
            plt.plot(rel_perf.index, rel_perf.values, label="Base Ckpt", color='black', marker='o', linestyle='--', markersize=8, linewidth=1.5, alpha=0.8)

        # Plot LoRA experiments
        for exp, frac_pct in lora_exps:
            exp_df = master_df[master_df['experiment'] == exp]
            valid_datasets = exp_df[exp_df['label'].isin(baseline_df.index)]
            if valid_datasets.empty: continue
            
            rel_perf = valid_datasets.set_index('label')['mAP50_95'] / baseline_df['mAP50_95'] * 100
            rel_perf = rel_perf.replace([float('inf'), float('-inf')], float('nan')).reindex(baseline_df.index)
            
            legend_label = f"LoRA {frac_pct}%"
            color = bright_colors[color_idx % len(bright_colors)]
            color_idx += 1
            
            plt.plot(rel_perf.index, rel_perf.values, label=legend_label, color=color, marker='*', linestyle='-', markersize=12, linewidth=1.5, alpha=0.8)

        plt.axhline(y=100, color='black', linestyle='-', label='Baseline (100%)')
        plt.axhline(y=90, color='gray', linestyle='--', label='90% Threshold')
        plt.axhline(y=50, color='red', linestyle='--', label='50% Threshold')
        
        # Fix duplicate legend entries
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if None in by_label: del by_label[None]
        
        plt.xticks(rotation=90, ha='center', fontsize=9)
        plt.yticks(fontsize=9)
        plt.ylabel("Relative Performance (% of Baseline)", fontsize=11)
        plt.xlabel("Dataset", fontsize=11)
        plt.title(f"Generalization Performance (Target: {target_group})", fontsize=14)
        plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10, title="Experiment Type")
        plt.tight_layout()
        plt.savefig(f"generalization_lora_and_8node_{target_group}{suffix}.png", dpi=180, bbox_inches="tight")
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
            
            is_lora = 'lora' in exp.lower()
            marker_symbol = 'star' if is_lora else 'circle'
            line_dash = 'dash' if is_lora else 'solid'
            marker_size = 12 if is_lora else 8
            
            fig.add_trace(go.Scatter(
                x=rel_perf.index,
                y=rel_perf.values,
                mode='lines+markers',
                name=exp,
                text=hover_text,
                hoverinfo='text',
                line=dict(color=colors[i], width=1, dash=line_dash),
                marker=dict(size=marker_size, symbol=marker_symbol)
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
        
        fig.write_html(f"generalization_relative_performance{suffix}.html")
    except ImportError:
        print("Plotly is not installed. Skipping interactive HTML plot. Run `uv add plotly` to enable.")

def parse_args():
    parser = argparse.ArgumentParser(description="Track Generalization Performance")
    parser.add_argument("--baseline", help="Path to baseline HTML or CSV report")
    parser.add_argument("--add-exp", help="Path to new experiment HTML or CSV report")
    parser.add_argument("--exp-name", help="Name of the new experiment")
    parser.add_argument("--master-csv", default="generalization_tracking.csv", help="Path to master tracking CSV")
    parser.add_argument("--metric", default="both", help="Metric to track (BBOX, SEGM, or both)")
    return parser.parse_args()

def main():
    args = parse_args()
    baseline_name = "Baseline"
    
    if bool(args.add_exp) != bool(args.exp_name):
        raise SystemExit("Error: Both --add-exp and --exp-name must be provided together.")
        
    metrics_to_process = ["BBOX", "SEGM"] if args.metric.lower() == "both" else [args.metric.upper()]
    
    for metric in metrics_to_process:
        suffix = "_segm" if metric == "SEGM" else ""
        current_csv = args.master_csv.replace(".csv", f"{suffix}.csv") if metric == "SEGM" else args.master_csv
        
        if args.baseline:
            try:
                base_df = parse_metrics_file(args.baseline, metric)
                if not base_df.empty:
                    update_master_csv(current_csv, baseline_name, base_df)
                    print(f"Updated baseline using {args.baseline} for metric {metric}")
                else:
                    print(f"Warning: No {metric} data found in {args.baseline}")
            except Exception as e:
                print(f"Error parsing baseline file for metric {metric}: {e}")
            
        if args.add_exp and args.exp_name:
            try:
                exp_df = parse_metrics_file(args.add_exp, metric)
                if not exp_df.empty:
                    update_master_csv(current_csv, args.exp_name, exp_df)
                    print(f"Added experiment {args.exp_name} from {args.add_exp} for metric {metric}")
                else:
                    print(f"Warning: No {metric} data found in {args.add_exp}")
            except Exception as e:
                print(f"Error parsing experiment file for metric {metric}: {e}")
            
        if not os.path.exists(current_csv):
            print(f"Error: Master CSV {current_csv} does not exist. Skipping plots for metric {metric}.")
            continue
            
        master_df = pd.read_csv(current_csv)
        # Check if baseline is in the un-commented portion of the data
        valid_df = master_df[~master_df['dataset'].astype(str).str.startswith('#')]
        if baseline_name not in valid_df['experiment'].values:
            print(f"Error: No baseline found in master CSV {current_csv}. Skipping plots for metric {metric}.")
            continue
            
        generate_plot(master_df, baseline_name, metric)
        print(f"Plots saved to generalization_relative_performance_lines{suffix}.png and generalization_lora_and_8node_<target>{suffix}.png")

if __name__ == "__main__":
    main()

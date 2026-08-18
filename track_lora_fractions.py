import argparse
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def parse_args():
    parser = argparse.ArgumentParser(description="Plot LoRA data fraction experiments")
    parser.add_argument("--master-csv", default="generalization_tracking.csv", help="Path to master tracking CSV")
    parser.add_argument("--target", help="Explicit target dataset name (optional)")
    parser.add_argument("--anchors", default="A549,MC38,HS675", help="Comma-separated additional anchor datasets")
    parser.add_argument("--base-exp", default="Baseline", help="Baseline experiment name (unused in heatmap)")
    parser.add_argument("--metric", default="both", help="Metric to track (BBOX, SEGM, or both)")
    return parser.parse_args()

def extract_fraction(exp_name):
    match = re.search(r'(?:_|-|^)([0-1]?\.[0-9]+|[0-9]+(?:pct|%))(?:_|-|$)', exp_name.lower())
    if match:
        val_str = match.group(1)
        if 'pct' in val_str or '%' in val_str:
            val = val_str.replace('pct', '').replace('%', '')
            return float(val) / 100.0
        return float(val_str)
    return None

def get_target_name(exp_name, target_arg):
    if target_arg:
        return target_arg
    name = re.sub(r'lora', '', exp_name, flags=re.IGNORECASE)
    name = re.sub(r'(?:_|-|^)([0-1]?\.[0-9]+|[0-9]+(?:pct|%))(?:_|-|$)', '_', name, count=1, flags=re.IGNORECASE)
    name = re.sub(r'[_\-]+', '_', name).strip('_-')
    return name if name else "Target"

def is_8_node(e):
    e_low = e.lower()
    has_all_3 = 'a549' in e_low and 'mc38' in e_low and 'hs675' in e_low
    has_other = 'moc22' in e_low or 'astro' in e_low
    return has_all_3 and not has_other
    
def short_name(d):
    m = re.match(r"^(\d{6,8})_(.*)$", str(d))
    date_part, rest = (m.group(1), m.group(2)) if m else ("", str(d))
    rest = rest.replace("_4_class", "").replace("_10x", "")
    date_short = date_part[2:] if len(date_part) == 8 else date_part
    return f"{rest} ({date_short})" if date_short else rest

def generate_plots_for_metric(args, metric):
    suffix = "_segm" if metric == "SEGM" else ""
    current_csv = args.master_csv.replace(".csv", f"{suffix}.csv") if metric == "SEGM" else args.master_csv

    if not os.path.exists(current_csv):
        print(f"Error: {current_csv} not found. Skipping {metric}.")
        return

    df = pd.read_csv(current_csv)
    df = df[~df['dataset'].astype(str).str.startswith('#')]
    df = df[df['split_type'] == 'test_ds']

    eight_node_exps = df[df['experiment'].apply(is_8_node)]
    e_df = pd.DataFrame()
    anchors = []
    if not eight_node_exps.empty:
        e_exp = eight_node_exps['experiment'].iloc[0]
        e_df = df[df['experiment'] == e_exp]
        train_ds_str = e_df['train_datasets'].iloc[0]
        if pd.notna(train_ds_str) and train_ds_str != "":
            anchors = [x.strip() for x in str(train_ds_str).split(',')]
    else:
        print("Warning: 8-node baseline not found in CSV.")

    lora_exps = df[df['experiment'].str.contains('lora', case=False, na=False)]
    
    valid_exps = []
    inferred_targets = set()
    
    for exp in lora_exps['experiment'].unique():
        fraction = extract_fraction(exp)
        if fraction is None:
            continue
            
        inferred = get_target_name(exp, None)
        inferred_targets.add(inferred)
        valid_exps.append((exp, fraction, inferred))

    target_name = args.target
    if target_name is None:
        if len(inferred_targets) > 1:
            import sys
            print(f"Error: Multiple target datasets inferred {inferred_targets}. Please provide --target explicitly.")
            sys.exit(1)
        elif len(inferred_targets) == 1:
            target_name = inferred_targets.pop()
        else:
            print("No valid LoRA fractions found to plot.")
            return

    unique_ds = df['dataset'].unique()
    if target_name not in unique_ds:
        for ds in unique_ds:
            if target_name.lower() in str(ds).lower():
                target_name = str(ds)
                break
                
    # Now filter valid_exps to only include those matching the target dataset
    filtered_exps = []
    for exp, fraction, inferred in valid_exps:
        # Match inferred target to the resolved target_name (ignoring hyphens/underscores)
        inferred_clean = inferred.replace('_', '').replace('-', '').lower()
        target_clean = target_name.replace('_', '').replace('-', '').lower()
        if inferred_clean in target_clean:
            filtered_exps.append((exp, fraction))
            
    if not filtered_exps:
        print(f"No LoRA experiments found for target: {target_name}")
        return

    anchors = [a for a in anchors if a != target_name]

    data_rows = []
    
    if not e_df.empty:
        col_name = "8-Node Base"
        t_row = e_df[e_df['dataset'] == target_name]
        t_val = t_row['mAP50_95'].values[0] if not t_row.empty else None
        data_rows.append({'Dataset': short_name(target_name), col_name: t_val, 'Type': 'Target'})
        
        for anchor in sorted(anchors):
            a_row = e_df[e_df['dataset'] == anchor]
            a_val = a_row['mAP50_95'].values[0] if not a_row.empty else None
            data_rows.append({'Dataset': short_name(anchor), col_name: a_val, 'Type': 'Anchor'})
                
    if not data_rows:
        data_rows.append({'Dataset': short_name(target_name), 'Type': 'Target'})
        for anchor in sorted(anchors):
            data_rows.append({'Dataset': short_name(anchor), 'Type': 'Anchor'})
            
    heatmap_df = pd.DataFrame(data_rows)
    
    valid_exps = sorted(filtered_exps, key=lambda x: x[1])
    
    orig_map = {short_name(x): x for x in anchors + [target_name]}
    
    for exp, fraction in valid_exps:
        fraction_pct = int(round(fraction * 100))
        col_name = f"LoRA {fraction_pct}%"
        exp_df = df[df['experiment'] == exp]
        
        for idx, row in heatmap_df.iterrows():
            ds_short = row['Dataset']
            ds_original = orig_map.get(ds_short)
            
            val = None
            if ds_original is not None:
                val_row = exp_df[exp_df['dataset'] == ds_original]
                if not val_row.empty:
                    val = val_row['mAP50_95'].values[0]
                    
            heatmap_df.at[idx, col_name] = val
            
    if heatmap_df.empty:
        print("No valid data to plot.")
        return

    heatmap_df = heatmap_df.set_index('Dataset')
    plot_df = heatmap_df.drop(columns=['Type'])
    plot_df = plot_df.dropna(axis=1, how='all')
    
    if plot_df.empty:
        print("No valid metrics found to populate the heatmap.")
        return

    plt.figure(figsize=(10, len(plot_df) * 0.6 + 2))
    ax = sns.heatmap(plot_df.astype(float), annot=True, fmt=".3f", cmap="YlGnBu", cbar_kws={'label': 'mAP@0.5-0.95'}, linewidths=.5)
    
    for tick_label in ax.get_yticklabels():
        if tick_label.get_text() == short_name(target_name):
            tick_label.set_weight("bold")
            
    plt.title(f"LoRA Fine-tuning Performance Heatmap\nTarget: {short_name(target_name)}", pad=20)
    plt.xlabel("Experiment")
    plt.ylabel("Evaluation Dataset")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"lora_fraction_heatmap_{target_name}{suffix}.png", dpi=180)
    plt.close()

    # Create Line Plots
    # We will use plot_df which has the datasets as rows and experiments as columns.
    
    frac_cols = [c for c in plot_df.columns if c.startswith("LoRA")]
    ranks = sorted(list(set([x[2] for x in valid_exps])))
    
    if frac_cols:
        # Colors and styles
        cmap = plt.get_cmap('tab20')
        colors = [cmap(i) for i in range(20)]
        
        # 1. Absolute Plot
        plt.figure(figsize=(12, 8))
        
        target_short = short_name(target_name)
        color_idx = 0
        
        for ds_name in plot_df.index:
            is_target = (ds_name == target_short)
            c = colors[color_idx % len(colors)]
            color_idx += 1
            
            for rank in ranks:
                rank_cols = [col for col in plot_df.columns if col.startswith(f"LoRA r{rank}")]
                if not rank_cols:
                    continue
                    
                def get_frac(col):
                    m = re.search(r'(\d+)%', col)
                    return int(m.group(1)) if m else 0
                rank_cols = sorted(rank_cols, key=get_frac)
                
                x_vals = [get_frac(col) for col in rank_cols]
                y_vals = plot_df.loc[ds_name, rank_cols].values
                
                if rank == 64:
                    marker = '*' if is_target else 'o'
                    linestyle = '-' if is_target else '--'
                else:
                    marker = 'X' if is_target else 's'
                    linestyle = '-.' if is_target else ':'
                    
                linewidth = 2.5 if is_target else 1.5
                label = f"{ds_name} (r{rank})"
                
                plt.plot(x_vals, y_vals, label=label, color=c, marker=marker, linestyle=linestyle, linewidth=linewidth, markersize=8, alpha=0.8)
            
            # 8-Node Baseline
            if "8-Node Base" in plot_df.columns and not pd.isna(plot_df.loc[ds_name, "8-Node Base"]):
                base_val = plot_df.loc[ds_name, "8-Node Base"]
                plt.axhline(y=base_val, color=c, linestyle=':', alpha=0.6, label=f"{ds_name} (Base)")
                
        plt.xlabel("Data Fraction (%)")
        plt.ylabel("mAP@0.5-0.95")
        plt.title(f"LoRA Fine-tuning: Absolute Performance vs Data Fraction\nTarget: {target_short}")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"lora_fraction_absolute_lines_{target_name}{suffix}.png", dpi=180)
        plt.close()
        
        # 2. Relative Plot (vs 8-Node Base)
        if "8-Node Base" in plot_df.columns:
            plt.figure(figsize=(12, 8))
            color_idx = 0
            
            for ds_name in plot_df.index:
                if pd.isna(plot_df.loc[ds_name, "8-Node Base"]):
                    continue
                    
                base_val = plot_df.loc[ds_name, "8-Node Base"]
                is_target = (ds_name == target_short)
                c = colors[color_idx % len(colors)]
                color_idx += 1
                
                for rank in ranks:
                    rank_cols = [col for col in plot_df.columns if col.startswith(f"LoRA r{rank}")]
                    if not rank_cols:
                        continue
                        
                    def get_frac(col):
                        m = re.search(r'(\d+)%', col)
                        return int(m.group(1)) if m else 0
                    rank_cols = sorted(rank_cols, key=get_frac)
                    
                    x_vals = [get_frac(col) for col in rank_cols]
                    y_vals = plot_df.loc[ds_name, rank_cols].values - base_val
                    
                    if rank == 64:
                        marker = '*' if is_target else 'o'
                        linestyle = '-' if is_target else '--'
                    else:
                        marker = 'X' if is_target else 's'
                        linestyle = '-.' if is_target else ':'
                        
                    linewidth = 2.5 if is_target else 1.5
                    label = f"{ds_name} (r{rank})"
                    
                    plt.plot(x_vals, y_vals, label=label, color=c, marker=marker, linestyle=linestyle, linewidth=linewidth, markersize=8, alpha=0.8)
                
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3, label='8-Node Base (0 Delta)')
            plt.xlabel("Data Fraction (%)")
            plt.ylabel("Delta mAP@0.5-0.95")
            plt.title(f"LoRA Fine-tuning: Relative to 8-Node Baseline\nTarget: {target_short}")
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"lora_fraction_relative_lines_{target_name}{suffix}.png", dpi=180)
            plt.close()

    print(f"Plots generated successfully: lora_fraction_heatmap_{target_name}{suffix}.png, lora_fraction_absolute_lines_{target_name}{suffix}.png, lora_fraction_relative_lines_{target_name}{suffix}.png")

def main():
    args = parse_args()
    metrics_to_process = ["BBOX", "SEGM"] if args.metric.lower() == "both" else [args.metric.upper()]
    
    for metric in metrics_to_process:
        print(f"\n--- Generating LoRA plots for {metric} ---")
        generate_plots_for_metric(args, metric)

if __name__ == "__main__":
    main()
import argparse
import os
import re
import pandas as pd
import numpy as np
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

def extract_rank(exp_name):
    match = re.search(r'(?:_|-|^)(?:r|rank)(\d+)(?:_|-|$)', exp_name.lower())
    if match:
        return int(match.group(1))
    return 64

def get_target_name(exp_name, target_arg):
    if target_arg:
        return target_arg
    name = re.sub(r'lora', '', exp_name, flags=re.IGNORECASE)
    name = re.sub(r'(?:_|-|^)([0-1]?\.[0-9]+|[0-9]+(?:pct|%))(?:_|-|$)', '_', name, count=1, flags=re.IGNORECASE)
    name = re.sub(r'(?:_|-|^)(?:r|rank)(\d+)(?:_|-|$)', '_', name, count=1, flags=re.IGNORECASE)
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
            
        rank = extract_rank(exp)
        inferred = get_target_name(exp, None)
        inferred_targets.add(inferred)
        valid_exps.append((exp, fraction, rank, inferred))

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
    for exp, fraction, rank, inferred in valid_exps:
        # Match inferred target to the resolved target_name (ignoring hyphens/underscores)
        inferred_clean = inferred.replace('_', '').replace('-', '').lower()
        target_clean = target_name.replace('_', '').replace('-', '').lower()
        if inferred_clean in target_clean:
            filtered_exps.append((exp, fraction, rank))
            
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
    
    valid_exps = sorted(filtered_exps, key=lambda x: (x[1], x[2]))
    
    orig_map = {short_name(x): x for x in anchors + [target_name]}
    
    for exp, fraction, rank in valid_exps:
        fraction_pct = int(round(fraction * 100))
        col_name = f"LoRA r{rank} {fraction_pct}%"
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

    out_dir = os.path.join("lora_plots", args.target)
    os.makedirs(out_dir, exist_ok=True)

    default_rank = 64
    heatmap_cols = [c for c in plot_df.columns if c == "8-Node Base" or c.startswith(f"LoRA r{default_rank}")]
    heatmap_plot_df = plot_df[heatmap_cols].copy()
    heatmap_plot_df.columns = [c.replace(f"r{default_rank} ", "") for c in heatmap_plot_df.columns]

    if heatmap_plot_df.empty or len(heatmap_plot_df.columns) <= 1:
        print(f"Warning: No valid metrics found for default rank {default_rank} to populate the heatmap.")
    else:
        plt.figure(figsize=(10, len(heatmap_plot_df) * 0.6 + 2))
        ax = sns.heatmap(heatmap_plot_df.astype(float), annot=True, fmt=".3f", cmap="YlGnBu", cbar_kws={'label': 'mAP@0.5-0.95'}, linewidths=.5)
        
        for tick_label in ax.get_yticklabels():
            if tick_label.get_text() == short_name(target_name):
                tick_label.set_weight("bold")
                
        plt.title(f"LoRA Fine-tuning Performance Heatmap\nTarget: {short_name(target_name)}", pad=20)
        plt.xlabel("Experiment")
        plt.ylabel("Evaluation Dataset")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"lora_fraction_heatmap_{target_name}{suffix}.png"), dpi=180)
        plt.close()

    # Create Line Plots
    # We will use plot_df which has the datasets as rows and experiments as columns.
    
    frac_cols = [c for c in plot_df.columns if c.startswith("LoRA")]
    ranks = sorted(list(set([x[2] for x in valid_exps])))
    
    if frac_cols:
        # Colors and styles
        cmap = plt.get_cmap('tab20')
        colors = [cmap(i) for i in range(20)]
        target_short = short_name(target_name)
        
        # 1 & 2. Absolute and Relative Plots for DEFAULT RANK (r64) ONLY
        default_rank = 64
        rank_cols = [col for col in plot_df.columns if col.startswith(f"LoRA r{default_rank}")]
        if rank_cols:
            def get_frac(col):
                m = re.search(r'(\d+)%', col)
                return int(m.group(1)) if m else 0
                
            rank_cols = sorted(rank_cols, key=get_frac)
            x_vals = [get_frac(col) for col in rank_cols]
            
            # Absolute Plot
            plt.figure(figsize=(10, 6))
            color_idx = 0
            
            for ds_name in plot_df.index:
                is_target = (ds_name == target_short)
                c = colors[color_idx % len(colors)]
                color_idx += 1
                
                y_vals = plot_df.loc[ds_name, rank_cols].values
                marker = '*' if is_target else 'o'
                linewidth = 2.5 if is_target else 1.5
                linestyle = '-' if is_target else '--'
                
                plt.plot(x_vals, y_vals, label=f"{ds_name}", color=c, marker=marker, linestyle=linestyle, linewidth=linewidth, markersize=8)
                
                if "8-Node Base" in plot_df.columns and not pd.isna(plot_df.loc[ds_name, "8-Node Base"]):
                    base_val = plot_df.loc[ds_name, "8-Node Base"]
                    plt.axhline(y=base_val, color=c, linestyle=':', alpha=0.6, label=f"{ds_name} (Base)" if is_target else None)
            
            # Fix duplicate legend entries for None
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            if None in by_label:
                del by_label[None]
            
            plt.xlabel("Data Fraction (%)")
            plt.ylabel("mAP@0.5-0.95")
            plt.title(f"LoRA Fine-tuning: Absolute Performance vs Data Fraction\nTarget: {target_short}")
            plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"lora_fraction_absolute_lines_{target_name}{suffix}.png"), dpi=180)
            plt.close()
            
            # Relative Plot
            if "8-Node Base" in plot_df.columns:
                plt.figure(figsize=(10, 6))
                color_idx = 0
                for ds_name in plot_df.index:
                    if pd.isna(plot_df.loc[ds_name, "8-Node Base"]):
                        continue
                        
                    base_val = plot_df.loc[ds_name, "8-Node Base"]
                    is_target = (ds_name == target_short)
                    c = colors[color_idx % len(colors)]
                    color_idx += 1
                    
                    y_vals = plot_df.loc[ds_name, rank_cols].values - base_val
                    marker = '*' if is_target else 'o'
                    linewidth = 2.5 if is_target else 1.5
                    linestyle = '-' if is_target else '--'
                    
                    plt.plot(x_vals, y_vals, label=f"{ds_name}", color=c, marker=marker, linestyle=linestyle, linewidth=linewidth, markersize=8)
                    
                plt.axhline(y=0, color='black', linestyle='-', alpha=0.3, label='8-Node Base (0 Delta)')
                plt.xlabel("Data Fraction (%)")
                plt.ylabel("Delta mAP@0.5-0.95")
                plt.title(f"LoRA Fine-tuning: Relative to 8-Node Baseline\nTarget: {target_short}")
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, f"lora_fraction_relative_lines_{target_name}{suffix}.png"), dpi=180)
                plt.close()

        # 3. Rank Comparison Plot
        if len(ranks) > 1:
            plt.figure(figsize=(10, 6))
            anchor_rows = [r for r in plot_df.index if r != target_short]
            rank_colors = {ranks[i]: colors[i * 2] for i in range(len(ranks))}
            
            for rank in ranks:
                rank_cols = [col for col in plot_df.columns if col.startswith(f"LoRA r{rank}")]
                if not rank_cols:
                    continue
                
                def get_frac(col):
                    m = re.search(r'(\d+)%', col)
                    return int(m.group(1)) if m else 0
                    
                rank_cols = sorted(rank_cols, key=get_frac)
                x_vals = [get_frac(col) for col in rank_cols]
                c = rank_colors[rank]
                
                if target_short in plot_df.index:
                    y_target = plot_df.loc[target_short, rank_cols].values
                    plt.plot(x_vals, y_target, label=f"Target (r{rank})", color=c, marker='*', linestyle='-', linewidth=2.5, markersize=10)
                
                if anchor_rows:
                    y_anchors = plot_df.loc[anchor_rows, rank_cols].mean(axis=0).values
                    plt.plot(x_vals, y_anchors, label=f"Anchors Avg (r{rank})", color=c, marker='o', linestyle='--', linewidth=1.5, markersize=8)
            
            if "8-Node Base" in plot_df.columns:
                if target_short in plot_df.index and not pd.isna(plot_df.loc[target_short, "8-Node Base"]):
                    base_target = plot_df.loc[target_short, "8-Node Base"]
                    plt.axhline(y=base_target, color='black', linestyle=':', alpha=0.6, label="Base Ckpt (Target)")
                if anchor_rows:
                    base_anchors = plot_df.loc[anchor_rows, "8-Node Base"].mean()
                    plt.axhline(y=base_anchors, color='gray', linestyle=':', alpha=0.6, label="Base Ckpt (Anchors Avg)")
            
            plt.xlabel("Data Fraction (%)")
            plt.ylabel("mAP@0.5-0.95")
            plt.title(f"LoRA Rank Comparison: Target & Anchor Avg\nTarget: {target_short}")
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"lora_fraction_rank_comparison_{target_name}{suffix}.png"), dpi=180)
            plt.close()

            # 4. Rank Comparison Heatmap on Anchor Datasets
            if anchor_rows:
                anchor_plot_df = plot_df.loc[anchor_rows].copy()
                
                def sort_col(c):
                    if c == "8-Node Base": return (-1, -1)
                    m_frac = re.search(r'(\d+)%', c)
                    m_rank = re.search(r'r(\d+)', c)
                    f = int(m_frac.group(1)) if m_frac else 0
                    r = int(m_rank.group(1)) if m_rank else 0
                    return (f, r)
                    
                anchor_plot_df = anchor_plot_df.reindex(sorted(anchor_plot_df.columns, key=sort_col), axis=1)
                
                plt.figure(figsize=(max(10, len(anchor_plot_df.columns) * 0.8), len(anchor_plot_df) * 0.6 + 2))
                sns.heatmap(anchor_plot_df.astype(float), annot=True, fmt=".3f", cmap="YlGnBu", cbar_kws={'label': 'mAP@0.5-0.95'}, linewidths=.5)
                
                plt.title(f"LoRA Rank Comparison Heatmap (Anchors Only)\nTarget: {target_short}", pad=20)
                plt.xlabel("Experiment")
                plt.ylabel("Anchor Dataset")
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, f"lora_rank_comparison_heatmap_anchors_{target_name}{suffix}.png"), dpi=180)
                plt.close()

            # 5. Rank Comparison Bar Plot on Target Dataset
            if target_short in plot_df.index:
                plt.figure(figsize=(10, 6))
                
                def get_frac(col):
                    m = re.search(r'(\d+)%', col)
                    return int(m.group(1)) if m else 0
                
                fractions = sorted(list(set([get_frac(c) for c in plot_df.columns if c.startswith("LoRA")])))
                x = np.arange(len(fractions))
                width = 0.35
                
                if "8-Node Base" in plot_df.columns and not pd.isna(plot_df.loc[target_short, "8-Node Base"]):
                    base_val = plot_df.loc[target_short, "8-Node Base"]
                    plt.axhline(y=base_val, color='black', linestyle='--', alpha=0.6, label="Base Ckpt")
                
                for i, rank in enumerate(ranks):
                    rank_vals = []
                    for frac in fractions:
                        col = f"LoRA r{rank} {frac}%"
                        if col in plot_df.columns and not pd.isna(plot_df.loc[target_short, col]):
                            rank_vals.append(plot_df.loc[target_short, col])
                        else:
                            rank_vals.append(0)
                    
                    offset = (i - len(ranks)/2 + 0.5) * width
                    c = colors[(i * 2) % 20]
                    plt.bar(x + offset, rank_vals, width, label=f"r{rank}", color=c, alpha=0.8)
                    
                plt.xlabel("Data Fraction (%)")
                plt.ylabel("mAP@0.5-0.95")
                plt.title(f"LoRA Rank Comparison: Target Dataset Performance\nTarget: {target_short}")
                plt.xticks(x, [f"{f}%" for f in fractions])
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.grid(True, alpha=0.3, axis='y')
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, f"lora_rank_comparison_bar_target_{target_name}{suffix}.png"), dpi=180)
                plt.close()

            # 6. Rank Comparison Plot on Anchor Datasets (Grid)
            if anchor_rows:
                num_anchors = len(anchor_rows)
                cols = min(3, num_anchors)
                rows = (num_anchors + cols - 1) // cols
                
                fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
                if num_anchors == 1:
                    axes = [axes]
                else:
                    axes = axes.flatten()
                
                for idx, anchor in enumerate(anchor_rows):
                    ax = axes[idx]
                    
                    if "8-Node Base" in plot_df.columns and not pd.isna(plot_df.loc[anchor, "8-Node Base"]):
                        base_val = plot_df.loc[anchor, "8-Node Base"]
                        ax.axhline(y=base_val, color='black', linestyle='--', alpha=0.6, label="Base Ckpt")
                        
                    for i, rank in enumerate(ranks):
                        rank_vals = []
                        for frac in fractions:
                            col = f"LoRA r{rank} {frac}%"
                            if col in plot_df.columns and not pd.isna(plot_df.loc[anchor, col]):
                                rank_vals.append(plot_df.loc[anchor, col])
                            else:
                                rank_vals.append(None)
                                
                        c = colors[(i * 2) % 20]
                        
                        valid_x = [pos for pos, v in zip(x, rank_vals) if v is not None]
                        valid_y = [v for v in rank_vals if v is not None]
                        
                        marker = '*' if rank == 64 else 'X'
                        linestyle = '-' if rank == 64 else '-.'
                        
                        ax.plot(valid_x, valid_y, label=f"r{rank}", color=c, marker=marker, linestyle=linestyle, linewidth=2, markersize=8)
                        
                    ax.set_title(anchor)
                    ax.set_xticks(x)
                    ax.set_xticklabels([f"{f}%" for f in fractions])
                    if idx == 0:
                        ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                for idx in range(num_anchors, len(axes)):
                    fig.delaxes(axes[idx])
                    
                fig.suptitle(f"LoRA Rank Comparison: Zero-Shot Performance\nTarget: {target_short}", y=1.02, fontsize=16)
                fig.tight_layout()
                plt.savefig(os.path.join(out_dir, f"lora_rank_comparison_anchors_grid_{target_name}{suffix}.png"), dpi=180, bbox_inches='tight')
                plt.close()

    print(f"Plots generated successfully in '{out_dir}': lora_fraction_heatmap, absolute_lines, relative_lines, rank_comparison, etc.")

def main():
    args = parse_args()
    metrics_to_process = ["BBOX", "SEGM"] if args.metric.lower() == "both" else [args.metric.upper()]
    
    for metric in metrics_to_process:
        print(f"\n--- Generating LoRA plots for {metric} ---")
        generate_plots_for_metric(args, metric)

if __name__ == "__main__":
    main()
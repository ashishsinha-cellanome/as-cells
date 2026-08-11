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

def main():
    args = parse_args()
    if not os.path.exists(args.master_csv):
        print(f"Error: {args.master_csv} not found.")
        return

    df = pd.read_csv(args.master_csv)
    df = df[~df['dataset'].astype(str).str.startswith('#')]
    df = df[df['split_type'] == 'test_ds']

    # Anchor resolution: Find any dataset that matches the --anchors substrings
    additional_anchors = [a.strip() for a in args.anchors.split(',')]
    expanded_anchors = set()
    for ds in df['dataset'].unique():
        for anchor_sub in additional_anchors:
            if anchor_sub.lower() in str(ds).lower():
                expanded_anchors.add(str(ds))
    anchors = list(expanded_anchors)

    lora_exps = df[df['experiment'].str.contains('lora', case=False, na=False)]
    
    valid_exps = []
    inferred_targets = set()
    
    for exp in lora_exps['experiment'].unique():
        fraction = extract_fraction(exp)
        if fraction is None:
            continue
        valid_exps.append((exp, fraction))
        if args.target is None:
            inferred_targets.add(get_target_name(exp, None))

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

    eight_node_exps = df[df['experiment'].apply(is_8_node)]
    e_df = pd.DataFrame()
    if not eight_node_exps.empty:
        e_exp = eight_node_exps['experiment'].iloc[0]
        e_df = df[df['experiment'] == e_exp]
    else:
        print("Warning: 8-node baseline not found in CSV.")

    data_rows = []
    
    if not e_df.empty:
        col_name = "8-Node Base"
        t_row = e_df[e_df['dataset'] == target_name]
        t_val = t_row['mAP50_95'].values[0] if not t_row.empty else None
        data_rows.append({'Dataset': short_name(target_name), col_name: t_val, 'Type': 'Target'})
        
        for anchor in sorted(anchors):
            a_row = e_df[e_df['dataset'] == anchor]
            a_val = a_row['mAP50_95'].values[0] if not a_row.empty else None
            if a_val is not None:
                data_rows.append({'Dataset': short_name(anchor), col_name: a_val, 'Type': 'Anchor'})
                
    if not data_rows:
        data_rows.append({'Dataset': short_name(target_name), 'Type': 'Target'})
        for anchor in sorted(anchors):
            data_rows.append({'Dataset': short_name(anchor), 'Type': 'Anchor'})
            
    heatmap_df = pd.DataFrame(data_rows)
    
    valid_exps = sorted(valid_exps, key=lambda x: x[1])
    
    for exp, fraction in valid_exps:
        fraction_pct = int(round(fraction * 100))
        col_name = f"LoRA {fraction_pct}%"
        exp_df = df[df['experiment'] == exp]
        
        for idx, row in heatmap_df.iterrows():
            ds_short = row['Dataset']
            ds_original = target_name if ds_short == short_name(target_name) else None
            if ds_original is None:
                for anchor in anchors:
                    if short_name(anchor) == ds_short:
                        ds_original = anchor
                        break
            
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
    plt.savefig("lora_fraction_heatmap.png", dpi=180)
    plt.close()

    print("Plots generated successfully: lora_fraction_heatmap.png")

if __name__ == "__main__":
    main()
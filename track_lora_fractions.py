import argparse
import os
import re
import pandas as pd
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description="Plot LoRA data fraction experiments")
    parser.add_argument("--master-csv", default="generalization_tracking.csv", help="Path to master tracking CSV")
    parser.add_argument("--target", help="Explicit target dataset name (optional)")
    parser.add_argument("--anchors", default="A549,MC38,HS675", help="Comma-separated additional anchor datasets")
    parser.add_argument("--base-exp", default="Baseline", help="Baseline experiment name")
    return parser.parse_args()

def extract_fraction(exp_name):
    match = re.search(r'([0-9]*\.?[0-9]+)\s*(?:pct|%)?', exp_name.lower())
    if match:
        val = match.group(1)
        if 'pct' in exp_name.lower() or '%' in exp_name:
            return float(val) / 100.0
        return float(val)
    return None

def get_target_name(exp_name, target_arg):
    if target_arg:
        return target_arg
    name = re.sub(r'lora', '', exp_name, flags=re.IGNORECASE)
    name = re.sub(r'[0-9]*\.?[0-9]+\s*(?:pct|%)?', '', name, flags=re.IGNORECASE)
    name = re.sub(r'^_+|_+$', '', name)
    name = name.replace('__', '_')
    return name if name else "Target"

def is_8_node(e):
    e_low = e.lower()
    has_all_3 = 'a549' in e_low and 'mc38' in e_low and 'hs675' in e_low
    has_8node = '8 node' in e_low or '8node' in e_low or '8_node' in e_low
    return has_all_3 or has_8node

def main():
    args = parse_args()
    if not os.path.exists(args.master_csv):
        print(f"Error: {args.master_csv} not found.")
        return

    df = pd.read_csv(args.master_csv)
    df = df[~df['dataset'].astype(str).str.startswith('#')]
    df = df[df['split_type'] == 'test_ds']

    base_df = df[df['experiment'] == args.base_exp]
    if base_df.empty:
        print(f"Warning: Baseline '{args.base_exp}' not found in {args.master_csv}. Relative plots vs baseline will be skipped.")
        train_datasets = []
    else:
        train_ds_str = base_df['train_datasets'].iloc[0]
        train_datasets = str(train_ds_str).split(',') if pd.notna(train_ds_str) and train_ds_str != "" else []
        
    additional_anchors = [a.strip() for a in args.anchors.split(',')]
    anchors = list(set(train_datasets + additional_anchors))

    lora_exps = df[df['experiment'].str.contains('lora', case=False, na=False)]
    
    results = []
    target_name = args.target

    for exp in lora_exps['experiment'].unique():
        fraction = extract_fraction(exp)
        if fraction is None:
            print(f"Warning: Skipping {exp}, no parseable fraction found.")
            continue
        
        if target_name is None:
            target_name = get_target_name(exp, None)
            
        exp_df = df[df['experiment'] == exp]
        
        target_mAP = None
        target_row = exp_df[exp_df['dataset'] == target_name]
        if not target_row.empty:
            target_mAP = target_row['mAP50_95'].values[0]
        else:
            print(f"Warning: Target dataset {target_name} not found in experiment {exp}. Dropping fraction {fraction}.")
            continue
            
        anchor_df = exp_df[exp_df['dataset'].isin(anchors)]
        anchor_mAP = anchor_df['mAP50_95'].mean() if not anchor_df.empty else None
        
        results.append({
            'fraction': fraction,
            'target_mAP': target_mAP,
            'anchor_mAP': anchor_mAP
        })
    
    if not results:
        print("No valid LoRA fractions found to plot.")
        return
        
    results_df = pd.DataFrame(results).sort_values('fraction')
    
    base_target_mAP, base_anchor_mAP = None, None
    if not base_df.empty:
        t_row = base_df[base_df['dataset'] == target_name]
        if not t_row.empty:
            base_target_mAP = t_row['mAP50_95'].values[0]
        a_row = base_df[base_df['dataset'].isin(anchors)]
        if not a_row.empty:
            base_anchor_mAP = a_row['mAP50_95'].mean()

    eight_node_target_mAP, eight_node_anchor_mAP = None, None
    eight_node_exps = df[df['experiment'].apply(is_8_node)]
    if not eight_node_exps.empty:
        e_exp = eight_node_exps['experiment'].iloc[0]
        e_df = df[df['experiment'] == e_exp]
        t_row = e_df[e_df['dataset'] == target_name]
        if not t_row.empty:
            eight_node_target_mAP = t_row['mAP50_95'].values[0]
        a_row = e_df[e_df['dataset'].isin(anchors)]
        if not a_row.empty:
            eight_node_anchor_mAP = a_row['mAP50_95'].mean()
    else:
        print("Warning: 8-node baseline not found in CSV. Relative plots vs 8-node will be skipped.")

    plt.style.use('seaborn-v0_8-colorblind')
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    target_color = colors[0]
    anchor_color = colors[1]

    # 1. Absolute Plot
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['fraction'], results_df['target_mAP'], marker='o', label=f'Target ({target_name})', color=target_color)
    plt.plot(results_df['fraction'], results_df['anchor_mAP'], marker='s', linestyle='--', label='Anchors Average', color=anchor_color)
    if base_target_mAP is not None:
        plt.axhline(y=base_target_mAP, color=target_color, linestyle=':', alpha=0.5, label='Base Ckpt Target')
    if base_anchor_mAP is not None:
        plt.axhline(y=base_anchor_mAP, color=anchor_color, linestyle=':', alpha=0.5, label='Base Ckpt Anchors')
    
    plt.xlabel("Data Fraction")
    plt.ylabel("mAP@0.5-0.95")
    plt.title("LoRA Fine-tuning: Absolute Performance vs Data Fraction")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("lora_fraction_absolute.png", dpi=180)
    plt.close()

    # 2. Relative to Baseline (5-node)
    if base_target_mAP is not None and base_anchor_mAP is not None:
        plt.figure(figsize=(10, 6))
        plt.plot(results_df['fraction'], results_df['target_mAP'] - base_target_mAP, marker='o', label=f'Target Delta ({target_name})', color=target_color)
        plt.plot(results_df['fraction'], results_df['anchor_mAP'] - base_anchor_mAP, marker='s', linestyle='--', label='Anchors Delta', color=anchor_color)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3, label='Base Ckpt (0 Delta)')
        plt.xlabel("Data Fraction")
        plt.ylabel("Delta mAP@0.5-0.95")
        plt.title("LoRA Fine-tuning: Relative to 5-node Baseline")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("lora_fraction_relative_to_baseline.png", dpi=180)
        plt.close()
        
    # 3. Relative to 8-node
    if eight_node_target_mAP is not None and eight_node_anchor_mAP is not None:
        plt.figure(figsize=(10, 6))
        plt.plot(results_df['fraction'], results_df['target_mAP'] - eight_node_target_mAP, marker='o', label=f'Target Delta ({target_name})', color=target_color)
        plt.plot(results_df['fraction'], results_df['anchor_mAP'] - eight_node_anchor_mAP, marker='s', linestyle='--', label='Anchors Delta', color=anchor_color)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3, label='8-node Ckpt (0 Delta)')
        plt.xlabel("Data Fraction")
        plt.ylabel("Delta mAP@0.5-0.95")
        plt.title("LoRA Fine-tuning: Relative to 8-node Baseline")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("lora_fraction_relative_to_8node.png", dpi=180)
        plt.close()
        
    print("Plots generated successfully.")

if __name__ == "__main__":
    main()

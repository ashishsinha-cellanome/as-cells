import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import re

def main():
    # Set a clean, aesthetic style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    df = pd.read_csv("generalization_tracking.csv")
    df = df[~df['dataset'].astype(str).str.startswith('#')]
    df = df[df['split_type'] == 'test_ds']

    baseline = df[df['experiment'] == 'Baseline'][['dataset', 'mAP50_95']].set_index('dataset')

    results = []
    for exp in df['experiment'].unique():
        if exp == 'Baseline': continue
        
        exp_df = df[df['experiment'] == exp][['dataset', 'mAP50_95']].set_index('dataset')
        rel_perf = (exp_df['mAP50_95'] / baseline['mAP50_95']) * 100
        
        # Extract node count for coloring
        m = re.search(r'(\d+)\s+Node', exp)
        node_count = int(m.group(1)) if m else 0
        
        results.append({
            'experiment': exp,
            'mean': rel_perf.mean(),
            'std': rel_perf.std(),
            'nodes': node_count
        })

    results_df = pd.DataFrame(results).sort_values('mean', ascending=True)

    # Okabe-Ito Colorblind-Friendly Palette mapped to Node counts
    color_map = {
        2: '#999999',  # Grey
        5: '#E69F00',  # Orange
        6: '#56B4E9',  # Sky Blue
        7: '#009E73',  # Bluish Green
        8: '#0072B2'   # Deep Blue
    }
    
    # Assign colors based on node count
    bar_colors = [color_map.get(n, '#CC79A7') for n in results_df['nodes']]

    fig, ax = plt.subplots(figsize=(12, 8.5))
    
    # Plot horizontal bars
    bars = ax.barh(results_df['experiment'], results_df['mean'], xerr=results_df['std'],
                   capsize=4, color=bar_colors, edgecolor='none', alpha=0.9,
                   error_kw={'ecolor': '#333333', 'elinewidth': 1.5, 'markeredgewidth': 1.5})

    # Baseline line (Vermillion/Red-Orange from Okabe-Ito)
    ax.axvline(x=100, color='#D55E00', linestyle='--', linewidth=2, zorder=0)

    # Aesthetics and Formatting
    ax.set_xlabel('Mean Relative Performance (%)', fontsize=13, fontweight='bold', labelpad=12)
    ax.set_ylabel('Experiment Configuration', fontsize=13, fontweight='bold', labelpad=12)
    ax.set_title('Generalization Performance across Experiments', 
                 fontsize=16, fontweight='bold', pad=15)
                 
    ax.tick_params(axis='y', labelsize=11)
    ax.tick_params(axis='x', labelsize=11)
    ax.set_xlim(0, 115)
    
    ax.grid(axis='x', linestyle=':', alpha=0.7, color='gray')
    ax.grid(axis='y', visible=False)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#dddddd')
    ax.spines['bottom'].set_color('#dddddd')

    # Value labels inside the bars
    for bar in bars:
        width = bar.get_width()
        # Ensure text is inside the bar
        if width > 15:
            ax.text(width - 3, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', 
                     ha='right', va='center', color='white', fontweight='bold', fontsize=10)
        else:
            ax.text(width + 3, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', 
                     ha='left', va='center', color='black', fontweight='bold', fontsize=10)

    # Custom Legend
    legend_patches = [mpatches.Patch(color=color_map[k], label=f'{k} Nodes') 
                      for k in sorted(color_map.keys()) if k in results_df['nodes'].values]
    
    # Add a dummy line for the baseline in legend
    import matplotlib.lines as mlines
    baseline_line = mlines.Line2D([], [], color='#D55E00', linestyle='--', linewidth=2, label='Baseline (100%)')
    legend_patches.append(baseline_line)
    
    ax.legend(handles=legend_patches, loc='lower right', frameon=True, 
              fontsize=11, title="Legend", title_fontsize=12, fancybox=True, shadow=True)

    plt.tight_layout()
    plt.savefig('generalization_stats_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved generalization_stats_summary.png")

if __name__ == "__main__":
    main()

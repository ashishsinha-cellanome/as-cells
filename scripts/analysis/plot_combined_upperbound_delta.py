import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pandas as pd
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import seaborn as sns
import re

def parse_html(file_path, frac_label):
    with open(file_path, 'r') as f:
        html_content = f.read()

    soup = BeautifulSoup(html_content, 'html.parser')

    data = []
    current_dataset = None
    current_metric_type = None

    for element in soup.find_all(['h2', 'p', 'table']):
        if element.name == 'h2' and 'Dataset:' in element.text:
            current_dataset = element.text.replace('Dataset:', '').strip()
            if 'train_ds/' in current_dataset:
                current_dataset = current_dataset.split('train_ds/')[1].split('/')[0]
            elif 'domain_ds/' in current_dataset:
                current_dataset = current_dataset.split('domain_ds/')[1].split('/')[0]
            elif 'test_ds/' in current_dataset:
                current_dataset = current_dataset.split('test_ds/')[1].split('/')[0]
        elif element.name == 'p' and 'Metric Type:' in element.text:
            current_metric_type = element.text.replace('Metric Type:', '').strip()
        elif element.name == 'table':
            if current_dataset and current_metric_type and 'BBOX' in current_metric_type:
                headers = [th.text.strip() for th in element.find_all('th')]
                rows = element.find('tbody').find_all('tr')
                for row in rows:
                    cols = [td.text.strip() for td in row.find_all('td')]
                    if cols[0] == 'all':
                        row_data = dict(zip(headers, cols))
                        data.append({
                            'Dataset': current_dataset,
                            'Fraction': frac_label,
                            'mAP@0.5-0.95': float(row_data['mAP@0.5-0.95'])
                        })
    return pd.DataFrame(data)

df_100 = parse_html('coverage_exp/upperbound_fullFT_100_percent_wandb.html', '100% Data')
df_50 = parse_html('coverage_exp/upperbound_fullFT_frac0.5.html', '50% Data')
df_25 = parse_html('coverage_exp/upperbound_fullFT_frac0.25.html', '25% Data')

def short_name(d):
    m = re.match(r"^(\d{6,8})_(.*)$", str(d))
    date_part, rest = (m.group(1), m.group(2)) if m else ("", str(d))
    rest = rest.replace("_4_class", "").replace("_10x", "").strip('_')
    date_short = date_part[2:] if len(date_part) == 8 else date_part
    return f"{rest} ({date_short})" if date_short else rest

df_100['Dataset_Short'] = df_100['Dataset'].apply(short_name)
df_50['Dataset_Short'] = df_50['Dataset'].apply(short_name)
df_25['Dataset_Short'] = df_25['Dataset'].apply(short_name)

# Merge and calculate Delta
df_merged = pd.merge(df_100, df_50, on=['Dataset', 'Dataset_Short'], suffixes=('_100', '_50'))
df_merged = pd.merge(df_merged, df_25, on=['Dataset', 'Dataset_Short'])
df_merged = df_merged.rename(columns={'mAP@0.5-0.95': 'mAP@0.5-0.95_25'})

# Drop duplicates
df_merged = df_merged.drop_duplicates(subset=['Dataset_Short'])

# Calculate relative difference vs 100% baseline
df_merged['Relative_Delta_Pct_50'] = (df_merged['mAP@0.5-0.95_50'] - df_merged['mAP@0.5-0.95_100']) / df_merged['mAP@0.5-0.95_100'] * 100.0
df_merged['Relative_Delta_Pct_25'] = (df_merged['mAP@0.5-0.95_25'] - df_merged['mAP@0.5-0.95_100']) / df_merged['mAP@0.5-0.95_100'] * 100.0

# Sort datasets by 100% performance to match the previous plots
df_merged = df_merged.sort_values('mAP@0.5-0.95_100', ascending=False)

plt.figure(figsize=(14, 8))
sns.set_theme(style="whitegrid", rc={"axes.labelsize": 14, "axes.titlesize": 16, "xtick.labelsize": 10, "ytick.labelsize": 12})

# Plot as a line plot
plt.plot(df_merged['Dataset_Short'], df_merged['Relative_Delta_Pct_50'], marker='o', linestyle='-', linewidth=2, markersize=8, label='50% Data', color='#D55E00')
plt.plot(df_merged['Dataset_Short'], df_merged['Relative_Delta_Pct_25'], marker='s', linestyle='-', linewidth=2, markersize=8, label='25% Data', color='#009E73')

plt.axhline(0, color='black', linewidth=1.5, linestyle='--', label='100% Data (Baseline)')

for x, y_50, y_25 in zip(df_merged['Dataset_Short'], df_merged['Relative_Delta_Pct_50'], df_merged['Relative_Delta_Pct_25']):
    # offset by a small fixed amount in points so it's always placed above the marker
    plt.annotate(f'{y_50:.1f}%', (x, y_50), textcoords="offset points", xytext=(0, 8),
                 ha='center', va='bottom', fontsize=10, fontweight='bold', color='#D55E00',
                 bbox=dict(facecolor='white', edgecolor='none', alpha=0.6, pad=0.5))
    plt.annotate(f'{y_25:.1f}%', (x, y_25), textcoords="offset points", xytext=(0, -15),
                 ha='center', va='top', fontsize=10, fontweight='bold', color='#009E73',
                 bbox=dict(facecolor='white', edgecolor='none', alpha=0.6, pad=0.5))

plt.title('Relative Performance Delta vs 100% Data Scale', fontsize=18)
plt.xlabel('Dataset', fontsize=14)
plt.ylabel('Relative Delta in mAP@0.5-0.95 (%)', fontsize=14)
plt.xticks(rotation=90, ha='center')
plt.legend(title='Data Fraction', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=12, title_fontsize=14)
plt.tight_layout()
plt.savefig('coverage_exp/upperbound_combined_fractions_delta_lineplot.png', dpi=600, bbox_inches='tight')
print("Plot saved to coverage_exp/upperbound_combined_fractions_delta_lineplot.png")

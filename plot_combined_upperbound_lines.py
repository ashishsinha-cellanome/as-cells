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

df = pd.concat([df_100, df_50, df_25], ignore_index=True)
df = df.drop_duplicates(subset=['Dataset', 'Fraction'])

def short_name(d):
    m = re.match(r"^(\d{6,8})_(.*)$", str(d))
    date_part, rest = (m.group(1), m.group(2)) if m else ("", str(d))
    rest = rest.replace("_4_class", "").replace("_10x", "").strip('_')
    date_short = date_part[2:] if len(date_part) == 8 else date_part
    return f"{rest} ({date_short})" if date_short else rest

df['Dataset_Short'] = df['Dataset'].apply(short_name)

# Sort datasets by 100% performance
df_100_sorted = df[df['Fraction'] == '100% Data'].sort_values('mAP@0.5-0.95', ascending=False)
dataset_order = df_100_sorted['Dataset_Short'].tolist()

plt.figure(figsize=(14, 8))
sns.set_theme(style="whitegrid", rc={"axes.labelsize": 14, "axes.titlesize": 16, "xtick.labelsize": 10, "ytick.labelsize": 12})
colors = {"100% Data": "#0072B2", "50% Data": "#E69F00", "25% Data": "#009E73"} # Okabe-Ito Blue, Orange, Green

for frac in ['100% Data', '50% Data', '25% Data']:
    df_frac = df[df['Fraction'] == frac].set_index('Dataset_Short').reindex(dataset_order)
    plt.plot(df_frac.index, df_frac['mAP@0.5-0.95'], marker='o', linestyle='-', linewidth=2, markersize=8, label=frac, color=colors[frac])

plt.title('Full Finetuning on Adhered Cell-Lines with Varied Data Scales', fontsize=18)
plt.xlabel('Dataset', fontsize=14)
plt.ylabel('mAP@0.5-0.95', fontsize=14)
plt.xticks(rotation=90, ha='center')
plt.legend(title='Data Fraction', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=12, title_fontsize=14)
plt.tight_layout()
plt.savefig('coverage_exp/upperbound_combined_fractions_mAP_lineplot.png', dpi=300, bbox_inches='tight')
print("Plot saved to coverage_exp/upperbound_combined_fractions_mAP_lineplot.png")

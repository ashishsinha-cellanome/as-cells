import os
import re
import csv
import sys
import glob
import argparse
from html.parser import HTMLParser
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# =========================================================
# 1. CLI Argument Parsing (Robust)
# =========================================================
parser = argparse.ArgumentParser(description="Parse RF-DETR HTML report and generate metrics plots.")
parser.add_argument("input_html", nargs="?", default=None, 
                    help="Path to the input HTML file. If not provided, the script will look for an HTML file in the current directory.")
parser.add_argument("--out-csv", default="metrics_tidy.csv", help="Name of the output CSV file (default: metrics_tidy.csv)")
parser.add_argument("--out-prefix", default="plot_", help="Prefix for the output plot images (default: plot_)")
parser.add_argument("--title-suffix", default="", help="Optional suffix to add to plot titles (e.g., ' (Phase 2)')")
parser.add_argument("--in-domain", default=["a549"], nargs="+", help="One or more substrings to identify the in-domain dataset(s) (e.g., 'a549', 'hela'). Default: ['a549']")

args = parser.parse_args()

# Auto-detect HTML file if not explicitly provided
INPUT_HTML = args.input_html
if INPUT_HTML is None:
    html_files = glob.glob("*.html")
    if not html_files:
        sys.exit("ERROR: No input HTML file provided, and no .html files found in the current directory.")
    INPUT_HTML = html_files[0]
    print(f"No input file specified. Auto-detected and using: '{INPUT_HTML}'")

OUTPUT_CSV = args.out_csv
OUT_PREFIX = args.out_prefix
TITLE_SUFFIX = args.title_suffix
IN_DOMAINS = args.in_domain
IN_DOMAIN_STR = ", ".join(IN_DOMAINS)


# =========================================================
# 2. HTML Parser Implementation (Standard Library Only)
# =========================================================
class ReportHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows = []
        self.warnings = []
        
        # State tracking
        self.in_h2 = False
        self.in_p = False
        self.in_tbody = False
        self.in_tr = False
        self.in_td = False
        
        # Extracted context
        self.cur_split_type = None
        self.cur_dataset = None
        self.cur_metric_type = None
        
        # Temporary parsing buffers
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
                full = text.replace("Dataset:", "").strip()
                segments = [s for s in full.split("/") if s]
                if len(segments) < 2:
                    self.warnings.append(f"Could not parse dataset path: '{full}'")
                    self.cur_dataset, self.cur_split_type = None, None
                else:
                    self.cur_split_type = segments[0]
                    self.cur_dataset = segments[1]
        
        elif tag == 'p':
            self.in_p = False
            text = "".join(self.temp_text).strip()
            if text.startswith("Metric Type:"):
                if "BBOX" in text.upper():
                    self.cur_metric_type = "BBOX"
                elif "SEGM" in text.upper() or "SEGMENTATION" in text.upper():
                    self.cur_metric_type = "SEGM"
                else:
                    self.cur_metric_type = None
                    self.warnings.append(f"Unrecognized metric type context: '{text}'")
                    
        elif tag == 'tbody':
            self.in_tbody = False
            
        elif tag == 'tr':
            self.in_tr = False
            if self.in_tbody and self.current_row:
                if len(self.current_row) >= 6:
                    cls, p, r, f1, map50, map50_95 = self.current_row[:6]
                    try:
                        self.rows.append({
                            "dataset": self.cur_dataset,
                            "split_type": self.cur_split_type,
                            "metric_type": self.cur_metric_type,
                            "class": cls,
                            "P": float(p),
                            "R": float(r),
                            "F1": float(f1),
                            "mAP50": float(map50),
                            "mAP50_95": float(map50_95)
                        })
                    except ValueError as e:
                        self.warnings.append(f"Skipped row containing non-numeric values {self.current_row}: {e}")
                        
        elif tag == 'td':
            self.in_td = False
            self.current_row.append("".join(self.temp_text).strip())

    def handle_data(self, data):
        if self.in_h2 or self.in_p or self.in_td:
            self.temp_text.append(data)


# =========================================================
# 3. Extract Data from HTML & Save CSV
# =========================================================
if not os.path.exists(INPUT_HTML):
    sys.exit(f"ERROR: Input HTML file '{INPUT_HTML}' not found.")

print(f"Reading and parsing HTML: '{INPUT_HTML}'...")
with open(INPUT_HTML, "r", encoding="utf-8") as f:
    html_content = f.read()

parser = ReportHTMLParser()
parser.feed(html_content)

if not parser.rows:
    sys.exit("ERROR: No structured data rows could be parsed. Check HTML format.")

# Deduplicate identical repeats
seen = set()
deduped_rows = []
dup_count = 0
for r in parser.rows:
    key = (r["dataset"], r["split_type"], r["metric_type"], r["class"])
    if key in seen:
        dup_count += 1
        continue
    seen.add(key)
    deduped_rows.append(r)

# Write to Tidy CSV
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(deduped_rows[0].keys()))
    writer.writeheader()
    writer.writerows(deduped_rows)

print(f"Parsed {len(parser.rows)} rows -> {len(deduped_rows)} unique rows ({dup_count} exact duplicates removed)")
print(f"Saved tidy table to: '{OUTPUT_CSV}'\n")

if parser.warnings:
    print(f"Warnings found during parsing ({len(parser.warnings)}):")
    for w in parser.warnings[:5]:
        print(f"  - {w}")
    if len(parser.warnings) > 5:
        print(f"  - ...and {len(parser.warnings) - 5} more")


# =========================================================
# 4. Data Visualization Pipeline (Matplotlib)
# =========================================================
# Okabe-Ito colorblind-safe palette
BLACK   = "#000000"
ORANGE  = "#E69F00"
SKY     = "#56B4E9"
GREEN   = "#009E73"
YELLOW  = "#F0E442"
BLUE    = "#0072B2"
VERM    = "#D55E00"
PURPLE  = "#CC79A7"
GREY    = "#999999"

IN_DOMAIN_COLOR = BLUE      # model's own train_ds test split
ZERO_SHOT_COLOR = ORANGE    # zero-shot cross-dataset test_ds

plt.rcParams.update({
    "font.size": 10,
    "axes.edgecolor": "#444444",
    "axes.labelcolor": "#222222",
    "text.color": "#222222",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "axes.grid": True,
    "grid.color": "#e0e0e0",
    "grid.linewidth": 0.6,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "svg.fonttype": "none",
})

# Load the newly created CSV
df = pd.read_csv(OUTPUT_CSV)
df_all = df[df["class"] == "all"].copy()

# Fix for when HTML contains duplicate test_ds and train_ds files.
# We keep only 'test_ds' splits to deduplicate evaluations.
df_all = df_all[df_all["split_type"] == "test_ds"].copy()

# Identify in-domain datasets dynamically (supports multiple via regex OR)
pattern = '|'.join(IN_DOMAINS)
df_all["domain"] = np.where(df_all["dataset"].str.contains(pattern, case=False, na=False),
                             f"In-domain ({IN_DOMAIN_STR})",
                             "Zero-shot (unseen)")

def short_name(d):
    m = re.match(r"^(\d{6,8})_(.*)$", str(d))
    date_part, rest = (m.group(1), m.group(2)) if m else ("", str(d))
    rest = rest.replace("_4_class", "").replace("_10x", "")
    date_short = date_part[2:] if len(date_part) == 8 else date_part
    return f"{rest} ({date_short})" if date_short else rest

df_all["label"] = df_all["dataset"].apply(short_name)

bbox = df_all[df_all.metric_type == "BBOX"].set_index("dataset")
segm = df_all[df_all.metric_type == "SEGM"].set_index("dataset")

# ---------------------------------------------------------
# Plot 1: mAP@0.5:0.95 (BBOX, class=all) ranked bar chart
# ---------------------------------------------------------
d = bbox.sort_values("mAP50_95", ascending=True)
fig, ax = plt.subplots(figsize=(9, 8))
colors = [IN_DOMAIN_COLOR if v == f"In-domain ({IN_DOMAIN_STR})" else ZERO_SHOT_COLOR for v in d["domain"]]
bars = ax.barh(d["label"], d["mAP50_95"], color=colors, edgecolor="white", height=0.68)
for bar, val in zip(bars, d["mAP50_95"]):
    ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, f"{val:.3f}",
            va="center", ha="left", fontsize=8.5, color="#222222")
ax.set_xlabel("mAP@0.5:0.95 (BBOX, class = all)")
ax.set_title(f"RF-DETR Cross-Dataset Generalization{TITLE_SUFFIX}\nBounding-Box mAP@0.5:0.95 by Test Dataset", fontsize=12, weight="bold", loc="left")
ax.set_xlim(0, max(d["mAP50_95"]) * 1.18)
legend_handles = [mpatches.Patch(color=IN_DOMAIN_COLOR, label=f"In-domain ({IN_DOMAIN_STR})"),
                  mpatches.Patch(color=ZERO_SHOT_COLOR, label="Zero-shot (unseen test datasets)")]
ax.legend(handles=legend_handles, loc="lower right", frameon=False)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(f"{OUT_PREFIX}1_map5095_bbox_ranked.png", dpi=180, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------
# Plot 1b: mAP@0.5:0.95 (SEGM, class=all) ranked bar chart
# ---------------------------------------------------------
d = segm.sort_values("mAP50_95", ascending=True)
fig, ax = plt.subplots(figsize=(9, 8))
colors = [IN_DOMAIN_COLOR if v == f"In-domain ({IN_DOMAIN_STR})" else ZERO_SHOT_COLOR for v in d["domain"]]
bars = ax.barh(d["label"], d["mAP50_95"], color=colors, edgecolor="white", height=0.68)
for bar, val in zip(bars, d["mAP50_95"]):
    ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, f"{val:.3f}",
            va="center", ha="left", fontsize=8.5, color="#222222")
ax.set_xlabel("mAP@0.5:0.95 (BBOX, class = all)")
ax.set_title(f"RF-DETR Cross-Dataset Generalization{TITLE_SUFFIX}\nBounding-Box mAP@0.5:0.95 by Test Dataset", fontsize=12, weight="bold", loc="left")
ax.set_xlim(0, max(d["mAP50_95"]) * 1.18)
ax.legend(handles=legend_handles, loc="lower right", frameon=False)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(f"{OUT_PREFIX}1_map5095_segm_ranked.png", dpi=180, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------
# Plot 2: mAP@0.5 (BBOX, class=all) ranked bar chart
# ---------------------------------------------------------
d = bbox.sort_values("mAP50", ascending=True)
fig, ax = plt.subplots(figsize=(9, 8))
colors = [IN_DOMAIN_COLOR if v == f"In-domain ({IN_DOMAIN_STR})" else ZERO_SHOT_COLOR for v in d["domain"]]
bars = ax.barh(d["label"], d["mAP50"], color=colors, edgecolor="white", height=0.68)
for bar, val in zip(bars, d["mAP50"]):
    ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, f"{val:.3f}",
            va="center", ha="left", fontsize=8.5, color="#222222")
ax.set_xlabel("mAP@0.5 (BBOX, class = all)")
ax.set_title(f"RF-DETR Cross-Dataset Generalization{TITLE_SUFFIX}\nBounding-Box mAP@0.5 by Test Dataset", fontsize=12, weight="bold", loc="left")
ax.set_xlim(0, max(d["mAP50"]) * 1.15)
ax.legend(handles=legend_handles, loc="lower right", frameon=False)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(f"{OUT_PREFIX}2_map50_bbox_ranked.png", dpi=180, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------
# Plot 2b: mAP@0.5 (SEGM, class=all) ranked bar chart
# ---------------------------------------------------------
d = segm.sort_values("mAP50", ascending=True)
fig, ax = plt.subplots(figsize=(9, 8))
colors = [IN_DOMAIN_COLOR if v == f"In-domain ({IN_DOMAIN_STR})" else ZERO_SHOT_COLOR for v in d["domain"]]
bars = ax.barh(d["label"], d["mAP50"], color=colors, edgecolor="white", height=0.68)
for bar, val in zip(bars, d["mAP50"]):
    ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, f"{val:.3f}",
            va="center", ha="left", fontsize=8.5, color="#222222")
ax.set_xlabel("mAP@0.5 (BBOX, class = all)")
ax.set_title(f"RF-DETR Cross-Dataset Generalization{TITLE_SUFFIX}\nBounding-Box mAP@0.5 by Test Dataset", fontsize=12, weight="bold", loc="left")
ax.set_xlim(0, max(d["mAP50"]) * 1.15)
ax.legend(handles=legend_handles, loc="lower right", frameon=False)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(f"{OUT_PREFIX}2_map50_segm_ranked.png", dpi=180, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------
# Plot 3: BBOX vs SEGM dumbbell plot (mAP@0.5:0.95)
# ---------------------------------------------------------
merged = bbox[["label", "domain", "mAP50_95"]].rename(columns={"mAP50_95": "bbox"})
merged["segm"] = segm["mAP50_95"]
merged = merged.sort_values("bbox", ascending=True)

fig, ax = plt.subplots(figsize=(9, 8))
y = np.arange(len(merged))
for yi, (idx, row) in zip(y, merged.iterrows()):
    ax.plot([row["bbox"], row["segm"]], [yi, yi], color="#bbbbbb", zorder=1, lw=1.6)
ax.scatter(merged["bbox"], y, color=BLUE, label="BBOX", zorder=2, s=55)
ax.scatter(merged["segm"], y, color=VERM, label="SEGM", zorder=2, s=55, marker="D")
ax.set_yticks(y)
ax.set_yticklabels(merged["label"])
ax.set_xlabel("mAP@0.5:0.95 (class = all)")
ax.set_title(f"Detection (BBOX) vs Segmentation (SEGM) mAP@0.5:0.95{TITLE_SUFFIX}\nby Test Dataset", fontsize=12, weight="bold", loc="left")
ax.legend(frameon=False, loc="lower right")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(f"{OUT_PREFIX}3_bbox_vs_segm_dumbbell.png", dpi=180, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------
# Plot 4: Precision vs Recall scatter (BBOX, class=all)
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 7))
for dom, color, marker in [(f"In-domain ({IN_DOMAIN_STR})", IN_DOMAIN_COLOR, "s"),
                            ("Zero-shot (unseen)", ZERO_SHOT_COLOR, "o")]:
    sub = bbox[bbox.domain == dom]
    ax.scatter(sub["R"], sub["P"], color=color, marker=marker, s=90,
               edgecolor="white", linewidth=0.8, label=dom, zorder=3)

for idx, row in bbox.iterrows():
    ax.annotate(row["label"], (row["R"], row["P"]), fontsize=7.2,
                xytext=(4, 3), textcoords="offset points", color="#444444")

# Iso-F1 Curves
f1_levels = [0.2, 0.4, 0.6, 0.8]
r_range = np.linspace(0.01, 1, 200)
for f1 in f1_levels:
    p_curve = (f1 * r_range) / (2 * r_range - f1 + 1e-9)
    p_curve = np.where((p_curve > 0) & (p_curve <= 1), p_curve, np.nan)
    ax.plot(r_range, p_curve, color="#dddddd", lw=0.9, zorder=1)
    valid = ~np.isnan(p_curve)
    if valid.any():
        xi = r_range[valid][-1]
        yi = p_curve[valid][-1]
        ax.text(xi, yi, f"F1={f1}", fontsize=7, color="#aaaaaa", ha="right", va="bottom")

ax.set_xlabel("Recall (BBOX, class = all)")
ax.set_ylabel("Precision (BBOX, class = all)")
ax.set_title(f"Precision vs. Recall Across Test Datasets{TITLE_SUFFIX}\n(grey lines = iso-F1 contours)", fontsize=12, weight="bold", loc="left")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.legend(frameon=False, loc="upper left")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(f"{OUT_PREFIX}4_precision_recall_scatter.png", dpi=180, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------
# Plot 5: Heatmap of all metrics (BBOX, class=all)
# ---------------------------------------------------------
metrics_cols = ["P", "R", "F1", "mAP50", "mAP50_95"]
heat_df = bbox.sort_values("mAP50_95", ascending=False)[["label"] + metrics_cols].set_index("label")

fig, ax = plt.subplots(figsize=(7, 9))
data = heat_df.values
im = ax.imshow(data, aspect="auto", cmap="cividis", vmin=0, vmax=1)
ax.set_xticks(range(len(metrics_cols)))
ax.set_xticklabels(["Precision", "Recall", "F1", "mAP@0.5", "mAP@0.5:0.95"], rotation=30, ha="right")
ax.set_yticks(range(len(heat_df)))
ax.set_yticklabels(heat_df.index)
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        val = data[i, j]
        txt_color = "white" if val < 0.55 else "black"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color=txt_color)
cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
cbar.set_label("Score")
ax.set_title(f"BBOX Metrics Heatmap (class = all){TITLE_SUFFIX}\nsorted by mAP@0.5:0.95", fontsize=12, weight="bold", loc="left")
plt.tight_layout()
plt.savefig(f"{OUT_PREFIX}5_heatmap_bbox_all_metrics.png", dpi=180, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------
# Plot 6: Per-class mAP@0.5:0.95 grouped bars
# ---------------------------------------------------------
df_cls = df[(df.metric_type == "BBOX") & (df["class"] != "all")].copy()
# Fixed logic for test_ds extraction
df_cls = df_cls[df_cls["split_type"] == "test_ds"]

label_map = df_all.drop_duplicates("dataset").set_index("dataset")["label"]
df_cls["label"] = df_cls["dataset"].map(label_map)
order = bbox.sort_values("mAP50_95", ascending=False)["label"].tolist()

fig, ax = plt.subplots(figsize=(11, 7))
width = 0.38
x = np.arange(len(order))
class_colors = {"cell": SKY, "cell-adhered": VERM, "soma": GREEN}
plotted_classes = []
for i, cls in enumerate(["cell", "cell-adhered", "soma"]):
    vals = []
    for lab in order:
        row = df_cls[(df_cls.label == lab) & (df_cls["class"] == cls)]
        vals.append(row["mAP50_95"].values[0] if len(row) else np.nan)
    if all(np.isnan(vals)):
        continue
    offset = (i - 1) * width * 0.7
    ax.bar(x + offset, vals, width * 0.65, label=cls, color=class_colors[cls])
    plotted_classes.append(cls)

ax.set_xticks(x)
ax.set_xticklabels(order, rotation=60, ha="right", fontsize=8)
ax.set_ylabel("mAP@0.5:0.95 (BBOX)")
ax.set_title(f"Per-Class BBOX mAP@0.5:0.95 by Test Dataset{TITLE_SUFFIX}", fontsize=12, weight="bold", loc="left")
ax.legend(frameon=False, title="Class")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(f"{OUT_PREFIX}6_per_class_map5095.png", dpi=180, bbox_inches="tight")
plt.close()

print("All plots saved successfully.")
print("\nSummary table (BBOX, class=all), sorted by mAP@0.5:0.95:")
print(bbox.sort_values("mAP50_95", ascending=False)[["label","domain","P","R","F1","mAP50","mAP50_95"]].to_string(index=False))

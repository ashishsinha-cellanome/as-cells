# Interactive Coverage Tracking and Performance Dashboard

## Architecture Overview
We will create an interactive, standalone single-page web dashboard (`coverage_dashboard.html`) utilizing an Atmospheric visual theme, a clean layout with no faux browser chrome, and roman-only headers.

To ensure the dashboard loads instantly and remains self-contained, we will write a Python compilation script `compile_dashboard.py` which:
1. Parses arborescence reports, node lists, and edges in the root directory (e.g., `coverage_arborescence_report_*.txt` files) for DINOv2 Base and Large models across all thresholds (0.4, 0.5, 0.6, and default).
2. Parses accuracy metrics (mAP@50, mAP@50-95, true positives, false positives, false negatives, precision, recall, and F1) from `PHASE2_meet_all_metrics.csv` for models (`rf_detr_seg`, `yolov26`, `yolo`) and classes.
3. Parses zero-shot generalization experiment results from `generalization_tracking.csv`.
4. Parses compute and complexity profiles (GFLOPS, parameter counts, native FPS, compiled FPS, and inference/postprocessing latencies) from `run_architectures_benchmark_report.csv`.
5. Compiles and injects this structured data into an HTML template (`dashboard_template.html`), exporting a standalone `coverage_dashboard.html` file.

To support live updates as experiments run, the compilation script will support a `--watch` mode to monitor files for updates and rebuild automatically. Additionally, when hosted on a local HTTP server, the dashboard will support dynamic hot-reloading by fetching the raw CSVs/TXTs directly and parsing them in-browser using PapaParse.

## Out-of-Scope
The following features are explicitly out-of-scope for this project:
1. **Interactive/Persistent Database:** The dashboard will rely entirely on static pre-compiled JSON or direct static CSV/TXT file parses. No backend database (e.g., SQLite, PostgreSQL) will be introduced.
2. **User Authentication or Access Control:** The dashboard is a standalone static HTML file. No login, user roles, or security authentication will be implemented.
3. **Live Server-Side WebSockets:** Real-time hot-reloading will not use WebSockets. When watched or run under a local server, reloading will happen via client-side fetch or browser refresh.
4. **Model Training or Execution:** The compilation script will not trigger, execute, or retrain any machine learning models. It only processes pre-generated CSV and TXT report files.

## Components and Responsibilities
- **Data Compiler (`compile_dashboard.py`):**
  - **Arborescence Report Parser:** Reads the text files in the root directory to extract ranked lists of out-coverage datasets and parent-child tree edges, including status (`STRONG`/`WEAK`) and coverage percentages.
  - **Metrics Parser:** Reads `PHASE2_meet_all_metrics.csv` and structures performance metrics by model, dataset, and class.
  - **Generalization Parser:** Reads `generalization_tracking.csv` to format the baseline and multi-node experiment mAP values.
  - **Complexity Parser:** Reads `run_architectures_benchmark_report.csv` to map hardware latencies, parameter counts, and GFLOPS for each configuration.
  - **Watcher Loop:** Uses a polling file monitor (triggered via a `--watch` CLI flag) to recompile the dashboard on any CSV or TXT file update.
- **Frontend Presentation (`coverage_dashboard.html`):**
  - **Interactive Tree Explorer (D3.js / HTML canvas):** Renders the full hierarchical arborescence tree dynamically. The user will select DINOv2 Base vs. Large, use a selector for thresholds (0.4, 0.5, 0.6, and default) to observe components dissolve, and hover over parent-child paths for details.
  - **Compute Complexity Hub:** Displays a scatter plot of GFLOPS vs. Latency alongside a datatable of FPS and postprocessing profiling, which will support sorting and backbone model comparison.
  - **Model Evaluation Grid:** Interactive datatables with dropdown filters for Model, Class, and Dataset to view and sort precision, recall, F1, and mAP metrics.
  - **Generalization Trend Tracker:** Displays interactive charts comparing multi-node zero-shot performance drop-offs vs. full baselines on unseen datasets.

## Data Flow
1. **At Compile Time:**
   - User invokes `python compile_dashboard.py`.
   - The Python script reads raw files from the root directory (`coverage_arborescence_report_*.txt`, `PHASE2_meet_all_metrics.csv`, `generalization_tracking.csv`, and `run_architectures_benchmark_report.csv`).
   - The script parses and encodes this raw data into JSON blobs.
   - The JSON strings are injected into the template file `dashboard_template.html` to generate `coverage_dashboard.html`.
2. **At Runtime (HTTP Server Hot-Reload):**
   - User visits `http://localhost:8000/coverage_dashboard.html` via Python's HTTP server.
   - The client-side JavaScript detects the HTTP/HTTPS protocol and immediately initiates parallel fetches to the raw CSV/TXT files on disk.
   - The frontend parses the live raw files on-the-fly, bypassing the compiled data to reflect newly run experiment results instantly upon browser refresh.

## Error Handling and Edge Cases
- **Malformed CSV/TXT Entries:** If a CSV contains blank or NaN metric values (e.g. `-1.0` in evaluation CSVs representing inactive classes), the parser will treat these as "Not Applicable (N/A)" and hide them from visualization lists rather than crashing or showing invalid negative rates.
- **Missing Asset Fallback:** If specific arborescence plots (PNGs) are missing when rendering, the dashboard will fall back to displaying the dynamically generated D3/CSS tree with a small non-blocking message.
- **CORS Local Access Restrictions:** If the HTML file is opened directly via the `file://` protocol, the JavaScript will catch the fetch security errors and transparently fall back to the pre-compiled, embedded JSON data.

## Testing Approach
- **Parsing Verification:** We will verify that the compilation script parses 100% of the nodes, edges, complexity parameters, and generalization scores without throwing key-errors.
- **Responsive Layout Verification:** We will verify the UI displays cleanly on mobile widths (320px, 375px, 414px) and large desktop monitors (root overflow-x set to clip, text stays fully readable inside grid tracks).
- **Interactive Toggles Verification:** We will manually test selecting base/large models, sliding thresholds, and toggling legend series to ensure visual rendering updates smoothly without lag or layout jumps.

## Documentation Impact
- Feature / user-facing docs introduced: none
- Materially amended existing docs: none
- Derived / memory docs invalidated: none

## External References
This specification relies on the schema and formatting of several external files. We recommend inlining their exact expected schemas and sample formats to ensure alignment between parsing code and data generators:

1. **Arborescence Report Format (`coverage_arborescence_report_*.txt`):**
   - Header metadata (e.g., `Root Node (Broadest Dataset): {dataset} (Mean Out-Coverage: {value})`)
   - `Ranked Nodes` section under a table headers: `Rank`, `Node`, `Mean Out-Coverage`
   - `Tree Edges` section under table headers: `Source`, `Target`, `Coverage`, `Status`
   - Example segment:
     ```text
     Root Node (Broadest Dataset): 20240625_mc38_10x_caged_4_class (Mean Out-Coverage: 0.554)
     Ranked Nodes (by Mean Out-Coverage):
     1     20240625_mc38_10x_caged_4_class0.554
     Tree Edges:
     Source               -> Target               | Coverage   | Status    
     20240625_mc38_10x_caged_4_class -> 20240905_u87-adhered_10x_caged_4_class | 0.843      | STRONG
     ```

2. **Metrics CSV Schema (`PHASE2_meet_all_metrics.csv`):**
   - Columns: `Model`, `Dataset`, `Class`, `mAP@50`, `mAP@50-95`, `TP`, `FP`, `FN`, `Precision`, `Recall`, `F1`
   - Example Row:
     `rf_detr_seg,20240703_neuron-adhered_10x_caged_4_class,cell,0.9081,0.5639,22152,8232,1196,0.7291,0.9487,0.8245`

3. **Generalization Master CSV Schema (`generalization_tracking.csv`):**
   - Columns: `dataset`, `split_type`, `mAP50_95`, `experiment`, `train_datasets`
   - Expected `split_type` values: `train_ds`, `test_ds`
   - Example Row:
     `20241212_preadipocytes-adhered_10x_uncaged_4_class,test_ds,0.456,c1_fibro_to_preadipo,20240509_Hs675Tfibroblasts_10x_caged_4_class`

4. **Architecture Benchmark Schema (`run_architectures_benchmark_report.csv`):**
   - Columns: `Model / Config`, `Batch Size`, `Params (M)`, `Input Size`, `GFLOPS/img`, `Forward Time/img (ms)`, `Postproc Time/img (ms)`, `Total Time/img (ms)`, `Native FPS`, `Compiled FPS`
   - Example Row:
     `RF-DETR base,1,31.2,672,42.9,23.16,0.43,23.60,42.4,150.5`

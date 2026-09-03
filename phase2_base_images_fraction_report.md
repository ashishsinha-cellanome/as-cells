# Phase 2 Fine-Tuning: Base Images per Data Fraction

A base image represents the original full 4K field of view. During training, the pipeline dynamically extracts 32 distinct square crops from each chosen base image. The table below shows the exact number of base images kept during fine-tuning across all 21 adhered datasets when scaling the `target_data_frac`.

| Dataset | Total Base Images | Frac=0.01 | Frac=0.05 | Frac=0.10 | Frac=0.25 | Frac=0.50 | Frac=1.00 |
|---------|------------------|-----------|-----------|-----------|-----------|-----------|-----------|
| `20240422_neuron-adhered_10x_uncaged_4_class` | 648 | 6 | 32 | 64 | **162** | 324 | 648 |
| `20240509_Hs675Tfibroblasts_10x_caged_4_class` | 648 | 6 | 32 | 64 | **162** | 324 | 648 |
| `20240509_hela-adhered_10x_caged_4_class` | 648 | 6 | 32 | 64 | **162** | 324 | 648 |
| `20240515_DC-adhered_10x_caged_4_class` | 303 | 3 | 15 | 30 | **75** | 151 | 303 |
| `20240516_DC-adhered_10x_caged_4_class` | 303 | 3 | 15 | 30 | **75** | 151 | 303 |
| `20240624_mc38_10x_caged_4_class` | 648 | 6 | 32 | 64 | **162** | 324 | 648 |
| `20240624_mc38_10x_uncaged_4_class` | 1296 | 12 | 64 | 129 | **324** | 648 | 1296 |
| `20240625_mc38_10x_caged_4_class` | 648 | 6 | 32 | 64 | **162** | 324 | 648 |
| `20240703_neuron-adhered_10x_caged_4_class` | 648 | 6 | 32 | 64 | **162** | 324 | 648 |
| `20240704_neuron-adhered_10x_caged_4_class` | 648 | 6 | 32 | 64 | **162** | 324 | 648 |
| `20240905_u87-adhered_10x_caged_4_class` | 624 | 6 | 31 | 62 | **156** | 312 | 624 |
| `20240924_enteric-glia-adhered_10x_uncaged_4_class` | 631 | 6 | 31 | 63 | **157** | 315 | 631 |
| `20241212_preadipocytes-adhered_10x_uncaged_4_class` | 453 | 4 | 22 | 45 | **113** | 226 | 453 |
| `20250108_neuron-adhered_10x_uncaged_4_class` | 648 | 6 | 32 | 64 | **162** | 324 | 648 |
| `20250227_preadipocytes-adhered_10x_caged_4_class` | 648 | 6 | 32 | 64 | **162** | 324 | 648 |
| `20250305_neuron-adhered_10x_uncaged_4_class` | 1942 | 19 | 97 | 194 | **485** | 971 | 1942 |
| `20250820_c8d1a_astrocytes-adherent_10x_caged_4_class`| 161 | 1 | 8 | 16 | **40** | 80 | 161 |
| `20250917_moc22-adhered_10x_caged_4_class` | 648 | 6 | 32 | 64 | **162** | 324 | 648 |
| `20260316_a549-tomm20-gfp-adhered_10x_caged_at_4x_4_class`| 648 | 6 | 32 | 64 | **162** | 324 | 648 |
| `231212_imr90_multichannel_overlay_4_class` | 391 | 3 | 19 | 39 | **97** | 195 | 391 |
| `240213_imr90_multichannel_overlay_4_class` | 267 | 2 | 13 | 26 | **66** | 133 | 267 |

### Translation to Actual Crop Counts
Because `train_rfdetr_phase2.py` sets `crops_per_base = 32`, for a standard 648-image target like `20250108_neuron-adhered_10x_uncaged_4_class`:
* **Frac 0.01**: 6 base images × 32 crops = **192 total training images**
* **Frac 0.25 (Current upperbound Run)**: 162 base images × 32 crops = **5,184 total training images** 
* **Frac 0.50 (Prior upperbound Run)**: 324 base images × 32 crops = **10,368 total training images** 
* **Frac 1.00**: 648 base images × 32 crops = **20,736 total training images**
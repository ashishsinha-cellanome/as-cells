# Topology Analysis Report
Threshold for Coverage: Distance < 0.2 (> 80% Coverage)

## Analysis for K=5
### Topological Roles
**Roots (Ultimate Supersets - Best for Generalizing Downward):**

**Leaves (Ultimate Subsets - Narrow Domains):**

**Internal Nodes (Clusters/Mid-level):**

**Isolates (Independent Domains):**
- `20240509_Hs675Tfibroblasts_10x_caged_4_class`
- `20240509_hela-adhered_10x_caged_4_class`
- `20240515_DC-adhered_10x_caged_4_class`
- `20240516_DC-adhered_10x_caged_4_class`
- `20240624_mc38_10x_caged_4_class`
- `20240624_mc38_10x_uncaged_4_class`
- `20240625_mc38_10x_caged_4_class`
- `20240905_u87-adhered_10x_caged_4_class`
- `20240924_enteric-glia-adhered_10x_uncaged_4_class`
- `20241212_preadipocytes-adhered_10x_uncaged_4_class`
- `20250227_preadipocytes-adhered_10x_caged_4_class`
- `20250820_c8d1a_astrocytes-adherent_10x_caged_4_class`
- `20250917_moc22-adhered_10x_caged_4_class`
- `20260316_a549-tomm20-gfp-adhered_10x_caged_at_4x_4_class`
- `231212_imr90_multichannel_overlay_4_class`
- `240213_imr90_multichannel_overlay_4_class`

### Hierarchical Tree & Coverage (% Covered by Parent)

---

## Analysis for K=10
### Topological Roles
**Roots (Ultimate Supersets - Best for Generalizing Downward):**
- `20240624_mc38_10x_uncaged_4_class` (Covers 1 subsets)
- `20260316_a549-tomm20-gfp-adhered_10x_caged_at_4x_4_class` (Covers 1 subsets)

**Leaves (Ultimate Subsets - Narrow Domains):**
- `20240624_mc38_10x_caged_4_class` (Covered by 2 supersets)

**Internal Nodes (Clusters/Mid-level):**
- `20240515_DC-adhered_10x_caged_4_class`
- `20240516_DC-adhered_10x_caged_4_class`

**Isolates (Independent Domains):**
- `20240509_Hs675Tfibroblasts_10x_caged_4_class`
- `20240509_hela-adhered_10x_caged_4_class`
- `20240625_mc38_10x_caged_4_class`
- `20240905_u87-adhered_10x_caged_4_class`
- `20240924_enteric-glia-adhered_10x_uncaged_4_class`
- `20241212_preadipocytes-adhered_10x_uncaged_4_class`
- `20250227_preadipocytes-adhered_10x_caged_4_class`
- `20250820_c8d1a_astrocytes-adherent_10x_caged_4_class`
- `20250917_moc22-adhered_10x_caged_4_class`
- `231212_imr90_multichannel_overlay_4_class`
- `240213_imr90_multichannel_overlay_4_class`

### Hierarchical Tree & Coverage (% Covered by Parent)
- **[ROOT]** `20240624_mc38_10x_uncaged_4_class`
```text
[ROOT] 20240624_mc38_10x_uncaged_4_class
└── [LEAF] `20240624_mc38_10x_caged_4_class` (Coverage: 96.7%)
```
- **[ROOT]** `20260316_a549-tomm20-gfp-adhered_10x_caged_at_4x_4_class`
```text
[ROOT] 20260316_a549-tomm20-gfp-adhered_10x_caged_at_4x_4_class
└── [LEAF] `20240624_mc38_10x_caged_4_class` (Coverage: 86.1%)
```

---

## Analysis for K=15
### Topological Roles
**Roots (Ultimate Supersets - Best for Generalizing Downward):**
- `20240509_hela-adhered_10x_caged_4_class` (Covers 1 subsets)
- `20240624_mc38_10x_uncaged_4_class` (Covers 1 subsets)
- `20240905_u87-adhered_10x_caged_4_class` (Covers 2 subsets)
- `20260316_a549-tomm20-gfp-adhered_10x_caged_at_4x_4_class` (Covers 1 subsets)

**Leaves (Ultimate Subsets - Narrow Domains):**
- `20240624_mc38_10x_caged_4_class` (Covered by 6 supersets)
- `20240625_mc38_10x_caged_4_class` (Covered by 1 supersets)
- `20250820_c8d1a_astrocytes-adherent_10x_caged_4_class` (Covered by 1 supersets)

**Internal Nodes (Clusters/Mid-level):**
- `20240515_DC-adhered_10x_caged_4_class`
- `20240516_DC-adhered_10x_caged_4_class`

**Isolates (Independent Domains):**
- `20240509_Hs675Tfibroblasts_10x_caged_4_class`
- `20240924_enteric-glia-adhered_10x_uncaged_4_class`
- `20241212_preadipocytes-adhered_10x_uncaged_4_class`
- `20250227_preadipocytes-adhered_10x_caged_4_class`
- `20250917_moc22-adhered_10x_caged_4_class`
- `231212_imr90_multichannel_overlay_4_class`
- `240213_imr90_multichannel_overlay_4_class`

### Hierarchical Tree & Coverage (% Covered by Parent)
- **[ROOT]** `20240509_hela-adhered_10x_caged_4_class`
```text
[ROOT] 20240509_hela-adhered_10x_caged_4_class
└── [LEAF] `20240624_mc38_10x_caged_4_class` (Coverage: 80.7%)
```
- **[ROOT]** `20240624_mc38_10x_uncaged_4_class`
```text
[ROOT] 20240624_mc38_10x_uncaged_4_class
└── [LEAF] `20240624_mc38_10x_caged_4_class` (Coverage: 99.7%)
```
- **[ROOT]** `20240905_u87-adhered_10x_caged_4_class`
```text
[ROOT] 20240905_u87-adhered_10x_caged_4_class
├── [LEAF] `20240624_mc38_10x_caged_4_class` (Coverage: 86.2%)
└── [LEAF] `20240625_mc38_10x_caged_4_class` (Coverage: 84.0%)
```
- **[ROOT]** `20260316_a549-tomm20-gfp-adhered_10x_caged_at_4x_4_class`
```text
[ROOT] 20260316_a549-tomm20-gfp-adhered_10x_caged_at_4x_4_class
└── [LEAF] `20240624_mc38_10x_caged_4_class` (Coverage: 93.7%)
```

---

## Analysis for K=30
### Topological Roles
**Roots (Ultimate Supersets - Best for Generalizing Downward):**
- `20240509_hela-adhered_10x_caged_4_class` (Covers 1 subsets)

**Leaves (Ultimate Subsets - Narrow Domains):**
- `20240509_Hs675Tfibroblasts_10x_caged_4_class` (Covered by 2 supersets)
- `231212_imr90_multichannel_overlay_4_class` (Covered by 1 supersets)

**Internal Nodes (Clusters/Mid-level):**
- `20240515_DC-adhered_10x_caged_4_class`
- `20240516_DC-adhered_10x_caged_4_class`
- `20240624_mc38_10x_caged_4_class`
- `20240624_mc38_10x_uncaged_4_class`
- `20240625_mc38_10x_caged_4_class`
- `20240905_u87-adhered_10x_caged_4_class`
- `20241212_preadipocytes-adhered_10x_uncaged_4_class`
- `20250227_preadipocytes-adhered_10x_caged_4_class`
- `20250820_c8d1a_astrocytes-adherent_10x_caged_4_class`
- `20250917_moc22-adhered_10x_caged_4_class`
- `20260316_a549-tomm20-gfp-adhered_10x_caged_at_4x_4_class`

**Isolates (Independent Domains):**
- `20240924_enteric-glia-adhered_10x_uncaged_4_class`
- `240213_imr90_multichannel_overlay_4_class`

### Hierarchical Tree & Coverage (% Covered by Parent)
- **[ROOT]** `20240509_hela-adhered_10x_caged_4_class`
```text
[ROOT] 20240509_hela-adhered_10x_caged_4_class
└── [NODE] `20240624_mc38_10x_caged_4_class` (Coverage: 91.2%)
    └── [NODE] `20240624_mc38_10x_uncaged_4_class` (Coverage: 85.1%)
        ├── [NODE] `20240624_mc38_10x_caged_4_class` (Coverage: 100.0%)
        ├── [NODE] `20240625_mc38_10x_caged_4_class` (Coverage: 87.5%)
        │   ├── [NODE] `20240624_mc38_10x_caged_4_class` (Coverage: 92.5%)
        │   ├── [NODE] `20240905_u87-adhered_10x_caged_4_class` (Coverage: 91.5%)
        │   │   ├── [NODE] `20240624_mc38_10x_caged_4_class` (Coverage: 96.4%)
        │   │   ├── [NODE] `20240625_mc38_10x_caged_4_class` (Coverage: 96.1%)
        │   │   └── [NODE] `20240624_mc38_10x_uncaged_4_class` (Coverage: 85.8%)
        │   └── [LEAF] `20240509_Hs675Tfibroblasts_10x_caged_4_class` (Coverage: 85.2%)
        └── [NODE] `20260316_a549-tomm20-gfp-adhered_10x_caged_at_4x_4_class` (Coverage: 84.8%)
            ├── [NODE] `20240624_mc38_10x_caged_4_class` (Coverage: 98.8%)
            ├── [NODE] `20240624_mc38_10x_uncaged_4_class` (Coverage: 92.7%)
            └── [NODE] `20250917_moc22-adhered_10x_caged_4_class` (Coverage: 82.8%)
                ├── [NODE] `20240624_mc38_10x_caged_4_class` (Coverage: 92.5%)
                └── [NODE] `20260316_a549-tomm20-gfp-adhered_10x_caged_at_4x_4_class` (Coverage: 86.9%)
```

---

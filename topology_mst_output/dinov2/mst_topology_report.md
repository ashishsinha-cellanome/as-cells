# Optimal Minimum Spanning Tree Topology Report
This report constructs a fully connected Directed Tree (Minimum Spanning Arborescence) for all datasets.
Every node is assigned to its *optimal* parent (the dataset that covers it best), ensuring zero isolates.
The global root of the tree is the dataset that naturally provides the broadest overarching coverage for the entire domain.

## Hierarchical Tree for K=5

```text
[ROOT] 20240924_enteric-glia-adhered_10x_uncaged_4_class
└── [NODE] `20240905_u87-adhered_10x_caged_4_class` (Coverage: 29.8%) ⚠️ WEAK LINK
    ├── [NODE] `20240625_mc38_10x_caged_4_class` (Coverage: 45.4%) ⚠️ WEAK LINK
    │   └── [NODE] `20250227_preadipocytes-adhered_10x_caged_4_class` (Coverage: 31.0%) ⚠️ WEAK LINK
    │       ├── [LEAF] `20241212_preadipocytes-adhered_10x_uncaged_4_class` (Coverage: 36.0%) ⚠️ WEAK LINK
    │       ├── [LEAF] `231212_imr90_multichannel_overlay_4_class` (Coverage: 16.3%) ⚠️ WEAK LINK
    │       └── [LEAF] `20240509_Hs675Tfibroblasts_10x_caged_4_class` (Coverage: 14.7%) ⚠️ WEAK LINK
    ├── [NODE] `20240624_mc38_10x_uncaged_4_class` (Coverage: 33.1%) ⚠️ WEAK LINK
    │   ├── [LEAF] `20240624_mc38_10x_caged_4_class` (Coverage: 64.9%)
    │   ├── [NODE] `20260316_a549-tomm20-gfp-adhered_10x_caged_at_4x_4_class` (Coverage: 42.2%) ⚠️ WEAK LINK
    │   │   └── [LEAF] `20250917_moc22-adhered_10x_caged_4_class` (Coverage: 42.7%) ⚠️ WEAK LINK
    │   └── [NODE] `20240515_DC-adhered_10x_caged_4_class` (Coverage: 24.0%) ⚠️ WEAK LINK
    │       └── [NODE] `20240516_DC-adhered_10x_caged_4_class` (Coverage: 78.0%)
    │           ├── [LEAF] `20250820_c8d1a_astrocytes-adherent_10x_caged_4_class` (Coverage: 46.9%) ⚠️ WEAK LINK
    │           └── [LEAF] `20240509_hela-adhered_10x_caged_4_class` (Coverage: 19.5%) ⚠️ WEAK LINK
    └── [LEAF] `240213_imr90_multichannel_overlay_4_class` (Coverage: 13.5%) ⚠️ WEAK LINK
```

## Hierarchical Tree for K=10

```text
[ROOT] 20240924_enteric-glia-adhered_10x_uncaged_4_class
└── [NODE] `20240625_mc38_10x_caged_4_class` (Coverage: 53.1%)
    ├── [NODE] `20240905_u87-adhered_10x_caged_4_class` (Coverage: 64.9%)
    │   └── [NODE] `20240624_mc38_10x_uncaged_4_class` (Coverage: 54.9%)
    │       ├── [LEAF] `20240624_mc38_10x_caged_4_class` (Coverage: 97.4%)
    │       ├── [NODE] `20260316_a549-tomm20-gfp-adhered_10x_caged_at_4x_4_class` (Coverage: 59.1%)
    │       │   └── [LEAF] `20250917_moc22-adhered_10x_caged_4_class` (Coverage: 59.6%)
    │       └── [NODE] `20240515_DC-adhered_10x_caged_4_class` (Coverage: 39.5%) ⚠️ WEAK LINK
    │           ├── [NODE] `20240516_DC-adhered_10x_caged_4_class` (Coverage: 92.4%)
    │           │   └── [LEAF] `20240509_hela-adhered_10x_caged_4_class` (Coverage: 34.8%) ⚠️ WEAK LINK
    │           └── [LEAF] `20250820_c8d1a_astrocytes-adherent_10x_caged_4_class` (Coverage: 70.8%)
    ├── [NODE] `20250227_preadipocytes-adhered_10x_caged_4_class` (Coverage: 47.3%) ⚠️ WEAK LINK
    │   ├── [LEAF] `20241212_preadipocytes-adhered_10x_uncaged_4_class` (Coverage: 54.8%)
    │   ├── [LEAF] `231212_imr90_multichannel_overlay_4_class` (Coverage: 48.1%) ⚠️ WEAK LINK
    │   └── [LEAF] `20240509_Hs675Tfibroblasts_10x_caged_4_class` (Coverage: 47.3%) ⚠️ WEAK LINK
    └── [LEAF] `240213_imr90_multichannel_overlay_4_class` (Coverage: 42.2%) ⚠️ WEAK LINK
```

## Hierarchical Tree for K=15

```text
[ROOT] 20240924_enteric-glia-adhered_10x_uncaged_4_class
└── [NODE] `20240625_mc38_10x_caged_4_class` (Coverage: 63.6%)
    ├── [NODE] `20240905_u87-adhered_10x_caged_4_class` (Coverage: 75.8%)
    │   └── [NODE] `20240624_mc38_10x_uncaged_4_class` (Coverage: 66.2%)
    │       ├── [LEAF] `20240624_mc38_10x_caged_4_class` (Coverage: 99.7%)
    │       ├── [NODE] `20260316_a549-tomm20-gfp-adhered_10x_caged_at_4x_4_class` (Coverage: 69.2%)
    │       │   └── [LEAF] `20250917_moc22-adhered_10x_caged_4_class` (Coverage: 69.5%)
    │       └── [NODE] `20240515_DC-adhered_10x_caged_4_class` (Coverage: 49.9%) ⚠️ WEAK LINK
    │           ├── [NODE] `20240516_DC-adhered_10x_caged_4_class` (Coverage: 96.6%)
    │           │   └── [LEAF] `20240509_hela-adhered_10x_caged_4_class` (Coverage: 42.8%) ⚠️ WEAK LINK
    │           └── [LEAF] `20250820_c8d1a_astrocytes-adherent_10x_caged_4_class` (Coverage: 81.9%)
    ├── [LEAF] `240213_imr90_multichannel_overlay_4_class` (Coverage: 58.0%)
    └── [NODE] `20250227_preadipocytes-adhered_10x_caged_4_class` (Coverage: 57.8%)
        ├── [LEAF] `20240509_Hs675Tfibroblasts_10x_caged_4_class` (Coverage: 66.5%)
        ├── [LEAF] `20241212_preadipocytes-adhered_10x_uncaged_4_class` (Coverage: 65.2%)
        └── [LEAF] `231212_imr90_multichannel_overlay_4_class` (Coverage: 62.9%)
```

## Hierarchical Tree for K=30

```text
[ROOT] 20240924_enteric-glia-adhered_10x_uncaged_4_class
└── [NODE] `20240625_mc38_10x_caged_4_class` (Coverage: 79.3%)
    ├── [NODE] `20240905_u87-adhered_10x_caged_4_class` (Coverage: 90.6%)
    │   └── [NODE] `20240624_mc38_10x_uncaged_4_class` (Coverage: 81.8%)
    │       ├── [LEAF] `20240624_mc38_10x_caged_4_class` (Coverage: 100.0%)
    │       ├── [NODE] `20260316_a549-tomm20-gfp-adhered_10x_caged_at_4x_4_class` (Coverage: 84.8%)
    │       │   └── [LEAF] `20250917_moc22-adhered_10x_caged_4_class` (Coverage: 83.5%)
    │       ├── [NODE] `20240515_DC-adhered_10x_caged_4_class` (Coverage: 69.9%)
    │       │   ├── [LEAF] `20240516_DC-adhered_10x_caged_4_class` (Coverage: 99.7%)
    │       │   └── [LEAF] `20250820_c8d1a_astrocytes-adherent_10x_caged_4_class` (Coverage: 93.0%)
    │       └── [LEAF] `20240509_hela-adhered_10x_caged_4_class` (Coverage: 63.5%)
    ├── [LEAF] `240213_imr90_multichannel_overlay_4_class` (Coverage: 79.0%)
    └── [NODE] `20250227_preadipocytes-adhered_10x_caged_4_class` (Coverage: 76.2%)
        ├── [LEAF] `20240509_Hs675Tfibroblasts_10x_caged_4_class` (Coverage: 89.7%)
        ├── [LEAF] `231212_imr90_multichannel_overlay_4_class` (Coverage: 82.7%)
        └── [LEAF] `20241212_preadipocytes-adhered_10x_uncaged_4_class` (Coverage: 81.1%)
```

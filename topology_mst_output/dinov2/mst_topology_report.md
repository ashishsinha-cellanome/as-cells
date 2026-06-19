# Optimal Minimum Spanning Tree Topology Report
This report constructs a fully connected Directed Tree (Minimum Spanning Arborescence) for all datasets.
Every node is assigned to its *optimal* parent (the dataset that covers it best), ensuring zero isolates.
The global root of the tree is the dataset that naturally provides the broadest overarching coverage for the entire domain.

## Hierarchical Tree for K=5

```text
[ROOT] 20240924_enteric-glia-adhered_10x_uncaged_4_class
└── [NODE] `20240905_u87-adhered_10x_caged_4_class` (Coverage: 31.4%) ⚠️ WEAK LINK
    ├── [NODE] `20240625_mc38_10x_caged_4_class` (Coverage: 46.0%) ⚠️ WEAK LINK
    │   └── [NODE] `20250227_preadipocytes-adhered_10x_caged_4_class` (Coverage: 30.8%) ⚠️ WEAK LINK
    │       ├── [LEAF] `20241212_preadipocytes-adhered_10x_uncaged_4_class` (Coverage: 38.9%) ⚠️ WEAK LINK
    │       ├── [LEAF] `231212_imr90_multichannel_overlay_4_class` (Coverage: 16.3%) ⚠️ WEAK LINK
    │       └── [LEAF] `20240509_Hs675Tfibroblasts_10x_caged_4_class` (Coverage: 15.3%) ⚠️ WEAK LINK
    ├── [NODE] `20240624_mc38_10x_uncaged_4_class` (Coverage: 36.8%) ⚠️ WEAK LINK
    │   ├── [LEAF] `20240624_mc38_10x_caged_4_class` (Coverage: 63.7%)
    │   ├── [NODE] `20260316_a549-tomm20-gfp-adhered_10x_caged_at_4x_4_class` (Coverage: 42.0%) ⚠️ WEAK LINK
    │   │   └── [LEAF] `20250917_moc22-adhered_10x_caged_4_class` (Coverage: 43.7%) ⚠️ WEAK LINK
    │   └── [NODE] `20240515_DC-adhered_10x_caged_4_class` (Coverage: 25.3%) ⚠️ WEAK LINK
    │       ├── [NODE] `20240516_DC-adhered_10x_caged_4_class` (Coverage: 76.4%)
    │       │   └── [LEAF] `20240509_hela-adhered_10x_caged_4_class` (Coverage: 18.5%) ⚠️ WEAK LINK
    │       └── [LEAF] `20250820_c8d1a_astrocytes-adherent_10x_caged_4_class` (Coverage: 45.6%) ⚠️ WEAK LINK
    └── [LEAF] `240213_imr90_multichannel_overlay_4_class` (Coverage: 12.8%) ⚠️ WEAK LINK
```

## Hierarchical Tree for K=10

```text
[ROOT] 20240924_enteric-glia-adhered_10x_uncaged_4_class
└── [NODE] `20240905_u87-adhered_10x_caged_4_class` (Coverage: 46.8%) ⚠️ WEAK LINK
    ├── [NODE] `20240625_mc38_10x_caged_4_class` (Coverage: 71.1%)
    │   ├── [NODE] `20250227_preadipocytes-adhered_10x_caged_4_class` (Coverage: 45.3%) ⚠️ WEAK LINK
    │   │   ├── [LEAF] `20241212_preadipocytes-adhered_10x_uncaged_4_class` (Coverage: 58.5%)
    │   │   ├── [LEAF] `231212_imr90_multichannel_overlay_4_class` (Coverage: 49.4%) ⚠️ WEAK LINK
    │   │   └── [LEAF] `20240509_Hs675Tfibroblasts_10x_caged_4_class` (Coverage: 45.9%) ⚠️ WEAK LINK
    │   └── [LEAF] `240213_imr90_multichannel_overlay_4_class` (Coverage: 40.7%) ⚠️ WEAK LINK
    └── [NODE] `20240624_mc38_10x_uncaged_4_class` (Coverage: 60.3%)
        ├── [LEAF] `20240624_mc38_10x_caged_4_class` (Coverage: 96.2%)
        ├── [NODE] `20260316_a549-tomm20-gfp-adhered_10x_caged_at_4x_4_class` (Coverage: 59.8%)
        │   └── [LEAF] `20250917_moc22-adhered_10x_caged_4_class` (Coverage: 60.5%)
        └── [NODE] `20240515_DC-adhered_10x_caged_4_class` (Coverage: 39.4%) ⚠️ WEAK LINK
            ├── [NODE] `20240516_DC-adhered_10x_caged_4_class` (Coverage: 92.3%)
            │   └── [LEAF] `20240509_hela-adhered_10x_caged_4_class` (Coverage: 34.0%) ⚠️ WEAK LINK
            └── [LEAF] `20250820_c8d1a_astrocytes-adherent_10x_caged_4_class` (Coverage: 69.7%)
```

## Hierarchical Tree for K=15

```text
[ROOT] 20240924_enteric-glia-adhered_10x_uncaged_4_class
└── [NODE] `20240905_u87-adhered_10x_caged_4_class` (Coverage: 56.2%)
    ├── [NODE] `20240625_mc38_10x_caged_4_class` (Coverage: 83.5%)
    │   └── [NODE] `20250227_preadipocytes-adhered_10x_caged_4_class` (Coverage: 55.5%)
    │       ├── [LEAF] `20240509_Hs675Tfibroblasts_10x_caged_4_class` (Coverage: 69.1%)
    │       ├── [LEAF] `20241212_preadipocytes-adhered_10x_uncaged_4_class` (Coverage: 68.2%)
    │       └── [LEAF] `231212_imr90_multichannel_overlay_4_class` (Coverage: 63.8%)
    ├── [NODE] `20240624_mc38_10x_uncaged_4_class` (Coverage: 72.5%)
    │   ├── [LEAF] `20240624_mc38_10x_caged_4_class` (Coverage: 99.7%)
    │   ├── [NODE] `20260316_a549-tomm20-gfp-adhered_10x_caged_at_4x_4_class` (Coverage: 69.3%)
    │   │   └── [LEAF] `20250917_moc22-adhered_10x_caged_4_class` (Coverage: 69.8%)
    │   └── [NODE] `20240515_DC-adhered_10x_caged_4_class` (Coverage: 49.7%) ⚠️ WEAK LINK
    │       ├── [NODE] `20240516_DC-adhered_10x_caged_4_class` (Coverage: 97.0%)
    │       │   └── [LEAF] `20240509_hela-adhered_10x_caged_4_class` (Coverage: 42.5%) ⚠️ WEAK LINK
    │       └── [LEAF] `20250820_c8d1a_astrocytes-adherent_10x_caged_4_class` (Coverage: 80.5%)
    └── [LEAF] `240213_imr90_multichannel_overlay_4_class` (Coverage: 55.5%)
```

## Hierarchical Tree for K=30

```text
[ROOT] 20240924_enteric-glia-adhered_10x_uncaged_4_class
└── [NODE] `20240625_mc38_10x_caged_4_class` (Coverage: 77.5%)
    ├── [NODE] `20240905_u87-adhered_10x_caged_4_class` (Coverage: 89.8%)
    │   └── [NODE] `20240624_mc38_10x_uncaged_4_class` (Coverage: 88.0%)
    │       ├── [LEAF] `20240624_mc38_10x_caged_4_class` (Coverage: 100.0%)
    │       ├── [NODE] `20260316_a549-tomm20-gfp-adhered_10x_caged_at_4x_4_class` (Coverage: 84.9%)
    │       │   └── [LEAF] `20250917_moc22-adhered_10x_caged_4_class` (Coverage: 82.7%)
    │       ├── [NODE] `20240515_DC-adhered_10x_caged_4_class` (Coverage: 68.1%)
    │       │   ├── [LEAF] `20240516_DC-adhered_10x_caged_4_class` (Coverage: 99.7%)
    │       │   └── [LEAF] `20250820_c8d1a_astrocytes-adherent_10x_caged_4_class` (Coverage: 92.9%)
    │       └── [LEAF] `20240509_hela-adhered_10x_caged_4_class` (Coverage: 61.9%)
    ├── [LEAF] `240213_imr90_multichannel_overlay_4_class` (Coverage: 78.5%)
    └── [NODE] `20250227_preadipocytes-adhered_10x_caged_4_class` (Coverage: 73.5%)
        ├── [LEAF] `20240509_Hs675Tfibroblasts_10x_caged_4_class` (Coverage: 91.2%)
        ├── [LEAF] `20241212_preadipocytes-adhered_10x_uncaged_4_class` (Coverage: 84.2%)
        └── [LEAF] `231212_imr90_multichannel_overlay_4_class` (Coverage: 82.8%)
```

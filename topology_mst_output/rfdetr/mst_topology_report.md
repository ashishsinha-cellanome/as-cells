# Optimal Minimum Spanning Tree Topology Report
This report constructs a fully connected Directed Tree (Minimum Spanning Arborescence) for all datasets.
Every node is assigned to its *optimal* parent (the dataset that covers it best), ensuring zero isolates.
The global root of the tree is the dataset that naturally provides the broadest overarching coverage for the entire domain.

## Hierarchical Tree for K=5

```text
[ROOT] 20250917_moc22-adhered_10x_caged_4_class
└── [NODE] `20260316_a549-tomm20-gfp-adhered_10x_caged_at_4x_4_class` (Coverage: 70.4%)
    └── [NODE] `20240624_mc38_10x_uncaged_4_class` (Coverage: 52.6%)
        ├── [LEAF] `20240624_mc38_10x_caged_4_class` (Coverage: 82.4%)
        └── [NODE] `20240905_u87-adhered_10x_caged_4_class` (Coverage: 61.2%)
            └── [NODE] `20240625_mc38_10x_caged_4_class` (Coverage: 67.7%)
                ├── [NODE] `20250227_preadipocytes-adhered_10x_caged_4_class` (Coverage: 55.7%)
                │   ├── [LEAF] `20241212_preadipocytes-adhered_10x_uncaged_4_class` (Coverage: 60.7%)
                │   ├── [LEAF] `20240509_Hs675Tfibroblasts_10x_caged_4_class` (Coverage: 45.3%) ⚠️ WEAK LINK
                │   └── [NODE] `231212_imr90_multichannel_overlay_4_class` (Coverage: 32.5%) ⚠️ WEAK LINK
                │       └── [LEAF] `240213_imr90_multichannel_overlay_4_class` (Coverage: 27.7%) ⚠️ WEAK LINK
                ├── [LEAF] `20240924_enteric-glia-adhered_10x_uncaged_4_class` (Coverage: 48.3%) ⚠️ WEAK LINK
                └── [NODE] `20240509_hela-adhered_10x_caged_4_class` (Coverage: 38.5%) ⚠️ WEAK LINK
                    └── [NODE] `20240515_DC-adhered_10x_caged_4_class` (Coverage: 32.3%) ⚠️ WEAK LINK
                        ├── [LEAF] `20240516_DC-adhered_10x_caged_4_class` (Coverage: 90.5%)
                        └── [LEAF] `20250820_c8d1a_astrocytes-adherent_10x_caged_4_class` (Coverage: 55.6%)
```

## Hierarchical Tree for K=10

```text
[ROOT] 20250917_moc22-adhered_10x_caged_4_class
└── [NODE] `20260316_a549-tomm20-gfp-adhered_10x_caged_at_4x_4_class` (Coverage: 90.8%)
    └── [NODE] `20240624_mc38_10x_uncaged_4_class` (Coverage: 76.6%)
        ├── [LEAF] `20240624_mc38_10x_caged_4_class` (Coverage: 99.7%)
        └── [NODE] `20240905_u87-adhered_10x_caged_4_class` (Coverage: 80.2%)
            └── [NODE] `20240625_mc38_10x_caged_4_class` (Coverage: 89.3%)
                ├── [NODE] `20250227_preadipocytes-adhered_10x_caged_4_class` (Coverage: 76.7%)
                │   ├── [LEAF] `20240509_Hs675Tfibroblasts_10x_caged_4_class` (Coverage: 78.9%)
                │   ├── [LEAF] `20241212_preadipocytes-adhered_10x_uncaged_4_class` (Coverage: 77.9%)
                │   └── [NODE] `231212_imr90_multichannel_overlay_4_class` (Coverage: 67.5%)
                │       └── [LEAF] `240213_imr90_multichannel_overlay_4_class` (Coverage: 52.6%)
                ├── [LEAF] `20240924_enteric-glia-adhered_10x_uncaged_4_class` (Coverage: 71.2%)
                └── [NODE] `20240509_hela-adhered_10x_caged_4_class` (Coverage: 66.9%)
                    └── [NODE] `20240516_DC-adhered_10x_caged_4_class` (Coverage: 49.6%) ⚠️ WEAK LINK
                        └── [NODE] `20240515_DC-adhered_10x_caged_4_class` (Coverage: 98.7%)
                            └── [LEAF] `20250820_c8d1a_astrocytes-adherent_10x_caged_4_class` (Coverage: 78.8%)
```

## Hierarchical Tree for K=15

```text
[ROOT] 20250917_moc22-adhered_10x_caged_4_class
└── [NODE] `20260316_a549-tomm20-gfp-adhered_10x_caged_at_4x_4_class` (Coverage: 96.2%)
    ├── [NODE] `20240624_mc38_10x_uncaged_4_class` (Coverage: 86.6%)
    │   ├── [LEAF] `20240624_mc38_10x_caged_4_class` (Coverage: 100.0%)
    │   └── [NODE] `20240905_u87-adhered_10x_caged_4_class` (Coverage: 90.2%)
    │       ├── [NODE] `20240625_mc38_10x_caged_4_class` (Coverage: 96.2%)
    │       │   ├── [NODE] `20250227_preadipocytes-adhered_10x_caged_4_class` (Coverage: 86.9%)
    │       │   │   ├── [LEAF] `20240509_Hs675Tfibroblasts_10x_caged_4_class` (Coverage: 93.5%)
    │       │   │   ├── [LEAF] `20241212_preadipocytes-adhered_10x_uncaged_4_class` (Coverage: 85.3%)
    │       │   │   └── [LEAF] `231212_imr90_multichannel_overlay_4_class` (Coverage: 79.7%)
    │       │   ├── [LEAF] `20240509_hela-adhered_10x_caged_4_class` (Coverage: 82.1%)
    │       │   └── [LEAF] `20240924_enteric-glia-adhered_10x_uncaged_4_class` (Coverage: 81.6%)
    │       └── [LEAF] `240213_imr90_multichannel_overlay_4_class` (Coverage: 67.0%)
    └── [NODE] `20240516_DC-adhered_10x_caged_4_class` (Coverage: 61.6%)
        └── [NODE] `20240515_DC-adhered_10x_caged_4_class` (Coverage: 99.8%)
            └── [LEAF] `20250820_c8d1a_astrocytes-adherent_10x_caged_4_class` (Coverage: 87.9%)
```

## Hierarchical Tree for K=30

```text
[ROOT] 20240515_DC-adhered_10x_caged_4_class
├── [LEAF] `20240516_DC-adhered_10x_caged_4_class` (Coverage: 100.0%)
├── [NODE] `20240624_mc38_10x_caged_4_class` (Coverage: 98.8%)
│   └── [NODE] `20240624_mc38_10x_uncaged_4_class` (Coverage: 95.5%)
│       ├── [NODE] `20240905_u87-adhered_10x_caged_4_class` (Coverage: 97.8%)
│       │   ├── [NODE] `20240625_mc38_10x_caged_4_class` (Coverage: 100.0%)
│       │   │   ├── [NODE] `20250227_preadipocytes-adhered_10x_caged_4_class` (Coverage: 97.4%)
│       │   │   │   ├── [LEAF] `20240509_Hs675Tfibroblasts_10x_caged_4_class` (Coverage: 99.8%)
│       │   │   │   ├── [LEAF] `231212_imr90_multichannel_overlay_4_class` (Coverage: 94.3%)
│       │   │   │   └── [LEAF] `20241212_preadipocytes-adhered_10x_uncaged_4_class` (Coverage: 93.2%)
│       │   │   ├── [LEAF] `20240509_hela-adhered_10x_caged_4_class` (Coverage: 96.6%)
│       │   │   └── [LEAF] `20240924_enteric-glia-adhered_10x_uncaged_4_class` (Coverage: 95.3%)
│       │   └── [LEAF] `240213_imr90_multichannel_overlay_4_class` (Coverage: 87.5%)
│       └── [NODE] `20260316_a549-tomm20-gfp-adhered_10x_caged_at_4x_4_class` (Coverage: 92.0%)
│           └── [LEAF] `20250917_moc22-adhered_10x_caged_4_class` (Coverage: 94.5%)
└── [LEAF] `20250820_c8d1a_astrocytes-adherent_10x_caged_4_class` (Coverage: 97.0%)
```

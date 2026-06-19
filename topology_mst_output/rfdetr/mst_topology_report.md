# Optimal Minimum Spanning Tree Topology Report
This report constructs a fully connected Directed Tree (Minimum Spanning Arborescence) for all datasets.
Every node is assigned to its *optimal* parent (the dataset that covers it best), ensuring zero isolates.
The global root of the tree is the dataset that naturally provides the broadest overarching coverage for the entire domain.

## Hierarchical Tree for K=5

```text
[ROOT] 20250917_moc22-adhered_10x_caged_4_class
└── [NODE] `20260316_a549-tomm20-gfp-adhered_10x_caged_at_4x_4_class` (Coverage: 70.7%)
    └── [NODE] `20240624_mc38_10x_uncaged_4_class` (Coverage: 50.4%)
        ├── [LEAF] `20240624_mc38_10x_caged_4_class` (Coverage: 83.4%)
        └── [NODE] `20240905_u87-adhered_10x_caged_4_class` (Coverage: 61.8%)
            └── [NODE] `20240625_mc38_10x_caged_4_class` (Coverage: 68.9%)
                ├── [NODE] `20250227_preadipocytes-adhered_10x_caged_4_class` (Coverage: 53.3%)
                │   ├── [LEAF] `20241212_preadipocytes-adhered_10x_uncaged_4_class` (Coverage: 59.2%)
                │   ├── [LEAF] `20240509_Hs675Tfibroblasts_10x_caged_4_class` (Coverage: 45.3%) ⚠️ WEAK LINK
                │   └── [NODE] `231212_imr90_multichannel_overlay_4_class` (Coverage: 33.8%) ⚠️ WEAK LINK
                │       └── [LEAF] `240213_imr90_multichannel_overlay_4_class` (Coverage: 27.7%) ⚠️ WEAK LINK
                ├── [LEAF] `20240924_enteric-glia-adhered_10x_uncaged_4_class` (Coverage: 50.7%)
                └── [NODE] `20240509_hela-adhered_10x_caged_4_class` (Coverage: 37.4%) ⚠️ WEAK LINK
                    └── [NODE] `20240516_DC-adhered_10x_caged_4_class` (Coverage: 32.2%) ⚠️ WEAK LINK
                        └── [NODE] `20240515_DC-adhered_10x_caged_4_class` (Coverage: 91.5%)
                            └── [LEAF] `20250820_c8d1a_astrocytes-adherent_10x_caged_4_class` (Coverage: 56.1%)
```

## Hierarchical Tree for K=10

```text
[ROOT] 20250917_moc22-adhered_10x_caged_4_class
└── [NODE] `20260316_a549-tomm20-gfp-adhered_10x_caged_at_4x_4_class` (Coverage: 90.1%)
    └── [NODE] `20240624_mc38_10x_uncaged_4_class` (Coverage: 75.8%)
        ├── [LEAF] `20240624_mc38_10x_caged_4_class` (Coverage: 99.6%)
        └── [NODE] `20240905_u87-adhered_10x_caged_4_class` (Coverage: 81.9%)
            └── [NODE] `20240625_mc38_10x_caged_4_class` (Coverage: 90.9%)
                ├── [NODE] `20250227_preadipocytes-adhered_10x_caged_4_class` (Coverage: 73.6%)
                │   ├── [LEAF] `20240509_Hs675Tfibroblasts_10x_caged_4_class` (Coverage: 80.1%)
                │   ├── [LEAF] `20241212_preadipocytes-adhered_10x_uncaged_4_class` (Coverage: 77.2%)
                │   └── [NODE] `231212_imr90_multichannel_overlay_4_class` (Coverage: 67.1%)
                │       └── [LEAF] `240213_imr90_multichannel_overlay_4_class` (Coverage: 52.6%)
                ├── [LEAF] `20240924_enteric-glia-adhered_10x_uncaged_4_class` (Coverage: 72.9%)
                └── [NODE] `20240509_hela-adhered_10x_caged_4_class` (Coverage: 65.9%)
                    └── [NODE] `20240516_DC-adhered_10x_caged_4_class` (Coverage: 49.8%) ⚠️ WEAK LINK
                        └── [NODE] `20240515_DC-adhered_10x_caged_4_class` (Coverage: 99.0%)
                            └── [LEAF] `20250820_c8d1a_astrocytes-adherent_10x_caged_4_class` (Coverage: 79.7%)
```

## Hierarchical Tree for K=15

```text
[ROOT] 20250917_moc22-adhered_10x_caged_4_class
└── [NODE] `20260316_a549-tomm20-gfp-adhered_10x_caged_at_4x_4_class` (Coverage: 95.7%)
    └── [NODE] `20240624_mc38_10x_uncaged_4_class` (Coverage: 86.5%)
        ├── [LEAF] `20240624_mc38_10x_caged_4_class` (Coverage: 100.0%)
        └── [NODE] `20240905_u87-adhered_10x_caged_4_class` (Coverage: 90.0%)
            ├── [NODE] `20240625_mc38_10x_caged_4_class` (Coverage: 97.1%)
            │   ├── [LEAF] `20240924_enteric-glia-adhered_10x_uncaged_4_class` (Coverage: 83.9%)
            │   ├── [NODE] `20250227_preadipocytes-adhered_10x_caged_4_class` (Coverage: 83.1%)
            │   │   ├── [LEAF] `20240509_Hs675Tfibroblasts_10x_caged_4_class` (Coverage: 94.4%)
            │   │   ├── [LEAF] `20241212_preadipocytes-adhered_10x_uncaged_4_class` (Coverage: 84.3%)
            │   │   └── [LEAF] `231212_imr90_multichannel_overlay_4_class` (Coverage: 79.6%)
            │   └── [NODE] `20240509_hela-adhered_10x_caged_4_class` (Coverage: 80.9%)
            │       └── [NODE] `20240516_DC-adhered_10x_caged_4_class` (Coverage: 62.2%)
            │           └── [NODE] `20240515_DC-adhered_10x_caged_4_class` (Coverage: 99.7%)
            │               └── [LEAF] `20250820_c8d1a_astrocytes-adherent_10x_caged_4_class` (Coverage: 88.8%)
            └── [LEAF] `240213_imr90_multichannel_overlay_4_class` (Coverage: 67.3%)
```

## Hierarchical Tree for K=30

```text
[ROOT] 20240515_DC-adhered_10x_caged_4_class
├── [LEAF] `20240516_DC-adhered_10x_caged_4_class` (Coverage: 100.0%)
├── [NODE] `20240624_mc38_10x_caged_4_class` (Coverage: 98.2%)
│   └── [NODE] `20240624_mc38_10x_uncaged_4_class` (Coverage: 95.1%)
│       ├── [NODE] `20240905_u87-adhered_10x_caged_4_class` (Coverage: 97.4%)
│       │   ├── [NODE] `20240625_mc38_10x_caged_4_class` (Coverage: 100.0%)
│       │   │   ├── [LEAF] `20240924_enteric-glia-adhered_10x_uncaged_4_class` (Coverage: 96.4%)
│       │   │   ├── [LEAF] `20240509_hela-adhered_10x_caged_4_class` (Coverage: 96.1%)
│       │   │   └── [NODE] `20240509_Hs675Tfibroblasts_10x_caged_4_class` (Coverage: 95.8%)
│       │   │       └── [NODE] `20250227_preadipocytes-adhered_10x_caged_4_class` (Coverage: 99.5%)
│       │   │           ├── [LEAF] `231212_imr90_multichannel_overlay_4_class` (Coverage: 95.2%)
│       │   │           └── [LEAF] `20241212_preadipocytes-adhered_10x_uncaged_4_class` (Coverage: 91.8%)
│       │   └── [LEAF] `240213_imr90_multichannel_overlay_4_class` (Coverage: 89.1%)
│       └── [NODE] `20260316_a549-tomm20-gfp-adhered_10x_caged_at_4x_4_class` (Coverage: 92.4%)
│           └── [LEAF] `20250917_moc22-adhered_10x_caged_4_class` (Coverage: 93.9%)
└── [LEAF] `20250820_c8d1a_astrocytes-adherent_10x_caged_4_class` (Coverage: 98.0%)
```

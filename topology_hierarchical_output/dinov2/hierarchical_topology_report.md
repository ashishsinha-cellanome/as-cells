# Hierarchical Clustering Topology Report

This report constructs a tree based on standard Agglomerative Hierarchical Clustering (what seaborn.clustermap and scipy dendrograms use).
Because linkage requires symmetric distances, we use the average coverage between the two directions: `(Coverage(X->Y) + Coverage(Y->X)) / 2`.
Each internal node shows the average mutual coverage between its two merged sub-clusters.

## Hierarchical Tree for K=5

```text
[GLOBAL ROOT] (Merged at 6.2% mutual coverage)
├── [CLUSTER] (Merged at 11.2% mutual coverage)
│   ├── [CLUSTER] (Merged at 14.4% mutual coverage)
│   │   ├── [CLUSTER] (Merged at 18.7% mutual coverage)
│   │   │   ├── [CLUSTER] (Merged at 21.2% mutual coverage)
│   │   │   │   ├── [CLUSTER] (Merged at 24.4% mutual coverage)
│   │   │   │   │   ├── [CLUSTER] (Merged at 31.5% mutual coverage)
│   │   │   │   │   │   ├── [CLUSTER] (Merged at 44.2% mutual coverage)
│   │   │   │   │   │   │   ├── [LEAF] `20240624_mc38_10x_caged_4_class`
│   │   │   │   │   │   │   └── [LEAF] `20240624_mc38_10x_uncaged_4_class`
│   │   │   │   │   │   └── [CLUSTER] (Merged at 43.3% mutual coverage)
│   │   │   │   │   │       ├── [LEAF] `20240625_mc38_10x_caged_4_class`
│   │   │   │   │   │       └── [LEAF] `20240905_u87-adhered_10x_caged_4_class`
│   │   │   │   │   └── [CLUSTER] (Merged at 39.5% mutual coverage)
│   │   │   │   │       ├── [LEAF] `20250917_moc22-adhered_10x_caged_4_class`
│   │   │   │   │       └── [LEAF] `20260316_a549-tomm20-gfp-adhered_10x_caged_at_4x_4_class`
│   │   │   │   └── [CLUSTER] (Merged at 39.3% mutual coverage)
│   │   │   │       ├── [CLUSTER] (Merged at 79.0% mutual coverage)
│   │   │   │       │   ├── [LEAF] `20240515_DC-adhered_10x_caged_4_class`
│   │   │   │       │   └── [LEAF] `20240516_DC-adhered_10x_caged_4_class`
│   │   │   │       └── [LEAF] `20250820_c8d1a_astrocytes-adherent_10x_caged_4_class`
│   │   │   └── [LEAF] `20240509_hela-adhered_10x_caged_4_class`
│   │   └── [LEAF] `20240924_enteric-glia-adhered_10x_uncaged_4_class`
│   └── [CLUSTER] (Merged at 15.7% mutual coverage)
│       ├── [CLUSTER] (Merged at 41.3% mutual coverage)
│       │   ├── [LEAF] `20241212_preadipocytes-adhered_10x_uncaged_4_class`
│       │   └── [LEAF] `20250227_preadipocytes-adhered_10x_caged_4_class`
│       └── [LEAF] `20240509_Hs675Tfibroblasts_10x_caged_4_class`
└── [CLUSTER] (Merged at 8.7% mutual coverage)
    ├── [LEAF] `231212_imr90_multichannel_overlay_4_class`
    └── [LEAF] `240213_imr90_multichannel_overlay_4_class`
```

## Hierarchical Tree for K=10

```text
[GLOBAL ROOT] (Merged at 18.4% mutual coverage)
├── [CLUSTER] (Merged at 24.0% mutual coverage)
│   ├── [CLUSTER] (Merged at 32.2% mutual coverage)
│   │   ├── [CLUSTER] (Merged at 34.4% mutual coverage)
│   │   │   ├── [CLUSTER] (Merged at 37.0% mutual coverage)
│   │   │   │   ├── [CLUSTER] (Merged at 48.0% mutual coverage)
│   │   │   │   │   ├── [CLUSTER] (Merged at 59.2% mutual coverage)
│   │   │   │   │   │   ├── [CLUSTER] (Merged at 70.6% mutual coverage)
│   │   │   │   │   │   │   ├── [LEAF] `20240624_mc38_10x_caged_4_class`
│   │   │   │   │   │   │   └── [LEAF] `20240624_mc38_10x_uncaged_4_class`
│   │   │   │   │   │   └── [LEAF] `20260316_a549-tomm20-gfp-adhered_10x_caged_at_4x_4_class`
│   │   │   │   │   └── [CLUSTER] (Merged at 66.6% mutual coverage)
│   │   │   │   │       ├── [LEAF] `20240625_mc38_10x_caged_4_class`
│   │   │   │   │       └── [LEAF] `20240905_u87-adhered_10x_caged_4_class`
│   │   │   │   └── [CLUSTER] (Merged at 57.1% mutual coverage)
│   │   │   │       ├── [CLUSTER] (Merged at 93.2% mutual coverage)
│   │   │   │       │   ├── [LEAF] `20240515_DC-adhered_10x_caged_4_class`
│   │   │   │       │   └── [LEAF] `20240516_DC-adhered_10x_caged_4_class`
│   │   │   │       └── [LEAF] `20250820_c8d1a_astrocytes-adherent_10x_caged_4_class`
│   │   │   └── [LEAF] `20240509_hela-adhered_10x_caged_4_class`
│   │   └── [LEAF] `20250917_moc22-adhered_10x_caged_4_class`
│   └── [LEAF] `20240924_enteric-glia-adhered_10x_uncaged_4_class`
└── [CLUSTER] (Merged at 22.7% mutual coverage)
    ├── [CLUSTER] (Merged at 33.9% mutual coverage)
    │   ├── [CLUSTER] (Merged at 60.9% mutual coverage)
    │   │   ├── [LEAF] `20241212_preadipocytes-adhered_10x_uncaged_4_class`
    │   │   └── [LEAF] `20250227_preadipocytes-adhered_10x_caged_4_class`
    │   └── [LEAF] `20240509_Hs675Tfibroblasts_10x_caged_4_class`
    └── [CLUSTER] (Merged at 30.2% mutual coverage)
        ├── [LEAF] `231212_imr90_multichannel_overlay_4_class`
        └── [LEAF] `240213_imr90_multichannel_overlay_4_class`
```

## Hierarchical Tree for K=15

```text
[GLOBAL ROOT] (Merged at 24.8% mutual coverage)
├── [CLUSTER] (Merged at 29.9% mutual coverage)
│   ├── [CLUSTER] (Merged at 39.7% mutual coverage)
│   │   ├── [CLUSTER] (Merged at 43.2% mutual coverage)
│   │   │   ├── [CLUSTER] (Merged at 45.7% mutual coverage)
│   │   │   │   ├── [CLUSTER] (Merged at 57.4% mutual coverage)
│   │   │   │   │   ├── [CLUSTER] (Merged at 67.9% mutual coverage)
│   │   │   │   │   │   ├── [CLUSTER] (Merged at 79.7% mutual coverage)
│   │   │   │   │   │   │   ├── [LEAF] `20240624_mc38_10x_caged_4_class`
│   │   │   │   │   │   │   └── [LEAF] `20240624_mc38_10x_uncaged_4_class`
│   │   │   │   │   │   └── [LEAF] `20260316_a549-tomm20-gfp-adhered_10x_caged_at_4x_4_class`
│   │   │   │   │   └── [CLUSTER] (Merged at 78.2% mutual coverage)
│   │   │   │   │       ├── [LEAF] `20240625_mc38_10x_caged_4_class`
│   │   │   │   │       └── [LEAF] `20240905_u87-adhered_10x_caged_4_class`
│   │   │   │   └── [CLUSTER] (Merged at 67.4% mutual coverage)
│   │   │   │       ├── [CLUSTER] (Merged at 97.2% mutual coverage)
│   │   │   │       │   ├── [LEAF] `20240515_DC-adhered_10x_caged_4_class`
│   │   │   │       │   └── [LEAF] `20240516_DC-adhered_10x_caged_4_class`
│   │   │   │       └── [LEAF] `20250820_c8d1a_astrocytes-adherent_10x_caged_4_class`
│   │   │   └── [LEAF] `20240509_hela-adhered_10x_caged_4_class`
│   │   └── [LEAF] `20250917_moc22-adhered_10x_caged_4_class`
│   └── [LEAF] `20240924_enteric-glia-adhered_10x_uncaged_4_class`
└── [CLUSTER] (Merged at 32.4% mutual coverage)
    ├── [CLUSTER] (Merged at 47.6% mutual coverage)
    │   ├── [CLUSTER] (Merged at 72.3% mutual coverage)
    │   │   ├── [LEAF] `20241212_preadipocytes-adhered_10x_uncaged_4_class`
    │   │   └── [LEAF] `20250227_preadipocytes-adhered_10x_caged_4_class`
    │   └── [LEAF] `20240509_Hs675Tfibroblasts_10x_caged_4_class`
    └── [CLUSTER] (Merged at 45.5% mutual coverage)
        ├── [LEAF] `231212_imr90_multichannel_overlay_4_class`
        └── [LEAF] `240213_imr90_multichannel_overlay_4_class`
```

## Hierarchical Tree for K=30

```text
[GLOBAL ROOT] (Merged at 37.1% mutual coverage)
├── [CLUSTER] (Merged at 41.7% mutual coverage)
│   ├── [CLUSTER] (Merged at 57.2% mutual coverage)
│   │   ├── [CLUSTER] (Merged at 58.6% mutual coverage)
│   │   │   ├── [CLUSTER] (Merged at 62.3% mutual coverage)
│   │   │   │   ├── [CLUSTER] (Merged at 76.6% mutual coverage)
│   │   │   │   │   ├── [CLUSTER] (Merged at 92.8% mutual coverage)
│   │   │   │   │   │   ├── [LEAF] `20240625_mc38_10x_caged_4_class`
│   │   │   │   │   │   └── [LEAF] `20240905_u87-adhered_10x_caged_4_class`
│   │   │   │   │   └── [CLUSTER] (Merged at 92.5% mutual coverage)
│   │   │   │   │       ├── [LEAF] `20240624_mc38_10x_caged_4_class`
│   │   │   │   │       └── [LEAF] `20240624_mc38_10x_uncaged_4_class`
│   │   │   │   └── [CLUSTER] (Merged at 81.7% mutual coverage)
│   │   │   │       ├── [LEAF] `20250917_moc22-adhered_10x_caged_4_class`
│   │   │   │       └── [LEAF] `20260316_a549-tomm20-gfp-adhered_10x_caged_at_4x_4_class`
│   │   │   └── [CLUSTER] (Merged at 81.3% mutual coverage)
│   │   │       ├── [CLUSTER] (Merged at 99.7% mutual coverage)
│   │   │       │   ├── [LEAF] `20240515_DC-adhered_10x_caged_4_class`
│   │   │       │   └── [LEAF] `20240516_DC-adhered_10x_caged_4_class`
│   │   │       └── [LEAF] `20250820_c8d1a_astrocytes-adherent_10x_caged_4_class`
│   │   └── [LEAF] `20240509_hela-adhered_10x_caged_4_class`
│   └── [LEAF] `20240924_enteric-glia-adhered_10x_uncaged_4_class`
└── [CLUSTER] (Merged at 49.9% mutual coverage)
    ├── [CLUSTER] (Merged at 65.9% mutual coverage)
    │   ├── [CLUSTER] (Merged at 87.1% mutual coverage)
    │   │   ├── [LEAF] `20241212_preadipocytes-adhered_10x_uncaged_4_class`
    │   │   └── [LEAF] `20250227_preadipocytes-adhered_10x_caged_4_class`
    │   └── [LEAF] `20240509_Hs675Tfibroblasts_10x_caged_4_class`
    └── [CLUSTER] (Merged at 65.9% mutual coverage)
        ├── [LEAF] `231212_imr90_multichannel_overlay_4_class`
        └── [LEAF] `240213_imr90_multichannel_overlay_4_class`
```

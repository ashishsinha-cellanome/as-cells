# Hierarchical Clustering Topology Report

This report constructs a tree based on standard Agglomerative Hierarchical Clustering (what seaborn.clustermap and scipy dendrograms use).
Because linkage requires symmetric distances, we use the average coverage between the two directions: `(Coverage(X->Y) + Coverage(Y->X)) / 2`.
Each internal node shows the average mutual coverage between its two merged sub-clusters.

## Hierarchical Tree for K=5

```text
[GLOBAL ROOT] (Merged at 17.3% mutual coverage)
├── [CLUSTER] (Merged at 28.2% mutual coverage)
│   ├── [CLUSTER] (Merged at 31.1% mutual coverage)
│   │   ├── [CLUSTER] (Merged at 37.6% mutual coverage)
│   │   │   ├── [CLUSTER] (Merged at 42.7% mutual coverage)
│   │   │   │   ├── [CLUSTER] (Merged at 51.7% mutual coverage)
│   │   │   │   │   ├── [CLUSTER] (Merged at 64.0% mutual coverage)
│   │   │   │   │   │   ├── [LEAF] `20240625_mc38_10x_caged_4_class`
│   │   │   │   │   │   └── [LEAF] `20240905_u87-adhered_10x_caged_4_class`
│   │   │   │   │   └── [LEAF] `20240924_enteric-glia-adhered_10x_uncaged_4_class`
│   │   │   │   └── [CLUSTER] (Merged at 59.5% mutual coverage)
│   │   │   │       ├── [LEAF] `20240624_mc38_10x_caged_4_class`
│   │   │   │       └── [LEAF] `20240624_mc38_10x_uncaged_4_class`
│   │   │   └── [LEAF] `20240509_hela-adhered_10x_caged_4_class`
│   │   └── [CLUSTER] (Merged at 67.4% mutual coverage)
│   │       ├── [LEAF] `20250917_moc22-adhered_10x_caged_4_class`
│   │       └── [LEAF] `20260316_a549-tomm20-gfp-adhered_10x_caged_at_4x_4_class`
│   └── [CLUSTER] (Merged at 46.1% mutual coverage)
│       ├── [CLUSTER] (Merged at 90.8% mutual coverage)
│       │   ├── [LEAF] `20240515_DC-adhered_10x_caged_4_class`
│       │   └── [LEAF] `20240516_DC-adhered_10x_caged_4_class`
│       └── [LEAF] `20250820_c8d1a_astrocytes-adherent_10x_caged_4_class`
└── [CLUSTER] (Merged at 20.8% mutual coverage)
    ├── [CLUSTER] (Merged at 25.1% mutual coverage)
    │   ├── [CLUSTER] (Merged at 46.5% mutual coverage)
    │   │   ├── [CLUSTER] (Merged at 62.4% mutual coverage)
    │   │   │   ├── [LEAF] `20241212_preadipocytes-adhered_10x_uncaged_4_class`
    │   │   │   └── [LEAF] `20250227_preadipocytes-adhered_10x_caged_4_class`
    │   │   └── [LEAF] `20240509_Hs675Tfibroblasts_10x_caged_4_class`
    │   └── [LEAF] `231212_imr90_multichannel_overlay_4_class`
    └── [LEAF] `240213_imr90_multichannel_overlay_4_class`
```

## Hierarchical Tree for K=10

```text
[GLOBAL ROOT] (Merged at 29.3% mutual coverage)
├── [CLUSTER] (Merged at 43.6% mutual coverage)
│   ├── [CLUSTER] (Merged at 46.8% mutual coverage)
│   │   ├── [CLUSTER] (Merged at 58.5% mutual coverage)
│   │   │   ├── [CLUSTER] (Merged at 61.3% mutual coverage)
│   │   │   │   ├── [CLUSTER] (Merged at 72.9% mutual coverage)
│   │   │   │   │   ├── [CLUSTER] (Merged at 85.6% mutual coverage)
│   │   │   │   │   │   ├── [LEAF] `20240625_mc38_10x_caged_4_class`
│   │   │   │   │   │   └── [LEAF] `20240905_u87-adhered_10x_caged_4_class`
│   │   │   │   │   └── [LEAF] `20240924_enteric-glia-adhered_10x_uncaged_4_class`
│   │   │   │   └── [CLUSTER] (Merged at 80.7% mutual coverage)
│   │   │   │       ├── [LEAF] `20240624_mc38_10x_caged_4_class`
│   │   │   │       └── [LEAF] `20240624_mc38_10x_uncaged_4_class`
│   │   │   └── [LEAF] `20240509_hela-adhered_10x_caged_4_class`
│   │   └── [CLUSTER] (Merged at 85.7% mutual coverage)
│   │       ├── [LEAF] `20250917_moc22-adhered_10x_caged_4_class`
│   │       └── [LEAF] `20260316_a549-tomm20-gfp-adhered_10x_caged_at_4x_4_class`
│   └── [CLUSTER] (Merged at 66.2% mutual coverage)
│       ├── [CLUSTER] (Merged at 98.8% mutual coverage)
│       │   ├── [LEAF] `20240515_DC-adhered_10x_caged_4_class`
│       │   └── [LEAF] `20240516_DC-adhered_10x_caged_4_class`
│       └── [LEAF] `20250820_c8d1a_astrocytes-adherent_10x_caged_4_class`
└── [CLUSTER] (Merged at 42.9% mutual coverage)
    ├── [CLUSTER] (Merged at 71.3% mutual coverage)
    │   ├── [CLUSTER] (Merged at 83.6% mutual coverage)
    │   │   ├── [LEAF] `20240509_Hs675Tfibroblasts_10x_caged_4_class`
    │   │   └── [LEAF] `20250227_preadipocytes-adhered_10x_caged_4_class`
    │   └── [LEAF] `20241212_preadipocytes-adhered_10x_uncaged_4_class`
    └── [CLUSTER] (Merged at 51.6% mutual coverage)
        ├── [LEAF] `231212_imr90_multichannel_overlay_4_class`
        └── [LEAF] `240213_imr90_multichannel_overlay_4_class`
```

## Hierarchical Tree for K=15

```text
[GLOBAL ROOT] (Merged at 39.8% mutual coverage)
├── [CLUSTER] (Merged at 44.7% mutual coverage)
│   ├── [CLUSTER] (Merged at 56.2% mutual coverage)
│   │   ├── [CLUSTER] (Merged at 70.0% mutual coverage)
│   │   │   ├── [CLUSTER] (Merged at 83.2% mutual coverage)
│   │   │   │   ├── [CLUSTER] (Merged at 93.7% mutual coverage)
│   │   │   │   │   ├── [LEAF] `20240625_mc38_10x_caged_4_class`
│   │   │   │   │   └── [LEAF] `20240905_u87-adhered_10x_caged_4_class`
│   │   │   │   └── [LEAF] `20240924_enteric-glia-adhered_10x_uncaged_4_class`
│   │   │   └── [CLUSTER] (Merged at 71.6% mutual coverage)
│   │   │       ├── [CLUSTER] (Merged at 88.4% mutual coverage)
│   │   │       │   ├── [LEAF] `20240624_mc38_10x_caged_4_class`
│   │   │       │   └── [LEAF] `20240624_mc38_10x_uncaged_4_class`
│   │   │       └── [LEAF] `20240509_hela-adhered_10x_caged_4_class`
│   │   └── [CLUSTER] (Merged at 82.7% mutual coverage)
│   │       ├── [CLUSTER] (Merged at 93.7% mutual coverage)
│   │       │   ├── [LEAF] `20240509_Hs675Tfibroblasts_10x_caged_4_class`
│   │       │   └── [LEAF] `20250227_preadipocytes-adhered_10x_caged_4_class`
│   │       └── [LEAF] `20241212_preadipocytes-adhered_10x_uncaged_4_class`
│   └── [CLUSTER] (Merged at 67.0% mutual coverage)
│       ├── [LEAF] `231212_imr90_multichannel_overlay_4_class`
│       └── [LEAF] `240213_imr90_multichannel_overlay_4_class`
└── [CLUSTER] (Merged at 50.4% mutual coverage)
    ├── [CLUSTER] (Merged at 76.7% mutual coverage)
    │   ├── [CLUSTER] (Merged at 99.8% mutual coverage)
    │   │   ├── [LEAF] `20240515_DC-adhered_10x_caged_4_class`
    │   │   └── [LEAF] `20240516_DC-adhered_10x_caged_4_class`
    │   └── [LEAF] `20250820_c8d1a_astrocytes-adherent_10x_caged_4_class`
    └── [CLUSTER] (Merged at 91.7% mutual coverage)
        ├── [LEAF] `20250917_moc22-adhered_10x_caged_4_class`
        └── [LEAF] `20260316_a549-tomm20-gfp-adhered_10x_caged_at_4x_4_class`
```

## Hierarchical Tree for K=30

```text
[GLOBAL ROOT] (Merged at 50.6% mutual coverage)
├── [CLUSTER] (Merged at 68.3% mutual coverage)
│   ├── [CLUSTER] (Merged at 70.0% mutual coverage)
│   │   ├── [CLUSTER] (Merged at 84.0% mutual coverage)
│   │   │   ├── [CLUSTER] (Merged at 93.9% mutual coverage)
│   │   │   │   ├── [CLUSTER] (Merged at 99.3% mutual coverage)
│   │   │   │   │   ├── [LEAF] `20240625_mc38_10x_caged_4_class`
│   │   │   │   │   └── [LEAF] `20240905_u87-adhered_10x_caged_4_class`
│   │   │   │   └── [LEAF] `20240924_enteric-glia-adhered_10x_uncaged_4_class`
│   │   │   └── [CLUSTER] (Merged at 86.5% mutual coverage)
│   │   │       ├── [CLUSTER] (Merged at 97.4% mutual coverage)
│   │   │       │   ├── [LEAF] `20240624_mc38_10x_caged_4_class`
│   │   │       │   └── [LEAF] `20240624_mc38_10x_uncaged_4_class`
│   │   │       └── [LEAF] `20240509_hela-adhered_10x_caged_4_class`
│   │   └── [CLUSTER] (Merged at 97.8% mutual coverage)
│   │       ├── [LEAF] `20250917_moc22-adhered_10x_caged_4_class`
│   │       └── [LEAF] `20260316_a549-tomm20-gfp-adhered_10x_caged_at_4x_4_class`
│   └── [CLUSTER] (Merged at 89.7% mutual coverage)
│       ├── [CLUSTER] (Merged at 100.0% mutual coverage)
│       │   ├── [LEAF] `20240515_DC-adhered_10x_caged_4_class`
│       │   └── [LEAF] `20240516_DC-adhered_10x_caged_4_class`
│       └── [LEAF] `20250820_c8d1a_astrocytes-adherent_10x_caged_4_class`
└── [CLUSTER] (Merged at 74.9% mutual coverage)
    ├── [CLUSTER] (Merged at 93.5% mutual coverage)
    │   ├── [CLUSTER] (Merged at 99.7% mutual coverage)
    │   │   ├── [LEAF] `20240509_Hs675Tfibroblasts_10x_caged_4_class`
    │   │   └── [LEAF] `20250227_preadipocytes-adhered_10x_caged_4_class`
    │   └── [LEAF] `20241212_preadipocytes-adhered_10x_uncaged_4_class`
    └── [CLUSTER] (Merged at 87.8% mutual coverage)
        ├── [LEAF] `231212_imr90_multichannel_overlay_4_class`
        └── [LEAF] `240213_imr90_multichannel_overlay_4_class`
```

# Combined Hierarchical Clustering Topology Report (RF-DETR vs DINOv2)

This report constructs a tree based on standard Agglomerative Hierarchical Clustering. Because linkage requires symmetric distances, we use the average coverage between the two directions. Each internal node shows the dataset that provides the best coverage acting as the parent.

---

# Part 1: RF-DETR Hierarchical Topologies

## Hierarchical Tree for K=5

```text
[GLOBAL ROOT] `20240516_DC-adhered_10x_caged_4_class`
├── [LEAF] `20240515_DC-adhered_10x_caged_4_class` (Covered by parent at 91.2%)
├── [LEAF] `20250820_c8d1a_astrocytes-adherent_10x_caged_4_class` (Covered by parent at 51.9%)
├── [NODE] `20240509_hela-adhered_10x_caged_4_class` (Covered by parent at 32.6%) ⚠️ WEAK LINK
│   ├── [NODE] `20240624_mc38_10x_uncaged_4_class` (Covered by parent at 46.2%) ⚠️ WEAK LINK
│   │   ├── [LEAF] `20240624_mc38_10x_caged_4_class` (Covered by parent at 82.4%)
│   │   └── [NODE] `20240924_enteric-glia-adhered_10x_uncaged_4_class` (Covered by parent at 40.6%) ⚠️ WEAK LINK
│   │       └── [NODE] `20240905_u87-adhered_10x_caged_4_class` (Covered by parent at 49.2%) ⚠️ WEAK LINK
│   │           └── [LEAF] `20240625_mc38_10x_caged_4_class` (Covered by parent at 67.7%)
│   └── [NODE] `20250917_moc22-adhered_10x_caged_4_class` (Covered by parent at 22.7%) ⚠️ WEAK LINK
│       └── [LEAF] `20260316_a549-tomm20-gfp-adhered_10x_caged_at_4x_4_class` (Covered by parent at 71.0%)
└── [NODE] `20240509_Hs675Tfibroblasts_10x_caged_4_class` (Covered by parent at 4.5%) ⚠️ WEAK LINK
    ├── [NODE] `20241212_preadipocytes-adhered_10x_uncaged_4_class` (Covered by parent at 43.6%) ⚠️ WEAK LINK
    │   └── [LEAF] `20250227_preadipocytes-adhered_10x_caged_4_class` (Covered by parent at 63.1%)
    ├── [LEAF] `231212_imr90_multichannel_overlay_4_class` (Covered by parent at 25.9%) ⚠️ WEAK LINK
    └── [LEAF] `240213_imr90_multichannel_overlay_4_class` (Covered by parent at 25.5%) ⚠️ WEAK LINK
```

## Hierarchical Tree for K=10

```text
[GLOBAL ROOT] `20240515_DC-adhered_10x_caged_4_class`
├── [LEAF] `20240516_DC-adhered_10x_caged_4_class` (Covered by parent at 99.0%)
├── [LEAF] `20250820_c8d1a_astrocytes-adherent_10x_caged_4_class` (Covered by parent at 78.7%)
├── [NODE] `20240509_hela-adhered_10x_caged_4_class` (Covered by parent at 51.1%)
│   ├── [NODE] `20240624_mc38_10x_uncaged_4_class` (Covered by parent at 66.5%)
│   │   ├── [LEAF] `20240624_mc38_10x_caged_4_class` (Covered by parent at 99.3%)
│   │   └── [NODE] `20240924_enteric-glia-adhered_10x_uncaged_4_class` (Covered by parent at 62.1%)
│   │       └── [NODE] `20240905_u87-adhered_10x_caged_4_class` (Covered by parent at 69.5%)
│   │           └── [LEAF] `20240625_mc38_10x_caged_4_class` (Covered by parent at 90.1%)
│   └── [NODE] `20240509_Hs675Tfibroblasts_10x_caged_4_class` (Covered by parent at 29.9%) ⚠️ WEAK LINK
│       ├── [LEAF] `20250227_preadipocytes-adhered_10x_caged_4_class` (Covered by parent at 85.1%)
│       └── [LEAF] `20241212_preadipocytes-adhered_10x_uncaged_4_class` (Covered by parent at 64.0%)
├── [NODE] `20250917_moc22-adhered_10x_caged_4_class` (Covered by parent at 36.2%) ⚠️ WEAK LINK
│   └── [LEAF] `20260316_a549-tomm20-gfp-adhered_10x_caged_at_4x_4_class` (Covered by parent at 90.0%)
└── [NODE] `231212_imr90_multichannel_overlay_4_class` (Covered by parent at 4.8%) ⚠️ WEAK LINK
    └── [LEAF] `240213_imr90_multichannel_overlay_4_class` (Covered by parent at 52.6%)
```

## Hierarchical Tree for K=15

```text
[GLOBAL ROOT] `20240515_DC-adhered_10x_caged_4_class`
├── [LEAF] `20240516_DC-adhered_10x_caged_4_class` (Covered by parent at 99.9%)
├── [LEAF] `20250820_c8d1a_astrocytes-adherent_10x_caged_4_class` (Covered by parent at 87.9%)
├── [NODE] `20250917_moc22-adhered_10x_caged_4_class` (Covered by parent at 43.7%) ⚠️ WEAK LINK
│   ├── [LEAF] `20260316_a549-tomm20-gfp-adhered_10x_caged_at_4x_4_class` (Covered by parent at 95.3%)
│   └── [NODE] `20240905_u87-adhered_10x_caged_4_class` (Covered by parent at 52.5%)
│       ├── [LEAF] `20240625_mc38_10x_caged_4_class` (Covered by parent at 96.5%)
│       ├── [LEAF] `20240924_enteric-glia-adhered_10x_uncaged_4_class` (Covered by parent at 79.1%)
│       └── [NODE] `20240509_hela-adhered_10x_caged_4_class` (Covered by parent at 72.7%)
│           └── [NODE] `20240624_mc38_10x_uncaged_4_class` (Covered by parent at 77.2%)
│               └── [LEAF] `20240624_mc38_10x_caged_4_class` (Covered by parent at 100.0%)
└── [NODE] `20241212_preadipocytes-adhered_10x_uncaged_4_class` (Covered by parent at 15.1%) ⚠️ WEAK LINK
    ├── [NODE] `20240509_Hs675Tfibroblasts_10x_caged_4_class` (Covered by parent at 77.3%)
    │   └── [LEAF] `20250227_preadipocytes-adhered_10x_caged_4_class` (Covered by parent at 93.8%)
    └── [NODE] `240213_imr90_multichannel_overlay_4_class` (Covered by parent at 54.9%)
        └── [LEAF] `231212_imr90_multichannel_overlay_4_class` (Covered by parent at 68.4%)
```

## Hierarchical Tree for K=30

```text
[GLOBAL ROOT] `20250917_moc22-adhered_10x_caged_4_class`
├── [LEAF] `20260316_a549-tomm20-gfp-adhered_10x_caged_at_4x_4_class` (Covered by parent at 99.4%)
├── [NODE] `20240624_mc38_10x_uncaged_4_class` (Covered by parent at 87.2%)
│   ├── [LEAF] `20240624_mc38_10x_caged_4_class` (Covered by parent at 100.0%)
│   ├── [NODE] `20240905_u87-adhered_10x_caged_4_class` (Covered by parent at 98.7%)
│   │   ├── [LEAF] `20240625_mc38_10x_caged_4_class` (Covered by parent at 99.7%)
│   │   └── [LEAF] `20240924_enteric-glia-adhered_10x_uncaged_4_class` (Covered by parent at 94.2%)
│   ├── [LEAF] `20240509_hela-adhered_10x_caged_4_class` (Covered by parent at 94.4%)
│   └── [NODE] `20240515_DC-adhered_10x_caged_4_class` (Covered by parent at 79.4%)
│       ├── [LEAF] `20240516_DC-adhered_10x_caged_4_class` (Covered by parent at 100.0%)
│       └── [LEAF] `20250820_c8d1a_astrocytes-adherent_10x_caged_4_class` (Covered by parent at 97.7%)
└── [NODE] `20241212_preadipocytes-adhered_10x_uncaged_4_class` (Covered by parent at 22.0%) ⚠️ WEAK LINK
    ├── [NODE] `20240509_Hs675Tfibroblasts_10x_caged_4_class` (Covered by parent at 92.0%)
    │   └── [LEAF] `20250227_preadipocytes-adhered_10x_caged_4_class` (Covered by parent at 99.4%)
    └── [NODE] `240213_imr90_multichannel_overlay_4_class` (Covered by parent at 77.4%)
        └── [LEAF] `231212_imr90_multichannel_overlay_4_class` (Covered by parent at 91.1%)
```

---

# Part 2: DINOv2 Hierarchical Topologies

## Hierarchical Tree for K=5

```text
[GLOBAL ROOT] `20240924_enteric-glia-adhered_10x_uncaged_4_class`
├── [NODE] `20240509_Hs675Tfibroblasts_10x_caged_4_class` (Covered by parent at 9.7%) ⚠️ WEAK LINK
│   └── [NODE] `20241212_preadipocytes-adhered_10x_uncaged_4_class` (Covered by parent at 11.9%) ⚠️ WEAK LINK
│       └── [LEAF] `20250227_preadipocytes-adhered_10x_caged_4_class` (Covered by parent at 49.7%) ⚠️ WEAK LINK
├── [NODE] `20240509_hela-adhered_10x_caged_4_class` (Covered by parent at 8.8%) ⚠️ WEAK LINK
│   └── [NODE] `20240516_DC-adhered_10x_caged_4_class` (Covered by parent at 20.9%) ⚠️ WEAK LINK
│       ├── [LEAF] `20240515_DC-adhered_10x_caged_4_class` (Covered by parent at 80.8%)
│       ├── [LEAF] `20250820_c8d1a_astrocytes-adherent_10x_caged_4_class` (Covered by parent at 47.2%) ⚠️ WEAK LINK
│       └── [NODE] `20250917_moc22-adhered_10x_caged_4_class` (Covered by parent at 20.1%) ⚠️ WEAK LINK
│           ├── [LEAF] `20260316_a549-tomm20-gfp-adhered_10x_caged_at_4x_4_class` (Covered by parent at 40.4%) ⚠️ WEAK LINK
│           └── [NODE] `20240624_mc38_10x_uncaged_4_class` (Covered by parent at 24.9%) ⚠️ WEAK LINK
│               ├── [LEAF] `20240624_mc38_10x_caged_4_class` (Covered by parent at 65.1%)
│               └── [NODE] `20240905_u87-adhered_10x_caged_4_class` (Covered by parent at 38.1%) ⚠️ WEAK LINK
│                   └── [LEAF] `20240625_mc38_10x_caged_4_class` (Covered by parent at 45.0%) ⚠️ WEAK LINK
└── [NODE] `231212_imr90_multichannel_overlay_4_class` (Covered by parent at 5.3%) ⚠️ WEAK LINK
    └── [LEAF] `240213_imr90_multichannel_overlay_4_class` (Covered by parent at 11.6%) ⚠️ WEAK LINK
```

## Hierarchical Tree for K=10

```text
[GLOBAL ROOT] `20240924_enteric-glia-adhered_10x_uncaged_4_class`
├── [NODE] `20240516_DC-adhered_10x_caged_4_class` (Covered by parent at 18.3%) ⚠️ WEAK LINK
│   ├── [LEAF] `20240515_DC-adhered_10x_caged_4_class` (Covered by parent at 93.9%)
│   ├── [LEAF] `20250820_c8d1a_astrocytes-adherent_10x_caged_4_class` (Covered by parent at 70.3%)
│   ├── [NODE] `20260316_a549-tomm20-gfp-adhered_10x_caged_at_4x_4_class` (Covered by parent at 42.4%) ⚠️ WEAK LINK
│   │   ├── [NODE] `20240624_mc38_10x_uncaged_4_class` (Covered by parent at 65.8%)
│   │   │   └── [LEAF] `20240624_mc38_10x_caged_4_class` (Covered by parent at 95.8%)
│   │   └── [NODE] `20240905_u87-adhered_10x_caged_4_class` (Covered by parent at 46.9%) ⚠️ WEAK LINK
│   │       └── [LEAF] `20240625_mc38_10x_caged_4_class` (Covered by parent at 72.5%)
│   ├── [LEAF] `20240509_hela-adhered_10x_caged_4_class` (Covered by parent at 34.5%) ⚠️ WEAK LINK
│   └── [LEAF] `20250917_moc22-adhered_10x_caged_4_class` (Covered by parent at 30.4%) ⚠️ WEAK LINK
└── [NODE] `20241212_preadipocytes-adhered_10x_uncaged_4_class` (Covered by parent at 10.3%) ⚠️ WEAK LINK
    ├── [LEAF] `20250227_preadipocytes-adhered_10x_caged_4_class` (Covered by parent at 69.7%)
    ├── [LEAF] `20240509_Hs675Tfibroblasts_10x_caged_4_class` (Covered by parent at 26.8%) ⚠️ WEAK LINK
    └── [NODE] `231212_imr90_multichannel_overlay_4_class` (Covered by parent at 20.3%) ⚠️ WEAK LINK
        └── [LEAF] `240213_imr90_multichannel_overlay_4_class` (Covered by parent at 33.0%) ⚠️ WEAK LINK
```

## Hierarchical Tree for K=15

```text
[GLOBAL ROOT] `20240924_enteric-glia-adhered_10x_uncaged_4_class`
├── [NODE] `20240509_hela-adhered_10x_caged_4_class` (Covered by parent at 26.5%) ⚠️ WEAK LINK
│   └── [NODE] `20240516_DC-adhered_10x_caged_4_class` (Covered by parent at 44.4%) ⚠️ WEAK LINK
│       ├── [LEAF] `20240515_DC-adhered_10x_caged_4_class` (Covered by parent at 97.9%)
│       ├── [LEAF] `20250820_c8d1a_astrocytes-adherent_10x_caged_4_class` (Covered by parent at 80.7%)
│       └── [NODE] `20250917_moc22-adhered_10x_caged_4_class` (Covered by parent at 37.2%) ⚠️ WEAK LINK
│           ├── [LEAF] `20260316_a549-tomm20-gfp-adhered_10x_caged_at_4x_4_class` (Covered by parent at 70.7%)
│           └── [NODE] `20240905_u87-adhered_10x_caged_4_class` (Covered by parent at 34.5%) ⚠️ WEAK LINK
│               ├── [LEAF] `20240625_mc38_10x_caged_4_class` (Covered by parent at 84.0%)
│               └── [NODE] `20240624_mc38_10x_uncaged_4_class` (Covered by parent at 68.8%)
│                   └── [LEAF] `20240624_mc38_10x_caged_4_class` (Covered by parent at 99.6%)
└── [NODE] `20241212_preadipocytes-adhered_10x_uncaged_4_class` (Covered by parent at 13.2%) ⚠️ WEAK LINK
    ├── [LEAF] `20250227_preadipocytes-adhered_10x_caged_4_class` (Covered by parent at 79.1%)
    ├── [LEAF] `20240509_Hs675Tfibroblasts_10x_caged_4_class` (Covered by parent at 41.8%) ⚠️ WEAK LINK
    └── [NODE] `240213_imr90_multichannel_overlay_4_class` (Covered by parent at 19.2%) ⚠️ WEAK LINK
        └── [LEAF] `231212_imr90_multichannel_overlay_4_class` (Covered by parent at 46.6%) ⚠️ WEAK LINK
```

## Hierarchical Tree for K=30

```text
[GLOBAL ROOT] `20240924_enteric-glia-adhered_10x_uncaged_4_class`
├── [NODE] `20240509_hela-adhered_10x_caged_4_class` (Covered by parent at 41.5%) ⚠️ WEAK LINK
│   ├── [NODE] `20240516_DC-adhered_10x_caged_4_class` (Covered by parent at 63.9%)
│   │   ├── [LEAF] `20240515_DC-adhered_10x_caged_4_class` (Covered by parent at 99.9%)
│   │   └── [LEAF] `20250820_c8d1a_astrocytes-adherent_10x_caged_4_class` (Covered by parent at 92.3%)
│   └── [NODE] `20250917_moc22-adhered_10x_caged_4_class` (Covered by parent at 46.9%) ⚠️ WEAK LINK
│       ├── [LEAF] `20260316_a549-tomm20-gfp-adhered_10x_caged_at_4x_4_class` (Covered by parent at 85.4%)
│       └── [NODE] `20240905_u87-adhered_10x_caged_4_class` (Covered by parent at 47.5%) ⚠️ WEAK LINK
│           ├── [LEAF] `20240625_mc38_10x_caged_4_class` (Covered by parent at 96.7%)
│           └── [NODE] `20240624_mc38_10x_uncaged_4_class` (Covered by parent at 85.0%)
│               └── [LEAF] `20240624_mc38_10x_caged_4_class` (Covered by parent at 100.0%)
└── [NODE] `20241212_preadipocytes-adhered_10x_uncaged_4_class` (Covered by parent at 20.1%) ⚠️ WEAK LINK
    ├── [LEAF] `20250227_preadipocytes-adhered_10x_caged_4_class` (Covered by parent at 91.7%)
    ├── [LEAF] `20240509_Hs675Tfibroblasts_10x_caged_4_class` (Covered by parent at 64.2%)
    └── [NODE] `240213_imr90_multichannel_overlay_4_class` (Covered by parent at 37.9%) ⚠️ WEAK LINK
        └── [LEAF] `231212_imr90_multichannel_overlay_4_class` (Covered by parent at 69.4%)
```

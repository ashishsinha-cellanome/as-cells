# Coverage Arborescence Data Splits
This file tracks the generated Hydra data configs based on local motifs extracted from the Tree Topology.
| Config YAML | Motif Type | Train Datasets | Test Datasets | Coverage (Train -> Test) |
|-------------|------------|----------------|---------------|--------------------------|
| `motif_01_mc38_downward.yaml` | Downward | 1 dataset(s) | 2 dataset(s) | 0.800 |
| `motif_02_mc38_to_mc38_upward.yaml` | Upward Single | 1 dataset(s) | 1 dataset(s) | 0.971 |
| `motif_03_u87-adhered_to_mc38_upward.yaml` | Upward Single | 1 dataset(s) | 1 dataset(s) | 0.717 |
| `motif_04_multichildren_to_mc38_upward.yaml` | Upward Multiple | 2 dataset(s) | 1 dataset(s) | 0.971 |
| `motif_05_mc38_to_u87-adhered_sibling.yaml` | Sibling | 1 dataset(s) | 1 dataset(s) | 0.670 |
| `motif_06_u87-adhered_to_mc38_sibling.yaml` | Sibling | 1 dataset(s) | 1 dataset(s) | 0.705 |
| `motif_07_mc38_downward.yaml` | Downward | 1 dataset(s) | 2 dataset(s) | 0.561 |
| `motif_08_a549-tomm20-gfp-adhered_to_mc38_upward.yaml` | Upward Single | 1 dataset(s) | 1 dataset(s) | 0.766 |
| `motif_09_hela-adhered_to_mc38_upward.yaml` | Upward Single | 1 dataset(s) | 1 dataset(s) | 0.594 |
| `motif_10_multichildren_to_mc38_upward.yaml` | Upward Multiple | 2 dataset(s) | 1 dataset(s) | 0.766 |
| `motif_11_a549-tomm20-gfp-adhered_to_hela-adhered_sibling.yaml` | Sibling | 1 dataset(s) | 1 dataset(s) | 0.327 |
| `motif_12_hela-adhered_to_a549-tomm20-gfp-adhered_sibling.yaml` | Sibling | 1 dataset(s) | 1 dataset(s) | 0.401 |
| `motif_13_u87-adhered_downward.yaml` | Downward | 1 dataset(s) | 1 dataset(s) | 0.843 |
| `motif_14_mc38_to_u87-adhered_upward.yaml` | Upward Single | 1 dataset(s) | 1 dataset(s) | 0.759 |
| `motif_15_mc38_downward.yaml` | Downward | 1 dataset(s) | 3 dataset(s) | 0.484 |
| `motif_16_Hs675Tfibroblasts_to_mc38_upward.yaml` | Upward Single | 1 dataset(s) | 1 dataset(s) | 0.683 |
| `motif_17_c8d1a_to_mc38_upward.yaml` | Upward Single | 1 dataset(s) | 1 dataset(s) | 0.639 |
| `motif_18_enteric-glia-adhered_to_mc38_upward.yaml` | Upward Single | 1 dataset(s) | 1 dataset(s) | 0.627 |
| `motif_19_multichildren_to_mc38_upward.yaml` | Upward Multiple | 3 dataset(s) | 1 dataset(s) | 0.683 |
| `motif_20_Hs675Tfibroblasts_to_c8d1a_sibling.yaml` | Sibling | 1 dataset(s) | 1 dataset(s) | 0.364 |
| `motif_21_c8d1a_to_Hs675Tfibroblasts_sibling.yaml` | Sibling | 1 dataset(s) | 1 dataset(s) | 0.498 |
| `motif_22_Hs675Tfibroblasts_to_enteric-glia-adhered_sibling.yaml` | Sibling | 1 dataset(s) | 1 dataset(s) | 0.234 |
| `motif_23_enteric-glia-adhered_to_Hs675Tfibroblasts_sibling.yaml` | Sibling | 1 dataset(s) | 1 dataset(s) | 0.561 |
| `motif_24_c8d1a_to_enteric-glia-adhered_sibling.yaml` | Sibling | 1 dataset(s) | 1 dataset(s) | 0.209 |
| `motif_25_enteric-glia-adhered_to_c8d1a_sibling.yaml` | Sibling | 1 dataset(s) | 1 dataset(s) | 0.332 |
| `motif_26_Hs675Tfibroblasts_downward.yaml` | Downward | 1 dataset(s) | 2 dataset(s) | 0.593 |
| `motif_27_imr90_to_Hs675Tfibroblasts_upward.yaml` | Upward Single | 1 dataset(s) | 1 dataset(s) | 0.733 |
| `motif_28_preadipocytes-adhered_to_Hs675Tfibroblasts_upward.yaml` | Upward Single | 1 dataset(s) | 1 dataset(s) | 0.718 |
| `motif_29_multichildren_to_Hs675Tfibroblasts_upward.yaml` | Upward Multiple | 2 dataset(s) | 1 dataset(s) | 0.733 |
| `motif_30_imr90_to_preadipocytes-adhered_sibling.yaml` | Sibling | 1 dataset(s) | 1 dataset(s) | 0.664 |
| `motif_31_preadipocytes-adhered_to_imr90_sibling.yaml` | Sibling | 1 dataset(s) | 1 dataset(s) | 0.592 |
| `motif_32_preadipocytes-adhered_downward.yaml` | Downward | 1 dataset(s) | 1 dataset(s) | 0.600 |
| `motif_33_preadipocytes-adhered_to_preadipocytes-adhered_upward.yaml` | Upward Single | 1 dataset(s) | 1 dataset(s) | 0.749 |
| `motif_34_a549-tomm20-gfp-adhered_downward.yaml` | Downward | 1 dataset(s) | 2 dataset(s) | 0.439 |
| `motif_35_moc22-adhered_to_a549-tomm20-gfp-adhered_upward.yaml` | Upward Single | 1 dataset(s) | 1 dataset(s) | 0.681 |
| `motif_36_neuron-adhered_to_a549-tomm20-gfp-adhered_upward.yaml` | Upward Single | 1 dataset(s) | 1 dataset(s) | 0.508 |
| `motif_37_multichildren_to_a549-tomm20-gfp-adhered_upward.yaml` | Upward Multiple | 2 dataset(s) | 1 dataset(s) | 0.681 |
| `motif_38_moc22-adhered_to_neuron-adhered_sibling.yaml` | Sibling | 1 dataset(s) | 1 dataset(s) | 0.143 |
| `motif_39_neuron-adhered_to_moc22-adhered_sibling.yaml` | Sibling | 1 dataset(s) | 1 dataset(s) | 0.402 |
| `motif_40_imr90_downward.yaml` | Downward | 1 dataset(s) | 1 dataset(s) | 0.667 |
| `motif_41_imr90_to_imr90_upward.yaml` | Upward Single | 1 dataset(s) | 1 dataset(s) | 0.671 |
| `motif_42_c8d1a_downward.yaml` | Downward | 1 dataset(s) | 1 dataset(s) | 0.649 |
| `motif_43_DC-adhered_to_c8d1a_upward.yaml` | Upward Single | 1 dataset(s) | 1 dataset(s) | 0.695 |
| `motif_44_DC-adhered_downward.yaml` | Downward | 1 dataset(s) | 1 dataset(s) | 0.943 |
| `motif_45_DC-adhered_to_DC-adhered_upward.yaml` | Upward Single | 1 dataset(s) | 1 dataset(s) | 0.948 |
| `motif_46_neuron-adhered_downward.yaml` | Downward | 1 dataset(s) | 1 dataset(s) | 0.816 |
| `motif_47_neuron-adhered_to_neuron-adhered_upward.yaml` | Upward Single | 1 dataset(s) | 1 dataset(s) | 0.822 |
| `motif_48_neuron-adhered_downward.yaml` | Downward | 1 dataset(s) | 2 dataset(s) | 0.629 |
| `motif_49_neuron-adhered_to_neuron-adhered_upward.yaml` | Upward Single | 1 dataset(s) | 1 dataset(s) | 0.824 |
| `motif_50_neuron-adhered_to_neuron-adhered_upward.yaml` | Upward Single | 1 dataset(s) | 1 dataset(s) | 0.694 |
| `motif_51_multichildren_to_neuron-adhered_upward.yaml` | Upward Multiple | 2 dataset(s) | 1 dataset(s) | 0.824 |
| `motif_52_neuron-adhered_to_neuron-adhered_sibling.yaml` | Sibling | 1 dataset(s) | 1 dataset(s) | 0.793 |
| `motif_53_neuron-adhered_to_neuron-adhered_sibling.yaml` | Sibling | 1 dataset(s) | 1 dataset(s) | 0.669 |
| `motif_54_neuron-adhered_downward.yaml` | Downward | 1 dataset(s) | 1 dataset(s) | 0.993 |
| `motif_55_neuron-adhered_to_neuron-adhered_upward.yaml` | Upward Single | 1 dataset(s) | 1 dataset(s) | 0.996 |

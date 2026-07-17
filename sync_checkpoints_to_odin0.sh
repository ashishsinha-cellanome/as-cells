#!/bin/bash

# NOTE: Run this script from the VULCAN server to transfer files directly to ODIN0.

# ---------------- CONFIGURATION ----------------
ODIN0_USER="ubuntu"                              # Update if your username on odin0 is different
ODIN0_HOST="odin0"                               # Update with the actual address of odin0 if needed
DEST_BASE="/mnt/personal/cellanome/checkpoints/" # Update this to the folder where you want to save them on odin0
# -----------------------------------------------

echo "Starting transfer of checkpoints from vulcan to odin0..."

# List of formatted "FolderName|SourcePath"
# Spaces, commas, and special characters in Model+Config have been replaced with underscores.
declare -a CHECKPOINTS=(
  "RF-DETR_base|/project/aip-robsc/asinha/cellanome/DATA/checkpoints/rf_detr_base_2026-03-14_21_4342634_21-23/ckpts/rfdetr-ema-09-val_map_ema0.7284.ckpt"
  "RF-DETR_medium|/project/aip-robsc/asinha/cellanome/DATA/checkpoints/rf_detr_medium_2026-03-15_11_4345647_11-51/ckpts/rfdetr-ema-09-val_map_ema0.7293.ckpt"
  "RT-DETR-v1_resnet50|/project/aip-robsc/asinha/cellanome/DATA/checkpoints/rtdetr_v1_resnet50_2026-02-27_4134642_09-46/ckpts/rtdetr-ema-17-val_map0.5446.ckpt"
  "RT-DETR-v2_w_ResNet50_300_queries_stage_4|/project/aip-robsc/asinha/cellanome/DATA/checkpoints/rtdetr_v2_resnet50_backbone_2026-03-23_11_4455655_11-50/ckpts/rtdetr-regular-epoch13-val_map0.6747.ckpt"
  "RT-DETR-v2_w_ResNet50_600_queries_stage_4|/project/aip-robsc/asinha/cellanome/DATA/checkpoints/rtdetr_v2_resnet50_backbone_2026-03-23_07_4454455_07-06/ckpts/rtdetr-regular-epoch23-val_map0.6778.ckpt"
  "RT-DETR-v2_w_ResNet50_300_queries_stage_0|/project/aip-robsc/asinha/cellanome/DATA/checkpoints/rtdetr_v2_resnet50_backbone_2026-03-23_12_4455910_12-30/ckpts/rtdetr-regular-epoch10-val_map0.6894.ckpt"
  "RT-DETR-v2_w_ResNet50_600_queries_stage_0|/project/aip-robsc/asinha/cellanome/DATA/checkpoints/rtdetr_v2_resnet50_backbone_2026-03-23_12_4455903_12-30/ckpts/rtdetr-regular-epoch10-val_map0.6885.ckpt"
  "RT-DETR-v2_w_ResNet50_300_queries_stage_2|/project/aip-robsc/asinha/cellanome/DATA/checkpoints/rtdetr_v2_resnet50_backbone_2026-03-23_12_4455919_12-30/ckpts/rtdetr-regular-epoch23-val_map0.6855.ckpt"
  "RT-DETR-v2_w_ResNet50_600_queries_stage_2|/project/aip-robsc/asinha/cellanome/DATA/checkpoints/rtdetr_v2_resnet50_backbone_2026-03-23_12_4455928_12-30/ckpts/rtdetr-regular-epoch14-val_map0.6849.ckpt"
  "RT-DETR-v2_w_Dinov2_fused_fpn_300_q_3_7_11|/project/aip-robsc/asinha/cellanome/DATA/checkpoints/rtdetr_v2_dinov2-base_backbone_2026-03-31_18_4568833_18-26/ckpts/rtdetr-regular-epoch20-val_map0.6866.ckpt"
  "RT-DETRv2_w_Dinov2_simple_fpn_300_Q_3_7_11|/project/aip-robsc/asinha/cellanome/DATA/checkpoints/rtdetr_v2_dinov2-base_backbone_2026-04-01_03_4570921_03-17/ckpts/rtdetr-regular-epoch20-val_map0.6768.ckpt"
  "RT-DETRv2_w_Dinov2_tiny_fpn_300_Q_3_7_11|/project/aip-robsc/asinha/cellanome/DATA/checkpoints/rtdetr_v2_dinov2-base_backbone_2026-04-01_19_4582635_19-01/ckpts/rtdetr-regular-epoch10-val_map0.6821.ckpt"
  "Mask2Former_dinov2_tinyFPN|/project/aip-robsc/asinha/cellanome/DATA/checkpoints/mask2former_dinov2-base_2026-04-16_05_4706797_05-56/ckpts/mask2former-ema-epoch20-val_map_ema0.0000.ckpt"
  "Mask2Former_dinvo2_SFP|/project/aip-robsc/asinha/cellanome/DATA/checkpoints/mask2former_dinov2-base_2026-04-16_10_4704603_10-54/ckpts/mask2former-epoch18-val_map0.0000.ckpt"
  "DEIMv2-X_num_denoising_0|/project/aip-robsc/asinha/cellanome/DATA/checkpoints/deimv2_x_2026-04-15_06_4696051_06-48/ckpts/deimv2-ema-29-val_map_ema0.6635.ckpt"
  "DEIMv2-M_num_denoising_100|/project/aip-robsc/asinha/cellanome/DATA/checkpoints/deimv2_m_2026-04-07_19_4633064_19-37/ckpts/deimv2-ema-00-val_map_ema-1.0000.ckpt"
  "DEIMv2-M_num_denoising_0|/project/aip-robsc/asinha/cellanome/DATA/checkpoints/deimv2_m_2026-04-15_06_4696666_06-24/ckpts/deimv2-ema-26-val_map_ema0.6555.ckpt"
)

# Iterate over each item
for item in "${CHECKPOINTS[@]}"; do
  # Extract folder name and source path
  FOLDER_NAME="${item%%|*}"
  SRC_PATH="${item##*|}"

  echo "--------------------------------------------------------"
  echo "Processing: $FOLDER_NAME"

  if [ ! -f "$SRC_PATH" ]; then
    echo "Warning: File does not exist: $SRC_PATH"
    continue
  fi

  # Create the target directory on odin0
  ssh "${ODIN0_USER}@${ODIN0_HOST}" "mkdir -p '${DEST_BASE}/${FOLDER_NAME}'"

  # Rsync the checkpoint file to the target directory
  rsync -avzP "$SRC_PATH" "${ODIN0_USER}@${ODIN0_HOST}:${DEST_BASE}/${FOLDER_NAME}/"
done

echo "--------------------------------------------------------"
echo "All transfers completed."

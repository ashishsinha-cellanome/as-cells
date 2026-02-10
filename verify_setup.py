#!/usr/bin/env python3
"""
Verification script for RT-DETR training setup.
- Checks the dataloader and visualizes a batch of data.
- Performs an overfitting test on a single batch to ensure the model can learn.
"""

import argparse
import yaml
import torch
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

from train_rt_detr import setup_model, setup_data
from transformers.image_transforms import center_to_corners_format

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def convert_bbox_yolo_to_pascal(boxes, image_size):
    """
    Convert bounding boxes from YOLO format (x_center, y_center, width, height)
    to Pascal VOC format (x_min, y_min, x_max, y_max).
    """
    boxes = center_to_corners_format(boxes)
    height, width = image_size
    boxes = boxes * torch.tensor([[width, height, width, height]])
    return boxes

def visualize_batch(batch, processor, label_map, num_samples=8):
    """Visualize a few samples from a batch."""
    print("\nVisualizing a batch of data...")
    
    pixel_values = batch['pixel_values']
    labels = batch['labels']
    COLORS = [(0, 0, 0), (0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 255, 0), (255, 0, 255)]
    
    # Denormalize images
    mean = torch.tensor(processor.image_mean).view(1, 3, 1, 1)
    std = torch.tensor(processor.image_std).view(1, 3, 1, 1)
    images = (pixel_values * std) + mean
    images = (images * 255).byte().permute(0, 2, 3, 1).cpu().numpy()
    
    num_samples = min(num_samples, len(images))
    if num_samples == 0:
        print("No samples to visualize.")
        return

    # Handle single sample case
    if num_samples == 1:
        nrows, ncols = 1, 1
        figsize = (5, 5)
    else:
        # Create a grid with 2 rows
        nrows = 2
        # Calculate columns needed (handles odd/even num_samples)
        ncols = (num_samples + 1) // 2 
        figsize = (5 * ncols, 5 * nrows) # Width, Height

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    
    # Flatten axes array for easy 1D indexing, and handle single sample case
    if num_samples == 1:
        axes = [axes] # Make it iterable
    else:
        axes = axes.flatten() 

    for i in range(num_samples):
        img = images[i].copy()
        img_labels = labels[i]
        
        boxes = convert_bbox_yolo_to_pascal(
            img_labels['boxes'],
            (img_labels['size'][0].item(), img_labels['size'][1].item())
        ).numpy().astype(int)
        
        class_ids = img_labels['class_labels'].numpy()
        
        for box, class_id in zip(boxes, class_ids):
            xmin, ymin, xmax, ymax = box

            color = COLORS[(class_id + 1) % len(COLORS)] # add 1 to be consistent with Mask R-CNN colors/labels

            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
            
            class_name = label_map.get(class_id, "Unknown")
            cv2.putText(img, class_name, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
        # --- FIX: Use the flattened axis ---
        ax = axes[i]
        ax.imshow(img)
        ax.set_title(f"Sample {i+1}")
        ax.axis('off')

    # --- ADDITION: Turn off any unused subplots ---
    # This handles cases like 7 samples in a 2x4 grid
    for j in range(num_samples, nrows * ncols):
        axes[j].axis('off')
        
    plt.tight_layout()
    plt.savefig("dataloader_verification_batch.png")
    print("✓ Batch visualization saved to 'dataloader_verification_batch.png'")
    plt.show()

def run_overfit_test(model, batch, num_steps=500):
    """Run an overfitting test on a single batch."""
    print(f"\nRunning overfitting test for {num_steps} steps...")
    
    # Move model and data to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    pixel_values = batch["pixel_values"].to(device)
    labels = [{k: v.to(device) for k, v in sample.items()} for sample in batch["labels"]]
    
    # Get optimizer
    optimizer_dict = model.configure_optimizers()
    optimizer = optimizer_dict['optimizer']
    
    losses = []
    
    pbar = tqdm(range(num_steps), desc="Overfitting test")
    for step in pbar:
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model.model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        pbar.set_postfix({"loss": loss.item()})
        
    print("✓ Overfitting test complete.")
    
    # Plot loss
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title("Overfitting Test Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig("overfitting_test_loss.png")
    print("✓ Overfitting test loss plot saved to 'overfitting_test_loss.png'")
    plt.show()
    
    # Check if loss decreased
    if len(losses) > 1 and losses[0] > losses[-1]:
        print(f"SUCCESS: Loss decreased from {losses[0]:.4f} to {losses[-1]:.4f}")
    else:
        print(f"WARNING: Loss did not decrease significantly. Initial: {losses[0]:.4f}, Final: {losses[-1]:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Verification script for RT-DETR training setup.")
    parser.add_argument(
        '--config','-c',
        type=str,
        default='configs/rt_detr_dinov2_config.yaml',
        help='Path to configuration file'
    )
    # breakpoint()
    args = parser.parse_args()
    
    print("="*80)
    print("Starting Training Setup Verification")
    print("="*80)
    
    # Load config
    print(f"\nLoading configuration from: {args.config}")
    config = load_config(args.config)
    
    # --- 1. DataLoader Verification ---
    print("\n--- Step 1: Verifying DataLoader ---")
    
    # Force num_workers to 0 to avoid multiprocessing issues
    # config['training']['num_workers'] = 0
    # print("\nForcing num_workers=0 for verification to avoid multiprocessing issues.\n")
    
    # Setup model and data module
    # We need the processor from the model setup for the data module
    model, processor = setup_model(config)
    data_module = setup_data(config, processor)
    
    # Get a batch
    data_module.prepare_data()
    data_module.setup(stage='fit')
    train_loader = data_module.train_dataloader()
    
    try:
        batch = next(iter(train_loader))
        print("✓ Successfully loaded one batch of data.")
    except Exception as e:
        print(f"ERROR: Failed to load a batch of data: {e}")
        return
        
    # Check batch contents
    print(f"  - Pixel values shape: {batch['pixel_values'].shape}")
    print(f"  - Labels type: {type(batch['labels'])}")
    print(f"  - Number of samples in batch: {len(batch['labels'])}")
    
    # Visualize batch
    visualize_batch(batch, processor, config['model']['label_map'])
    
    # --- 2. Model Overfitting Test ---
    print("\n--- Step 2: Verifying Model Overfitting ---")
    run_overfit_test(model, batch)
    
    print("\n" + "="*80)
    print("Verification script finished.")
    print("="*80)

if __name__ == "__main__":
    main()

import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from transformers.models.mask2former.modeling_mask2former import Mask2FormerHungarianMatcher, sample_point, pair_wise_sigmoid_cross_entropy_loss, pair_wise_dice_loss
import numpy as np

def memory_efficient_matcher_forward(
    self,
    masks_queries_logits: torch.Tensor,
    class_queries_logits: torch.Tensor,
    mask_labels: torch.Tensor,
    class_labels: torch.Tensor,
) -> list[tuple[torch.Tensor]]:
    indices = []
    batch_size = masks_queries_logits.shape[0]
    
    for i in range(batch_size):
        pred_probs = class_queries_logits[i].softmax(-1)
        pred_mask = masks_queries_logits[i]

        cost_class = -pred_probs[:, class_labels[i]]
        
        # Keep ground truth on GPU as bool
        target_mask_bool = mask_labels[i]
        num_gt = target_mask_bool.shape[0]
        
        pred_mask = pred_mask[:, None]
        point_coordinates = torch.rand(1, self.num_points, 2, device=pred_mask.device)
        pred_coordinates = point_coordinates.repeat(pred_mask.shape[0], 1, 1)
        pred_mask_sampled = sample_point(pred_mask, pred_coordinates, align_corners=False).squeeze(1)
        
        # Process targets in chunks to prevent 7.6GB OOM from grid_sample
        CHUNK_SIZE = 256
        cost_mask_list = []
        cost_dice_list = []
        
        target_coordinates_chunk = point_coordinates.repeat(CHUNK_SIZE, 1, 1)
        
        for chunk_start in range(0, num_gt, CHUNK_SIZE):
            chunk_end = min(num_gt, chunk_start + CHUNK_SIZE)
            actual_chunk_size = chunk_end - chunk_start
            
            # Only convert the chunk to float32
            target_mask_chunk = target_mask_bool[chunk_start:chunk_end].to(pred_mask.dtype)[:, None]
            
            if actual_chunk_size == CHUNK_SIZE:
                coords = target_coordinates_chunk
            else:
                coords = point_coordinates.repeat(actual_chunk_size, 1, 1)
                
            target_mask_sampled_chunk = sample_point(target_mask_chunk, coords, align_corners=False).squeeze(1)
            
            # Compute losses for this chunk
            # pred_mask_sampled: (num_queries, num_points)
            # target_mask_sampled_chunk: (chunk_size, num_points)
            cost_mask_chunk = pair_wise_sigmoid_cross_entropy_loss(pred_mask_sampled, target_mask_sampled_chunk)
            cost_dice_chunk = pair_wise_dice_loss(pred_mask_sampled, target_mask_sampled_chunk)
            
            cost_mask_list.append(cost_mask_chunk)
            cost_dice_list.append(cost_dice_chunk)
            
            # Free memory
            del target_mask_chunk, target_mask_sampled_chunk
            
        if len(cost_mask_list) > 0:
            cost_mask = torch.cat(cost_mask_list, dim=1)
            cost_dice = torch.cat(cost_dice_list, dim=1)
        else:
            cost_mask = torch.empty((pred_mask.shape[0], 0), device=pred_mask.device)
            cost_dice = torch.empty((pred_mask.shape[0], 0), device=pred_mask.device)
            
        cost_matrix = self.cost_mask * cost_mask + self.cost_class * cost_class + self.cost_dice * cost_dice
        cost_matrix = torch.minimum(cost_matrix, torch.tensor(1e10))
        cost_matrix = torch.maximum(cost_matrix, torch.tensor(-1e10))
        cost_matrix = torch.nan_to_num(cost_matrix, 0)
        
        assigned_indices = linear_sum_assignment(cost_matrix.cpu())
        indices.append(assigned_indices)

    matched_indices = [
        (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices
    ]
    return matched_indices

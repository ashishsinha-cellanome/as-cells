import rfdetr.utilities.box_ops
import rfdetr.models.matcher
import rfdetr.models.criterion

from utils.rfdetr_patches import box_ops as patched_box_ops
from utils.rfdetr_patches import matcher as patched_matcher
from utils.rfdetr_patches import criterion as patched_criterion

def apply_patches():
    # Patch box_ops functions
    rfdetr.utilities.box_ops.box_iou = patched_box_ops.box_iou
    rfdetr.utilities.box_ops.generalized_box_iou = patched_box_ops.generalized_box_iou
    rfdetr.utilities.box_ops.batch_sigmoid_ce_loss = patched_box_ops.batch_sigmoid_ce_loss

    # Patch HungarianMatcher.forward
    rfdetr.models.matcher.HungarianMatcher.forward = patched_matcher.HungarianMatcher.forward

    # Patch SetCriterion.loss_masks
    rfdetr.models.criterion.SetCriterion.loss_masks = patched_criterion.SetCriterion.loss_masks

    from utils.distributed_utils import rank_zero_print
    rank_zero_print("[Startup] ✓ Successfully applied RF-DETR numerical stability patches.")

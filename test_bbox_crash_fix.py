import torch
import transformers.loss.loss_for_object_detection as loss_utils

def test_fix():
    # Degenerate boxes that would normally cause a ValueError
    # x0=0.5, y0=0.5, x1=0.4, y1=0.6 (x1 < x0)
    boxes1 = torch.tensor([[0.5, 0.5, 0.4, 0.6]]) 
    boxes2 = torch.tensor([[0.5, 0.5, 0.7, 0.7]])

    print("--- Testing WITHOUT patch ---")
    try:
        iou = loss_utils.generalized_box_iou(boxes1, boxes2)
        print("Wait, it didn't crash? This shouldn't happen unless it's already patched.")
    except ValueError as e:
        print(f"Caught expected crash: {e}")

    print("\n--- Applying patch ---")
    def patched_generalized_box_iou(boxes1, boxes2):
        # Ensure x2 >= x1 and y2 >= y1 without in-place modification
        boxes1 = torch.cat([boxes1[..., :2], torch.max(boxes1[..., 2:], boxes1[..., :2])], dim=-1)
        boxes2 = torch.cat([boxes2[..., :2], torch.max(boxes2[..., 2:], boxes2[..., :2])], dim=-1)
        return loss_utils.original_generalized_box_iou(boxes1, boxes2)

    if not hasattr(loss_utils, 'original_generalized_box_iou'):
        loss_utils.original_generalized_box_iou = loss_utils.generalized_box_iou
        loss_utils.generalized_box_iou = patched_generalized_box_iou
    
    # Also patch it in loss_rt_detr if it was already imported
    try:
        import transformers.loss.loss_rt_detr as loss_rt_detr
        loss_rt_detr.generalized_box_iou = patched_generalized_box_iou
    except (ImportError, AttributeError):
        pass

    print("\n--- Testing WITH patch ---")
    try:
        iou = loss_utils.generalized_box_iou(boxes1, boxes2)
        print("Success! GIoU calculated:", iou)
        
        # Verify it handles double degenerate as well
        boxes1_v2 = torch.tensor([[0.5, 0.5, 0.1, 0.1]])
        iou_v2 = loss_utils.generalized_box_iou(boxes1_v2, boxes2)
        print("Success! Double degenerate GIoU calculated:", iou_v2)
        
    except ValueError as e:
        print(f"FAILED: Still crashing with patch: {e}")

if __name__ == "__main__":
    test_fix()

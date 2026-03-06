import numpy as np
import torch
from sahi.models.base import DetectionModel
from sahi.predict import get_sliced_prediction
from sahi.prediction import ObjectPrediction
from sahi.utils.compatibility import fix_full_shape_list

class LightningDetectionModel(DetectionModel):
    """
    Generic SAHI adapter for our Lightning Modules.
    It expects a `predict_fn` that takes a numpy array (slice) 
    and returns a dictionary with 'boxes' (xyxy), 'scores', and 'labels'.
    """
    def __init__(self, predict_fn, **kwargs):
        self.predict_fn = predict_fn
        super().__init__(**kwargs)

    def load_model(self):
        # Model is already loaded in the Lightning Module, so this is a no-op
        pass

    def perform_inference(self, image: np.ndarray):
        """
        Args:
            image: np.ndarray of shape (H, W, 3) (RGB)
        """
        # Our predict_fn should handle all necessary model-specific PyTorch logic,
        # like tensor conversion, normalization, device placement, and forward pass.
        preds = self.predict_fn(image)
        self._original_predictions = preds

    @property
    def num_categories(self):
        return 80 # Generic fallback, not strictly used in our pipeline if id2label is minimal

    def _create_object_prediction_list_from_original_predictions(
        self,
        shift_amount_list: list | None = None,
        full_shape_list: list | None = None,
    ):
        original_predictions = self._original_predictions
        shift_amount_list = fix_full_shape_list(shift_amount_list)
        full_shape_list = fix_full_shape_list(full_shape_list)

        if not shift_amount_list:
            shift_amount_list = [[0, 0] for _ in range(1)]
        if not full_shape_list:
            full_shape_list = [None for _ in range(1)]

        object_prediction_list = []

        # original_predictions is expected to be a dict:
        # {
        #   'boxes': torch.Tensor or np.ndarray of shape (N, 4) in xyxy format,
        #   'scores': torch.Tensor or np.ndarray of shape (N,),
        #   'labels': torch.Tensor or np.ndarray of shape (N,)
        # }
        
        boxes = original_predictions.get('boxes', [])
        scores = original_predictions.get('scores', [])
        labels = original_predictions.get('labels', [])

        # Convert to numpy if they are torch tensors
        if torch.is_tensor(boxes): boxes = boxes.cpu().numpy()
        if torch.is_tensor(scores): scores = scores.cpu().numpy()
        if torch.is_tensor(labels): labels = labels.cpu().numpy()

        for i in range(len(boxes)):
            box = boxes[i]
            score = scores[i]
            category_id = int(labels[i])
            category_name = self.category_mapping.get(str(category_id), str(category_id))

            # Apply shift
            shifted_box = [
                box[0] + shift_amount_list[0][0],
                box[1] + shift_amount_list[0][1],
                box[2] + shift_amount_list[0][0],
                box[3] + shift_amount_list[0][1]
            ]

            object_prediction_list.append(
                ObjectPrediction(
                    bbox=shifted_box,
                    category_id=category_id,
                    score=score,
                    bool_mask=None,
                    category_name=category_name,
                    shift_amount=shift_amount_list[0],
                    full_shape=full_shape_list[0],
                )
            )

        return object_prediction_list


def resolve_sahi_overlap(sahi_cfg, slice_dim: int, is_height: bool) -> float:
    """
    Resolve overlap configuration.
    Precedence: ratio > px > default(0.2).
    """
    ratio_key = 'overlap_height_ratio' if is_height else 'overlap_width_ratio'
    px_key = 'overlap_height_px' if is_height else 'overlap_width_px'
    
    ratio_val = getattr(sahi_cfg, ratio_key, None)
    if ratio_val is not None:
        return min(float(ratio_val), 0.99)
        
    px_val = getattr(sahi_cfg, px_key, None)
    if px_val is not None and slice_dim > 0:
        ratio = float(px_val) / float(slice_dim)
        return min(ratio, 0.99)
        
    return 0.2 # Fallback


def run_sahi_sliced_eval(image, predict_fn, sahi_config, input_size):
    """
    Runs SAHI sliced inference using the provided image and generic model wrapper.
    
    Args:
        image: PIL Image or np.ndarray (RGB)
        predict_fn: Callable that takes a numpy array and returns standard predictions dict
        sahi_config: OmegaConf dict containing SAHI parameters
        input_size: The default slice size (from config.model.input_size)
    """
    slice_height = sahi_config.get('slice_height')
    if slice_height is None: slice_height = input_size
    
    slice_width = sahi_config.get('slice_width')
    if slice_width is None: slice_width = input_size

    overlap_height_ratio = resolve_sahi_overlap(sahi_config, slice_height, is_height=True)
    overlap_width_ratio = resolve_sahi_overlap(sahi_config, slice_width, is_height=False)

    model = LightningDetectionModel(
        predict_fn=predict_fn,
        confidence_threshold=0.001, # Pass highly confident and low confident predictions, let the metric eval filter them
        device='cpu', # Device management is handled internally by predict_fn
    )

    result = get_sliced_prediction(
        image,
        model,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio,
        perform_standard_pred=sahi_config.get('perform_standard_pred', True),
        postprocess_type=sahi_config.get('postprocess_type', 'NMS'),
        postprocess_match_metric=sahi_config.get('postprocess_match_metric', 'IOU'),
        postprocess_match_threshold=sahi_config.get('postprocess_match_threshold', 0.5),
        postprocess_class_agnostic=sahi_config.get('postprocess_class_agnostic', False),
        verbose=sahi_config.get('verbose', 0)
    )

    # Convert SAHI result back to standard PyTorch dictionaries
    # (boxes, scores, labels) on CPU
    object_prediction_list = result.object_prediction_list
    
    boxes = []
    scores = []
    labels = []
    for obj in object_prediction_list:
        boxes.append([obj.bbox.minx, obj.bbox.miny, obj.bbox.maxx, obj.bbox.maxy])
        scores.append(obj.score.value)
        labels.append(obj.category.id)

    if len(boxes) > 0:
        boxes = torch.tensor(boxes, dtype=torch.float32)
        scores = torch.tensor(scores, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
    else:
        boxes = torch.empty((0, 4), dtype=torch.float32)
        scores = torch.empty((0,), dtype=torch.float32)
        labels = torch.empty((0,), dtype=torch.int64)

    return {
        "boxes": boxes,
        "scores": scores,
        "labels": labels
    }

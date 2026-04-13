from models.rf_detr_lightning_module import RFDETRLightningModule
from utils.coco_eval_utils import (
    convert_preds_to_coco,
    gather_outputs_across_processes,
    broadcast_object,
    compute_coco_metrics,
)


class RFDETRSegLightningModule(RFDETRLightningModule):
    """PyTorch Lightning module for RF-DETR Segmentation training/evaluation."""

    def on_validation_epoch_end(self):
        viz_predictions = None

        def _compute_and_log(outputs_list, prefix_name, base_prefix, suffix=""):
            all_outputs = gather_outputs_across_processes(outputs_list)
            merged = self._merge_predictions_map(all_outputs)
            metrics = {}
            if self.trainer.is_global_zero:
                predictions = []
                image_ids = []
                for batch_out in all_outputs:
                    predictions.extend(
                        convert_preds_to_coco(
                            batch_out["predictions"], model_to_coco=self.model_to_coco
                        )
                    )
                    image_ids.extend(batch_out["image_ids"])

                if predictions:
                    metrics = compute_coco_metrics(
                        coco_gt=self.val_coco_gt,
                        predictions=predictions,
                        image_ids=sorted(set(image_ids)),
                        max_detections=int(self.config.model.max_detections),
                        label_map=self.config.model.label_map,
                        prefix=f"{prefix_name} performance",
                        iou_type="segm",
                        metric_prefix="segm",
                    )
                else:
                    metrics = {"segm_map": 0.0, "segm_map_50": 0.0, "segm_map_75": 0.0}

            metrics = broadcast_object(metrics, src=0)
            for key, value in metrics.items():
                self.log(
                    f"{base_prefix}/{key}{suffix}",
                    value,
                    prog_bar=(key in {"segm_map", "segm_map_50"}),
                    sync_dist=True,
                )
                if key == "segm_map":
                    self.log(
                        f"{base_prefix}_{key}{suffix}",
                        value,
                        prog_bar=False,
                        sync_dist=True,
                    )

            outputs_list.clear()
            return merged

        # 1. Standard Whole
        if self.validation_step_outputs:
            viz_predictions = _compute_and_log(
                self.validation_step_outputs, "Val", "val", ""
            )

        # 2. Standard Sliced
        if self.validation_step_outputs_sliced:
            viz_predictions = _compute_and_log(
                self.validation_step_outputs_sliced, "Val Sliced", "val", "_sliced"
            )

        # 3. EMA Whole
        if (
            hasattr(self, "validation_step_outputs_ema")
            and self.validation_step_outputs_ema
        ):
            viz_predictions = _compute_and_log(
                self.validation_step_outputs_ema, "Val EMA", "val", "_ema"
            )

        # 4. EMA Sliced
        if (
            hasattr(self, "validation_step_outputs_sliced_ema")
            and self.validation_step_outputs_sliced_ema
        ):
            viz_predictions = _compute_and_log(
                self.validation_step_outputs_sliced_ema,
                "Val Sliced EMA",
                "val",
                "_sliced_ema",
            )

        if self.trainer.is_global_zero and viz_predictions is not None:
            self._visualize_aggregated_predictions(viz_predictions, split="val")
            self.print(
                f"[VAL] Completed validation for epoch {self.current_epoch + 1}."
            )

    def on_test_epoch_end(self):
        viz_predictions = None

        def _compute_and_log(outputs_list, prefix_name, base_prefix, suffix=""):
            all_outputs = gather_outputs_across_processes(outputs_list)
            merged = self._merge_predictions_map(all_outputs)
            metrics = {}
            if self.trainer.is_global_zero:
                predictions = []
                image_ids = []
                for batch_out in all_outputs:
                    predictions.extend(
                        convert_preds_to_coco(
                            batch_out["predictions"], model_to_coco=self.model_to_coco
                        )
                    )
                    image_ids.extend(batch_out["image_ids"])

                if predictions:
                    metrics = compute_coco_metrics(
                        coco_gt=self.test_coco_gt,
                        predictions=predictions,
                        image_ids=sorted(set(image_ids)),
                        max_detections=int(self.config.model.max_detections),
                        label_map=self.config.model.label_map,
                        prefix=f"{prefix_name} performance",
                        iou_type="segm",
                        metric_prefix="segm",
                    )
                else:
                    metrics = {"segm_map": 0.0, "segm_map_50": 0.0, "segm_map_75": 0.0}

            metrics = broadcast_object(metrics, src=0)
            for key, value in metrics.items():
                self.log(
                    f"{base_prefix}/{key}{suffix}",
                    value,
                    prog_bar=(key in {"segm_map", "segm_map_50"}),
                    sync_dist=True,
                )

            outputs_list.clear()
            return merged

        # 1. Standard Whole
        if self.test_step_outputs:
            viz_predictions = _compute_and_log(
                self.test_step_outputs, "Test", "test", ""
            )

        # 2. Standard Sliced
        if self.test_step_outputs_sliced:
            viz_predictions = _compute_and_log(
                self.test_step_outputs_sliced, "Test Sliced", "test", "_sliced"
            )

        # 3. EMA Whole
        if hasattr(self, "test_step_outputs_ema") and self.test_step_outputs_ema:
            viz_predictions = _compute_and_log(
                self.test_step_outputs_ema, "Test EMA", "test", "_ema"
            )

        # 4. EMA Sliced
        if (
            hasattr(self, "test_step_outputs_sliced_ema")
            and self.test_step_outputs_sliced_ema
        ):
            viz_predictions = _compute_and_log(
                self.test_step_outputs_sliced_ema,
                "Test Sliced EMA",
                "test",
                "_sliced_ema",
            )

        if self.trainer.is_global_zero and viz_predictions is not None:
            self._visualize_aggregated_predictions(viz_predictions, split="test")

    def configure_optimizers(self):
        res = super().configure_optimizers()
        if "lr_scheduler" in res and res["lr_scheduler"].get("monitor") == "val/map":
            res["lr_scheduler"]["monitor"] = "val/segm_map"
        return res

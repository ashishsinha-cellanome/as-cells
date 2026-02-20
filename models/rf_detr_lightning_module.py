import torch
import pytorch_lightning as pl

from utils.coco_eval_utils import (
    to_cpu_device,
    convert_preds_to_coco,
    gather_outputs_across_processes,
    broadcast_object,
    compute_coco_metrics,
)


class RFDETRLightningModule(pl.LightningModule):
    """PyTorch Lightning module for RF-DETR training/evaluation."""

    def __init__(self, model, criterion, postprocess, config, val_coco_gt=None, test_coco_gt=None):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.postprocess = postprocess
        self.config = config
        self.val_coco_gt = val_coco_gt
        self.test_coco_gt = test_coco_gt

        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, samples, targets=None):
        return self.model(samples, targets)

    def _move_targets(self, targets):
        return [{k: v.to(self.device) for k, v in target.items()} for target in targets]

    def _compute_loss(self, outputs, targets):
        loss_dict = self.criterion(outputs, targets)
        weight_dict = self.criterion.weight_dict
        loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict if k in weight_dict)
        return loss, loss_dict, weight_dict

    def _log_loss_dict(self, split: str, loss_dict, weight_dict):
        """
        Log all loss terms from RF-DETR criterion.
        For weighted terms we log both unscaled and scaled values.
        """
        for key, value in loss_dict.items():
            self.log(f"{split}/{key}_unscaled", value, on_step=False, on_epoch=True, sync_dist=True)
            if key in weight_dict:
                self.log(f"{split}/{key}", value * weight_dict[key], on_step=False, on_epoch=True, sync_dist=True)
            else:
                self.log(f"{split}/{key}", value, on_step=False, on_epoch=True, sync_dist=True)

    def training_step(self, batch, batch_idx):
        samples, targets = batch
        samples = samples.to(self.device)
        targets = self._move_targets(targets)

        outputs = self.model(samples, targets)
        loss, loss_dict, weight_dict = self._compute_loss(outputs, targets)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self._log_loss_dict("train", loss_dict, weight_dict)

        return loss

    def _collect_batch_predictions(self, outputs, targets):
        orig_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        post = self.postprocess(outputs, orig_sizes)
        post = [to_cpu_device(pred) for pred in post]
        result_map = {
            int(target["image_id"].item()): pred for target, pred in zip(targets, post)
        }
        image_ids = [int(target["image_id"].item()) for target in targets]
        return result_map, image_ids

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        samples, targets = batch
        samples = samples.to(self.device)
        targets = self._move_targets(targets)

        outputs = self.model(samples)
        loss, loss_dict, weight_dict = self._compute_loss(outputs, targets)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self._log_loss_dict("val", loss_dict, weight_dict)

        predictions, image_ids = self._collect_batch_predictions(outputs, targets)
        self.validation_step_outputs.append({"predictions": predictions, "image_ids": image_ids})
        return {"predictions": predictions, "image_ids": image_ids}

    def on_validation_epoch_end(self):
        all_outputs = gather_outputs_across_processes(self.validation_step_outputs)
        metrics = {}
        if self.trainer.is_global_zero:
            val_predictions = []
            val_image_ids = []
            for batch_out in all_outputs:
                val_predictions.extend(convert_preds_to_coco(batch_out["predictions"]))
                val_image_ids.extend(batch_out["image_ids"])

            if len(val_predictions) > 0:
                metrics = compute_coco_metrics(
                    coco_gt=self.val_coco_gt,
                    predictions=val_predictions,
                    image_ids=list(set(val_image_ids)),
                    max_detections=int(self.config.model.max_detections),
                    label_map=self.config.model.label_map,
                )
            else:
                metrics = {"map": 0.0, "map_50": 0.0, "map_75": 0.0}

        metrics = broadcast_object(metrics, src=0)
        for key, value in metrics.items():
            self.log(f"val/{key}", value, prog_bar=(key in {"map", "map_50"}), sync_dist=True)
            if key == "map":
                self.log("val_map", value, prog_bar=False, sync_dist=True)

        self.validation_step_outputs.clear()

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        samples, targets = batch
        samples = samples.to(self.device)
        targets = self._move_targets(targets)

        outputs = self.model(samples)
        predictions, image_ids = self._collect_batch_predictions(outputs, targets)
        self.test_step_outputs.append({"predictions": predictions, "image_ids": image_ids})
        return {"predictions": predictions, "image_ids": image_ids}

    def on_test_epoch_end(self):
        all_outputs = gather_outputs_across_processes(self.test_step_outputs)
        metrics = {}
        if self.trainer.is_global_zero:
            test_predictions = []
            test_image_ids = []
            for batch_out in all_outputs:
                test_predictions.extend(convert_preds_to_coco(batch_out["predictions"]))
                test_image_ids.extend(batch_out["image_ids"])

            if len(test_predictions) > 0:
                metrics = compute_coco_metrics(
                    coco_gt=self.test_coco_gt,
                    predictions=test_predictions,
                    image_ids=list(set(test_image_ids)),
                    max_detections=int(self.config.model.max_detections),
                    label_map=self.config.model.label_map,
                )
            else:
                metrics = {"map": 0.0, "map_50": 0.0, "map_75": 0.0}

        metrics = broadcast_object(metrics, src=0)
        for key, value in metrics.items():
            self.log(f"test/{key}", value, prog_bar=(key in {"map", "map_50"}), sync_dist=True)

        self.test_step_outputs.clear()

    @torch.no_grad()
    def predict_batch(self, samples, score_threshold=0.25):
        """Simple inference helper used by external scripts."""
        self.model.eval()
        if hasattr(samples, "to"):
            samples = samples.to(self.device)
        outputs = self.model(samples)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        if "pred_boxes" in outputs:
            # RF-DETR postprocess expects target sizes.
            if hasattr(samples, "tensors"):
                bsz = samples.tensors.shape[0]
                h, w = samples.tensors.shape[-2:]
            else:
                bsz = samples.shape[0]
                h, w = samples.shape[-2:]
            target_sizes = torch.tensor([[h, w]] * bsz, device=self.device)
            preds = self.postprocess(outputs, target_sizes)
            filtered = []
            for pred in preds:
                keep = pred["scores"] >= score_threshold
                filtered.append({k: v[keep] for k, v in pred.items()})
            return filtered
        return outputs

    def configure_optimizers(self):
        opt_config = self.config.optimizer.optimizer
        sch_config = self.config.scheduler

        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=float(opt_config.lr),
            weight_decay=float(opt_config.weight_decay),
        )

        if sch_config.type == "reduce_lr_on_plateau":
            scheduler = {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode="max",
                    factor=float(sch_config.factor),
                    patience=int(sch_config.patience),
                ),
                "monitor": "val/map",
                "interval": "epoch",
                "frequency": 1,
            }
        elif sch_config.type == "cosine":
            total_steps = max(1, self.trainer.estimated_stepping_batches)
            scheduler = {
                "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=total_steps,
                    eta_min=float(sch_config.eta_min),
                ),
                "interval": "step",
            }
        elif sch_config.type == "onecycle":
            total_steps = max(1, self.trainer.estimated_stepping_batches)
            scheduler = {
                "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=float(opt_config.lr),
                    total_steps=total_steps,
                    pct_start=float(sch_config.pct_start),
                    anneal_strategy="cos",
                    div_factor=25.0,
                    final_div_factor=1e3,
                ),
                "interval": "step",
            }
        else:
            scheduler = {
                "scheduler": torch.optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=max(1, int(getattr(sch_config, "step_size", 10))),
                    gamma=float(getattr(sch_config, "gamma", 0.1)),
                ),
                "interval": "epoch",
            }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

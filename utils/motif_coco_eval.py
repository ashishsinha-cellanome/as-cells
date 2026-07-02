import torch
from utils.detailed_coco_eval import DetailedCocoEvalCallback

class MotifCocoEvalCallback(DetailedCocoEvalCallback):
    def __init__(self, val_dataloader_names, test_dataloader_names, val_get_coco_gt_fns, test_get_coco_gt_fns, label_map=None):
        super().__init__()
        self.val_dataloader_names = val_dataloader_names
        self.test_dataloader_names = test_dataloader_names
        self.val_get_coco_gt_fns = val_get_coco_gt_fns
        self.test_get_coco_gt_fns = test_get_coco_gt_fns
        self.label_map = label_map
        self.val_outputs = {}
        self.val_outputs_ema = {}
        self.test_outputs = {}
        self.test_outputs_ema = {}
        
        for i in range(len(val_dataloader_names)):
            self.val_outputs[i] = []
            self.val_outputs_ema[i] = []
            
        for i in range(len(test_dataloader_names)):
            self.test_outputs[i] = []
            self.test_outputs_ema[i] = []

    def on_validation_epoch_start(self, trainer, pl_module):
        for k in self.val_outputs:
            self.val_outputs[k].clear()
            self.val_outputs_ema[k].clear()
        self._ensure_metadata(trainer, pl_module)

    def on_test_epoch_start(self, trainer, pl_module):
        for k in self.test_outputs:
            self.test_outputs[k].clear()
            self.test_outputs_ema[k].clear()
        self._ensure_metadata(trainer, pl_module)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self._accumulate_batch(outputs, self.val_outputs[dataloader_idx])
        ema_cb = self._get_ema_callback(trainer)
        if ema_cb is not None:
            self._evaluate_ema(ema_cb, pl_module, batch, self.val_outputs_ema[dataloader_idx])

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self._accumulate_batch(outputs, self.test_outputs[dataloader_idx])
        ema_cb = self._get_ema_callback(trainer)
        if ema_cb is not None:
            self._evaluate_ema(ema_cb, pl_module, batch, self.test_outputs_ema[dataloader_idx])

    def on_validation_epoch_end(self, trainer, pl_module):
        for dl_idx, dl_name in enumerate(self.val_dataloader_names):
            coco_gt = self.val_get_coco_gt_fns[dl_idx]()
            if not coco_gt or len(self.val_outputs[dl_idx]) == 0:
                continue
                
            prefix = f"val/{dl_name}"
            print(f"\n[MotifCocoEvalCallback] Computing Validation Metrics for {dl_name}...")
            metrics_bbox, metrics_segm = self._compute_and_log(trainer, pl_module, self.val_outputs[dl_idx], coco_gt, prefix, "")
            
            if dl_name == "train_ds/merged":
                bbox_map = metrics_bbox.get("detailed_bbox_map", 0.0)
                pl_module.log("val/mAP_50_95", bbox_map, sync_dist=True, prog_bar=True)
                trainer.callback_metrics["val/mAP_50_95"] = torch.tensor(bbox_map)
                
                segm_map = metrics_segm.get("detailed_segm_map", 0.0)
                pl_module.log("val/segm_mAP_50_95", segm_map, sync_dist=True)
                trainer.callback_metrics["val/segm_mAP_50_95"] = torch.tensor(segm_map)
            
            if len(self.val_outputs_ema[dl_idx]) > 0:
                print(f"\n[MotifCocoEvalCallback] Computing Validation EMA Metrics for {dl_name}...")
                metrics_bbox_ema, metrics_segm_ema = self._compute_and_log(trainer, pl_module, self.val_outputs_ema[dl_idx], coco_gt, prefix, "_ema")
                
                if dl_name == "train_ds/merged":
                    ema_bbox = metrics_bbox_ema.get("detailed_bbox_map", 0.0)
                    pl_module.log("val/ema_mAP_50_95", ema_bbox, sync_dist=True, prog_bar=True)
                    trainer.callback_metrics["val/ema_mAP_50_95"] = torch.tensor(ema_bbox)
                    
                    ema_segm = metrics_segm_ema.get("detailed_segm_map", 0.0)
                    pl_module.log("val/ema_segm_mAP_50_95", ema_segm, sync_dist=True)
                    trainer.callback_metrics["val/ema_segm_mAP_50_95"] = torch.tensor(ema_segm)

    def on_test_epoch_end(self, trainer, pl_module):
        for dl_idx, dl_name in enumerate(self.test_dataloader_names):
            coco_gt = self.test_get_coco_gt_fns[dl_idx]()
            if not coco_gt or len(self.test_outputs[dl_idx]) == 0:
                continue
                
            prefix = f"test/{dl_name}"
            print(f"\n[MotifCocoEvalCallback] Computing Test Metrics for {dl_name}...")
            metrics_bbox, metrics_segm = self._compute_and_log(trainer, pl_module, self.test_outputs[dl_idx], coco_gt, prefix, "")
            
            if dl_name == "train_ds/merged":
                bbox_map = metrics_bbox.get("detailed_bbox_map", 0.0)
                pl_module.log("test/mAP_50_95", bbox_map, sync_dist=True, prog_bar=True)
                trainer.callback_metrics["test/mAP_50_95"] = torch.tensor(bbox_map)
                
                segm_map = metrics_segm.get("detailed_segm_map", 0.0)
                pl_module.log("test/segm_mAP_50_95", segm_map, sync_dist=True)
                trainer.callback_metrics["test/segm_mAP_50_95"] = torch.tensor(segm_map)
            
            if len(self.test_outputs_ema[dl_idx]) > 0:
                print(f"\n[MotifCocoEvalCallback] Computing Test EMA Metrics for {dl_name}...")
                metrics_bbox_ema, metrics_segm_ema = self._compute_and_log(trainer, pl_module, self.test_outputs_ema[dl_idx], coco_gt, prefix, "_ema")
                
                if dl_name == "train_ds/merged":
                    ema_bbox = metrics_bbox_ema.get("detailed_bbox_map", 0.0)
                    pl_module.log("test/ema_mAP_50_95", ema_bbox, sync_dist=True, prog_bar=True)
                    trainer.callback_metrics["test/ema_mAP_50_95"] = torch.tensor(ema_bbox)
                    
                    ema_segm = metrics_segm_ema.get("detailed_segm_map", 0.0)
                    pl_module.log("test/ema_segm_mAP_50_95", ema_segm, sync_dist=True)
                    trainer.callback_metrics["test/ema_segm_mAP_50_95"] = torch.tensor(ema_segm)

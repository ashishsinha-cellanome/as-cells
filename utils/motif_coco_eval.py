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
            
        self.best_val_metrics = {
            "val/best_mAP_50_95": 0.0,
            "val/best_segm_mAP_50_95": 0.0,
            "val/best_ema_mAP_50_95": 0.0,
            "val/best_ema_segm_mAP_50_95": 0.0,
        }
        self.best_val_epochs = {
            "val/best_epoch_mAP_50_95": 0,
            "val/best_epoch_segm_mAP_50_95": 0,
            "val/best_epoch_ema_mAP_50_95": 0,
            "val/best_epoch_ema_segm_mAP_50_95": 0,
        }

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
        if ema_cb is not None and getattr(ema_cb, "_swapped_state_dict", None) is None:
            self._evaluate_ema(ema_cb, pl_module, batch, self.test_outputs_ema[dataloader_idx])

    def _visualize_predictions(self, trainer, pl_module, step_outputs, coco_gt, split, dl_name):
        import os, cv2
        from pathlib import Path
        from utils.coco_eval_utils import gather_outputs_across_processes
        
        # 1. Gather predictions from all GPUs to rank 0
        all_outputs = gather_outputs_across_processes(step_outputs)
        
        if not trainer.is_global_zero:
            return
            
        cfg = getattr(pl_module, "config", None)
        if cfg is None:
            return
            
        viz_every = cfg.checkpointing.get("visualize_every_n_epochs", 5)
        if (trainer.current_epoch + 1) % viz_every != 0 and split != "test":
            return
            
        max_samples = cfg.checkpointing.get("visualize_samples", 100)
        viz_dir = cfg.checkpointing.get("visualization_dir", "predictions")
        out_dir = getattr(pl_module.train_config, "output_dir", cfg.checkpointing.save_dir) if hasattr(pl_module, "train_config") else cfg.checkpointing.save_dir
        
        save_dir = os.path.join(out_dir, viz_dir, f"epoch_{(trainer.current_epoch + 1):03d}" if split != "test" else "test", split, dl_name.replace("/", "_"))
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"\n[VIZ] Saving {split.upper()} visualizations to: {save_dir}")
        
        all_preds = {}
        for batch_out in all_outputs:
            preds_map = batch_out.get("predictions", {})
            for k, v in preds_map.items():
                all_preds[k] = v
                
        saved_count = 0
        viz_threshold = float(cfg.model.get("draw_threshold", 0.4))
        
        data_path = Path(cfg.data.path)
        
        for img_id, preds in all_preds.items():
            if max_samples != -1 and saved_count >= max_samples:
                break
                
            img_info = coco_gt.loadImgs(img_id)
            if not img_info: continue
            img_info = img_info[0]
            
            img_file = img_info["file_name"]
            img_path = data_path / img_file
            if not img_path.exists():
                img_path = data_path / "images" / img_file
            if not img_path.exists():
                # train_ds/merged -> might have val_name folder
                val_name = getattr(cfg.data, 'val_name', 'val_new') if split == 'val' else getattr(cfg.data, 'test_name', 'test')
                img_path = data_path / "images" / val_name / img_file
            if not img_path.exists():
                continue
                
            img = cv2.imread(str(img_path))
            if img is None: continue
            
            gt_anns = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=[img_id]))
            gt_counts = {}
            for ann in gt_anns:
                x, y, w, h = ann["bbox"]
                if w > 0 and h > 0:
                    cv2.rectangle(img, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 2)
                    cat_name = self.label_map.get(ann["category_id"], str(ann["category_id"])) if self.label_map else str(ann["category_id"])
                    cv2.putText(img, f"GT: {cat_name}", (int(x), int(y)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    gt_counts[cat_name] = gt_counts.get(cat_name, 0) + 1
            
            pred_counts = {}
            if "boxes" in preds and len(preds["boxes"]) > 0:
                boxes = preds["boxes"]
                scores = preds["scores"]
                labels = preds["labels"]
                
                valid_idx = scores >= viz_threshold
                for box, score, lbl in zip(boxes[valid_idx], scores[valid_idx], labels[valid_idx]):
                    x1, y1, x2, y2 = box.tolist()
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                    lbl_val = lbl.item() if hasattr(lbl, 'item') else int(lbl)
                    cat_name = self.label_map.get(lbl_val, str(lbl_val)) if self.label_map else str(lbl_val)
                    cv2.putText(img, f"Pred: {cat_name} {score:.2f}", (int(x1), int(y1)-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    pred_counts[cat_name] = pred_counts.get(cat_name, 0) + 1
            
            text_y = 35
            all_classes = sorted(list(set(gt_counts.keys()) | set(pred_counts.keys())))
            
            for cls_name in all_classes:
                full_text_str = f"{cls_name}: {pred_counts.get(cls_name, 0)}/{gt_counts.get(cls_name, 0)}"
                (w_full, h_full), _ = cv2.getTextSize(full_text_str, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                
                text_x = img.shape[1] - w_full - 25
                
                lbl_part = f"{cls_name}: "
                (w_lbl, _), _ = cv2.getTextSize(lbl_part, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.putText(img, lbl_part, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                pred_part = f"{pred_counts.get(cls_name, 0)}"
                (w_pred, _), _ = cv2.getTextSize(pred_part, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.putText(img, pred_part, (text_x + w_lbl, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                slash_part = "/"
                (w_slash, _), _ = cv2.getTextSize(slash_part, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.putText(img, slash_part, (text_x + w_lbl + w_pred, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                gt_part = f"{gt_counts.get(cls_name, 0)}"
                cv2.putText(img, gt_part, (text_x + w_lbl + w_pred + w_slash, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                text_y += 30
            
            save_path = os.path.join(save_dir, f"viz_{img_file.replace('/', '_')}")
            cv2.imwrite(save_path, img)
            saved_count += 1

    def on_validation_epoch_end(self, trainer, pl_module):
        for dl_idx, dl_name in enumerate(self.val_dataloader_names):
            coco_gt = self.val_get_coco_gt_fns[dl_idx]()
            if not coco_gt or len(self.val_outputs[dl_idx]) == 0:
                continue
                
            prefix = f"val/{dl_name}"
            
            self._visualize_predictions(trainer, pl_module, self.val_outputs[dl_idx], coco_gt, "val", dl_name)
            
            print(f"\n[MotifCocoEvalCallback] Computing Validation Metrics for {dl_name}...")
            metrics_bbox, metrics_segm = self._compute_and_log(trainer, pl_module, self.val_outputs[dl_idx], coco_gt, prefix, "")
            
            if dl_name == "train_ds/merged":
                bbox_map = metrics_bbox.get("detailed_bbox_map", 0.0)
                pl_module.log("val/mAP_50_95", bbox_map, sync_dist=True, prog_bar=True)
                trainer.callback_metrics["val/mAP_50_95"] = torch.tensor(bbox_map)
                if bbox_map > self.best_val_metrics["val/best_mAP_50_95"]:
                    self.best_val_metrics["val/best_mAP_50_95"] = bbox_map
                    self.best_val_epochs["val/best_epoch_mAP_50_95"] = trainer.current_epoch + 1
                pl_module.log("val/best_mAP_50_95", self.best_val_metrics["val/best_mAP_50_95"], sync_dist=True)
                pl_module.log("val/best_epoch_mAP_50_95", float(self.best_val_epochs["val/best_epoch_mAP_50_95"]), sync_dist=True)
                
                segm_map = metrics_segm.get("detailed_segm_map", 0.0)
                pl_module.log("val/segm_mAP_50_95", segm_map, sync_dist=True)
                trainer.callback_metrics["val/segm_mAP_50_95"] = torch.tensor(segm_map)
                if segm_map > self.best_val_metrics["val/best_segm_mAP_50_95"]:
                    self.best_val_metrics["val/best_segm_mAP_50_95"] = segm_map
                    self.best_val_epochs["val/best_epoch_segm_mAP_50_95"] = trainer.current_epoch + 1
                pl_module.log("val/best_segm_mAP_50_95", self.best_val_metrics["val/best_segm_mAP_50_95"], sync_dist=True)
                pl_module.log("val/best_epoch_segm_mAP_50_95", float(self.best_val_epochs["val/best_epoch_segm_mAP_50_95"]), sync_dist=True)
            
            if len(self.val_outputs_ema[dl_idx]) > 0:
                self._visualize_predictions(trainer, pl_module, self.val_outputs_ema[dl_idx], coco_gt, "val", dl_name + "_ema")
                print(f"\n[MotifCocoEvalCallback] Computing Validation EMA Metrics for {dl_name}...")
                metrics_bbox_ema, metrics_segm_ema = self._compute_and_log(trainer, pl_module, self.val_outputs_ema[dl_idx], coco_gt, prefix, "_ema")
                
                if dl_name == "train_ds/merged":
                    ema_bbox = metrics_bbox_ema.get("detailed_bbox_map", 0.0)
                    pl_module.log("val/ema_mAP_50_95", ema_bbox, sync_dist=True, prog_bar=True)
                    trainer.callback_metrics["val/ema_mAP_50_95"] = torch.tensor(ema_bbox)
                    if ema_bbox > self.best_val_metrics["val/best_ema_mAP_50_95"]:
                        self.best_val_metrics["val/best_ema_mAP_50_95"] = ema_bbox
                        self.best_val_epochs["val/best_epoch_ema_mAP_50_95"] = trainer.current_epoch + 1
                    pl_module.log("val/best_ema_mAP_50_95", self.best_val_metrics["val/best_ema_mAP_50_95"], sync_dist=True)
                    pl_module.log("val/best_epoch_ema_mAP_50_95", float(self.best_val_epochs["val/best_epoch_ema_mAP_50_95"]), sync_dist=True)
                    
                    ema_segm = metrics_segm_ema.get("detailed_segm_map", 0.0)
                    pl_module.log("val/ema_segm_mAP_50_95", ema_segm, sync_dist=True)
                    trainer.callback_metrics["val/ema_segm_mAP_50_95"] = torch.tensor(ema_segm)
                    if ema_segm > self.best_val_metrics["val/best_ema_segm_mAP_50_95"]:
                        self.best_val_metrics["val/best_ema_segm_mAP_50_95"] = ema_segm
                        self.best_val_epochs["val/best_epoch_ema_segm_mAP_50_95"] = trainer.current_epoch + 1
                    pl_module.log("val/best_ema_segm_mAP_50_95", self.best_val_metrics["val/best_ema_segm_mAP_50_95"], sync_dist=True)
                    pl_module.log("val/best_epoch_ema_segm_mAP_50_95", float(self.best_val_epochs["val/best_epoch_ema_segm_mAP_50_95"]), sync_dist=True)

    def on_test_epoch_end(self, trainer, pl_module):
        ema_cb = self._get_ema_callback(trainer)
        
        test_group_scores = {
            "train_ds": {"mAP": [], "segm": [], "ema_mAP": [], "ema_segm": []},
            "test_ds": {"mAP": [], "segm": [], "ema_mAP": [], "ema_segm": []}
        }
        
        # Initialize markdown report
        ckpt_path = getattr(pl_module.config.initialization, "load_from_checkpoint", "Unknown")
        md_report = f"# Inference Summary Report\n**Checkpoint:** `{ckpt_path}`\n\n"
        
        for dl_idx, dl_name in enumerate(self.test_dataloader_names):
            coco_gt = self.test_get_coco_gt_fns[dl_idx]()
            if not coco_gt or len(self.test_outputs[dl_idx]) == 0:
                continue
                
            group = "train_ds" if dl_name.startswith("train_ds") else "test_ds"
            prefix = f"test/{dl_name}"
            
            self._visualize_predictions(trainer, pl_module, self.test_outputs[dl_idx], coco_gt, "test", dl_name)
            
            print(f"\n[MotifCocoEvalCallback] Computing Test Metrics for {dl_name}...")
            metrics_bbox, metrics_segm = self._compute_and_log(trainer, pl_module, self.test_outputs[dl_idx], coco_gt, prefix, "")
            
            # Add to Markdown Report
            md_report += f"## Dataset: `{dl_name}`\n"
            if "_markdown_table" in metrics_bbox:
                md_report += f"**Metric Type:** Regular Weights - Bounding Box (BBOX)\n\n"
                md_report += metrics_bbox["_markdown_table"] + "\n\n"
            if "_markdown_table" in metrics_segm:
                md_report += f"**Metric Type:** Regular Weights - Segmentation (SEGM)\n\n"
                md_report += metrics_segm["_markdown_table"] + "\n\n"
            
            bbox_map = metrics_bbox.get("detailed_bbox_map", 0.0)
            segm_map = metrics_segm.get("detailed_segm_map", 0.0)
            test_group_scores[group]["mAP"].append(bbox_map)
            if "detailed_segm_map" in metrics_segm:
                test_group_scores[group]["segm"].append(segm_map)
            
            pl_module.log(f"{prefix}/mAP_50_95", bbox_map, sync_dist=True)
            trainer.callback_metrics[f"{prefix}/mAP_50_95"] = torch.tensor(bbox_map)
            
            pl_module.log(f"{prefix}/segm_mAP_50_95", segm_map, sync_dist=True)
            trainer.callback_metrics[f"{prefix}/segm_mAP_50_95"] = torch.tensor(segm_map)
            
            if dl_idx == 0:
                pl_module.log("test/mAP_50_95", bbox_map, sync_dist=True, prog_bar=True)
                trainer.callback_metrics["test/mAP_50_95"] = torch.tensor(bbox_map)
                
                pl_module.log("test/segm_mAP_50_95", segm_map, sync_dist=True)
                trainer.callback_metrics["test/segm_mAP_50_95"] = torch.tensor(segm_map)
            
            if len(self.test_outputs_ema[dl_idx]) > 0:
                self._visualize_predictions(trainer, pl_module, self.test_outputs_ema[dl_idx], coco_gt, "test", dl_name + "_ema")
                print(f"\n[MotifCocoEvalCallback] Computing Test EMA Metrics for {dl_name}...")
                metrics_bbox_ema, metrics_segm_ema = self._compute_and_log(trainer, pl_module, self.test_outputs_ema[dl_idx], coco_gt, prefix, "_ema")
                
                # Add EMA to Markdown Report
                if "_markdown_table" in metrics_bbox_ema:
                    md_report += f"**Metric Type:** EMA Weights - Bounding Box (BBOX)\n\n"
                    md_report += metrics_bbox_ema["_markdown_table"] + "\n\n"
                if "_markdown_table" in metrics_segm_ema:
                    md_report += f"**Metric Type:** EMA Weights - Segmentation (SEGM)\n\n"
                    md_report += metrics_segm_ema["_markdown_table"] + "\n\n"
                
                ema_bbox = metrics_bbox_ema.get("detailed_bbox_map", 0.0)
                ema_segm = metrics_segm_ema.get("detailed_segm_map", 0.0)
                test_group_scores[group]["ema_mAP"].append(ema_bbox)
                if "detailed_segm_map" in metrics_segm_ema:
                    test_group_scores[group]["ema_segm"].append(ema_segm)
                
                pl_module.log(f"{prefix}/ema_mAP_50_95", ema_bbox, sync_dist=True)
                trainer.callback_metrics[f"{prefix}/ema_mAP_50_95"] = torch.tensor(ema_bbox)
                
                pl_module.log(f"{prefix}/ema_segm_mAP_50_95", ema_segm, sync_dist=True)
                trainer.callback_metrics[f"{prefix}/ema_segm_mAP_50_95"] = torch.tensor(ema_segm)
                
                if dl_idx == 0:
                    pl_module.log("test/ema_mAP_50_95", ema_bbox, sync_dist=True, prog_bar=True)
                    trainer.callback_metrics["test/ema_mAP_50_95"] = torch.tensor(ema_bbox)
                    
                    pl_module.log("test/ema_segm_mAP_50_95", ema_segm, sync_dist=True)
                    trainer.callback_metrics["test/ema_segm_mAP_50_95"] = torch.tensor(ema_segm)
            elif ema_cb is not None:
                pl_module.log(f"{prefix}/ema_mAP_50_95", bbox_map, sync_dist=True)
                trainer.callback_metrics[f"{prefix}/ema_mAP_50_95"] = torch.tensor(bbox_map)
                
                pl_module.log(f"{prefix}/ema_segm_mAP_50_95", segm_map, sync_dist=True)
                trainer.callback_metrics[f"{prefix}/ema_segm_mAP_50_95"] = torch.tensor(segm_map)
                
                if dl_idx == 0:
                    pl_module.log("test/ema_mAP_50_95", bbox_map, sync_dist=True, prog_bar=True)
                    trainer.callback_metrics["test/ema_mAP_50_95"] = torch.tensor(bbox_map)
                    
                    pl_module.log("test/ema_segm_mAP_50_95", segm_map, sync_dist=True)
                    trainer.callback_metrics["test/ema_segm_mAP_50_95"] = torch.tensor(segm_map)
                    
        for group, scores in test_group_scores.items():
            if len(scores["mAP"]) > 0:
                pl_module.log(f"test/best_{group}_mAP_50_95", max(scores["mAP"]), sync_dist=True)
                pl_module.log(f"test/avg_{group}_mAP_50_95", sum(scores["mAP"]) / len(scores["mAP"]), sync_dist=True)
            if len(scores["segm"]) > 0:
                pl_module.log(f"test/best_{group}_segm_mAP_50_95", max(scores["segm"]), sync_dist=True)
                pl_module.log(f"test/avg_{group}_segm_mAP_50_95", sum(scores["segm"]) / len(scores["segm"]), sync_dist=True)
            if len(scores["ema_mAP"]) > 0:
                pl_module.log(f"test/best_{group}_ema_mAP_50_95", max(scores["ema_mAP"]), sync_dist=True)
                pl_module.log(f"test/avg_{group}_ema_mAP_50_95", sum(scores["ema_mAP"]) / len(scores["ema_mAP"]), sync_dist=True)
            if len(scores["ema_segm"]) > 0:
                pl_module.log(f"test/best_{group}_ema_segm_mAP_50_95", max(scores["ema_segm"]), sync_dist=True)
                pl_module.log(f"test/avg_{group}_ema_segm_mAP_50_95", sum(scores["ema_segm"]) / len(scores["ema_segm"]), sync_dist=True)

        if trainer.is_global_zero:
            import os
            
            out_dir = getattr(pl_module.config.checkpointing, "save_dir", "output")
            if hasattr(pl_module, "train_config") and hasattr(pl_module.train_config, "output_dir"):
                out_dir = pl_module.train_config.output_dir
            elif hasattr(pl_module, "config") and hasattr(pl_module.config, "run_name"):
                out_dir = os.path.join(out_dir, "phase2", pl_module.config.run_name)
            
            os.makedirs(out_dir, exist_ok=True)
            report_path = os.path.join(out_dir, "inference_summary_report.md")
            
            with open(report_path, "w") as f:
                f.write(md_report)
                
            print("\n" + "="*80)
            print(md_report)
            print("="*80 + "\n")
            print(f"[MotifCocoEvalCallback] Saved Markdown report to: {report_path}")
            
            # Log to W&B
            if hasattr(trainer, "logger") and getattr(trainer.logger, "experiment", None):
                try:
                    import wandb
                    import markdown
                    html = markdown.markdown(md_report, extensions=["tables"])
                    trainer.logger.experiment.log({"Inference Summary Report": wandb.Html(html)})
                    print("[MotifCocoEvalCallback] Logged report to W&B")
                except ImportError:
                    print("[MotifCocoEvalCallback] Could not log report to W&B: `markdown` package not installed.")

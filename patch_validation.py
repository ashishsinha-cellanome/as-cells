import re

with open("models/deim_v2_lightning_module.py", "r") as f:
    content = f.read()

replacement = """    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        samples, targets = batch
        batch_size = int(samples.shape[0]) if isinstance(samples, torch.Tensor) else len(targets)
        samples = samples.to(self.device)
        targets = self._move_targets(targets)

        outputs = self.model(samples)
        
        # skip computing validation loss since DEIMCriterion expects aux_outputs
        
        eval_mode = self.config.get("eval_inference", {}).get("mode", "whole")

        predictions, image_ids = self._collect_batch_predictions(outputs, targets)

        if eval_mode in ["whole", "both"]:
            self.validation_step_outputs.append({"predictions": predictions, "image_ids": image_ids})"""

content = re.sub(
    r"    @torch\.no_grad\(\)\n    def validation_step\(self, batch, batch_idx\):.*?self\.validation_step_outputs\.append\(\{\"predictions\": predictions, \"image_ids\": image_ids\}\)",
    replacement,
    content,
    flags=re.DOTALL,
)

with open("models/deim_v2_lightning_module.py", "w") as f:
    f.write(content)

print("deim_v2_lightning_module.py updated successfully!")

import torch
import torch.nn as nn
import copy

def to_cpu_device(data):
    if isinstance(data, dict):
        return {k: to_cpu_device(v) for k, v in data.items()}
    if isinstance(data, list):
        return [to_cpu_device(v) for v in data]
    if isinstance(data, torch.Tensor):
        return data.detach().cpu()
    return data

class ModelEma(nn.Module):
    def __init__(self, model, decay=0.9999):
        super().__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = copy.deepcopy(model)
        self.module.eval()
        for p in self.module.parameters():
            p.requires_grad = False
        self.decay = decay

    def train(self, mode=True):
        """Force the module and its children to stay in evaluation mode."""
        self.training = False
        for module in self.children():
            module.train(False)
        return self

    def _update(self, model, update_fn):
        with torch.no_grad():
            model_params = dict(model.named_parameters())
            ema_params = dict(self.module.named_parameters())
            for name, param in ema_params.items():
                if name in model_params:
                    param.copy_(update_fn(param, model_params[name]))

            model_buffers = dict(model.named_buffers())
            ema_buffers = dict(self.module.named_buffers())
            for name, buffer in ema_buffers.items():
                if name in model_buffers:
                    buffer.copy_(update_fn(buffer, model_buffers[name]))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)

import pytorch_lightning as pl

class RTDETREMACallback(pl.Callback):
    """
    Exponential Moving Average callback for RT-DETR.
    Manages a ModelEma instance and handles synchronization and checkpointing.

    Args:
        decay: EMA decay rate (default: 0.9999)
        warmup_steps: Number of training steps before EMA updates start (default: 0)
                     Early training has noisy gradients, warmup lets model stabilize first
    """
    def __init__(self, decay=0.9999, warmup_steps=0):
        super().__init__()
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.ema_model = None
        self.warmup_completed = False

    def on_fit_start(self, trainer, pl_module):
        """Initialize EMA model and sync weights at the start of training."""
        if self.ema_model is None:
            pl_module.print(f"[EMA Callback] Initializing EMA with decay={self.decay}")
            if self.warmup_steps > 0:
                pl_module.print(f"[EMA Callback] Warmup enabled: EMA updates will start after {self.warmup_steps} steps")
            self.ema_model = ModelEma(pl_module.model, decay=self.decay)

        # Always sync at start of fit to ensure valid state
        pl_module.print("[EMA Callback] Synchronizing EMA weights with model weights...")
        self.ema_model.set(pl_module.model)

        # Verification check
        with torch.no_grad():
            all_equal = all(torch.equal(p1, p2) for p1, p2 in zip(pl_module.model.parameters(), self.ema_model.module.parameters()))
            if all_equal:
                pl_module.print("[EMA Callback] Verified: Model and EMA weights are identical.")
            else:
                pl_module.print("[EMA Callback] WARNING: Model and EMA weights differ after synchronization!")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Update EMA weights after each training step (after warmup period)."""
        if not self.ema_model:
            return

        # Check if warmup period has passed
        if trainer.global_step < self.warmup_steps:
            return  # Skip EMA update during warmup

        # Log when warmup completes (once)
        if not self.warmup_completed and trainer.global_step >= self.warmup_steps:
            self.warmup_completed = True
            if self.warmup_steps > 0:
                pl_module.print(f"[EMA Callback] Warmup completed at step {trainer.global_step}. Starting EMA updates...")

            # Verify EMA update is working by checking weight divergence after first update
            if trainer.global_step == self.warmup_steps:
                # Store a copy of first param before update
                first_param_before = next(self.ema_model.module.parameters()).clone()
                self.ema_model.update(pl_module.model)
                first_param_after = next(self.ema_model.module.parameters())

                if torch.allclose(first_param_before, first_param_after, atol=1e-9):
                    pl_module.print(f"⚠️  [EMA] WARNING: EMA weights did NOT change after update! Check implementation.")
                else:
                    diff = (first_param_after - first_param_before).abs().max().item()
                    pl_module.print(f"✓ [EMA] First update successful. Max param change: {diff:.2e}")
                return  # Already updated above

        # Update EMA model
        self.ema_model.update(pl_module.model)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        """Save EMA state into the main checkpoint."""
        if self.ema_model:
            # Save the internal module's state_dict
            checkpoint['ema_state_dict'] = self.ema_model.module.state_dict()

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        """Restore EMA state from the checkpoint."""
        if 'ema_state_dict' in checkpoint:
            if self.ema_model is None:
                self.ema_model = ModelEma(pl_module.model, decay=self.decay)
            self.ema_model.module.load_state_dict(checkpoint['ema_state_dict'])
            pl_module.print("[EMA Callback] Restored EMA state from checkpoint.")
import torch
import torch.nn as nn
import copy
import math


def to_cpu_device(data):
    if isinstance(data, dict):
        return {k: to_cpu_device(v) for k, v in data.items()}
    if isinstance(data, list):
        return [to_cpu_device(v) for v in data]
    if isinstance(data, torch.Tensor):
        return data.detach().cpu()
    return data


class ModelEma(nn.Module):
    def __init__(self, model, decay=0.9999, tau=None):
        super().__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = copy.deepcopy(model)
        self.module.eval()
        for p in self.module.parameters():
            p.requires_grad = False
        self.decay = decay
        self.tau = tau
        self.updates = 0

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
        self.updates += 1
        # Dynamic decay: starts fast (low d) and ramps up to self.decay (e.g. 0.9999)
        # Using exponential ramp-up as in official YOLOv5
        if self.tau is not None and self.tau > 0:
            d = self.decay * (1 - math.exp(-self.updates / self.tau))
        else:
            d = self.decay
        self._update(model, update_fn=lambda e, m: d * e + (1.0 - d) * m)
        return d

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


import pytorch_lightning as pl


class EMACallback(pl.Callback):
    """
    Exponential Moving Average callback for object detection models (RT-DETR, RF-DETR, YOLOv5).
    Manages a ModelEma instance and handles synchronization and checkpointing.

    Args:
        decay: Target EMA decay rate (default: 0.9999)
        warmup_steps: Number of training steps before EMA updates start (default: 0)
        tau: Decay ramp-up tau (default: None). Set to e.g. 2000 to enable YOLO-style ramp-up.
    """

    def __init__(self, decay=0.9999, warmup_steps=0, tau=None):
        super().__init__()
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.tau = tau
        self.ema_model = None
        self.warmup_completed = False

    def on_fit_start(self, trainer, pl_module):
        """Initialize EMA model and sync weights at the start of training."""
        if self.ema_model is None:
            pl_module.print(
                f"[EMA Callback] Initializing EMA with decay={self.decay}, tau={self.tau}"
            )
            if self.warmup_steps > 0:
                pl_module.print(
                    f"[EMA Callback] Warmup enabled: EMA updates will start after {self.warmup_steps} steps"
                )
            self.ema_model = ModelEma(pl_module.model, decay=self.decay, tau=self.tau)

        # Ensure EMA model is on the correct device
        self.ema_model.to(pl_module.device)

        # Always sync at start of fit to ensure valid state
        pl_module.print(
            "[EMA Callback] Synchronizing EMA weights with model weights..."
        )
        self.ema_model.set(pl_module.model)

        # Verification check
        with torch.no_grad():
            all_equal = all(
                torch.equal(p1, p2)
                for p1, p2 in zip(
                    pl_module.model.parameters(), self.ema_model.module.parameters()
                )
            )
            if all_equal:
                pl_module.print(
                    "[EMA Callback] Verified: Model and EMA weights are identical."
                )
            else:
                pl_module.print(
                    "[EMA Callback] WARNING: Model and EMA weights differ after synchronization!"
                )

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
                pl_module.print(
                    f"[EMA Callback] Warmup completed at step {trainer.global_step}. Starting EMA updates..."
                )

            # Verify EMA update is working by checking weight divergence after first update
            if trainer.global_step == self.warmup_steps:
                # Count parameters
                total_params = 0
                trainable_params = 0
                frozen_params = 0

                param_to_check = None
                param_name = "unknown"

                for name, p in pl_module.model.named_parameters():
                    total_params += 1
                    if p.requires_grad:
                        trainable_params += 1
                        if param_to_check is None:
                            # Use the first trainable parameter for verification
                            # Find corresponding EMA param
                            for (
                                ema_name,
                                ema_p,
                            ) in self.ema_model.module.named_parameters():
                                if ema_name == name:
                                    param_to_check = ema_p
                                    param_name = name
                                    break
                    else:
                        frozen_params += 1

                pl_module.print(
                    f"[EMA Check] Model status: {total_params} parameters total. {frozen_params} frozen (ignored), {trainable_params} trainable."
                )

                if param_to_check is None:
                    # Fallback if no trainable params found (unlikely)
                    param_to_check = next(self.ema_model.module.parameters())
                    param_name = "first_param (fallback)"
                    pl_module.print(
                        f"[EMA Check] WARNING: No trainable parameters found! Checking '{param_name}'."
                    )

                # Store a copy before update
                param_before = param_to_check.clone()
                self.ema_model.update(pl_module.model)

                # Get the same param after update
                # (We need to re-fetch it because update() might modify it in-place or replace it)
                param_after = None
                for ema_name, ema_p in self.ema_model.module.named_parameters():
                    if ema_name == param_name:
                        param_after = ema_p
                        break

                if param_after is None:
                    param_after = next(
                        self.ema_model.module.parameters()
                    )  # Should not happen

                if torch.allclose(param_before, param_after, atol=1e-9):
                    pl_module.print(
                        f"[EMA] WARNING: EMA weights ({param_name}) did NOT change after update! Check implementation."
                    )
                else:
                    diff = (param_after - param_before).abs().max().item()
                    pl_module.print(
                        f"[EMA Check] First update successful. Verified on 1 trainable parameter (max change: {diff:.2e})."
                    )
                return  # Already updated above

        # Update EMA model
        d = self.ema_model.update(pl_module.model)

        # Occasionally log the current decay value (every 100 steps)
        if trainer.global_step % 100 == 0:
            pl_module.log("ema/decay", d)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        """Save EMA state into the main checkpoint."""
        if self.ema_model:
            # Save the internal module's state_dict
            checkpoint["ema_state_dict"] = self.ema_model.module.state_dict()

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        """Restore EMA state from the checkpoint."""
        if "ema_state_dict" in checkpoint:
            if self.ema_model is None:
                self.ema_model = ModelEma(
                    pl_module.model, decay=self.decay, tau=self.tau
                )
            self.ema_model.module.load_state_dict(
                checkpoint["ema_state_dict"], strict=False
            )
            self.ema_model.to(pl_module.device)

            precision_mode = str(getattr(trainer, "precision", "")).lower()
            target_dtype = None
            if precision_mode in {"16-true", "16"}:
                target_dtype = torch.float16
            elif precision_mode in {"bf16-true", "bf16"}:
                target_dtype = torch.bfloat16

            if target_dtype is not None:
                self.ema_model.module.to(dtype=target_dtype)
            pl_module.print("[EMA Callback] Restored EMA state from checkpoint.")

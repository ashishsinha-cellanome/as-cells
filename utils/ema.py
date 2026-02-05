import copy
from collections import OrderedDict
from torch import nn
import torch

class ModelEma(nn.Module):
    """
    Model Exponential Moving Average from Taming Transformers
    """
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
            # Use named_parameters and named_buffers to ensure we update by key matching
            # this handles potential shared parameters correctly and is safer than zip(values)
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
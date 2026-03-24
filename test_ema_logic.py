import torch
from torch import nn
import copy


# Mock ModelEma class as in utils/ema.py
class ModelEma(nn.Module):
    def __init__(self, model, decay=0.9999):
        super().__init__()
        self.module = copy.deepcopy(model)
        self.module.eval()
        self.decay = decay

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(
                self.module.state_dict().values(), model.state_dict().values()
            ):
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(
            model, update_fn=lambda e, m: self.decay * e + (1.0 - self.decay) * m
        )


# Test model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 10, 3)
        self.bn = nn.BatchNorm2d(10)


def test_ema():
    model = SimpleModel()
    ema = ModelEma(model, decay=0.9)

    # Check initial weights are same
    print(
        "Initial weight equality:",
        torch.allclose(model.conv.weight, ema.module.conv.weight),
    )

    # Modify model weights
    with torch.no_grad():
        model.conv.weight.add_(1.0)

    # Update EMA
    ema.update(model)

    # Check if EMA changed
    print(
        "EMA weight changed after update:",
        not torch.allclose(ema.module.conv.weight, model.conv.weight),
    )
    print("EMA weight contains added value:", torch.any(ema.module.conv.weight > 0.05))

    # Check if direct modification of state_dict value updates parameter
    sd = ema.module.state_dict()
    val = sd["conv.weight"]
    val.add_(5.0)
    print(
        "Direct state_dict modification updates parameter:",
        torch.allclose(ema.module.conv.weight, val),
    )


if __name__ == "__main__":
    test_ema()

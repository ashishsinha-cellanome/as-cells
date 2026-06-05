from utils.peft_utils import apply_peft
import torch
import torch.nn as nn

class DummyYolo(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.Conv2d(16, 32, 3),
            nn.Linear(32, 10) # dummy detect
        )

m = DummyYolo()
for name, param in m.named_parameters():
    print(name)

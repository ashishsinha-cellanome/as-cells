import os
from transformers import Dinov2Config
from huggingface_hub import constants

print("HF cache dir before:", constants.HF_HUB_CACHE)
os.environ["HF_HOME"] = "/tmp/my_fake_cache"
print("HF cache dir after setting env:", constants.HF_HUB_CACHE)

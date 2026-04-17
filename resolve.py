import re

files = [
    "COCO Evaluation/mask2former_model.py",
    "Inference/mask2former_model.py",
    "test_dinov2_resolutions.py",
]

for f in files:
    with open(f, "r") as file:
        content = file.read()
    
    # We keep HEAD for formatting where appropriate, but for test_dinov2_resolutions we keep ashish.
    if "test_dinov2" in f:
        # Keep ashish
        content = re.sub(r'<<<<<<< HEAD.*?=======\n(.*?)\n>>>>>>> ashish', r'\1', content, flags=re.DOTALL)
    else:
        # Keep HEAD for formatting
        content = re.sub(r'<<<<<<< HEAD\n(.*?)\n=======\n.*?>>>>>>> ashish', r'\1', content, flags=re.DOTALL)

    with open(f, "w") as file:
        file.write(content)


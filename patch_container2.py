import re

with open("DEIMv2/engine/data/transforms/container.py", "r") as f:
    content = f.read()

replacement = """
    def _apply_transform(self, transform, sample):
        import torchvision.transforms.v2 as T
        # Check if the transform is derived from torchvision v2 Transform
        if isinstance(transform, T.Transform) and type(transform).__name__ not in ("Mosaic", "EmptyTransform"):
            if isinstance(sample, (tuple, list)):
                res = transform(*sample[:2])
                if isinstance(res, (tuple, list)):
                    return tuple(res) + tuple(sample[2:])
                return (res,) + tuple(sample[2:])
            return transform(sample)
        else:
            if isinstance(sample, (tuple, list)):
                res = transform(*sample)
                if isinstance(res, (tuple, list)) and len(res) == 2 and len(sample) > 2:
                    return tuple(res) + tuple(sample[2:])
                elif not isinstance(res, (tuple, list)):
                    return (res,)
                return res
            return transform(sample)
"""

content = re.sub(
    r"    def _apply_transform.*?return transform\(sample\)",
    replacement,
    content,
    flags=re.DOTALL,
)

with open("DEIMv2/engine/data/transforms/container.py", "w") as f:
    f.write(content)

print("container.py updated successfully!")

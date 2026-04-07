import re

with open("DEIMv2/engine/data/transforms/container.py", "r") as f:
    content = f.read()

replacement = """
    def _apply_transform(self, transform, sample):
        if type(transform).__module__.startswith("torchvision"):
            if isinstance(sample, (tuple, list)):
                res = transform(*sample[:2])
                if isinstance(res, (tuple, list)):
                    return tuple(res) + tuple(sample[2:])
                return (res,) + tuple(sample[2:])
            return transform(sample)
        else:
            if isinstance(sample, (tuple, list)):
                res = transform(*sample)
                # Some custom transforms might return just (img, target)
                # But typically they return the full tuple if they received it
                if isinstance(res, (tuple, list)) and len(res) == 2 and len(sample) > 2:
                    return tuple(res) + tuple(sample[2:])
                elif not isinstance(res, (tuple, list)):
                    return (res,)
                return res
            return transform(sample)

    def default_forward(self, *inputs: Any) -> Any:
        sample = inputs if len(inputs) > 1 else inputs[0]
        for transform in self.transforms:
            sample = self._apply_transform(transform, sample)
        return sample

    def stop_epoch_forward(self, *inputs: Any):
        sample = inputs if len(inputs) > 1 else inputs[0]
        dataset = sample[-1]
        cur_epoch = dataset.epoch
        policy_ops = self.policy["ops"]
        policy_epoch = self.policy["epoch"]

        if isinstance(policy_epoch, list) and len(policy_epoch) == 3:  # 4-stages
            if policy_epoch[0] <= cur_epoch < policy_epoch[1]:
                with_mosaic = (
                    random.random() <= self.mosaic_prob
                )  # Probility for Mosaic
            else:
                with_mosaic = False
            for transform in self.transforms:
                if (
                    type(transform).__name__ in policy_ops
                    and cur_epoch < policy_epoch[0]
                ):  # first stage: NoAug
                    pass
                elif (
                    type(transform).__name__ in policy_ops
                    and cur_epoch >= policy_epoch[-1]
                ):  # last stage: NoAug
                    pass
                else:
                    # Using Mosaic for [policy_epoch[0], policy_epoch[1]] with probability
                    if type(transform).__name__ == "Mosaic" and not with_mosaic:
                        pass
                    # Mosaic and Zoomout/IoUCrop can not be co-existed in the same sample
                    elif (
                        type(transform).__name__ == "RandomZoomOut"
                        or type(transform).__name__ == "RandomIoUCrop"
                    ) and with_mosaic:
                        pass
                    else:
                        sample = self._apply_transform(transform, sample)
        else:  # the default data scheduler
            for transform in self.transforms:
                if type(transform).__name__ in policy_ops and cur_epoch >= policy_epoch:
                    pass
                else:
                    sample = self._apply_transform(transform, sample)

        return sample

    def stop_sample_forward(self, *inputs: Any):
        sample = inputs if len(inputs) > 1 else inputs[0]
        dataset = sample[-1]

        cur_epoch = dataset.epoch
        policy_ops = self.policy["ops"]
        policy_sample = self.policy["sample"]

        for transform in self.transforms:
            if (
                type(transform).__name__ in policy_ops
                and self.global_samples >= policy_sample
            ):
                pass
            else:
                sample = self._apply_transform(transform, sample)

        self.global_samples += 1

        return sample
"""

content = re.sub(
    r"    def default_forward.*?return sample", replacement, content, flags=re.DOTALL
)

with open("DEIMv2/engine/data/transforms/container.py", "w") as f:
    f.write(content)

print("container.py updated successfully!")

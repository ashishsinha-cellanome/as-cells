import torch
from transformers import RTDetrForObjectDetection

try:
    print("Loading with 300 queries...")
    m1 = RTDetrForObjectDetection.from_pretrained(
        "PekingU/rtdetr_r50vd", num_queries=300, ignore_mismatched_sizes=True
    )
    p1 = {n: p.shape for n, p in m1.named_parameters()}

    print("Loading with 600 queries...")
    m2 = RTDetrForObjectDetection.from_pretrained(
        "PekingU/rtdetr_r50vd", num_queries=600, ignore_mismatched_sizes=True
    )
    p2 = {n: p.shape for n, p in m2.named_parameters()}

    diff = []
    for n in p1:
        if p1[n] != p2[n]:
            diff.append(n)

    print("\nParameters that change shape based on num_queries:")
    if not diff:
        print("NONE! No parameters change shape based on num_queries.")
        print(
            "This means num_queries only affects the size of dynamically generated tensors during the forward pass."
        )
    else:
        for n in diff:
            print(f"{n}: {p1[n]} -> {p2[n]}")

except Exception as e:
    print(f"Error: {e}")

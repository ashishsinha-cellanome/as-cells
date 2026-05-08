import torch
import time
import numpy as np
from rfdetr import RFDETRSegLarge
from omegaconf import OmegaConf
import argparse
import tabulate

def run_benchmark(inner_model, postprocess_fn, dummy_input, orig_sizes, batch_size, num_runs=100, label=""):
    print(f"--- Benchmarking {label} ---")
    print("Warming up...")
    with torch.no_grad():
        for _ in range(20):
            outputs = inner_model(dummy_input)
            _ = postprocess_fn(outputs, orig_sizes)
            
    torch.cuda.synchronize()
    print("Benchmarking Forward Pass + Post-processing...")
    
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_runs)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_runs)]
    
    with torch.no_grad():
        for i in range(num_runs):
            start_events[i].record()
            outputs = inner_model(dummy_input)
            _ = postprocess_fn(outputs, orig_sizes)
            end_events[i].record()
            
    torch.cuda.synchronize()
    
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)] # in milliseconds
    
    avg_time_ms = np.mean(times)
    p90_time_ms = np.percentile(times, 90)
    time_per_image_ms = avg_time_ms / batch_size
    fps = 1000.0 / time_per_image_ms
    
    print(f"  Avg time per batch:  {avg_time_ms:.2f} ms")
    print(f"  Avg time per image:  {time_per_image_ms:.2f} ms")
    print(f"  Throughput:          {fps:.2f} images/s")
    print("-" * 50)
    return {
        "Mode": label,
        "Batch Size": batch_size,
        "Time/Batch (ms)": f"{avg_time_ms:.2f}",
        "Time/Image (ms)": f"{time_per_image_ms:.2f}",
        "Throughput (FPS)": f"{fps:.2f}"
    }

def benchmark(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Benchmarking on {device}")
    
    print("Loading model...")
    model = RFDETRSegLarge(group_detr=1, compile=False, num_classes=4)
    
    ckpt_path = "output/checkpoint_best_regular.pth"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt.get("model", ckpt))
    model.model.model.load_state_dict(state_dict, strict=False)
    
    batch_size = args.batch_size
    num_runs = args.num_runs
    print(f"Batch size: {batch_size}, Num runs: {num_runs}")
    
    dummy_input = torch.randn(batch_size, 3, 504, 504, device=device)
    orig_sizes = torch.tensor([[504, 504] for _ in range(batch_size)], device=device)
    
    postprocess_fn = model.model.postprocess
    
    results = []
    
    if args.compile_mode in ["none", "all"]:
        # 1. Baseline
        inner_model = model.model.model
        inner_model.to(device)
        inner_model.eval()
        if torch.cuda.is_available():
            inner_model = inner_model.bfloat16()
            dummy_input = dummy_input.bfloat16()
            
        print("Computing FLOPS...")
        try:
            from fvcore.nn import FlopCountAnalysis, flop_count_table
            flops = FlopCountAnalysis(inner_model, dummy_input)
            gflops = flops.total() / 1e9
            print(flop_count_table(flops))
            print(f"Total GFLOPS (Batch Size {batch_size}): {gflops:.2f}")
            print(f"Total GFLOPS per image: {gflops / batch_size:.2f}")
        except Exception as e:
            print(f"Failed to compute FLOPS: {e}")
            
        res = run_benchmark(inner_model, postprocess_fn, dummy_input, orig_sizes, batch_size, num_runs=num_runs, label="Baseline (bfloat16)")
        results.append(res)

    if args.compile_mode in ["compile", "all"]:
        # 3. Torch Compile (reduce-overhead)
        print("Testing torch.compile (reduce-overhead)...")
        model = RFDETRSegLarge(group_detr=1, compile=False, num_classes=4)
        model.model.model.load_state_dict(state_dict, strict=False)
        inner_model = model.model.model.to(device).eval()
        postprocess_fn = model.model.postprocess
        if torch.cuda.is_available():
            inner_model = inner_model.bfloat16()
            dummy_input = dummy_input.bfloat16()
            
        compiled_model = torch.compile(inner_model, mode="reduce-overhead")
        print("Compiling model (this may take a few minutes)...")
        res = run_benchmark(compiled_model, postprocess_fn, dummy_input, orig_sizes, batch_size, num_runs=num_runs, label="torch.compile")
        results.append(res)

    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    print(tabulate.tabulate(results, headers="keys", tablefmt="pretty"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Inference")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for benchmark")
    parser.add_argument("--num-runs", type=int, default=100, help="Number of forward passes")
    parser.add_argument("--compile-mode", type=str, choices=["none", "compile", "all"], default="all", help="Which optimization to test")
    args = parser.parse_args()
    benchmark(args)

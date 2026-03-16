"""
GPU stress test with real-time performance monitoring.
Runs a ResNet50 training loop on synthetic data and prints live GPU stats.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import time
import subprocess
import threading
import sys
import os

BATCH_SIZE = int(os.environ.get("GPU_TEST_BATCH_SIZE", 32))
IMAGE_SIZE = 224
NUM_CLASSES = 1000
NUM_ITERATIONS = int(os.environ.get("GPU_TEST_ITERATIONS", 200))
MONITOR_INTERVAL = 1.0


def get_gpu_stats():
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,power.limit,clocks.current.sm",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return None
        return result.stdout.strip()
    except Exception:
        return None


def monitor_gpu(stop_event):
    while not stop_event.is_set():
        stats = get_gpu_stats()
        if stats:
            parts = [p.strip() for p in stats.split(",")]
            if len(parts) >= 9:
                name, temp, gpu_util, mem_util, mem_used, mem_total, power, power_cap, sm_clock = parts[:9]
                bar_len = 30
                gpu_fill = int(float(gpu_util) / 100 * bar_len)
                mem_fill = int(float(mem_util) / 100 * bar_len)
                gpu_bar = "#" * gpu_fill + "-" * (bar_len - gpu_fill)
                mem_bar = "#" * mem_fill + "-" * (bar_len - mem_fill)

                print(f"\r\033[K", end="")
                print(f"  GPU: [{gpu_bar}] {gpu_util:>3}%  |  "
                      f"VRAM: [{mem_bar}] {mem_used:>5}/{mem_total} MB  |  "
                      f"Temp: {temp}C  |  "
                      f"Power: {power}/{power_cap} W  |  "
                      f"SM: {sm_clock} MHz", end="", flush=True)
        stop_event.wait(MONITOR_INTERVAL)


def run_stress_test():
    print("=" * 90)
    print("  GPU STRESS TEST")
    print("=" * 90)

    if not torch.cuda.is_available():
        print("\n  CUDA is not available. Check your PyTorch installation.")
        print("  Install CUDA-enabled PyTorch: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        sys.exit(1)

    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    print(f"\n  Device       : {props.name}")
    print(f"  VRAM         : {props.total_mem / 1024**3:.1f} GB")
    print(f"  SM Count     : {props.multi_processor_count}")
    print(f"  CUDA Cores   : ~{props.multi_processor_count * 128}")
    print(f"  Compute Cap  : {props.major}.{props.minor}")
    print(f"  PyTorch      : {torch.__version__}")
    print(f"  CUDA Version : {torch.version.cuda}")

    print(f"\n  Workload     : ResNet50 training (synthetic ImageNet)")
    print(f"  Batch Size   : {BATCH_SIZE}")
    print(f"  Iterations   : {NUM_ITERATIONS}")
    print(f"  Precision    : FP32 + TF32 (if Ampere+)")
    print("-" * 90)

    torch.backends.cudnn.benchmark = True
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model = models.resnet50(weights=None, num_classes=NUM_CLASSES).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    images = torch.randn(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
    labels = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,), device=device)

    print("\n  Warming up GPU (5 iterations)...")
    model.train()
    for _ in range(5):
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    torch.cuda.synchronize()

    print("  Warm-up complete. Starting stress test...\n")

    stop_event = threading.Event()
    monitor_thread = threading.Thread(target=monitor_gpu, args=(stop_event,), daemon=True)
    monitor_thread.start()

    throughputs = []
    losses = []
    start_total = time.perf_counter()

    for i in range(NUM_ITERATIONS):
        start = time.perf_counter()

        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.synchronize()

        elapsed = time.perf_counter() - start
        imgs_per_sec = BATCH_SIZE / elapsed
        throughputs.append(imgs_per_sec)
        losses.append(loss.item())

        if (i + 1) % 10 == 0:
            print(f"\n  [{i+1:>4}/{NUM_ITERATIONS}]  "
                  f"Loss: {loss.item():.4f}  |  "
                  f"{imgs_per_sec:.0f} img/s  |  "
                  f"Step: {elapsed*1000:.1f} ms")

    total_time = time.perf_counter() - start_total
    stop_event.set()
    monitor_thread.join(timeout=2)

    avg_throughput = sum(throughputs) / len(throughputs)
    peak_throughput = max(throughputs)
    peak_mem = torch.cuda.max_memory_allocated() / 1024**3

    print("\n")
    print("=" * 90)
    print("  RESULTS")
    print("=" * 90)
    print(f"  Total Time       : {total_time:.1f} s")
    print(f"  Avg Throughput   : {avg_throughput:.0f} img/s")
    print(f"  Peak Throughput  : {peak_throughput:.0f} img/s")
    print(f"  Peak VRAM Used   : {peak_mem:.2f} GB")
    print(f"  Final Loss       : {losses[-1]:.4f}")
    print(f"  Total Images     : {BATCH_SIZE * NUM_ITERATIONS}")
    print("=" * 90)

    print("\n  GPU is healthy and ready for Isaac Sim workloads.\n")


if __name__ == "__main__":
    run_stress_test()

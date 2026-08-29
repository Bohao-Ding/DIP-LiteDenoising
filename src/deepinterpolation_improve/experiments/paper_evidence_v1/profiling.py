from __future__ import annotations

import copy
import statistics
import time
from typing import Any

import torch
import torch.nn as nn

from .blocks import fuse_reparameterised_convolutions


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def conv_profile(
    model: nn.Module,
    input_shape: tuple[int, int, int],
    device: str | torch.device = "cpu",
) -> dict[str, int | float]:
    """Profile convolution MACs/FLOPs for one batch element.

    FLOPs are reported as two operations per multiply-accumulate.  Pooling,
    interpolation, softmax, elementwise operations and memory traffic are not
    included.
    """

    selected_device = torch.device(device)
    model = model.to(selected_device).eval()
    totals = {"macs": 0, "activation_elements": 0}
    handles = []

    def hook(module: nn.Conv2d, _inputs, output: torch.Tensor) -> None:
        batch, out_channels, out_height, out_width = output.shape
        kernel_height, kernel_width = module.kernel_size
        per_output = (module.in_channels // module.groups) * kernel_height * kernel_width
        totals["macs"] += int(batch * out_channels * out_height * out_width * per_output)
        totals["activation_elements"] += int(output.numel())

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(hook))
    try:
        with torch.inference_mode():
            model(torch.zeros((1, *input_shape), device=selected_device))
    finally:
        for handle in handles:
            handle.remove()
    return {
        "parameters": parameter_count(model),
        "conv_macs": totals["macs"],
        "conv_flops": 2 * totals["macs"],
        "conv_gflops": 2 * totals["macs"] / 1e9,
        "conv_activation_elements": totals["activation_elements"],
    }


def inference_copy(model: nn.Module) -> nn.Module:
    deployed = copy.deepcopy(model).eval()
    return fuse_reparameterised_convolutions(deployed)


def benchmark_latency(
    model: nn.Module,
    input_shape: tuple[int, int, int],
    device: str,
    batch_size: int,
    warmup: int,
    repeats: int,
    precision: str = "fp32",
    channels_last: bool = False,
) -> dict[str, Any]:
    if warmup < 1 or repeats < 2:
        raise ValueError("Latency benchmark requires warmup >= 1 and repeats >= 2")
    selected_device = torch.device(device)
    dtype_by_precision = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    if precision not in dtype_by_precision:
        raise ValueError(f"Unsupported benchmark precision: {precision}")
    if selected_device.type == "cpu" and precision == "fp16":
        raise ValueError("FP16 latency benchmarking requires CUDA")
    dtype = dtype_by_precision[precision]
    deployed = inference_copy(model).to(device=selected_device, dtype=dtype).eval()
    inputs = torch.randn((int(batch_size), *input_shape), device=selected_device, dtype=dtype)
    if channels_last:
        deployed = deployed.to(memory_format=torch.channels_last)
        inputs = inputs.contiguous(memory_format=torch.channels_last)

    def synchronise() -> None:
        if selected_device.type == "cuda":
            torch.cuda.synchronize(selected_device)

    with torch.inference_mode():
        for _ in range(int(warmup)):
            deployed(inputs)
        synchronise()
        if selected_device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(selected_device)
            resident_memory_mb = float(torch.cuda.memory_allocated(selected_device) / (1024**2))
        else:
            resident_memory_mb = 0.0
        samples: list[float] = []
        for _ in range(int(repeats)):
            start = time.perf_counter()
            deployed(inputs)
            synchronise()
            samples.append((time.perf_counter() - start) * 1000.0)
    ordered = sorted(samples)
    p95_index = min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))
    peak_allocated_memory_mb = (
        float(torch.cuda.max_memory_allocated(selected_device) / (1024**2))
        if selected_device.type == "cuda"
        else 0.0
    )
    return {
        "device": str(selected_device),
        "device_name": torch.cuda.get_device_name(selected_device) if selected_device.type == "cuda" else "CPU",
        "batch_size": int(batch_size),
        "warmup": int(warmup),
        "repeats": int(repeats),
        "precision": precision,
        "channels_last": bool(channels_last),
        "latency_ms_p50": statistics.median(samples),
        "latency_ms_p95": ordered[p95_index],
        "latency_ms_mean": statistics.fmean(samples),
        "throughput_frames_per_second": int(batch_size) * 1000.0 / statistics.fmean(samples),
        "resident_memory_mb": resident_memory_mb,
        "incremental_peak_memory_mb": max(0.0, peak_allocated_memory_mb - resident_memory_mb),
        "peak_allocated_memory_mb": peak_allocated_memory_mb,
        "memory_scope": "isolated deployed model and benchmark input",
    }

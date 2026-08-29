from __future__ import annotations

import gc
import statistics
import subprocess
from pathlib import Path
from typing import Any

import torch

from .models import create_variant
from .profiling import benchmark_latency
from .study import ExperimentTask


def gpu_telemetry() -> dict[str, float | str | None]:
    command = [
        "nvidia-smi",
        "--query-gpu=temperature.gpu,power.draw,clocks.sm,clocks.mem,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=10)
        values = [value.strip() for value in completed.stdout.splitlines()[0].split(",")]
        if len(values) != 5:
            raise ValueError("unexpected nvidia-smi field count")
        return {
            "temperature_c": float(values[0]),
            "power_w": float(values[1]),
            "sm_clock_mhz": float(values[2]),
            "memory_clock_mhz": float(values[3]),
            "utilisation_percent": float(values[4]),
        }
    except (OSError, subprocess.SubprocessError, ValueError, IndexError) as exc:
        return {"telemetry_error": str(exc)}


def balanced_orders(labels: list[str], rounds: int) -> list[list[str]]:
    if not labels:
        raise ValueError("At least one model is required")
    if rounds < 1:
        raise ValueError("rounds must be >= 1")
    if rounds % len(labels) != 0:
        raise ValueError(
            "A cyclic order-balanced audit requires rounds to be a multiple "
            "of the number of models"
        )
    return [labels[offset % len(labels) :] + labels[: offset % len(labels)] for offset in range(rounds)]


def aggregate_latency_records(records: list[dict[str, Any]], baseline: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record["variant"]), []).append(record)
    if baseline not in grouped:
        raise ValueError(f"Missing latency baseline: {baseline}")

    output = []
    for variant, rows in sorted(grouped.items()):
        p50 = [float(row["latency_ms_p50"]) for row in rows]
        p95 = [float(row["latency_ms_p95"]) for row in rows]
        output.append(
            {
                "variant": variant,
                "rounds": len(rows),
                "median_of_round_p50_ms": statistics.median(p50),
                "sd_of_round_p50_ms": statistics.stdev(p50) if len(p50) > 1 else 0.0,
                "min_round_p50_ms": min(p50),
                "max_round_p50_ms": max(p50),
                "median_of_round_p95_ms": statistics.median(p95),
            }
        )
    baseline_p50 = next(
        row["median_of_round_p50_ms"] for row in output if row["variant"] == baseline
    )
    for row in output:
        row["p50_ratio_to_baseline"] = row["median_of_round_p50_ms"] / baseline_p50
    return output


def run_latency_audit(
    tasks: list[ExperimentTask],
    *,
    device: str = "cuda",
    rounds: int = 5,
    baseline: str = "baseline_b8d3",
    protocol_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if torch.device(device).type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("The interleaved FP16 latency audit requires CUDA")
    by_variant = {task.variant: task for task in tasks}
    if len(by_variant) != len(tasks):
        raise ValueError("Latency audit requires one task per model")
    labels = list(by_variant)
    if baseline not in by_variant:
        raise ValueError(f"Missing latency baseline: {baseline}")
    shapes = {(task.input_channels, *task.frame_shape) for task in tasks}
    if len(shapes) != 1:
        raise ValueError("All latency-audit tasks must share one input shape")
    input_shape = next(iter(shapes))

    models = {}
    for variant, task in by_variant.items():
        best_path = Path(task.output_dir) / "best.pt"
        if not best_path.is_file():
            raise FileNotFoundError(best_path)
        checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
        model = create_variant(variant, task.input_channels)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        models[variant] = model.eval()
        del checkpoint

    protocol = dict(protocol_override) if protocol_override is not None else dict(tasks[0].benchmark)
    if not bool(protocol.get("enabled", False)):
        raise ValueError("Latency benchmarking is disabled in the stage config")
    if int(protocol.get("batch_size", 0)) < 1:
        raise ValueError("Latency audit batch_size must be >= 1")
    if int(protocol.get("warmup", 0)) < 1 or int(protocol.get("repeats", 0)) < 2:
        raise ValueError("Latency audit requires warmup >= 1 and repeats >= 2")
    if str(protocol.get("precision", "")) not in {"fp16", "fp32", "bf16"}:
        raise ValueError("Latency audit precision must be fp16, fp32 or bf16")
    records = []
    orders = balanced_orders(labels, rounds)
    for round_index, order in enumerate(orders, start=1):
        for position, variant in enumerate(order, start=1):
            gc.collect()
            torch.cuda.empty_cache()
            telemetry_before = gpu_telemetry()
            measurement = benchmark_latency(
                models[variant],
                input_shape,
                device,
                int(protocol["batch_size"]),
                int(protocol["warmup"]),
                int(protocol["repeats"]),
                str(protocol["precision"]),
                bool(protocol["channels_last"]),
            )
            telemetry_after = gpu_telemetry()
            records.append(
                {
                    "round": round_index,
                    "position": position,
                    "variant": variant,
                    "telemetry_before": telemetry_before,
                    "telemetry_after": telemetry_after,
                    **measurement,
                }
            )
    return {
        "schema_version": 1,
        "method": "cyclic order-balanced checkpoint latency audit",
        "baseline": baseline,
        "rounds": rounds,
        "orders": orders,
        "protocol": protocol,
        "software": {
            "torch_version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
        },
        "records": records,
        "aggregates": aggregate_latency_records(records, baseline),
    }

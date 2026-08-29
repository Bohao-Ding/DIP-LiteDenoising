from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

import torch

from .engine import _write_json, _write_rows
from .latency_audit import run_latency_audit
from .paper_evidence import CORE_VARIANTS, EVIDENCE_ROOT, load_verified_freeze, paper_tasks


EXPECTED_DEVICE = "NVIDIA GeForce RTX 4060 Laptop GPU"
BASELINE = "paper_fullres_b8d3_mse"
CANDIDATE = "paper_subpixel_b8d3_mse"


def _condition_tasks() -> dict[tuple[str, str], list]:
    tasks = [task for task in paper_tasks("core") if task.seed == 0]
    grouped: dict[tuple[str, str], list] = {}
    for task in tasks:
        grouped.setdefault((task.recording, task.window), []).append(task)
    for condition, rows in grouped.items():
        if {task.variant for task in rows} != set(CORE_VARIANTS):
            raise ValueError(f"Incomplete latency model set for {condition}")
    if len(grouped) != 9:
        raise ValueError(f"Expected nine latency conditions, found {len(grouped)}")
    return grouped


def run_paper_latency(device: str = "cuda") -> dict[str, Any]:
    selected_device = torch.device(device)
    if selected_device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("paper-latency requires CUDA")
    device_name = torch.cuda.get_device_name(selected_device)
    if device_name != EXPECTED_DEVICE:
        raise RuntimeError(f"paper-latency requires {EXPECTED_DEVICE}; found {device_name}")
    grouped = _condition_tasks()
    load_verified_freeze([task for rows in grouped.values() for task in rows])
    raw_root = EVIDENCE_ROOT / "latency" / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    table_rows = []
    for (recording, window), tasks in sorted(grouped.items()):
        for batch_size in (1, 4):
            output = raw_root / f"{recording}_{window}_batch{batch_size}.json"
            if output.is_file():
                result = json.loads(output.read_text(encoding="utf-8"))
            else:
                protocol = {
                    "enabled": True,
                    "batch_size": batch_size,
                    "warmup": 100,
                    "repeats": 500,
                    "precision": "fp16",
                    "channels_last": True,
                    "source": "paper_evidence_v1 locked device protocol",
                }
                result = run_latency_audit(
                    tasks,
                    device=device,
                    rounds=5,
                    baseline=BASELINE,
                    protocol_override=protocol,
                )
                result.update({"recording": recording, "window": window, "device_required": EXPECTED_DEVICE})
                _write_json(result, output)
            for aggregate in result["aggregates"]:
                records = [row for row in result["records"] if row["variant"] == aggregate["variant"]]
                table_rows.append({
                    "recording": recording,
                    "window": window,
                    "batch_size": batch_size,
                    "variant": aggregate["variant"],
                    **aggregate,
                    "median_throughput_frames_per_second": statistics.median(
                        float(row["throughput_frames_per_second"]) for row in records
                    ),
                    "median_peak_allocated_memory_mb": statistics.median(
                        float(row["peak_allocated_memory_mb"]) for row in records
                    ),
                })
    latency_root = EVIDENCE_ROOT / "latency"
    _write_rows(table_rows, latency_root / "latency_condition_aggregates.csv")
    batch1 = [row for row in table_rows if row["batch_size"] == 1]
    by_key = {(row["recording"], row["window"], row["variant"]): row for row in batch1}
    ratios = []
    for recording, window in sorted({(row["recording"], row["window"]) for row in batch1}):
        candidate = by_key[(recording, window, CANDIDATE)]
        baseline = by_key[(recording, window, BASELINE)]
        ratios.append({
            "recording": recording,
            "window": window,
            "candidate": CANDIDATE,
            "baseline": BASELINE,
            "p50_ratio": float(candidate["median_of_round_p50_ms"]) / float(baseline["median_of_round_p50_ms"]),
        })
    _write_rows(ratios, latency_root / "subpixel_vs_fullres_batch1.csv")
    passed_conditions = sum(float(row["p50_ratio"]) <= 0.75 for row in ratios)
    summary = {
        "schema_version": 1,
        "device": device_name,
        "conditions": 9,
        "models": 5,
        "rounds_per_protocol": 5,
        "batch1_p50_ratios": ratios,
        "speed_gate": {
            "passed": passed_conditions >= 8,
            "threshold_ratio": 0.75,
            "required_conditions": 8,
            "passing_conditions": passed_conditions,
        },
    }
    _write_json(summary, latency_root / "latency_summary.json")
    return summary

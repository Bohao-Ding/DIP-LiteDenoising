from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from .analysis import (
    aggregate_comparisons,
    aggregate_frame_comparisons,
    audit_result_coverage,
    compare_with_reference,
    load_summaries,
    paired_frame_comparisons,
)
from .orchestration import (
    PROJECT_ROOT,
    STAGE_CONFIGS,
    execute_stage,
    parse_task_indices,
    select_stage_tasks,
    stage_snapshot,
)
from .latency_audit import run_latency_audit
from .paper_evidence import (
    FREEZE_PATH,
    build_paper_report,
    evaluate_paper_tasks,
    freeze_paper_evidence,
    paper_tasks,
)
from .paper_latency import run_paper_latency
from .paths import activate_path_overrides
from .study import expand_tasks, load_stage_config, sha256_file
from .validation import audit_stage_artifacts


def _run_checks() -> None:
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    source_root = str(PROJECT_ROOT / "src")
    existing_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = source_root + (os.pathsep + existing_pythonpath if existing_pythonpath else "")
    command = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"]
    subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=True)
    stages = {}
    for name, path in STAGE_CONFIGS.items():
        tasks = expand_tasks(path, validate_paths=False)
        stages[name] = stage_snapshot(tasks)
    print(json.dumps({"status": "passed", "training_started": False, "stages": stages}, indent=2))


def _data_check(args: argparse.Namespace) -> None:
    tasks = paper_tasks("all", validate_paths=True)
    datasets = sorted({task.dataset_root for task in tasks})
    splits = sorted({task.split_manifest for task in tasks})
    payload = {
        "status": "passed",
        "training_started": False,
        "tasks": len(tasks),
        "datasets": datasets,
        "splits": [
            {"path": path, "sha256": sha256_file(path)}
            for path in splits
        ],
    }
    print(json.dumps(payload, indent=2))


def _run_stage_command(args: argparse.Namespace) -> None:
    if args.execute and args.no_path_check:
        raise ValueError("--no-path-check cannot be combined with --execute")
    tasks = select_stage_tasks(
        args.command,
        args.models,
        args.tasks,
        validate_paths=not args.no_path_check,
    )
    snapshot = stage_snapshot(tasks)
    print(json.dumps(snapshot, indent=2))
    if not args.execute:
        print("Dry run only. Add --execute to start training.")
        return
    result = execute_stage(tasks, device=args.device, resume=args.resume)
    print(json.dumps(result, indent=2))


def _results_root(tasks) -> Path:
    return Path(tasks[0].output_dir).parents[3]


def _report(args: argparse.Namespace) -> None:
    tasks = select_stage_tasks(args.stage, args.models, "all")
    config = load_stage_config(STAGE_CONFIGS[args.stage])
    reference_variant = str(config.get("reference_variant", "baseline_b8d3"))
    root = _results_root(tasks)
    summaries = load_summaries(root)
    selected_models = {task.variant for task in tasks}
    summaries = [row for row in summaries if row.get("variant") in selected_models]
    coverage = audit_result_coverage(tasks, summaries)
    comparisons = (
        compare_with_reference(summaries, reference_variant, strict=False)
        if summaries
        else []
    )
    frame_comparisons = (
        paired_frame_comparisons(summaries, reference_variant, strict=False)
        if summaries
        else []
    )
    report: dict[str, Any] = {
        "stage": args.stage,
        "reference_variant": reference_variant,
        "results_root": str(root),
        "evaluation_splits": sorted(
            {str(task.training.get("evaluation_split", "test")) for task in tasks}
        ),
        "coverage": coverage,
        "artifact_audit": audit_stage_artifacts(tasks),
        "paired_results": comparisons,
        "model_aggregates": aggregate_comparisons(comparisons),
        "frame_results_vs_reference": frame_comparisons,
        "frame_aggregates_vs_reference": aggregate_frame_comparisons(frame_comparisons),
    }
    if args.stage in {
        "targeted",
        "innovate",
        "confirm",
        "next",
        "next-confirm",
        "next-multi",
    }:
        versus_subpixel = compare_with_reference(summaries, "subpixel_residual_b8d3")
        report["paired_results_vs_subpixel"] = [
            row for row in versus_subpixel if row["variant"] != "baseline_b8d3"
        ]
        report["frame_results_vs_baseline"] = paired_frame_comparisons(summaries, "baseline_b8d3")
        report["frame_results_vs_subpixel"] = [
            row
            for row in paired_frame_comparisons(summaries, "subpixel_residual_b8d3")
            if row["variant"] != "baseline_b8d3"
        ]
    if args.stage in {"innovate", "next", "next-confirm", "next-multi"}:
        versus_naf = compare_with_reference(summaries, "subpixel_naf_lite_b8d3")
        report["paired_results_vs_naf_lite"] = [
            row
            for row in versus_naf
            if row["variant"] not in {"baseline_b8d3", "subpixel_residual_b8d3"}
        ]
        report["frame_results_vs_naf_lite"] = [
            row
            for row in paired_frame_comparisons(summaries, "subpixel_naf_lite_b8d3")
            if row["variant"] not in {"baseline_b8d3", "subpixel_residual_b8d3"}
        ]
    rendered = json.dumps(report, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
        print(f"Wrote report to {output.resolve()}")
    else:
        print(rendered)


def _latency_audit(args: argparse.Namespace) -> None:
    tasks = select_stage_tasks(args.stage, args.models)
    if args.recording:
        tasks = [task for task in tasks if task.recording == args.recording]
    if args.window:
        tasks = [task for task in tasks if task.window == args.window]
    if not tasks:
        raise ValueError("No latency-audit tasks match the requested recording and window")
    conditions = {(task.recording, task.window) for task in tasks}
    if len(conditions) != 1:
        raise ValueError(
            "Latency audit requires exactly one recording-window condition; "
            "provide both --recording and --window"
        )
    rounds = int(args.rounds) if args.rounds is not None else len(tasks)
    output = Path(args.output) if args.output else PROJECT_ROOT / "artifacts" / f"{args.stage}_latency_audit.json"
    plan = {
        "stage": args.stage,
        "models": [task.variant for task in tasks],
        "rounds": rounds,
        "device": args.device,
        "recording": tasks[0].recording,
        "window": tasks[0].window,
        "output": str(output.resolve()),
    }
    print(json.dumps(plan, indent=2))
    if not args.execute:
        print("Dry run only. Add --execute to start the latency audit.")
        return
    protocol_override = None
    if args.standalone_protocol:
        protocol_override = {
            "enabled": True,
            "batch_size": 1,
            "warmup": 100,
            "repeats": 500,
            "precision": "fp16",
            "channels_last": True,
            "source": "explicit standalone latency protocol",
        }
    result = run_latency_audit(
        tasks,
        device=args.device,
        rounds=rounds,
        protocol_override=protocol_override,
    )
    result["recording"] = tasks[0].recording
    result["window"] = tasks[0].window
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote latency audit to {output.resolve()}")


def _paper_freeze(args: argparse.Namespace) -> None:
    tasks = paper_tasks("all", validate_paths=True)
    print(json.dumps({"training_tasks_required": len(tasks), "freeze_path": str(FREEZE_PATH)}, indent=2))
    if not args.execute:
        print("Dry run only. Add --execute to validate all training artifacts and create the freeze.")
        return
    payload = freeze_paper_evidence()
    print(json.dumps({"freeze_hash": payload["freeze_hash"], "frozen_tasks": len(payload["tasks"])}, indent=2))


def _paper_evaluate(args: argparse.Namespace) -> None:
    tasks = paper_tasks(args.scope, validate_paths=True)
    selected = parse_task_indices(args.tasks, len(tasks))
    tasks = [tasks[index] for index in selected]
    print(json.dumps({"scope": args.scope, "selected_tasks": len(tasks), "device": args.device}, indent=2))
    if not args.execute:
        print("Dry run only. Add --execute after paper-freeze to evaluate matched segments.")
        return
    print(json.dumps(evaluate_paper_tasks(tasks, args.device), indent=2))


def _paper_report(args: argparse.Namespace) -> None:
    print(json.dumps({"evidence_root": str(FREEZE_PATH.parent), "requires_frozen_evaluations": 252}, indent=2))
    if not args.execute:
        print("Dry run only. Add --execute to build locked evidence tables and the report.")
        return
    print(json.dumps(build_paper_report(), indent=2))


def _paper_latency(args: argparse.Namespace) -> None:
    print(json.dumps({"device": args.device, "conditions": 9, "models": 5, "batch_sizes": [1, 4]}, indent=2))
    if not args.execute:
        print("Dry run only. Add --execute after paper-freeze to run the device audit.")
        return
    print(json.dumps(run_paper_latency(args.device), indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DIP-LiteDenoising evidence workflow")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("check", help="Run CPU-only code and configuration checks without data")

    data_check = subparsers.add_parser("data-check", help="Validate data roots and frozen splits")
    data_check.add_argument("--paths", required=True, help="machine-local path override JSON")

    paper_freeze = subparsers.add_parser("paper-freeze", help="Freeze completed validation-only paper runs")
    paper_freeze.add_argument("--paths", help="machine-local path override JSON")
    paper_freeze.add_argument("--execute", action="store_true")

    paper_evaluate = subparsers.add_parser("paper-evaluate", help="Evaluate frozen checkpoints on matched segments")
    paper_evaluate.add_argument("--scope", choices=("core", "mechanism", "all"), default="all")
    paper_evaluate.add_argument("--tasks", default="all", help="all, comma-separated indices, or ranges")
    paper_evaluate.add_argument("--device", default="cuda")
    paper_evaluate.add_argument("--paths", help="machine-local path override JSON")
    paper_evaluate.add_argument("--execute", action="store_true")

    paper_report = subparsers.add_parser("paper-report", help="Build locked paper evidence tables")
    paper_report.add_argument("--paths", help="machine-local path override JSON")
    paper_report.add_argument("--execute", action="store_true")

    paper_latency = subparsers.add_parser("paper-latency", help="Run the locked RTX 4060 Laptop audit")
    paper_latency.add_argument("--device", default="cuda")
    paper_latency.add_argument("--paths", help="machine-local path override JSON")
    paper_latency.add_argument("--execute", action="store_true")

    latency = subparsers.add_parser("latency-audit", help="Run an order-balanced checkpoint latency audit")
    latency.add_argument(
        "--stage",
        choices=("targeted", "innovate", "confirm", "next", "next-confirm"),
        default="targeted",
    )
    latency.add_argument("--models", nargs="+")
    latency.add_argument("--recording")
    latency.add_argument("--window", choices=("w1", "w2", "w4"))
    latency.add_argument("--device", default="cuda")
    latency.add_argument("--paths", help="machine-local path override JSON")
    latency.add_argument(
        "--rounds",
        type=int,
        help="complete cyclic rounds; defaults to the number of selected models",
    )
    latency.add_argument("--output")
    latency.add_argument(
        "--standalone-protocol",
        action="store_true",
        help="use batch=1, FP16, channels-last, 100 warmups and 500 repeats",
    )
    latency.add_argument("--execute", action="store_true")

    explicit_models = {"confirm", "multi", "next-confirm", "next-multi"}
    for stage in (
        "quick",
        "targeted",
        "innovate",
        "confirm",
        "multi",
        "next",
        "next-confirm",
        "next-multi",
        "objective",
        "paper-core",
        "paper-mechanism",
    ):
        stage_parser = subparsers.add_parser(stage, help=f"Plan or execute the {stage} stage")
        stage_parser.add_argument("--models", nargs="+", required=stage in explicit_models)
        stage_parser.add_argument("--tasks", default="all", help="all, comma-separated indices, or ranges")
        stage_parser.add_argument("--device")
        stage_parser.add_argument("--paths", help="machine-local path override JSON")
        stage_parser.add_argument(
            "--no-path-check",
            action="store_true",
            help="enumerate tasks without requiring local data; execution is forbidden",
        )
        stage_parser.add_argument("--resume", action="store_true")
        stage_parser.add_argument("--execute", action="store_true")

    report = subparsers.add_parser("report", help="Audit result coverage and paired comparisons")
    report.add_argument("stage", choices=tuple(STAGE_CONFIGS))
    report.add_argument("--models", nargs="+")
    report.add_argument("--output")
    report.add_argument("--paths", help="machine-local path override JSON")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        activate_path_overrides(getattr(args, "paths", None))
        if args.command == "check":
            _run_checks()
        elif args.command == "data-check":
            _data_check(args)
        elif args.command == "latency-audit":
            _latency_audit(args)
        elif args.command == "report":
            _report(args)
        elif args.command == "paper-freeze":
            _paper_freeze(args)
        elif args.command == "paper-evaluate":
            _paper_evaluate(args)
        elif args.command == "paper-report":
            _paper_report(args)
        elif args.command == "paper-latency":
            _paper_latency(args)
        else:
            _run_stage_command(args)
    except (ValueError, RuntimeError, FileNotFoundError, FileExistsError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()

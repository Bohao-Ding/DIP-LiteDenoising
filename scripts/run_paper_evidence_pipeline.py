from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from deepinterpolation_improve.experiments.paper_evidence_v1.engine import _write_json  # noqa: E402
from deepinterpolation_improve.experiments.paper_evidence_v1.orchestration import (  # noqa: E402
    execute_stage,
    select_stage_tasks,
)
from deepinterpolation_improve.experiments.paper_evidence_v1.paper_evidence import (  # noqa: E402
    EVIDENCE_ROOT,
    FREEZE_PATH,
    build_paper_report,
    evaluate_paper_tasks,
    freeze_paper_evidence,
    load_verified_freeze,
    paper_tasks,
)
from deepinterpolation_improve.experiments.paper_evidence_v1.paper_latency import (  # noqa: E402
    run_paper_latency,
)
from deepinterpolation_improve.experiments.paper_evidence_v1.paths import (  # noqa: E402
    activate_path_overrides,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Resume the locked paper-evidence pipeline")
    parser.add_argument("--paths", required=True, help="machine-local path override JSON")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rscript", help="optional Rscript executable for regenerating Figure 6")
    parser.add_argument("--r-library", help="optional existing R package library")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    activate_path_overrides(args.paths)
    stages = {}
    stages["paper_core"] = execute_stage(
        select_stage_tasks("paper-core"), device=args.device, resume=True
    )
    stages["paper_mechanism"] = execute_stage(
        select_stage_tasks("paper-mechanism"), device=args.device, resume=True
    )
    if FREEZE_PATH.is_file():
        freeze = load_verified_freeze(paper_tasks("all"))
    else:
        freeze = freeze_paper_evidence()
    stages["evaluation"] = evaluate_paper_tasks(paper_tasks("all"), args.device)
    stages["latency"] = run_paper_latency(args.device)
    stages["report"] = build_paper_report()
    figure_generated = False
    if args.rscript:
        rscript = Path(args.rscript).resolve()
        if not rscript.is_file():
            raise FileNotFoundError(rscript)
        environment = dict(os.environ)
        if args.r_library:
            r_library = Path(args.r_library).resolve()
            if not r_library.is_dir():
                raise FileNotFoundError(r_library)
            environment["R_LIBS_USER"] = str(r_library)
        subprocess.run(
            [str(rscript), str(PROJECT_ROOT / "scripts" / "plot_paper_evidence_figure6.R")],
            cwd=PROJECT_ROOT,
            env=environment,
            check=True,
        )
        figure_generated = True
    summary = {
        "schema_version": 1,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "freeze_hash": freeze["freeze_hash"],
        "stages": stages,
        "figure6_generated": figure_generated,
    }
    EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
    _write_json(summary, EVIDENCE_ROOT / "pipeline_summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

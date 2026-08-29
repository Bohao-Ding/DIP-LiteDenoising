from __future__ import annotations

import pytest

from deepinterpolation_improve.experiments.paper_evidence_v1.cli import build_parser


def test_simple_launcher_defaults_to_dry_run() -> None:
    args = build_parser().parse_args(["quick"])
    assert args.command == "quick"
    assert args.execute is False
    assert args.tasks == "all"
    innovate = build_parser().parse_args(["innovate"])
    assert innovate.execute is False
    objective = build_parser().parse_args(["objective"])
    assert objective.execute is False
    assert objective.tasks == "all"
    paper_core = build_parser().parse_args(["paper-core"])
    assert paper_core.execute is False
    assert paper_core.tasks == "all"
    assert paper_core.no_path_check is False
    portable_core = build_parser().parse_args(
        ["paper-core", "--paths", "paths.local.json", "--no-path-check"]
    )
    assert portable_core.paths == "paths.local.json"
    assert portable_core.no_path_check is True
    data_check = build_parser().parse_args(
        ["data-check", "--paths", "paths.local.json"]
    )
    assert data_check.paths == "paths.local.json"
    paper_freeze = build_parser().parse_args(["paper-freeze"])
    assert paper_freeze.execute is False
    paper_evaluate = build_parser().parse_args(["paper-evaluate", "--scope", "core"])
    assert paper_evaluate.scope == "core"
    assert paper_evaluate.execute is False
    paper_latency = build_parser().parse_args(["paper-latency"])
    assert paper_latency.execute is False
    assert paper_latency.device == "cuda"
    latency = build_parser().parse_args(["latency-audit", "--stage", "innovate"])
    assert latency.rounds is None
    assert latency.standalone_protocol is False
    standalone = build_parser().parse_args(
        ["latency-audit", "--stage", "next", "--standalone-protocol"]
    )
    assert standalone.standalone_protocol is True
    confirm_latency = build_parser().parse_args(
        [
            "latency-audit",
            "--stage",
            "confirm",
            "--recording",
            "03.00",
            "--window",
            "w4",
        ]
    )
    assert confirm_latency.recording == "03.00"
    assert confirm_latency.window == "w4"


def test_confirm_and_multi_require_explicit_model_selection() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["confirm"])
    with pytest.raises(SystemExit):
        parser.parse_args(["multi"])
    args = parser.parse_args(["multi", "--models", "baseline_b8d3", "consensus_attention_b8d3"])
    assert args.models == ["baseline_b8d3", "consensus_attention_b8d3"]

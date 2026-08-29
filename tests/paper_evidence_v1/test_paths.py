from __future__ import annotations

import hashlib
import json
from pathlib import Path

from deepinterpolation_improve.experiments.paper_evidence_v1.paths import PATHS_ENV
from deepinterpolation_improve.experiments.paper_evidence_v1.study import expand_tasks


def test_machine_local_paths_are_resolved_and_recorded(
    tmp_path: Path, monkeypatch
) -> None:
    override = tmp_path / "paths.local.json"
    payload = {
        "schema_version": 1,
        "recording_roots": {
            "01.00": "data/neurofinder.01.00",
            "01.01": "data/neurofinder.01.01",
            "03.00": "data/neurofinder.03.00",
        },
        "split_pattern": "splits/{recording}/{window}.json",
    }
    override.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setenv(PATHS_ENV, str(override))

    root = Path(__file__).resolve().parents[2]
    tasks = expand_tasks(
        root / "configs" / "paper_evidence_v1" / "core.json",
        validate_paths=False,
    )

    digest = hashlib.sha256(override.read_bytes()).hexdigest()
    assert len(tasks) == 225
    assert {task.path_override for task in tasks} == {str(override.resolve())}
    assert {task.path_override_sha256 for task in tasks} == {digest}
    first = tasks[0]
    assert Path(first.dataset_root) == (
        tmp_path / "data" / "neurofinder.01.00"
    ).resolve()
    assert Path(first.split_manifest) == (
        tmp_path / "splits" / "01.00" / "w1.json"
    ).resolve()


def test_paper_model_state_dict_round_trips_strictly() -> None:
    from deepinterpolation_improve.experiments.paper_evidence_v1.models import (
        create_variant,
    )

    source = create_variant("paper_subpixel_b8d3_mse", 4)
    destination = create_variant("paper_subpixel_b8d3_mse", 4)
    keys = tuple(source.state_dict())
    assert keys
    assert keys == tuple(destination.state_dict())
    destination.load_state_dict(source.state_dict(), strict=True)

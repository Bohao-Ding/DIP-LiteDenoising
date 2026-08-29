"""Structural candidates for DeepInterpolation-style temporal prediction."""

from .models import create_variant, registered_variants, variant_spec
from .paper_evidence import (
    build_paper_report,
    evaluate_paper_tasks,
    freeze_paper_evidence,
    load_verified_freeze,
    paper_tasks,
)

__all__ = [
    "build_paper_report",
    "create_variant",
    "evaluate_paper_tasks",
    "freeze_paper_evidence",
    "load_verified_freeze",
    "paper_tasks",
    "registered_variants",
    "variant_spec",
]

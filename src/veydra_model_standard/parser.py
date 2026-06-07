"""Canonical parser API for Veydra model analysis.

This module is the public package boundary for parser/runtime analysis helpers.
It is intentionally self-contained within the veydra-model-standard package so
cloud services and local consumers can rely on one standalone distribution.
"""

from __future__ import annotations

from typing import Any, Dict

from .runtime_diagram_parser import (
    build_flow_and_feedback_from_source_map,
    build_flow_diagram_from_source_map,
    build_feedback_loops_from_flow_diagram,
)
from .veydra_ast_parser import (
    VeydraModelASTParser,
    parse_veydra_model_directory,
)


def analyze_structure_from_source_map(file_map: Dict[str, str], max_loop_length: int = 8) -> Dict[str, Any]:
    """Run canonical runtime parsing over in-memory source files."""
    return build_flow_and_feedback_from_source_map(file_map, max_loop_length=max_loop_length)


__all__ = [
    "VeydraModelASTParser",
    "parse_veydra_model_directory",
    "build_flow_diagram_from_source_map",
    "build_feedback_loops_from_flow_diagram",
    "build_flow_and_feedback_from_source_map",
    "analyze_structure_from_source_map",
]
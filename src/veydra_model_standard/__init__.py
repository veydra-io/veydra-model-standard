"""Veydra Model Standard package."""

from .base import (
    VeydraModelStandard,
    Submodel,
    SimulationContext,
    META_KEYS,
    DEFAULT_SUMMARY_STATS,
    compute_summary_stats,
    build_summary_table,
    build_stacked_table,
    apply_time_window,
)
from .parser import (
    VeydraModelASTParser,
    parse_veydra_model_directory,
    build_flow_diagram_from_source_map,
    build_feedback_loops_from_flow_diagram,
    build_flow_and_feedback_from_source_map,
    analyze_structure_from_source_map,
)

__version__ = "1.1.0"
__all__ = [
    "VeydraModelStandard",
    "Submodel",
    "SimulationContext",
    "META_KEYS",
    "DEFAULT_SUMMARY_STATS",
    "compute_summary_stats",
    "build_summary_table",
    "build_stacked_table",
    "apply_time_window",
    "VeydraModelASTParser",
    "parse_veydra_model_directory",
    "build_flow_diagram_from_source_map",
    "build_feedback_loops_from_flow_diagram",
    "build_flow_and_feedback_from_source_map",
    "analyze_structure_from_source_map",
]
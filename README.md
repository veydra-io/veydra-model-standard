# Veydra Model Standard

Standalone runtime and parser package for Veydra system dynamics models.

## Why This Repository Exists

This package defines a consistent foundation for model runtime behavior and structural analysis so model execution, parser outputs, and integration surfaces stay aligned across environments.

The goal is to make model behavior reproducible and inspectable in:

- local development workflows
- cloud execution services
- assistant-driven tooling and MCP integrations

## Objectives

- Establish one canonical package for runtime model contracts and parser logic
- Keep simulation and structure analysis outputs consistent across tools
- Support production cloud execution with predictable model interfaces
- Reduce parser drift between environments by centralizing shared logic

## What Sets This Apart

- Code-first canonical source of truth, rather than diagram-first authoring as the primary artifact
- Runtime and parser parity from the same package, instead of separate translation layers between diagrams and execution
- Flexible primitives with explicit contracts, avoiding tightly opinionated framework constraints on model structure
- Structured introspection outputs designed for automation, cloud services, and assistant tooling

## Features

- Standardized model runtime interface
- Built-in variable and parameter handling
- Simulation context and multi-scenario utilities
- Runtime structure parser helpers for flow diagrams and feedback loops
- AST-based model parsing utilities for model introspection
- Data formatting helpers for summaries and stacked outputs

## Installation

```bash
pip install veydra-model-standard
```

For development from local source:

```bash
pip install -e .
```

## Quick Start

```python
from veydra_model_standard import VeydraModelStandard

class MyModel(VeydraModelStandard):
    @classmethod
    def auto_discover_variables(cls):
        return {
            "demo.population": {
                "name": "Population",
                "category": "stock",
                "default": 1000,
                "units": "people",
                "description": "Total population stock",
            },
            "demo.growth_rate": {
                "name": "Growth rate",
                "category": "parameter",
                "default": 5,
                "units": "people per step",
                "description": "Linear growth per time step",
            }
        }

    def run_simulation(self, params):
        variables = self.auto_discover_variables()
        resolved = self._resolve_parameters(self._clean_params(params), variables)

        duration = int(resolved.get("simulation.duration", 10))
        growth = float(resolved["demo.growth_rate"])

        time = list(range(duration + 1))
        series = [variables["demo.population"]["default"] + growth * t for t in time]

        return {
            "success": True,
            "time": time,
            "stocks": {"demo.population": series},
            "flows": {},
        }


model = MyModel(params={})

# Single run
single = model.run_simulation({"simulation.duration": 5, "demo.growth_rate": 8})
print(single["stocks"]["demo.population"][-1])  # 1040.0

# Multi-scenario run with summary output
batch = model.run_multi_scenario(
    {
        "__output_format__": "summary",
        "__summary_variable__": "demo.population",
        "scenarios": [
            {"id": "low", "params": {"demo.growth_rate": 4, "simulation.duration": 5}},
            {"id": "high", "params": {"demo.growth_rate": 9, "simulation.duration": 5}},
        ],
    }
)
print(batch["summary"]["rows"])
```

## Parser API Surface

```python
from veydra_model_standard.parser import (
    parse_veydra_model_directory,
    build_flow_and_feedback_from_source_map,
)
```

Use these APIs when you need canonical structure analysis from Python source files.

## Development

```bash
# Install in editable mode
cd veydra-model-standard
pip install -e .

# Run tests
python -m pytest tests/
```

## License

This project is licensed under the Apache License 2.0.

- Commercial use is allowed.
- Veydra retains copyright ownership of this codebase.
- See [LICENSE](LICENSE) for full terms.
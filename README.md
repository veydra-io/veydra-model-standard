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

- One package for both runtime primitives and parser APIs
- Designed for cloud and local parity, not just notebook-only experimentation
- Supports direct model introspection workflows used by higher-level assistants
- Uses explicit, structured outputs intended for automation and downstream tooling

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
    def __init__(self, params):
        super().__init__(params)

    @classmethod
    def auto_discover_variables(cls):
        return {
            "stock.population": {
                "name": "Population",
                "category": "stock",
                "default": 1000,
                "units": "people",
                "description": "Total population stock",
            }
        }
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